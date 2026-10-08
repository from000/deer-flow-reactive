"""Run-explanation surface for `ReactiveGraph`.

Split out of ``graph.py`` to keep every engine module under the project's
1500-line contract. These methods answer "why did the engine do that?" for the
latest run: causal trace export, decision summaries, conflict explanation, and
the cost-avoided estimate. They read the graph's own recorded state
(``_trace`` / ``_conflict`` / ``_last_run_id`` plus the Driver's durable log)
and never change execution.
"""

from __future__ import annotations

import json
from typing import Any

from reactivegraph.errors import GraphBuildError


class ExplainMixin:
    """Mixed into :class:`ReactiveGraph`; relies on its documented attributes.

    The attributes below are declared here (not implemented) so a type checker
    validates this module standalone while the single owner of the state stays
    :class:`ReactiveGraph`.
    """

    definition: Any
    host: Any
    _trace: list[dict[str, Any]]
    _conflict: dict[str, Any] | None
    _last_run_id: str | None
    _computed_invalidations: dict[str, tuple[str, ...]]

    @staticmethod
    def _causal_trace_event(item: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        """Overridden by :class:`ReactiveGraph` (kept here for typing)."""
        raise NotImplementedError

    def export_trace(self, format: str = "json") -> Any:
        """Export the latest run as causal-trace v1 JSON or Graphviz DOT.

        The Python binding normalizes its in-memory `_trace` (fallback segment
        events or Driver span events) to the stable
        ``reactivegraph.causal-trace.v1`` schema. DOT is a lossy visualization;
        JSON is the canonical interchange format.
        """
        run_id = self._last_run_id
        if not isinstance(run_id, str):
            run_id = "latest"
        events = [self._causal_trace_event(item) for item in self._trace]
        trace = {"protocol": "reactivegraph.causal-trace.v1", "runId": run_id, "events": events}
        if format == "json":
            return trace
        if format != "dot":
            raise GraphBuildError(f"unsupported trace format: {format!r}")
        lines = [f"digraph {json.dumps(run_id)} {{", "  rankdir=LR;"]
        previous: str | None = None
        for index, event in enumerate(events, start=1):
            label = event.get("taskId") or event.get("selector") or event["kind"]
            node = f"e{index}"
            node_label = f"{index}: {event['kind']}:{label}"
            lines.append(f"  {node} [label={json.dumps(node_label)}, shape=box];")
            if previous is not None:
                lines.append(f"  {previous} -> {node};")
            previous = node
        lines.append("}")
        return "\n".join(lines) + "\n"

    def explain_run(self, run_id: str | None = None) -> dict[str, Any]:
        """Explain the most recent run's execution decisions.

        The fallback trace stores the latest run only. Driver ``stream(trace=True)``
        populates the same normalized span trace, so this API works for both
        execution modes. The summary counts executed tasks, pure skips,
        computed cache hits, and errors.

        With ``run_id`` and a durable Driver log, the query is historical and
        read-only; it works after process restart. Current durable records
        cover committed transactions; skip/computed decisions remain in-memory
        and the result says ``decisionsPersisted: false``.
        """
        if run_id is not None:
            if self.host is None:
                raise GraphBuildError("historical explain requires a DriverHost with durable log")
            result = self.host.request_internal("TRACE_QUERY", {"runId": run_id})
            return result.get("result", {}) if isinstance(result, dict) else {}
        task_ids = {item.get("task") for item in self._trace if isinstance(item.get("task"), str)}
        executed = sum(
            1
            for task_id in task_ids
            if any(
                item.get("event") == f"task:{task_id}" and not item.get("error")
                for item in self._trace
            )
        )
        skipped = sum(1 for item in self._trace if item.get("event") in {"cache_hit", "skip"})
        cache_hits = sum(1 for item in self._trace if item.get("event") == "cache_hit") + sum(
            1
            for item in self._trace
            if isinstance(item.get("event"), str)
            and item["event"].startswith("computed:")
            and item["event"].endswith(":hit")
        )
        errors = sum(1 for item in self._trace if item.get("error"))
        return {
            "events": list(self._trace),
            "executed": executed,
            "skipped": skipped,
            "cache_hits": cache_hits,
            "errors": errors,
        }

    def explain_conflict(self) -> dict[str, Any]:
        """Explain the most recent rejected parallel write conflict.

        The Driver attaches structured conflict metadata to the RUN error:
        loser, winner, path, and both declared and actual write sets. The
        decision is retained when the run raises, so callers can inspect it
        immediately after catching the execution error.
        """
        if self._conflict is None:
            raise GraphBuildError("no conflict decision found")
        conflict = dict(self._conflict)
        return {
            "kind": conflict.get("kind", "write_conflict"),
            "reason": conflict.get("reason", "unknown"),
            "task": conflict.get("taskId"),
            "winner": conflict.get("winnerTaskId"),
            "path": conflict.get("path"),
            "declared_writes": conflict.get("declaredWrites", {}),
            "actual_writes": conflict.get("actualWrites", {}),
        }

    def why_invalidated(self, selector_id: str) -> dict[str, Any]:
        """Explain why a computed selector was evaluated for the latest run."""
        computed = self.definition._computed_by_id.get(selector_id)
        if computed is None:
            raise GraphBuildError(f"unknown computed selector: {selector_id!r}")
        reads = self._computed_invalidations.get(selector_id)
        if reads is None:
            raise GraphBuildError(f"no invalidation decision found for {selector_id!r}")
        return {
            "selector": selector_id,
            "kind": "computed",
            "reason": "read_set_changed",
            "reads": list(reads),
        }

    def why_skipped(self, task_id: str) -> dict[str, Any]:
        """Explain why a pure task or computed selector was skipped."""
        for item in self._trace:
            if item.get("task") == task_id and item.get("event") in {"cache_hit", "skip"}:
                reason = item.get("reason", "fingerprint_unchanged")
                return {
                    "task": task_id,
                    "kind": "effect" if reason == "receipt_present" else "pure",
                    "reason": reason,
                }
            if item.get("computed_id") == task_id:
                return {
                    "selector": task_id,
                    "kind": "computed",
                    "reason": "read_set_unchanged",
                }
        raise GraphBuildError(f"no skip decision found for {task_id!r}")

    def cost_saved(self) -> dict[str, Any]:
        """Estimate handler, token, and USD costs avoided by reactive caching.

        Estimates are supplied per task and remain explicitly estimates; unknown
        cost is never silently treated as measured spend. A task with no cost
        metadata sets ``cost_known: False`` so reports remain honest.
        """
        explanation = self.explain_run()
        skipped_tasks = {
            item["task"]
            for item in self._trace
            if item.get("event") in {"cache_hit", "skip"}
            and isinstance(item.get("task"), str)
        }
        llm_calls_saved = sum(
            1
            for task_id in skipped_tasks
            if isinstance(task_id, str) and task_id.startswith(("llm", "model", "chat"))
        )
        tokens_in = sum(
            self.definition._task_by_id[task_id].estimated_tokens_in
            for task_id in skipped_tasks
            if task_id in self.definition._task_by_id
        )
        tokens_out = sum(
            self.definition._task_by_id[task_id].estimated_tokens_out
            for task_id in skipped_tasks
            if task_id in self.definition._task_by_id
        )
        usd = sum(
            self.definition._task_by_id[task_id].estimated_usd
            for task_id in skipped_tasks
            if task_id in self.definition._task_by_id
        )
        cost_known = all(
            bool(self.definition._task_by_id[task_id].estimated_usd > 0)
            for task_id in skipped_tasks
            if task_id in self.definition._task_by_id
        )
        return {
            "executions": explanation["executed"],
            "skipped": explanation["skipped"],
            "llm_calls_saved": llm_calls_saved,
            "tools_saved": sum(
                1
                for task_id in skipped_tasks
                if isinstance(task_id, str) and task_id.startswith("tool")
            ),
            "tokens_in_saved": tokens_in,
            "tokens_out_saved": tokens_out,
            "estimated_usd_saved": round(usd, 6),
            "cost_known": cost_known,
        }

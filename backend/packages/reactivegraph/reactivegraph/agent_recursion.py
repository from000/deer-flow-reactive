"""Recursion-limit accounting for the agent factory.

LangGraph charges one super-step per middleware/model/tools node and raises
``GraphRecursionError`` when the caller's ``recursion_limit`` is exhausted. The
engine reproduces that accounting in the compatibility layer because the
Driver's own ``maxRuns`` budget is a different, coarser bound.

Keeping this cluster here means ``create_agent`` only wires the graph; the
budget rules (validation, per-node charging, the "already exhausted" marker and
the upstream error shape) are testable without building an agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from reactivegraph.agent_state import (
    EXHAUSTED_KEY,
    MAX_RECURSION_LIMIT,
    RECURSION_LIMIT_KEY,
    STEP_COUNT_KEY,
)
from reactivegraph.errors import (
    ErrorCode,
    GraphRecursionError,
    create_error_message,
)
from reactivegraph.messages import (
    AIMessage,
    AnyMessage,
    convert_to_messages,
    message_to_foreign_dict,
)

# ``create_agent`` historically read these with the leading underscore; keep
# the aliases so the moved bodies stay byte-identical to their originals.
_MAX_RECURSION_LIMIT = MAX_RECURSION_LIMIT
_EXHAUSTED_KEY = EXHAUSTED_KEY
_RECURSION_LIMIT_KEY = RECURSION_LIMIT_KEY
_STEP_COUNT_KEY = STEP_COUNT_KEY

def _validate_recursion_limit(limit: Any) -> Any:
    """Validate a run's ``recursion_limit`` exactly the way upstream does.

    ``0``/negative raise ``ValueError``; a non-numeric value raises ``TypeError``
    from the comparison itself, which is what LangGraph's own check does.
    ``None`` means "use the compiled default" and is resolved by the caller.

    One bound is *ours*, not upstream's: the compiled task budgets and the
    fallback event guard are fixed wire values, so a run whose limit exceeds
    them would be truncated silently instead of raising. Fail closed instead.
    """
    if limit is None:
        return None
    if limit < 1:
        raise ValueError("recursion_limit must be at least 1")
    if limit > _MAX_RECURSION_LIMIT:
        raise ValueError(
            f"recursion_limit {limit!r} exceeds ReactiveGraph's compiled ceiling "
            f"of {_MAX_RECURSION_LIMIT}; a larger limit would be silently "
            "truncated by the engine's task budget"
        )
    return limit


def _run_recursion_limit(config: dict | None, default: Any) -> Any:
    """Resolve a run's limit: an explicit config value wins over the default.

    ``config['recursion_limit'] = None`` keeps the compiled default, matching
    upstream (probe-verified against langgraph 1.2.9).
    """
    limit = default
    if config:
        raw = config.get("recursion_limit")
        if raw is not None:
            limit = raw
    return _validate_recursion_limit(limit)


def _run_exhausted(state: Any) -> bool:
    """Whether a finished run spent its whole budget without stopping in time.

    Two independent signals, because the two paths notice exhaustion in
    different places: a task that was *refused* mid-run records
    ``_EXHAUSTED_KEY``; a run that converged normally still owes the
    terminating super-step upstream charges while discovering there is no more
    work (``count + 1``).
    """
    if not isinstance(state, dict):
        return False
    if state.get(_EXHAUSTED_KEY):
        return True
    limit = state.get(_RECURSION_LIMIT_KEY)
    if limit is None:
        return False
    count = int(state.get(_STEP_COUNT_KEY) or 0)
    return count + 1 > limit


def _graph_recursion_error(limit: Any) -> Exception:
    """Upstream's exact exhaustion message, troubleshooting link included."""
    error_type = cast("type[Exception]", GraphRecursionError)
    return error_type(
        create_error_message(
            message=(
                f"Recursion limit of {limit} reached without hitting a stop "
                "condition. You can increase the limit by setting the "
                "`recursion_limit` config key."
            ),
            error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
        )
    )


class _RecursionBudgetExceeded(Exception):
    """Internal control flow: this run spent more super-steps than allowed.

    It is deliberately *not* :class:`GraphRecursionError`: the node that would
    have run must not run, but the nodes that already ran have to keep their
    writes, because LangGraph commits every super-step before it notices the
    limit. The task catches this, returns a stopping outcome, and the
    compatibility layer raises the real error once the graph has stopped.
    """

    def __init__(self, node: str, limit: Any) -> None:
        super().__init__(node)
        self.node = node
        self.limit = limit
        # Filled in by :meth:`AgentGraph._compile._run_hook` when the refusal
        # lands part-way through a middleware chain: upstream compiles one node
        # per middleware instance, so the instances that already ran keep their
        # writes. Empty when the very first node was refused.
        self.partial: dict[str, Any] = {}


class _StepBudget:
    """LangGraph-compatible super-step accounting for one run.

    ``recursion_limit`` counts *super-steps*, not turns. Upstream advances its
    step counter once per loop iteration — including the final iteration that
    finds no runnable task — and refuses to prepare tasks once the counter
    exceeds the limit, so a graph that executes exactly ``limit`` nodes still
    raises. Exhaustion is only observable *after* those nodes committed.

    The engine runs one task per node, so this reproduces that arithmetic:
    every node is charged in upstream's order, and the terminating super-step
    is charged once the graph has stopped (``finish``). The spent count rides
    the graph state, because a task re-entry starts from committed values and
    would otherwise restart the count on every round.
    """

    def __init__(self, state: dict, default_limit: Any) -> None:
        raw = state.get(_RECURSION_LIMIT_KEY)
        self.limit = default_limit if raw is None else raw
        # ``count`` is the *tick index* upstream is currently on. The first
        # tick (index 0) is the one that seeds the input and prepares the
        # entry node, so an untouched run is at 0 and each node advances it.
        raw_count = state.get(_STEP_COUNT_KEY)
        self.count = 0 if raw_count is None else int(raw_count)

    def charge(self, node: str) -> None:
        """Advance one tick to prepare *node*, refusing if the budget is spent.

        ``SyncPregelLoop.tick`` runs while ``self.step <= self.stop`` where
        ``stop = recursion_limit`` for a fresh run, so tick *n* prepares a node
        iff ``n <= limit``. The node prepared at tick ``limit`` therefore still
        executes; only the tick after it is refused.
        """
        self.count += 1
        if self.count > self.limit:
            raise _RecursionBudgetExceeded(node, self.limit)

    def exhausted_update(self) -> dict[str, Any]:
        """The bookkeeping patch a task returns when its budget ran out."""
        return {_STEP_COUNT_KEY: self.count, _EXHAUSTED_KEY: True}

    def spent_update(self) -> dict[str, Any]:
        """The bookkeeping patch every completed task commits."""
        return {_STEP_COUNT_KEY: self.count}


class _BoundModel:
    """Wraps a caller-supplied chat model with the bind/bind_tools surface.

    Only the calls the agent loop actually makes are forwarded, so a fake model
    in tests needs the same two methods a real chat model exposes.
    """

    def __init__(self, model: Any, tools: Sequence[Any]) -> None:
        self._model = model
        self._tools = list(tools)
        self._bound: Any = None

    def bind(self) -> Any:
        """Return a callable bound to the given model, tools and options."""
        if self._tools and hasattr(self._model, "bind_tools"):
            self._bound = self._model.bind_tools(self._tools)
        elif hasattr(self._model, "bind"):
            self._bound = self._model.bind()
        else:  # pragma: no cover - defensive; fakes implement one of the two
            self._bound = self._model
        return self._bound

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        """Run *event* (or the graph) synchronously and return the result."""
        bound = self._bound if self._bound is not None else self.bind()
        # The model belongs to the host's world (e.g. a real LangChain chat
        # model), and its message coercion rejects our classes outright.
        # ``convert_to_messages`` accepts the flat ``{type, **dump}`` shape,
        # and adopting the reply keeps our reducers' isinstance checks valid.
        payload = [message_to_foreign_dict(message) for message in messages]
        return convert_to_messages([bound.invoke(payload)])[0]



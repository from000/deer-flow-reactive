"""Streaming adapter: LangGraph stream modes -> native engine frames.

The compatibility layer exposes LangGraph's ``stream``/``astream`` mode
language (``values``/``updates``/``messages``/``custom``/``tasks``/``debug``)
on top of the engine's raw frames. Keeping that translation here — rather than
inside the agent factory — means the factory only has to build and run the
graph, while the frame-to-chunk contract lives next to its own tests.

``_StreamCommitter`` persists supersteps as frames arrive so a host that stops
consuming mid-stream still leaves a recoverable thread.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from typing import Any

from reactivegraph.agent_state import PRIVATE_STATE_KEYS as _PRIVATE_STATE_KEYS

# ---------------------------------------------------------------------------
# Stream-mode adapter (LangGraph mode language -> native frames)
# ---------------------------------------------------------------------------

# Modes the real product asks for. ``messages-tuple`` is DeerFlow's public
# spelling of LangGraph's ``messages`` and is translated before we get here.
_SUPPORTED_STREAM_MODES = ("values", "updates", "messages", "custom", "tasks", "debug")


class UnsupportedStreamModeError(ValueError):
    """Raised when a caller asks for a stream mode we cannot honor."""

    def __init__(self, modes: Iterable[str]) -> None:
        self.modes = tuple(dict.fromkeys(modes))
        super().__init__(
            "Unsupported stream mode(s): " + ", ".join(self.modes)
        )


class _StreamCommitter:
    """Persist the run's supersteps while ``stream``/``astream`` is consumed.

    LangGraph's Pregel loop checkpoints after every superstep. The compatible
    streaming surface has to do the same even when the caller only asked for a
    narrower mode — a worker that requests ``custom`` still expects the turn to
    be recoverable — so the writer hangs off the *raw* ``values`` frames and is
    independent of ``stream_mode``.

    Writing as frames arrive (rather than in a ``finally``) matters because the
    host stops consuming on abort and leaves the generator open; a deferred
    write at that point never happens. Each write is chained onto the previous
    one via ``aget_head(stored)``, so the thread's lineage stays a chain rather
    than a set of orphaned roots.
    """

    def __init__(
        self,
        *,
        threads: Any,
        channels: Mapping,
        config: Mapping | None,
        head: Any,
        written: Sequence[str],
    ) -> None:
        self._threads = threads
        self._channels = channels
        self._config = config
        self._head = head
        self._written = list(written)
        self._first_commit = True

    @staticmethod
    def _state(frame: Any) -> dict[str, Any] | None:
        if not isinstance(frame, Mapping) or frame.get("eventType") != "values":
            return None
        state = (frame.get("payload") or {}).get("state")
        return state if isinstance(state, dict) and state else None

    def _touched(self, committed: Mapping) -> list[str]:
        """Channels to re-version for this superstep.

        The seeded input channels count only on the first commit: afterwards
        the adapter hands over the *full* state and the checkpointer derives
        what actually changed against its parent. Naming them again would
        re-version untouched channels on every frame.
        """
        if self._first_commit and self._written:
            return self._written
        return [key for key in committed if key in self._channels]

    def commit_frame(self, frame: Any) -> None:
        """Persist one engine frame to the thread checkpointer."""
        threads = self._threads
        state = self._state(frame)
        if threads is None or state is None:
            return
        committed = _public_state(state)
        stored = threads.commit(
            self._config,
            self._head,
            committed,
            written=self._touched(committed),
            metadata={"source": "loop", "step": 1, "parents": {}},
        )
        self._advance(stored, threads.get_head)

    async def acommit_frame(self, frame: Any) -> None:
        """Async twin of :meth:`commit_frame`."""
        threads = self._threads
        state = self._state(frame)
        if threads is None or state is None:
            return
        committed = _public_state(state)
        stored = await threads.acommit(
            self._config,
            self._head,
            committed,
            written=self._touched(committed),
            metadata={"source": "loop", "step": 1, "parents": {}},
        )
        await self._aadvance(stored, threads.aget_head)

    def _advance(self, stored: Any, read: Any) -> None:
        self._first_commit = False
        if stored is not None:
            self._head = read(stored)

    async def _aadvance(self, stored: Any, read: Any) -> None:
        self._first_commit = False
        if stored is not None:
            self._head = await read(stored)


def _normalize_stream_modes(
    stream_mode: str | Sequence[str] | None,
) -> tuple[tuple[str, ...], bool]:
    """Return ``(modes, as_list)``.

    ``as_list`` mirrors upstream's *shape* rule: a bare string (or ``None``)
    yields bare chunks, a list yields ``(mode, chunk)`` tuples — even when the
    list holds a single mode.
    """
    if stream_mode is None:
        return ("updates",), False
    if isinstance(stream_mode, str):
        raw = [stream_mode]
        as_list = False
    else:
        raw = list(stream_mode)
        as_list = True
    normalized = ["messages" if m == "messages-tuple" else str(m) for m in raw]
    unsupported = [m for m in normalized if m not in _SUPPORTED_STREAM_MODES]
    if unsupported:
        raise UnsupportedStreamModeError(unsupported)
    # Upstream de-duplicates while preserving first-seen order.
    return tuple(dict.fromkeys(normalized)), as_list


def _update_metadata(as_node: str | None) -> dict[str, Any]:
    """Upstream's ``update_state`` provenance metadata, plus the write node."""
    metadata: dict[str, Any] = {"source": "update", "step": 1, "parents": {}}
    if as_node:
        metadata["as_node"] = as_node
    return metadata


def _public_state(state: Any) -> dict[str, Any]:
    """Strip run-private bookkeeping keys from a returned state."""
    if not isinstance(state, dict):
        return state
    return {k: v for k, v in state.items() if k not in _PRIVATE_STATE_KEYS}


def _messages_delta(previous: Any, current: Any) -> list[Any]:
    """New messages appended between two ``values`` snapshots."""
    prev = previous.get("messages") if isinstance(previous, dict) else None
    cur = current.get("messages") if isinstance(current, dict) else None
    if not isinstance(cur, list):
        return []
    if not isinstance(prev, list):
        return list(cur)
    return list(cur[len(prev) :])


def _node_update(node: str, new_messages: list[Any], state: dict[str, Any]) -> dict[str, Any]:
    """Build one ``updates`` frame for *node* from its message delta."""
    update: dict[str, Any] = {}
    if new_messages:
        update["messages"] = new_messages
    for key, value in state.items():
        if key == "messages" or key in _PRIVATE_STATE_KEYS:
            continue
        update[key] = value
    return {node: update}


def _adapt_stream(
    raw: Iterator[dict],
    modes: tuple[str, ...],
    *,
    as_list: bool,
    integration: str,
) -> Iterator[Any]:
    """Translate native frames into the requested LangGraph stream modes.

    The fallback path reports ``task_end`` frames (authoritative task identity),
    the Driver path reports only ``values`` snapshots plus ``custom`` span
    frames. Both are handled here so a run yields the same mode chunks on
    either backend.
    """
    yield from _AdaptState(modes, as_list=as_list, integration=integration).feed(raw)


async def _aadapt_stream(
    raw: AsyncIterator[dict],
    modes: tuple[str, ...],
    *,
    as_list: bool,
    integration: str,
) -> AsyncIterator[Any]:
    """Async twin of :func:`_adapt_stream` — same semantics, no thread bridge."""
    state = _AdaptState(modes, as_list=as_list, integration=integration)
    async for frame in raw:
        for item in state.push(frame):
            yield item
    for item in state.finish():
        yield item


class _AdaptState:
    """Shared frame -> mode translation state for sync and async adapters."""

    def __init__(
        self, modes: tuple[str, ...], *, as_list: bool, integration: str
    ) -> None:
        self.modes = modes
        self.as_list = as_list
        self.integration = integration
        self.previous: dict[str, Any] = {}
        self.step = 0
        self.pending_task: str | None = None
        self.custom: list[Any] = []

    def feed(self, raw: Iterator[dict]) -> Iterator[Any]:
        """Yield mode chunks for every frame, then drain ``finish``."""
        for frame in raw:
            yield from self.push(frame)
        yield from self.finish()

    def finish(self) -> Iterator[Any]:
        """Flush buffered custom chunks at end-of-stream."""
        if "custom" in self.modes:
            for chunk in self.custom:
                yield _wrap("custom", chunk, self.as_list)
        self.custom = []

    def push(self, frame: dict) -> Iterator[Any]:
        """Translate one raw engine frame into zero or more mode chunks."""
        kind = str(frame.get("eventType") or "")
        handler = {
            "values": self._push_values,
            "task_start": self._push_task_start,
            "task_end": self._push_task_end,
            "custom": self._push_custom,
            "messages": self._push_message_frame,
        }.get(kind)
        if handler is None:
            return
        yield from handler(frame)

    def _push_values(self, frame: dict) -> Iterator[Any]:
        """``values``: emit the public state, message deltas and updates."""
        state = ((frame.get("payload") or {}).get("state")) or {}
        delta = _messages_delta(self.previous, state)
        first = not self.previous
        self.previous = state
        if "values" in self.modes:
            yield _wrap("values", _public_state(state), self.as_list)
        if not delta:
            return
        if "messages" in self.modes:
            for message in delta:
                self.step += 1
                yield _wrap(
                    "messages",
                    (message, self._metadata(message)),
                    self.as_list,
                )
        if "updates" in self.modes and not first:
            yield from self._push_update(delta, state)

    def _push_update(self, delta: list[Any], state: dict) -> Iterator[Any]:
        """Emit one ``updates`` chunk attributing *delta* to its node.

        ``task_end`` (fallback) gives the authoritative node; the Driver path
        has no task frames, so fall back to the message class — exact for this
        graph's model/tools split. Either way the payload is the *delta*,
        matching upstream, whose messages channel is reducer-appended.
        """
        self.step += 1
        node = self.pending_task or (
            "model" if _looks_like_model_message(delta) else "tools"
        )
        self.pending_task = None
        yield _wrap("updates", _node_update(node, delta, state), self.as_list)

    def _push_task_start(self, frame: dict) -> Iterator[Any]:
        """Remember the task that the immediately following values frame owns."""
        # Fallback path: the ``values`` frame for this task follows immediately
        # (fallback order is task_start → values → task_end), so this is the
        # authoritative attribution point.
        self.pending_task = str(frame.get("task") or "") or None
        return iter(())

    def _push_task_end(self, frame: dict) -> Iterator[Any]:
        self.pending_task = None
        return iter(())

    def _push_custom(self, frame: dict) -> Iterator[Any]:
        payload = frame.get("payload") or {}
        if str(payload.get("event", "")).startswith("span:"):
            # internal tracing frame, not a user custom chunk
            return iter(())
        self.custom.append(payload.get("payload"))
        return iter(())

    def _push_message_frame(self, frame: dict) -> Iterator[Any]:
        if "messages" in self.modes:
            payload = frame.get("payload") or {}
            yield _wrap("messages", (payload.get("message"), {}), self.as_list)

    def _metadata(self, message: Any) -> dict[str, Any]:
        return {
            "ls_integration": self.integration,
            "langgraph_step": self.step,
            "langgraph_node": "model"
            if message.__class__.__name__.startswith("AI")
            else "tools",
        }


def _wrap(mode: str, chunk: Any, as_list: bool) -> Any:
    return (mode, chunk) if as_list else chunk


def _looks_like_model_message(delta: list[Any]) -> bool:
    """Heuristic used only when the native frame carries no task identity."""
    return any(m.__class__.__name__.startswith("AI") for m in delta)



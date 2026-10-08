"""State-only ``StateGraph``: the surface hosts use for checkpoint mutation.

DeerFlow compiles a one-node graph whose only job is to apply
``update_state`` writes through a thread's channel table — rollback restore,
context compaction, and delta-resume linearization all do this. Those call
sites need exactly four things from the graph object:

* a channel table (``graph.channels``) and the schema it was compiled from
  (``graph.builder.schemas``), so the caller can tell reducer channels from
  last-value ones and wrap replacements in ``Overwrite``;
* ``update_state`` / ``aupdate_state``, folding writes through the reducers;
* ``get_state`` / ``aget_state`` and the history twins, materialising the
  stored checkpoint; and
* a ``checkpointer`` attribute the caller can set after compiling.

That is deliberately *not* a general graph runtime: there is no scheduling,
no routing, and no streaming. :meth:`CompiledStateGraph.invoke` runs the single
declared node once and commits the result, which is what "the mutation node
finishes immediately" means. Anything outside this surface raises
``NotImplementedError`` rather than pretending to work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from reactivegraph.channels import resolve_channels
from reactivegraph.thread_state import StateSnapshot, ThreadCheckpointer

__all__ = ("CompiledStateGraph", "StateGraph")

ENTRY_POINT_REQUIRED_MESSAGE = (
    "StateGraph has no entry point; call set_entry_point(node) before compile()"
)


def _adopt_messages(update: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a ``messages`` value as engine message objects.

    Host middleware returns LangChain's message classes; the reducer's
    ``isinstance`` checks only see ours, so adopt at the boundary or the write
    silently bypasses ``add_messages`` semantics.
    """
    from reactivegraph.messages import convert_to_messages

    messages = update.get("messages")
    if messages is None:
        return dict(update)
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        return dict(update)
    return {**update, "messages": list(convert_to_messages(messages))}


def _update_metadata(as_node: str | None) -> dict[str, Any]:
    """Upstream's ``update_state`` provenance metadata, plus the write node."""
    metadata: dict[str, Any] = {"source": "update", "step": 1, "parents": {}}
    if as_node:
        metadata["as_node"] = as_node
    return metadata


class _BuilderView:
    """``graph.builder`` — the schema table hosts read back."""

    def __init__(self, schemas: dict[type, type], channels: dict[str, Any]) -> None:
        self.schemas = schemas
        self.channels = channels


class CompiledStateGraph:
    """A compiled state-only graph bound to a checkpointer on first use."""

    def __init__(
        self,
        schema: type,
        nodes: Mapping[str, Any],
        entry_point: str,
        channels: dict[str, Any],
    ) -> None:
        self._schema = schema
        self._nodes = dict(nodes)
        self._entry_point = entry_point
        self.channels = channels
        self.builder = _BuilderView({schema: schema}, channels)
        self.checkpointer: Any = None
        self.store: Any = None
        # A graph compiled without a checkpointer is valid (upstream allows it
        # for stateless invocation). The thread adapter is therefore bound
        # lazily, and state reads/writes fail closed until one is supplied.
        self._threads: ThreadCheckpointer | None = None

    # -- binding -----------------------------------------------------------

    def _bind_checkpointer(self) -> ThreadCheckpointer:
        """Re-bind on every access: hosts assign ``graph.checkpointer`` late."""
        if self.checkpointer is None:
            raise ValueError(
                "No checkpointer set: bind a saver to graph.checkpointer before "
                "reading or writing state"
            )
        threads = self._threads
        if threads is None or threads._saver is not self.checkpointer:  # noqa: SLF001
            threads = ThreadCheckpointer(self.checkpointer, self.channels)
            self._threads = threads
        return threads

    # -- writes ------------------------------------------------------------

    def update_state(
        self,
        config: Mapping[str, Any] | None,
        values: Mapping[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply *values* to a thread as if *as_node* produced them."""
        threads = self._bind_checkpointer()
        head = threads.get_head(config)
        update = _adopt_messages(dict(values or {}))
        folded, written = threads.fold_input(
            threads.load_values(head), update
        )
        return threads.commit(
            config,
            head,
            folded,
            written=written,
            metadata=_update_metadata(as_node),
            task_id=task_id,
            delta_writes=update,
        )

    async def aupdate_state(
        self,
        config: Mapping[str, Any] | None,
        values: Mapping[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Async twin of :meth:`update_state`."""
        threads = self._bind_checkpointer()
        head = await threads.aget_head(config)
        update = _adopt_messages(dict(values or {}))
        folded, written = threads.fold_input(
            await threads.aload_values(head), update
        )
        return await threads.acommit(
            config,
            head,
            folded,
            written=written,
            metadata=_update_metadata(as_node),
            task_id=task_id,
            delta_writes=update,
        )

    # -- reads -------------------------------------------------------------

    def get_state(self, config: Mapping[str, Any] | None = None, **kwargs: Any) -> StateSnapshot:
        """Read the current thread state through the checkpointer."""
        del kwargs
        threads = self._bind_checkpointer()
        return threads.snapshot(threads.get_head(config), config)

    async def aget_state(
        self, config: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> StateSnapshot:
        """Async twin of :meth:`get_state`."""
        del kwargs
        threads = self._bind_checkpointer()
        return await threads.asnapshot(await threads.aget_head(config), config)

    def get_state_history(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[StateSnapshot]:
        """Iterate the thread's checkpoint history, newest first."""
        del kwargs
        threads = self._bind_checkpointer()
        return threads.history(config, before=before, limit=limit, filter=filter)

    def aget_state_history(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StateSnapshot]:
        """Async twin of :meth:`get_state_history`."""
        del kwargs
        threads = self._bind_checkpointer()
        return threads.ahistory(config, before=before, limit=limit, filter=filter)

    # -- the single node ---------------------------------------------------

    def _apply_node(self, state: dict[str, Any]) -> dict[str, Any]:
        node = self._nodes[self._entry_point]
        result = node(dict(state))
        return dict(result or {})

    def invoke(
        self,
        input: Mapping[str, Any] | Any,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run *event* (or the graph) synchronously and return the result."""
        del kwargs
        threads = self._bind_checkpointer()
        head = threads.get_head(config)
        seeded, _ = threads.fold_input(
            threads.load_values(head), _adopt_messages(dict(input or {}))
        )
        update = self._apply_node(seeded)
        folded, written = threads.fold_input(seeded, _adopt_messages(update))
        threads.commit(
            config,
            head,
            folded,
            written=written,
            metadata={"source": "loop", "step": 1, "parents": {}},
        )
        return folded

    async def ainvoke(
        self,
        input: Mapping[str, Any] | Any,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async twin of :meth:`invoke`."""
        del kwargs
        threads = self._bind_checkpointer()
        head = await threads.aget_head(config)
        seeded, _ = threads.fold_input(
            await threads.aload_values(head), _adopt_messages(dict(input or {}))
        )
        update = self._apply_node(seeded)
        folded, written = threads.fold_input(seeded, _adopt_messages(update))
        await threads.acommit(
            config,
            head,
            folded,
            written=written,
            metadata={"source": "loop", "step": 1, "parents": {}},
        )
        return folded


class StateGraph:
    """Minimal builder for a state-only mutation graph.

    Only the three calls the checkpoint-mutation path makes are implemented.
    Conditional edges, reducers configured at runtime, subgraphs, and streaming
    are out of scope and are rejected instead of half-honoured.
    """

    def __init__(self, schema: type, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise NotImplementedError(
                "reactivegraph.StateGraph supports only the state schema argument"
            )
        self._schema = schema
        self._nodes: dict[str, Any] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, action: Any, **kwargs: Any) -> StateGraph:
        """Register a node under *name* with its action function."""
        if kwargs:
            raise NotImplementedError(
                "reactivegraph.StateGraph.add_node does not support extra options"
            )
        if not callable(action):
            raise TypeError(f"node {name!r} action must be callable")
        if name in self._nodes:
            raise ValueError(f"duplicate node name: {name!r}")
        self._nodes[name] = action
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Mark *node* as the graph entry point."""
        if name not in self._nodes:
            raise ValueError(f"unknown node {name!r}: declare it with add_node first")
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> StateGraph:
        """Mark *node* as a terminal node."""
        if name not in self._nodes:
            raise ValueError(f"unknown node {name!r}: declare it with add_node first")
        return self

    def add_edge(self, start: str, end: str) -> StateGraph:
        """Add an unconditional edge between two nodes."""
        del start, end
        raise NotImplementedError(
            "reactivegraph.StateGraph is state-only: edges are not supported"
        )

    def compile(self, checkpointer: Any | None = None, **kwargs: Any) -> CompiledStateGraph:
        """Compile the builder into an executable graph."""
        if kwargs:
            raise NotImplementedError(
                "reactivegraph.StateGraph.compile does not support extra options"
            )
        if self._entry_point is None:
            raise ValueError(ENTRY_POINT_REQUIRED_MESSAGE)
        graph = CompiledStateGraph(
            self._schema,
            self._nodes,
            self._entry_point,
            resolve_channels([self._schema]),
        )
        graph.checkpointer = checkpointer
        return graph

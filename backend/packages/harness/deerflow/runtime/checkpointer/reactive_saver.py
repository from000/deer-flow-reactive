"""LangGraph ``BaseCheckpointSaver`` adapter over the ReactiveGraph store.

DeerFlow's graph execution still goes through LangGraph's Pregel loop, and
``langgraph.types.ensure_valid_checkpointer`` hard-rejects any checkpointer that
is not a ``BaseCheckpointSaver`` instance. This adapter therefore exists to
satisfy that isinstance contract while the storage semantics live in
``reactivegraph.checkpoint`` — one tested implementation shared by every host.

The translation is deliberately thin:

* ``put_writes``      LangGraph passes ``(channel, value)`` pairs plus a
                      separate ``task_id``; the ReactiveGraph store records
                      ``(task_id, channel, value)`` triples.
* ``get_tuple``/``list``  ReactiveGraph ``CheckpointTuple`` is a dataclass with
                      the same five fields; it is converted to LangGraph's
                      NamedTuple so consumers relying on the concrete type
                      keep working.

Anything this adapter cannot honour raises rather than silently degrading.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from reactivegraph import BaseCheckpointSaver, CheckpointTuple as LangGraphCheckpointTuple
from reactivegraph.checkpoint import CheckpointStore, CheckpointTuple

__all__ = ("ReactiveGraphSaver",)


class ReactiveGraphSaver(BaseCheckpointSaver[str]):
    """Store checkpoints through ReactiveGraph, present LangGraph's interface."""

    def __init__(self, store: CheckpointStore | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store if store is not None else CheckpointStore()

    # -- conversion --------------------------------------------------------

    @staticmethod
    def _to_langgraph(tup: CheckpointTuple | None) -> LangGraphCheckpointTuple | None:
        if tup is None:
            return None
        return LangGraphCheckpointTuple(
            config=tup.config,
            checkpoint=tup.checkpoint,
            metadata=tup.metadata,
            parent_config=tup.parent_config,
            pending_writes=list(tup.pending_writes),
        )

    # -- sync --------------------------------------------------------------

    def get_tuple(self, config: dict[str, Any]) -> LangGraphCheckpointTuple | None:
        return self._to_langgraph(self.store.get_tuple(config))

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[LangGraphCheckpointTuple]:
        for tup in self.store.list(config, before=before, limit=limit, filter=filter):
            converted = self._to_langgraph(tup)
            assert converted is not None
            yield converted

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        return self.store.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        triples = [(task_id, channel, value) for channel, value in writes]
        self.store.put_writes(config, triples, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self.store.delete_thread(thread_id)

    def get_next_version(self, current: Any, channel: str) -> int:
        return self.store.get_next_version(current, channel)

    # -- async -------------------------------------------------------------

    async def aget_tuple(self, config: dict[str, Any]) -> LangGraphCheckpointTuple | None:
        return self._to_langgraph(await self.store.aget_tuple(config))

    async def alist(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[LangGraphCheckpointTuple]:
        async for tup in self.store.alist(config, before=before, limit=limit, filter=filter):
            converted = self._to_langgraph(tup)
            assert converted is not None
            yield converted

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.store.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        triples = [(task_id, channel, value) for channel, value in writes]
        await self.store.aput_writes(config, triples, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await self.store.adelete_thread(thread_id)

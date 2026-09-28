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

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from reactivegraph import BaseCheckpointSaver
from reactivegraph import CheckpointTuple as LangGraphCheckpointTuple
from reactivegraph.checkpoint import CheckpointStore, CheckpointTuple

__all__ = ("ReactiveGraphSaver",)


class ReactiveGraphSaver(BaseCheckpointSaver[str]):
    """Store checkpoints through ReactiveGraph, present LangGraph's interface."""

    def __init__(self, store: CheckpointStore | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store if store is not None else CheckpointStore()
        # Durable ReactiveGraph stores own the wire format for their rows. The
        # delta-history cache reads this attribute, so it must not retain the
        # host's default serializer while the underlying store writes another.
        self.serde = getattr(self.store, "serde", self.serde)

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        setup = getattr(self.store, "setup", None)
        if setup is not None:
            setup()

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()

    async def asetup(self) -> None:
        setup = getattr(self.store, "asetup", None)
        if setup is not None:
            await setup()
            return
        setup = getattr(self.store, "setup", None)
        if setup is not None:
            await asyncio.to_thread(setup)

    async def aclose(self) -> None:
        close = getattr(self.store, "aclose", None)
        if close is not None:
            await close()
            return
        close = getattr(self.store, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

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

    # -- maintenance capabilities -----------------------------------------

    def storage_stats(self, thread_id: str) -> dict[str, int]:
        capability = getattr(self.store, "storage_stats", None)
        if capability is None:
            raise NotImplementedError(
                f"{type(self.store).__name__} does not support checkpoint storage stats"
            )
        return capability(thread_id)

    def checkpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        capability = getattr(self.store, "checkpoint_ids_with_writes", None)
        if capability is None:
            raise NotImplementedError(
                f"{type(self.store).__name__} does not support checkpoint write guards"
            )
        return capability(thread_id)

    def delete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        capability = getattr(self.store, "delete_checkpoint", None)
        if capability is None:
            raise NotImplementedError(
                f"{type(self.store).__name__} does not support checkpoint deletion"
            )
        capability(thread_id, key)

    def delete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        capability = getattr(self.store, "delete_unreachable_blobs", None)
        if capability is None:
            raise NotImplementedError(
                f"{type(self.store).__name__} does not support checkpoint blob GC"
            )
        capability(thread_id, survivor_versions)

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

    async def astorage_stats(self, thread_id: str) -> dict[str, int]:
        capability = getattr(self.store, "astorage_stats", None)
        if capability is not None:
            return await capability(thread_id)
        return await asyncio.to_thread(self.storage_stats, thread_id)

    async def acheckpoint_ids_with_writes(
        self, thread_id: str
    ) -> set[tuple[str, str]]:
        capability = getattr(self.store, "acheckpoint_ids_with_writes", None)
        if capability is not None:
            return await capability(thread_id)
        return await asyncio.to_thread(self.checkpoint_ids_with_writes, thread_id)

    async def adelete_checkpoint(
        self, thread_id: str, key: tuple[str, str]
    ) -> None:
        capability = getattr(self.store, "adelete_checkpoint", None)
        if capability is not None:
            await capability(thread_id, key)
            return
        await asyncio.to_thread(self.delete_checkpoint, thread_id, key)

    async def adelete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        capability = getattr(self.store, "adelete_unreachable_blobs", None)
        if capability is not None:
            await capability(thread_id, survivor_versions)
            return
        await asyncio.to_thread(
            self.delete_unreachable_blobs, thread_id, survivor_versions
        )

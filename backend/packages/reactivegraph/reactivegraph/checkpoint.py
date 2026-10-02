"""Durable checkpoint storage for ReactiveGraph.

This module owns the *storage* contract for graph checkpoints: immutable
snapshots, a parent lineage, pending writes, and lifecycle purge. It is
deliberately engine-agnostic and imports nothing from LangChain/LangGraph.

A host engine that demands a specific saver interface (for example LangGraph's
``BaseCheckpointSaver``) adapts :class:`MemoryCheckpointSaver` in its own
package. Keeping the semantics here means every host shares one tested
implementation instead of re-deriving it.

Native ``memory``, ``sqlite``, and ``postgres`` backends are provided.
:func:`get_checkpointer` rejects unknown backends rather than silently
degrading to process-local state, because that would be data loss presented as
durability.

Portions of this module are adapted from upstream `langgraph-checkpoint / oittaa/uuid6-python`
(https://github.com/langchain-ai/langgraph) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

# Reserved channels are keyed by a fixed negative index instead of their
# position in the batch. Regular writes are first-write-wins so replaying a
# retried task cannot duplicate them; reserved channels overwrite so the latest
# interrupt/error wins. Both behaviours are part of Pregel's retry contract.
WRITES_IDX_MAP: dict[str, int] = {
    "__error__": -1,
    "__scheduled__": -2,
    "__interrupt__": -3,
    "__resume__": -4,
}

__all__ = (
    "LATEST_CHECKPOINT_VERSION",
    "WRITES_IDX_MAP",
    "BaseCheckpointSaver",
    "CheckpointStore",
    "CheckpointTuple",
    "MemoryCheckpointSaver",
    "SqliteCheckpointSaver",
    "PostgresCheckpointSaver",
    "PendingWrite",
    "empty_checkpoint",
    "get_checkpointer",
    "reset_checkpointer",
    "uuid6",
)

V = TypeVar("V")

LATEST_CHECKPOINT_VERSION = 2
"""Checkpoint schema version written by :func:`empty_checkpoint`."""

_last_v6_timestamp: int | None = None


def uuid6(node: int | None = None, clock_seq: int | None = None) -> uuid.UUID:
    """Return a UUID version 6, ordered by generation time.

    Version 6 reorders the v1 fields so lexicographic order follows time,
    which is what makes checkpoint ids sortable in databases and object
    stores. Repeated calls never return the same id even when the clock has
    not advanced: the last timestamp is remembered and bumped.
    """
    global _last_v6_timestamp
    nanoseconds = time.time_ns()
    # 100-ns intervals between the UUID epoch (1582-10-15) and the Unix epoch.
    timestamp = nanoseconds // 100 + 0x01B21DD213814000
    if _last_v6_timestamp is not None and timestamp <= _last_v6_timestamp:
        timestamp = _last_v6_timestamp + 1
    _last_v6_timestamp = timestamp
    if clock_seq is None:
        clock_seq = random.getrandbits(14)
    if node is None:
        node = random.getrandbits(48)
    uuid_int = ((timestamp >> 12) & 0xFFFFFFFFFFFF) << 80
    uuid_int |= (timestamp & 0x0FFF) << 64
    uuid_int |= (clock_seq & 0x3FFF) << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    # The stdlib refuses ``version=6``; write the version nibble and the
    # RFC-4122 variant bits directly. ``UUID.version`` reads them back out of
    # the integer, so the result is a real v6 uuid.
    uuid_int &= ~(0xF << 76)
    uuid_int |= 6 << 76
    uuid_int &= ~(0xC000 << 48)
    uuid_int |= 0x8000 << 48
    return uuid.UUID(int=uuid_int)


def empty_checkpoint() -> dict[str, Any]:
    """Return a fresh, version-2 checkpoint with no channels or writes.

    Hosts use this as the initial snapshot for a thread and as the marker
    checkpoint when they need a new id/ts without running the graph. Every
    call allocates new mutable containers and a new ordered id.
    """
    return {
        "v": LATEST_CHECKPOINT_VERSION,
        "id": str(uuid6(clock_seq=-2)),
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }


@dataclass
class CheckpointTuple:
    """A stored checkpoint plus the lineage and writes around it.

    Mirrors the fields real consumers touch: ``config``, ``checkpoint``,
    ``metadata``, ``parent_config`` and ``pending_writes``. ``pending_writes``
    entries are ``(task_id, channel, value)`` triples.
    """

    config: dict[str, Any]
    checkpoint: dict[str, Any]
    metadata: dict[str, Any]
    parent_config: dict[str, Any] | None
    pending_writes: list[tuple[str, str, Any]]


PendingWrite = tuple[str, str, Any]
"""One pending write: ``(task_id, channel, value)``.

The tuple shape is LangGraph's, so a saver written against either engine can
read the other's ``pending_writes`` entries without translation.
"""


class _NativeBaseCheckpointSaver(Generic[V]):
    """LangGraph's ``BaseCheckpointSaver`` surface, implemented natively.

    This is the fallback used only when ``langgraph`` cannot be imported. The
    method bodies mirror upstream exactly where a host could observe a
    difference: unimplemented storage operations raise ``NotImplementedError``
    (fail closed) rather than silently returning empty state, ``get`` derives
    the checkpoint from ``get_tuple``, ``get_next_version`` does integer
    arithmetic, and ``get_delta_channel_history`` walks the parent chain the
    same way. ``serde`` is ``None`` here: the engine's own storage layer
    serialises with msgpack, and a host that needs typed serialisation must
    supply its own.
    """

    serde: Any = None

    def __init__(self, *, serde: Any | None = None) -> None:
        self.serde = serde

    @property
    def config_specs(self) -> list:
        return []

    def get(self, config: Mapping[str, Any]) -> dict[str, Any] | None:
        value = self.get_tuple(config)
        return value.checkpoint if value else None

    def get_tuple(self, config: Mapping[str, Any]) -> CheckpointTuple | None:
        raise NotImplementedError

    def list(
        self,
        config: Mapping[str, Any] | None,
        *,
        filter: Mapping[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError

    def put(
        self,
        config: Mapping[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def put_writes(
        self,
        config: Mapping[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError

    def delete_thread(self, thread_id: str) -> None:
        raise NotImplementedError

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        raise NotImplementedError

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        raise NotImplementedError

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError

    async def aget_tuple(self, config: Mapping[str, Any]) -> CheckpointTuple | None:
        raise NotImplementedError

    async def alist(
        self,
        config: Mapping[str, Any] | None,
        *,
        filter: Mapping[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ):
        raise NotImplementedError

    async def aput(
        self,
        config: Mapping[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    async def aput_writes(
        self,
        config: Mapping[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError

    async def adelete_thread(self, thread_id: str) -> None:
        raise NotImplementedError

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        raise NotImplementedError

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        raise NotImplementedError

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError

    def get_delta_channel_history(
        self, *, config: Mapping[str, Any], channels: Sequence[str]
    ) -> dict[str, Any]:
        """Walk the parent chain accumulating per-channel writes and seed."""
        if not channels:
            return {}
        collected: dict[str, list[PendingWrite]] = {c: [] for c in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)
        target = self.get_tuple(config)
        cursor = target.parent_config if target else None
        while cursor is not None and remaining:
            tup = self.get_tuple(cursor)
            if tup is None:
                break
            for write in reversed(tup.pending_writes or ()):
                channel = write[1]
                if channel in remaining:
                    collected[channel].append(write)
            for channel in list(remaining):
                if channel in (tup.checkpoint.get("channel_values") or {}):
                    seeds[channel] = tup.checkpoint["channel_values"][channel]
                    remaining.discard(channel)
            cursor = tup.parent_config
        result: dict[str, Any] = {}
        for channel in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[channel]))}
            if channel in seeds:
                entry["seed"] = seeds[channel]
            result[channel] = entry
        return result

    async def aget_delta_channel_history(
        self, *, config: Mapping[str, Any], channels: Sequence[str]
    ) -> dict[str, Any]:
        if not channels:
            return {}
        collected: dict[str, list[PendingWrite]] = {c: [] for c in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)
        target = await self.aget_tuple(config)
        cursor = target.parent_config if target else None
        while cursor is not None and remaining:
            tup = await self.aget_tuple(cursor)
            if tup is None:
                break
            for write in reversed(tup.pending_writes or ()):
                channel = write[1]
                if channel in remaining:
                    collected[channel].append(write)
            for channel in list(remaining):
                if channel in (tup.checkpoint.get("channel_values") or {}):
                    seeds[channel] = tup.checkpoint["channel_values"][channel]
                    remaining.discard(channel)
            cursor = tup.parent_config
        result: dict[str, Any] = {}
        for channel in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[channel]))}
            if channel in seeds:
                entry["seed"] = seeds[channel]
            result[channel] = entry
        return result

    def get_next_version(self, current: Any, channel: Any) -> Any:
        if isinstance(current, str):
            raise NotImplementedError
        if current is None:
            return 1
        return current + 1


try:  # pragma: no cover - exercised by the optional-integration tests
    from langgraph.checkpoint.base import BaseCheckpointSaver as BaseCheckpointSaver
except ImportError:  # pragma: no cover - langgraph is an optional host dep
    BaseCheckpointSaver = _NativeBaseCheckpointSaver  # type: ignore[misc, assignment, unused-ignore]


def _iter_writes(writes: Sequence[tuple[Any, ...]]) -> Iterator[tuple[str, Any]]:
    """Yield ``(channel, value)`` from either accepted write shape.

    Upstream's ``BaseCheckpointSaver.put_writes`` contract is
    ``Sequence[tuple[str, Any]]``; the engine's own storage layer has always
    stored ``(task_id, channel, value)`` triples and hands those back from
    ``get_tuple``. Accepting both keeps a host graph able to drive this saver
    without translating its writes, while the canonical triple remains the
    only thing callers ever read back.
    """
    for write in writes:
        if len(write) >= 3:
            yield write[1], write[2]
        else:
            yield write[0], write[1]


class CheckpointStore:
    """In-process checkpoint store with an append-only lineage.

    * ``get_tuple`` / ``aget_tuple``      latest-or-explicit checkpoint
    * ``list`` / ``alist``                newest-first history, cursor, limit, filter
    * ``put`` / ``aput``                  immutable snapshot + parent chain
    * ``put_writes`` / ``aput_writes``    pending writes per checkpoint
    * ``delete_thread`` / ``adelete_thread`` lifecycle purge
    * ``storage_stats`` / ``astorage_stats`` per-thread rows and bytes
    * ``delete_checkpoint`` / ``adelete_checkpoint`` one checkpoint + writes
    * ``delete_unreachable_blobs``        capability-compatible no-op here
    * ``get_next_version``                channel version arithmetic

    Checkpoints are immutable once written: ``put`` with an existing id is a
    no-op, matching an append-only lineage contract. Everything is deep-copied
    on the way in and out so callers cannot mutate stored state.

    Writes are keyed by ``(task_id, slot)`` where ``slot`` is the write's
    position in its batch, or a fixed negative index for a reserved channel.
    Replaying a retried task therefore cannot duplicate a regular write, while
    a repeated interrupt/error still overwrites the previous one.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str, str], CheckpointTuple] = {}
        self._writes: dict[tuple[str, str, str], dict[tuple[str, int], tuple[str, str, Any]]] = {}
        self._order: dict[tuple[str, str], list[str]] = {}
        self._latest: dict[tuple[str, str], str] = {}
        self._counter = 0
        self._lock = threading.RLock()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _ref(config: dict[str, Any] | None) -> tuple[str, str]:
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("config.configurable.thread_id is required")
        return str(thread_id), str(configurable.get("checkpoint_ns", "") or "")

    @staticmethod
    def _checkpoint_id(config: dict[str, Any] | None) -> str | None:
        configurable = (config or {}).get("configurable") or {}
        value = configurable.get("checkpoint_id")
        return str(value) if value else None

    @staticmethod
    def _with_checkpoint_id(config: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
        # Match LangGraph's savers: return the minimal durable identity rather
        # than deep-copying runtime config. The latter can contain locks,
        # callbacks, and other non-copyable execution state.
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("config.configurable.thread_id is required")
        return {
            "configurable": {
                "thread_id": str(thread_id),
                "checkpoint_ns": str(configurable.get("checkpoint_ns", "") or ""),
                "checkpoint_id": str(checkpoint_id),
            }
        }

    def _materialize(self, key: tuple[str, str, str], record: CheckpointTuple) -> CheckpointTuple:
        """Return a deep copy carrying the writes recorded for ``key``.

        Writes live in their own mapping keyed by ``(task_id, slot)`` so that
        replaying a retried task stays idempotent, so they are attached on read
        rather than stored on the checkpoint record. Callers must hold the lock.
        """
        out = copy.deepcopy(record)
        slots = self._writes.get(key)
        out.pending_writes = copy.deepcopy(list(slots.values())) if slots else []
        return out

    def new_checkpoint_id(self) -> str:
        """Monotonic, collision-safe id usable as ``checkpoint["id"]``."""
        with self._lock:
            self._counter += 1
            counter = self._counter
        return f"{time.time_ns():020d}-{counter:08d}"

    def get_next_version(self, current: Any, channel: str) -> int:
        del channel
        try:
            base = int(current) if current is not None else 0
        except (TypeError, ValueError):
            base = 0
        return base + 1

    # -- writes ------------------------------------------------------------

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        del new_versions
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            raise ValueError("checkpoint['id'] is required")
        checkpoint_id = str(checkpoint_id)
        thread_key = self._ref(config)
        parent_config = None
        parent_id = self._checkpoint_id(config)
        if parent_id is not None:
            parent = self._checkpoints.get((*thread_key, parent_id))
            if parent is not None:
                parent_config = copy.deepcopy(parent.config)
        stored_config = self._with_checkpoint_id(config, checkpoint_id)
        record = CheckpointTuple(
            config=stored_config,
            checkpoint=copy.deepcopy(checkpoint),
            metadata=copy.deepcopy(metadata or {}),
            parent_config=parent_config,
            pending_writes=[],
        )
        with self._lock:
            key = (*thread_key, checkpoint_id)
            existing = self._checkpoints.get(key)
            if existing is not None:
                # Immutable lineage: an id is written once.
                return copy.deepcopy(existing.config)
            self._checkpoints[key] = record
            self._order.setdefault(thread_key, []).append(checkpoint_id)
            self._latest[thread_key] = checkpoint_id
        return copy.deepcopy(stored_config)

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del task_path
        thread_key = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        if checkpoint_id is None:
            raise ValueError("config.configurable.checkpoint_id is required")
        key = (*thread_key, checkpoint_id)
        with self._lock:
            # LangGraph's Pregel commits a task's writes before the checkpoint
            # row itself (for example the ``__no_writes__`` marker). Keep the
            # write keyed by its checkpoint id; a later ``put`` makes it
            # visible through ``get_tuple``.
            slots = self._writes.setdefault(key, {})
            for index, (channel, value) in enumerate(_iter_writes(writes)):
                slot = WRITES_IDX_MAP.get(channel, index)
                if slot >= 0 and (task_id, slot) in slots:
                    # A regular write already occupies this position, so this is
                    # a replay of a retried task: keep the first write.
                    continue
                # Reserved channels use a negative slot, so re-writing the same
                # slot intentionally replaces the earlier value.
                slots[(task_id, slot)] = (task_id, channel, copy.deepcopy(value))

    def delete_thread(self, thread_id: str) -> None:
        target = str(thread_id)
        with self._lock:
            for key in [k for k in self._checkpoints if k[0] == target]:
                del self._checkpoints[key]
            for key in [k for k in self._writes if k[0] == target]:
                del self._writes[key]
            for order_key in [k for k in self._order if k[0] == target]:
                del self._order[order_key]
            for latest_key in [k for k in self._latest if k[0] == target]:
                del self._latest[latest_key]

    # -- maintenance -------------------------------------------------------

    @staticmethod
    def _logical_size(value: Any) -> int:
        """Best-effort byte accounting for native in-memory state."""
        from reactivegraph.serde import ReactiveSerializer

        try:
            return len(ReactiveSerializer().dumps_typed(value)[1])
        except TypeError:
            # Stats must never make an otherwise readable in-memory store
            # unusable when a host put a non-serializable metadata object in it.
            return len(repr(value).encode("utf-8", errors="replace"))

    def storage_stats(self, thread_id: str) -> dict[str, int]:
        """Return normalized per-thread row and byte counts.

        Native checkpoints are inline, so ``blob_rows`` and ``blob_bytes`` are
        intentionally zero. ``logical_checkpoint_bytes`` includes the inline
        checkpoint and metadata payload.
        """
        target = str(thread_id)
        checkpoint_rows = checkpoint_bytes = write_rows = write_bytes = 0
        with self._lock:
            for key, record in self._checkpoints.items():
                if key[0] != target:
                    continue
                checkpoint_rows += 1
                checkpoint_bytes += self._logical_size(record.checkpoint)
                checkpoint_bytes += self._logical_size(record.metadata)
            for key, slots in self._writes.items():
                if key[0] != target:
                    continue
                for write in slots.values():
                    write_rows += 1
                    write_bytes += self._logical_size(write[2])
        return {
            "logical_checkpoint_bytes": checkpoint_bytes,
            "logical_write_bytes": write_bytes,
            "checkpoint_rows": checkpoint_rows,
            "checkpoint_bytes": checkpoint_bytes,
            "blob_rows": 0,
            "blob_bytes": 0,
            "write_rows": write_rows,
            "write_bytes": write_bytes,
        }

    def checkpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        """Return ``(checkpoint_ns, checkpoint_id)`` pairs owning writes."""
        target = str(thread_id)
        with self._lock:
            return {
                (key[1], key[2])
                for key, slots in self._writes.items()
                if key[0] == target and slots
            }

    def delete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        """Delete one checkpoint and its writes, preserving other lineage."""
        target = str(thread_id)
        namespace, checkpoint_id = str(key[0]), str(key[1])
        record_key = (target, namespace, checkpoint_id)
        order_key = (target, namespace)
        with self._lock:
            self._checkpoints.pop(record_key, None)
            self._writes.pop(record_key, None)
            order = self._order.get(order_key)
            if order is not None:
                remaining = [item for item in order if item != checkpoint_id]
                if remaining:
                    self._order[order_key] = remaining
                else:
                    self._order.pop(order_key, None)
            if self._latest.get(order_key) == checkpoint_id:
                latest_order = self._order.get(order_key)
                if latest_order:
                    self._latest[order_key] = latest_order[-1]
                else:
                    self._latest.pop(order_key, None)

    def delete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        """No-op for the native inline checkpoint representation."""
        del thread_id, survivor_versions

    async def astorage_stats(self, thread_id: str) -> dict[str, int]:
        return await asyncio.to_thread(self.storage_stats, thread_id)

    async def acheckpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        return await asyncio.to_thread(self.checkpoint_ids_with_writes, thread_id)

    async def adelete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        return await asyncio.to_thread(self.delete_checkpoint, thread_id, key)

    async def adelete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        return await asyncio.to_thread(
            self.delete_unreachable_blobs, thread_id, survivor_versions
        )

    # -- reads -------------------------------------------------------------

    def get_tuple(self, config: dict[str, Any] | None) -> CheckpointTuple | None:
        if config is None:
            return None
        thread_key = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        with self._lock:
            if checkpoint_id is None:
                checkpoint_id = self._latest.get(thread_key)
            if checkpoint_id is None:
                return None
            record = self._checkpoints.get((*thread_key, checkpoint_id))
            if record is None:
                return None
            return self._materialize((*thread_key, checkpoint_id), record)

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[CheckpointTuple]:
        def latest_ts(key: tuple[str, str]) -> str:
            head = self._latest.get(key)
            record = self._checkpoints.get((*key, head)) if head is not None else None
            return str(record.checkpoint.get("ts") or "") if record is not None else ""

        with self._lock:
            if config is None:
                keys = sorted(self._order, key=latest_ts, reverse=True)
            else:
                thread_id, checkpoint_ns = self._ref(config)
                configurable = (config or {}).get("configurable") or {}
                if "checkpoint_ns" in configurable:
                    keys = [(thread_id, checkpoint_ns)]
                else:
                    # LangGraph's thread-wide listing spans every namespace.
                    # Only an explicitly supplied ``checkpoint_ns`` narrows it.
                    keys = sorted(
                        (key for key in self._order if key[0] == thread_id),
                        key=latest_ts,
                        reverse=True,
                    )
            for thread_key in keys:
                ids = list(self._order.get(thread_key, []))
                if before is not None:
                    before_id = self._checkpoint_id(before)
                    if before_id in ids:
                        # ``ids`` is oldest->newest; "before X" means older than X.
                        ids = ids[: ids.index(before_id)]
                for checkpoint_id in reversed(ids):
                    record = self._checkpoints.get((*thread_key, checkpoint_id))
                    if record is None:
                        continue
                    if filter and not all(record.metadata.get(k) == v for k, v in filter.items()):
                        continue
                    yield self._materialize((*thread_key, checkpoint_id), record)
                    if limit is not None:
                        limit -= 1
                        if limit <= 0:
                            return

    # -- async mirrors -----------------------------------------------------

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, before=None, limit=None, filter=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, before=before, limit=limit, filter=filter))
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)


class MemoryCheckpointSaver(CheckpointStore):
    """Named saver facade returned by :func:`get_checkpointer`.

    Same storage contract as :class:`CheckpointStore`; the distinct type marks
    the process-local backend so callers can isinstance-check what they got.
    """


class SqliteCheckpointSaver(CheckpointStore):
    """Durable checkpoint store backed by a LangGraph-compatible SQLite schema.

    The implementation owns its storage semantics; the adapter layer only makes
    it satisfy a host engine's isinstance contract. Rows use the same
    ``checkpoints``/``writes`` shape as LangGraph so an existing database can be
    opened without an offline migration.
    """

    def __init__(
        self,
        conn_string: str | Path = ":memory:",
        *,
        serde: Any | None = None,
    ) -> None:
        from reactivegraph.serde import ReactiveSerializer

        super().__init__()
        self.conn_string = str(conn_string)
        if self.conn_string != ":memory:":
            Path(self.conn_string).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self.serde = serde if serde is not None else ReactiveSerializer()
        self.conn = sqlite3.connect(
            self.conn_string,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self._db_lock = threading.RLock()
        self._closed = False
        self.setup()

    def setup(self) -> None:
        """Create the schema. Safe to call repeatedly."""
        with self._db_lock:
            if self._closed:
                raise RuntimeError("SQLite checkpoint saver is closed")
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE INDEX IF NOT EXISTS checkpoints_thread_ns_id_idx
                    ON checkpoints (thread_id, checkpoint_ns, checkpoint_id DESC);
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                CREATE INDEX IF NOT EXISTS writes_lookup_idx
                    ON writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx);
                """
            )

    def close(self) -> None:
        with self._db_lock:
            if not self._closed:
                self.conn.close()
                self._closed = True

    def __enter__(self) -> SqliteCheckpointSaver:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        if self._closed:
            raise RuntimeError("SQLite checkpoint saver is closed")
        return self.conn.execute(sql, tuple(params))

    def new_checkpoint_id(self) -> str:
        return str(uuid6())

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        del new_versions
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            raise ValueError("checkpoint['id'] is required")
        checkpoint_id = str(checkpoint_id)
        thread_id, checkpoint_ns = self._ref(config)
        parent_id = self._checkpoint_id(config)
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        metadata_bytes = json.dumps(
            metadata or {},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        stored_config = self._with_checkpoint_id(config, checkpoint_id)
        with self._db_lock:
            self._execute(
                """
                INSERT OR IGNORE INTO checkpoints (
                    thread_id, checkpoint_ns, checkpoint_id,
                    parent_checkpoint_id, type, checkpoint, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_id,
                    type_,
                    serialized_checkpoint,
                    metadata_bytes,
                ),
            )
        return copy.deepcopy(stored_config)

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del task_path
        thread_id, checkpoint_ns = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        if checkpoint_id is None:
            raise ValueError("config.configurable.checkpoint_id is required")
        with self._db_lock:
            # See CheckpointStore.put_writes: writes are allowed to precede
            # the checkpoint row because that is the order Pregel uses.
            for index, (channel, value) in enumerate(_iter_writes(writes)):
                slot = WRITES_IDX_MAP.get(channel, index)
                type_, serialized = self.serde.dumps_typed(value)
                operation = "INSERT OR REPLACE" if slot < 0 else "INSERT OR IGNORE"
                self._execute(
                    f"""
                    {operation} INTO writes (
                        thread_id, checkpoint_ns, checkpoint_id, task_id,
                        idx, channel, type, value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        str(task_id),
                        slot,
                        channel,
                        type_,
                        serialized,
                    ),
                )

    def get_tuple(self, config: dict[str, Any] | None) -> CheckpointTuple | None:
        if config is None:
            return None
        thread_id, checkpoint_ns = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        with self._db_lock:
            if checkpoint_id is None:
                row = self._execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
            else:
                row = self._execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_tuple(row)

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[CheckpointTuple]:
        where: list[str] = []
        params: list[Any] = []
        if config is not None:
            thread_id, checkpoint_ns = self._ref(config)
            configurable = (config or {}).get("configurable") or {}
            where.append("thread_id = ?")
            params.append(thread_id)
            if "checkpoint_ns" in configurable:
                where.append("checkpoint_ns = ?")
                params.append(checkpoint_ns)
        if before is not None:
            before_id = self._checkpoint_id(before)
            if before_id is not None:
                where.append("checkpoint_id < ?")
                params.append(before_id)
        sql = "SELECT * FROM checkpoints"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY checkpoint_id DESC"
        with self._db_lock:
            rows = self._execute(sql, params).fetchall()
            for row in rows:
                metadata = json.loads(row[6] or b"{}")
                if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
                yield self._row_to_tuple(row)
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        return

    def delete_thread(self, thread_id: str) -> None:
        target = str(thread_id)
        with self._db_lock:
            self._execute("DELETE FROM checkpoints WHERE thread_id = ?", (target,))
            self._execute("DELETE FROM writes WHERE thread_id = ?", (target,))

    # -- maintenance -------------------------------------------------------

    def storage_stats(self, thread_id: str) -> dict[str, int]:
        """Return normalized per-thread row and byte counts for SQLite."""
        target = str(thread_id)
        with self._db_lock:
            checkpoint_row = self._execute(
                """
                SELECT COUNT(*), COALESCE(SUM(
                    LENGTH(COALESCE(checkpoint, X'')) +
                    LENGTH(COALESCE(metadata, X''))
                ), 0)
                FROM checkpoints WHERE thread_id = ?
                """,
                (target,),
            ).fetchone()
            write_row = self._execute(
                """
                SELECT COUNT(*), COALESCE(SUM(LENGTH(COALESCE(value, X''))), 0)
                FROM writes WHERE thread_id = ?
                """,
                (target,),
            ).fetchone()
        checkpoint_rows = int(checkpoint_row[0])
        checkpoint_bytes = int(checkpoint_row[1] or 0)
        write_rows = int(write_row[0])
        write_bytes = int(write_row[1] or 0)
        return {
            "logical_checkpoint_bytes": checkpoint_bytes,
            "logical_write_bytes": write_bytes,
            "checkpoint_rows": checkpoint_rows,
            "checkpoint_bytes": checkpoint_bytes,
            "blob_rows": 0,
            "blob_bytes": 0,
            "write_rows": write_rows,
            "write_bytes": write_bytes,
        }

    def checkpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        target = str(thread_id)
        with self._db_lock:
            rows = self._execute(
                """
                SELECT DISTINCT checkpoint_ns, checkpoint_id
                FROM writes WHERE thread_id = ?
                """,
                (target,),
            ).fetchall()
        return {(str(row[0] or ""), str(row[1])) for row in rows}

    def delete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        target = str(thread_id)
        namespace, checkpoint_id = str(key[0]), str(key[1])
        with self._db_lock:
            # The two deletes are one destructive operation: a process crash
            # must not leave writes for a checkpoint row that no longer exists.
            self._execute("BEGIN IMMEDIATE")
            try:
                self._execute(
                    """
                    DELETE FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (target, namespace, checkpoint_id),
                )
                self._execute(
                    """
                    DELETE FROM writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (target, namespace, checkpoint_id),
                )
            except BaseException:
                self._execute("ROLLBACK")
                raise
            self._execute("COMMIT")

    def delete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        """No-op: SQLite checkpoint payloads are stored inline."""
        del thread_id, survivor_versions

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, before=None, limit=None, filter=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, before=before, limit=limit, filter=filter))
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def astorage_stats(self, thread_id: str) -> dict[str, int]:
        return await asyncio.to_thread(self.storage_stats, thread_id)

    async def acheckpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        return await asyncio.to_thread(self.checkpoint_ids_with_writes, thread_id)

    async def adelete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        return await asyncio.to_thread(self.delete_checkpoint, thread_id, key)

    async def adelete_unreachable_blobs(
        self, thread_id: str, survivor_versions: set[Any]
    ) -> None:
        return await asyncio.to_thread(
            self.delete_unreachable_blobs, thread_id, survivor_versions
        )

    def _row_to_tuple(self, row: sqlite3.Row) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
        metadata = json.loads(row["metadata"] or b"{}")
        pending = [
            (
                write["task_id"],
                write["channel"],
                self.serde.loads_typed((write["type"], write["value"])),
            )
            for write in self._execute(
                """
                SELECT task_id, channel, type, value FROM writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY task_id, idx
                """,
                (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]),
            ).fetchall()
        ]
        parent_config = (
            {
                "configurable": {
                    "thread_id": row[0],
                    "checkpoint_ns": row[1],
                    "checkpoint_id": row[3],
                }
            }
            if row[3]
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row[0],
                    "checkpoint_ns": row[1],
                    "checkpoint_id": row[2],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending,
        )



class PostgresCheckpointSaver(CheckpointStore):
    """Durable PostgreSQL checkpoint store using ReactiveGraph serialization.

    The table shape intentionally mirrors the SQLite saver so hosts can move
    between durable backends without changing checkpoint semantics. PostgreSQL
    bytea columns hold the same typed, no-pickle payloads.
    """

    def __init__(
        self,
        conn_string: str,
        *,
        schema: str = "public",
        serde: Any | None = None,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional production extra
            raise ImportError(
                "psycopg is required for the PostgreSQL checkpointer. "
                "Install it with: pip install 'reactivegraph[postgres]'"
            ) from exc

        super().__init__()
        self.conn_string = str(conn_string)
        self.schema = str(schema or "public")
        self.serde = serde
        self._psycopg = psycopg
        self.conn = psycopg.connect(self.conn_string, autocommit=False)
        self._db_lock = threading.RLock()
        self._closed = False
        self.setup()

    def setup(self) -> None:
        with self._db_lock:
            if self._closed:
                raise RuntimeError("PostgreSQL checkpoint saver is closed")
            with self.conn.transaction():
                # CREATE SCHEMA IF NOT EXISTS still races in PostgreSQL:
                # concurrent transactions can both observe a missing
                # namespace, then one fails on the unique catalog index.
                # Serialize bootstrap per target schema with a lock that
                # is released automatically when this transaction ends.
                self.conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                    ("reactivegraph_schema_bootstrap", self.schema),
                )
                self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                self.conn.execute(f'SET search_path TO "{self.schema}"')
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BYTEA,
                        metadata BYTEA,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS checkpoints_thread_ns_id_idx
                    ON checkpoints (thread_id, checkpoint_ns, checkpoint_id DESC)
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BYTEA,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """
                )

    def close(self) -> None:
        with self._db_lock:
            if not self._closed:
                self.conn.close()
                self._closed = True

    def __enter__(self) -> PostgresCheckpointSaver:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()):
        if self._closed:
            raise RuntimeError("PostgreSQL checkpoint saver is closed")
        self.conn.execute(f'SET search_path TO "{self.schema}"')
        return self.conn.execute(sql, tuple(params))

    def new_checkpoint_id(self) -> str:
        return str(uuid6())

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        del new_versions
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            raise ValueError("checkpoint['id'] is required")
        checkpoint_id = str(checkpoint_id)
        thread_id, checkpoint_ns = self._ref(config)
        parent_id = self._checkpoint_id(config)
        from reactivegraph.serde import ReactiveSerializer

        serde = self.serde if self.serde is not None else ReactiveSerializer()
        type_, serialized_checkpoint = serde.dumps_typed(checkpoint)
        metadata_bytes = json.dumps(
            metadata or {}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        stored_config = self._with_checkpoint_id(config, checkpoint_id)
        with self._db_lock:
            with self.conn.transaction():
                self._execute(
                    """
                    INSERT INTO checkpoints (
                        thread_id, checkpoint_ns, checkpoint_id,
                        parent_checkpoint_id, type, checkpoint, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO NOTHING
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        parent_id,
                        type_,
                        serialized_checkpoint,
                        metadata_bytes,
                    ),
                )
        return copy.deepcopy(stored_config)

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del task_path
        thread_id, checkpoint_ns = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        if checkpoint_id is None:
            raise ValueError("config.configurable.checkpoint_id is required")
        from reactivegraph.serde import ReactiveSerializer

        serde = self.serde if self.serde is not None else ReactiveSerializer()
        with self._db_lock:
            with self.conn.transaction():
                for index, (channel, value) in enumerate(_iter_writes(writes)):
                    slot = WRITES_IDX_MAP.get(channel, index)
                    type_, serialized = serde.dumps_typed(value)
                    sql = """
                        INSERT INTO writes (
                            thread_id, checkpoint_ns, checkpoint_id, task_id,
                            idx, channel, type, value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    if slot < 0:
                        sql += (
                            " ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, "
                            "task_id, idx) DO UPDATE SET channel = EXCLUDED.channel, "
                            "type = EXCLUDED.type, value = EXCLUDED.value"
                        )
                    else:
                        sql += (
                            " ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, "
                            "task_id, idx) DO NOTHING"
                        )
                    self._execute(
                        sql,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            str(task_id),
                            slot,
                            channel,
                            type_,
                            serialized,
                        ),
                    )

    def get_tuple(self, config: dict[str, Any] | None) -> CheckpointTuple | None:
        if config is None:
            return None
        thread_id, checkpoint_ns = self._ref(config)
        checkpoint_id = self._checkpoint_id(config)
        with self._db_lock:
            if checkpoint_id is None:
                row = self._execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
            else:
                row = self._execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_tuple(row)

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[CheckpointTuple]:
        where: list[str] = []
        params: list[Any] = []
        if config is not None:
            thread_id, checkpoint_ns = self._ref(config)
            configurable = (config or {}).get("configurable") or {}
            where.append("thread_id = %s")
            params.append(thread_id)
            if "checkpoint_ns" in configurable:
                where.append("checkpoint_ns = %s")
                params.append(checkpoint_ns)
        if before is not None:
            before_id = self._checkpoint_id(before)
            if before_id is not None:
                where.append("checkpoint_id < %s")
                params.append(before_id)
        sql = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, "
            "parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY checkpoint_id DESC"
        with self._db_lock:
            rows = self._execute(sql, params).fetchall()
            for row in rows:
                metadata = json.loads(row[6] or b"{}")
                if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
                yield self._row_to_tuple(row)
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        return

    def delete_thread(self, thread_id: str) -> None:
        target = str(thread_id)
        with self._db_lock:
            with self.conn.transaction():
                self._execute("DELETE FROM checkpoints WHERE thread_id = %s", (target,))
                self._execute("DELETE FROM writes WHERE thread_id = %s", (target,))

    def storage_stats(self, thread_id: str) -> dict[str, int]:
        target = str(thread_id)
        with self._db_lock:
            checkpoint_row = self._execute(
                """
                SELECT COUNT(*), COALESCE(SUM(
                    OCTET_LENGTH(COALESCE(checkpoint, ''::bytea)) +
                    OCTET_LENGTH(COALESCE(metadata, ''::bytea))
                ), 0)
                FROM checkpoints WHERE thread_id = %s
                """,
                (target,),
            ).fetchone()
            write_row = self._execute(
                """
                SELECT COUNT(*), COALESCE(SUM(OCTET_LENGTH(COALESCE(value, ''::bytea))), 0)
                FROM writes WHERE thread_id = %s
                """,
                (target,),
            ).fetchone()
        checkpoint_rows = int(checkpoint_row[0])
        checkpoint_bytes = int(checkpoint_row[1] or 0)
        write_rows = int(write_row[0])
        write_bytes = int(write_row[1] or 0)
        return {
            "logical_checkpoint_bytes": checkpoint_bytes,
            "logical_write_bytes": write_bytes,
            "checkpoint_rows": checkpoint_rows,
            "checkpoint_bytes": checkpoint_bytes,
            "blob_rows": 0,
            "blob_bytes": 0,
            "write_rows": write_rows,
            "write_bytes": write_bytes,
        }

    def checkpoint_ids_with_writes(self, thread_id: str) -> set[tuple[str, str]]:
        target = str(thread_id)
        with self._db_lock:
            rows = self._execute(
                "SELECT DISTINCT checkpoint_ns, checkpoint_id FROM writes WHERE thread_id = %s",
                (target,),
            ).fetchall()
        return {(str(row[0] or ""), str(row[1])) for row in rows}

    def delete_checkpoint(self, thread_id: str, key: tuple[str, str]) -> None:
        target = str(thread_id)
        namespace, checkpoint_id = str(key[0]), str(key[1])
        with self._db_lock:
            with self.conn.transaction():
                self._execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s "
                    "AND checkpoint_ns = %s AND checkpoint_id = %s",
                    (target, namespace, checkpoint_id),
                )
                self._execute(
                    "DELETE FROM writes WHERE thread_id = %s "
                    "AND checkpoint_ns = %s AND checkpoint_id = %s",
                    (target, namespace, checkpoint_id),
                )

    def delete_unreachable_blobs(self, thread_id: str, survivor_versions: set[Any]) -> None:
        del thread_id, survivor_versions

    def _row_to_tuple(self, row: Any) -> CheckpointTuple:
        from reactivegraph.serde import ReactiveSerializer

        serde = self.serde if self.serde is not None else ReactiveSerializer()
        checkpoint = serde.loads_typed((row[4], row[5]))
        metadata = json.loads(row[6] or b"{}")
        pending = [
            (
                write[0],
                write[1],
                serde.loads_typed((write[2], write[3])),
            )
            for write in self._execute(
                """
                SELECT task_id, channel, type, value FROM writes
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                ORDER BY task_id, idx
                """,
                (row[0], row[1], row[2]),
            ).fetchall()
        ]
        parent_config = (
            {
                "configurable": {
                    "thread_id": row[0],
                    "checkpoint_ns": row[1],
                    "checkpoint_id": row[3],
                }
            }
            if row[3]
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row[0],
                    "checkpoint_ns": row[1],
                    "checkpoint_id": row[2],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending,
        )

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, before=None, limit=None, filter=None):
        items = await asyncio.to_thread(
            lambda: list(
                self.list(config, before=before, limit=limit, filter=filter)
            )
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def astorage_stats(self, thread_id):
        return await asyncio.to_thread(self.storage_stats, thread_id)

    async def acheckpoint_ids_with_writes(self, thread_id):
        return await asyncio.to_thread(self.checkpoint_ids_with_writes, thread_id)

    async def adelete_checkpoint(self, thread_id, key):
        return await asyncio.to_thread(self.delete_checkpoint, thread_id, key)

    async def adelete_unreachable_blobs(self, thread_id, survivor_versions):
        return await asyncio.to_thread(self.delete_unreachable_blobs, thread_id, survivor_versions)


_checkpointer_lock = threading.Lock()
_checkpointer: CheckpointStore | None = None


def get_checkpointer(config: dict[str, Any] | Any | None = None) -> CheckpointStore:
    """Return the configured process-wide checkpointer singleton.

    ``memory``, ``sqlite``, and ``postgres`` are implemented natively. Unknown
    backends fail closed rather than silently degrading to process-local state.
    """
    global _checkpointer
    if config is None or not isinstance(config, dict):
        config = {"type": "memory"}
    backend = str(config.get("type", "memory"))
    if backend == "postgres" and not config.get("connection_string"):
        raise ValueError("checkpointer.connection_string is required for the postgres backend")
    if backend not in {"memory", "sqlite", "postgres"}:
        raise ValueError(f"unknown checkpointer type: {backend!r}")
    with _checkpointer_lock:
        if _checkpointer is None:
            if backend == "sqlite":
                connection_string = config.get("connection_string") or config.get("path")
                if not connection_string:
                    raise ValueError(
                        "checkpointer.connection_string is required for the sqlite backend"
                    )
                _checkpointer = SqliteCheckpointSaver(connection_string)
            elif backend == "postgres":
                _checkpointer = PostgresCheckpointSaver(
                    config["connection_string"], schema=config.get("schema", "public")
                )
            else:
                _checkpointer = MemoryCheckpointSaver()
        return _checkpointer


def reset_checkpointer() -> None:
    """Drop the singleton so the next ``get_checkpointer()`` builds a fresh one."""
    global _checkpointer
    with _checkpointer_lock:
        if isinstance(_checkpointer, (SqliteCheckpointSaver, PostgresCheckpointSaver)):
            _checkpointer.close()
        _checkpointer = None

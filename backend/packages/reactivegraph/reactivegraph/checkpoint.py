"""Durable checkpoint storage for ReactiveGraph.

This module owns the *storage* contract for graph checkpoints: immutable
snapshots, a parent lineage, pending writes, and lifecycle purge. It is
deliberately engine-agnostic and imports nothing from LangChain/LangGraph.

A host engine that demands a specific saver interface (for example LangGraph's
``BaseCheckpointSaver``) adapts :class:`MemoryCheckpointSaver` in its own
package. Keeping the semantics here means every host shares one tested
implementation instead of re-deriving it.

Scope today is the in-process ``memory`` backend. ``sqlite`` and ``postgres``
fail closed in :func:`get_checkpointer` rather than silently degrading to
process-local state, because that would be data loss presented as durability.
"""

from __future__ import annotations

import asyncio
import copy
import random
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
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
    BaseCheckpointSaver = _NativeBaseCheckpointSaver  # type: ignore[misc,assignment]


class CheckpointStore:
    """In-process checkpoint store with an append-only lineage.

    * ``get_tuple`` / ``aget_tuple``      latest-or-explicit checkpoint
    * ``list`` / ``alist``                newest-first history, cursor, limit, filter
    * ``put`` / ``aput``                  immutable snapshot + parent chain
    * ``put_writes`` / ``aput_writes``    pending writes per checkpoint
    * ``delete_thread`` / ``adelete_thread`` lifecycle purge
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
        out = copy.deepcopy(config or {})
        out.setdefault("configurable", {})
        out["configurable"]["checkpoint_id"] = checkpoint_id
        return out

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
            if key not in self._checkpoints:
                raise KeyError(
                    f"unknown checkpoint {checkpoint_id!r} for thread {thread_key[0]!r}"
                )
            slots = self._writes.setdefault(key, {})
            for index, (channel, value) in enumerate((w[1], w[2]) for w in writes):
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
            for key in [k for k in self._order if k[0] == target]:
                del self._order[key]
            for key in [k for k in self._latest if k[0] == target]:
                del self._latest[key]

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
            keys = (
                sorted(self._order, key=latest_ts, reverse=True)
                if config is None
                else [self._ref(config)]
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


_checkpointer_lock = threading.Lock()
_checkpointer: MemoryCheckpointSaver | None = None


def get_checkpointer(config: dict[str, Any] | Any | None = None) -> MemoryCheckpointSaver:
    """Return the process-wide memory checkpointer singleton.

    Only the ``memory`` backend is implemented natively today. ``sqlite`` and
    ``postgres`` fail closed with an explicit message: silently returning an
    in-process saver would be data loss dressed up as persistence parity.
    """
    global _checkpointer
    if config is None or not isinstance(config, dict):
        config = {"type": "memory"}
    backend = str(config.get("type", "memory"))
    if backend == "postgres" and not config.get("connection_string"):
        raise ValueError("checkpointer.connection_string is required for the postgres backend")
    if backend not in {"memory", "sqlite", "postgres"}:
        raise ValueError(f"unknown checkpointer type: {backend!r}")
    if backend != "memory":
        raise NotImplementedError(
            f"checkpointer backend {backend!r} is not ported to ReactiveGraph yet; "
            "only 'memory' is implemented. Refusing to silently fall back to process-local state."
        )
    with _checkpointer_lock:
        if _checkpointer is None:
            _checkpointer = MemoryCheckpointSaver()
        return _checkpointer


def reset_checkpointer() -> None:
    """Drop the singleton so the next ``get_checkpointer()`` builds a fresh one."""
    global _checkpointer
    with _checkpointer_lock:
        _checkpointer = None

"""Checkpoint primitives: schema types, ids and write normalisation.

``CheckpointTuple``, the id helpers and the version constant are shared by
every backend and by the engine's own checkpoint store. They live here so
``checkpoint.py`` can focus on the storage implementations while
``reactivegraph.checkpoint`` re-exports these names unchanged.

``BaseCheckpointSaver`` resolves to LangGraph's class when that host package is
installed and to the native fail-closed fallback otherwise.
"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

V = TypeVar("V")

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
        """Declared config keys accepted by this saver (empty by default)."""
        return []

    def get(self, config: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the checkpoint dict for *config*, or ``None`` when absent.

        Derived from :meth:`get_tuple` so subclasses only implement storage."""
        value = self.get_tuple(config)
        return value.checkpoint if value else None

    def get_tuple(self, config: Mapping[str, Any]) -> CheckpointTuple | None:
        """Return the :class:`CheckpointTuple` for *config*, or ``None``."""
        raise NotImplementedError

    def list(
        self,
        config: Mapping[str, Any] | None,
        *,
        filter: Mapping[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate checkpoint tuples newest-first, honouring optional filters."""
        raise NotImplementedError

    def put(
        self,
        config: Mapping[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Persist *checkpoint* and return the updated config mapping."""
        raise NotImplementedError

    def put_writes(
        self,
        config: Mapping[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Record pending writes for the task identified by *task_id*."""
        raise NotImplementedError

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint and write belonging to *thread_id*."""
        raise NotImplementedError

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete checkpoints produced by the given run ids."""
        raise NotImplementedError

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy the full checkpoint history to a new thread id."""
        raise NotImplementedError

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Drop old checkpoints for *thread_ids* per *strategy*."""
        raise NotImplementedError

    async def aget_tuple(self, config: Mapping[str, Any]) -> CheckpointTuple | None:
        """Async twin of :meth:`get_tuple`."""
        raise NotImplementedError

    async def alist(
        self,
        config: Mapping[str, Any] | None,
        *,
        filter: Mapping[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ):
        """Async twin of :meth:`list`."""
        raise NotImplementedError

    async def aput(
        self,
        config: Mapping[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Async twin of :meth:`put`."""
        raise NotImplementedError

    async def aput_writes(
        self,
        config: Mapping[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async twin of :meth:`put_writes`."""
        raise NotImplementedError

    async def adelete_thread(self, thread_id: str) -> None:
        """Async twin of :meth:`delete_thread`."""
        raise NotImplementedError

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Async twin of :meth:`delete_for_runs`."""
        raise NotImplementedError

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Async twin of :meth:`copy_thread`."""
        raise NotImplementedError

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Async twin of :meth:`prune`."""
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
        """Async twin of :meth:`get_delta_channel_history`."""
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
        """Return the version token for the next checkpoint."""
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



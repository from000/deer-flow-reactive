"""Thread-scoped checkpoint state for the ``create_agent`` compatibility layer.

A checkpointer is not a decoration: it is the thread's memory. LangGraph's
compiled graph loads the head checkpoint before a run, folds the caller's input
into the stored channel values through the channel reducers, writes the
resulting snapshot back, and exposes ``get_state`` / ``update_state`` /
``get_state_history`` over that lineage. ``create_agent`` must reproduce that,
because DeerFlow's product configuration is
``create_deerflow_agent(..., checkpointer=InMemorySaver())`` and its thread,
compaction, rollback, and resume paths all read state back through those three
methods.

Two saver families have to work, and both are duck-typed here rather than
imported:

* the engine's own :class:`reactivegraph.checkpoint.CheckpointStore`, and
* a host's real ``langgraph.checkpoint.base.BaseCheckpointSaver`` (DeerFlow
  passes ``InMemorySaver``/``AsyncSqliteSaver`` straight through).

They agree on the important shapes: ``get_tuple(config)`` returns a record with
``checkpoint``/``metadata``/``parent_config``/``pending_writes``, ``put`` takes
``(config, checkpoint, metadata, new_versions)`` and returns the config carrying
the new ``checkpoint_id``, and ``put_writes`` takes ``(config, writes, task_id)``
with ``(channel, value)`` pairs. Async twins mirror them.

Anything else fails closed. Silently treating an unknown object as "no
checkpointer" would make every turn start from a blank thread — data loss that
looks like a working graph.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NamedTuple

from reactivegraph.channels import _MISSING as _CHANNEL_EMPTY
from reactivegraph.channels import DeltaChannel, _host_delta_snapshot, channel_reducer

__all__ = (
    "NO_CHECKPOINTER_MESSAGE",
    "THREAD_ID_REQUIRED_MESSAGE",
    "CheckpointStateError",
    "StateSnapshot",
    "ThreadCheckpointer",
    "new_checkpoint_id",
)

NO_CHECKPOINTER_MESSAGE = "No checkpointer set"
"""Upstream's exact wording for state access without a checkpointer."""

THREAD_ID_REQUIRED_MESSAGE = (
    "Checkpointer requires one or more of the following 'configurable' keys: "
    "thread_id, checkpoint_ns, checkpoint_id"
)
"""Upstream's exact wording when a threaded call arrives without a thread id."""

LATEST_CHECKPOINT_VERSION = 2
DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT = 5000


class CheckpointStateError(ValueError):
    """A configured saver cannot honour a thread-state operation.

    Raised when the saver is a *structurally valid* checkpointer (it passed the
    ``get_tuple``/``put``/``put_writes`` probe) but cannot serve a specific
    request, e.g. ``get_state_history`` against a saver with no ``list()``.
    Invalid objects are rejected earlier with upstream's ``TypeError``.
    """


class StateSnapshot(NamedTuple):
    """The materialized thread state at one checkpoint.

    Field order matches ``langgraph.types.StateSnapshot``; consumers read
    ``values``/``next``/``config``/``metadata``/``created_at``/``parent_config``
    and DeerFlow additionally reads ``metadata`` for its channel-mode marker.
    ``tasks`` and ``interrupts`` are always empty here: this engine writes a
    finished checkpoint per run, so no task is left pending.
    """

    values: dict[str, Any]
    next: tuple[str, ...]
    config: dict[str, Any]
    metadata: dict[str, Any] | None
    created_at: str | None
    parent_config: dict[str, Any] | None
    tasks: tuple[Any, ...]
    interrupts: tuple[Any, ...]


def new_checkpoint_id() -> str:
    """A fresh, ordered checkpoint id for savers that do not mint their own."""
    from reactivegraph.checkpoint import uuid6

    return str(uuid6(clock_seq=-2))


def _empty_checkpoint(checkpoint_id: str) -> dict[str, Any]:
    return {
        "v": LATEST_CHECKPOINT_VERSION,
        "id": checkpoint_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }


def _as_delta_write(write: Sequence[Any]) -> tuple[Any, Any, Any]:
    """Normalize a pending write to ``(task_id, channel, value)``.

    Upstream stores triples; a duck-typed saver may store ``(channel, value)``
    pairs. ``DeltaChannel.replay_writes`` only reads the value, but the shape
    has to be uniform before it is handed over.
    """
    if len(write) >= 3:
        return write[0], write[1], write[2]
    return "", write[0], write[1]


def _normalize_delta_writes(writes: Any) -> list[tuple[Any, Any, Any]]:
    if not isinstance(writes, Sequence) or isinstance(writes, (str, bytes)):
        return []
    return [_as_delta_write(write) for write in writes if len(write) >= 2]


def _configurable(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    value = config.get("configurable")
    return dict(value) if isinstance(value, Mapping) else {}


def _config_snapshot(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Isolate a config envelope without copying the host objects inside it.

    A ``RunnableConfig`` can carry non-deep-copyable execution state (the
    run-scoped ``Runtime`` pins a store whose lock is a ``threading.RLock``).
    Copying the outer mapping and the ``configurable``/``metadata`` envelopes
    gives the caller a mutable snapshot without ever touching the values
    themselves, which is what upstream's ``StateSnapshot`` does as well.
    """
    if not isinstance(config, Mapping):
        return {}
    out = dict(config)
    for key in ("configurable", "metadata"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            out[key] = dict(nested)
    return out


def _merge_config_metadata(
    config: Mapping[str, Any] | None, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge caller config metadata into the stored checkpoint metadata.

    Mirrors upstream ``get_checkpoint_metadata``: only scalar values from
    ``config['metadata']``/``config['configurable']`` are copied, ``__*`` keys
    and LangGraph's internal bookkeeping keys are skipped, and an explicit
    metadata key always wins. DeerFlow records the thread's agent binding this
    way and reads it back off the snapshot.
    """
    merged = dict(metadata)
    if not isinstance(config, Mapping):
        return merged
    for source in (config.get("metadata"), config.get("configurable")):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if not isinstance(key, str) or key.startswith("__"):
                continue
            if key in merged or key in _EXCLUDED_METADATA_KEYS:
                continue
            if isinstance(value, str):
                merged[key] = value.replace("\u0000", "")
            elif isinstance(value, (int, bool, float)):
                merged[key] = value
    return merged


_EXCLUDED_METADATA_KEYS = frozenset(
    {
        "langgraph_node",
        "langgraph_triggers",
        "thread_id",
        "langgraph_step",
        "langgraph_checkpoint_ns",
        "checkpoint_id",
        "checkpoint_map",
        "checkpoint_ns",
        "langgraph_path",
    }
)


@dataclass
class _Head:
    """The loaded head checkpoint, normalized across both saver families."""

    config: dict[str, Any]
    checkpoint: dict[str, Any]
    metadata: dict[str, Any]
    parent_config: dict[str, Any] | None
    pending_writes: list[Any]
    thread_id: str
    checkpoint_ns: str


class ThreadCheckpointer:
    """Duck-typed adapter that binds one saver to one channel table.

    ``channels`` is the compiled channel table from
    :func:`reactivegraph.channels.resolve_channels`; it is what turns stored
    channel values plus the caller's input into the next checkpoint through the
    same reducers the graph itself commits with.
    """

    def __init__(self, saver: Any, channels: Mapping[str, Any]) -> None:
        self._saver = saver
        self._channels = dict(channels)
        self._validate()

    # -- capability probing ------------------------------------------------

    def _validate(self) -> None:
        """Reject a checkpointer we cannot drive, before any state is touched.

        The wording matches upstream ``langgraph.types.ensure_valid_checkpointer``
        (``TypeError``) so callers that already guard ``create_agent`` see the
        same failure; silently ignoring the object would make every threaded
        turn start a blank thread.

        A saver is drivable when it can complete *either* read/write family:
        the sync pair (``get_tuple``/``put``) or the async pair
        (``aget_tuple``/``aput``). Probing the sync trio only rejected
        async-only duck savers that upstream LangGraph drives happily — e.g.
        DeerFlow's race-injecting test wrappers, which forward just
        ``aget_tuple``/``aput``. ``put_writes``/``aput_writes`` are needed only
        when a caller commits with a ``task_id``, so they are guarded at the
        call site instead of being demanded up front.
        """
        sync_ok = all(
            callable(getattr(self._saver, name, None))
            for name in ("get_tuple", "put")
        )
        async_ok = all(
            callable(getattr(self._saver, name, None))
            for name in ("aget_tuple", "aput")
        )
        if not (sync_ok or async_ok):
            raise TypeError(
                "Invalid checkpointer provided. Expected an instance of "
                "`BaseCheckpointSaver`, `True`, `False`, or `None`. "
                f"Received {type(self._saver).__name__}. "
                "Pass a proper saver (e.g., InMemorySaver, AsyncPostgresSaver)."
            )

    def _supports_async(self) -> bool:
        """True when the saver can serve the async read/write family.

        ``aput_writes`` is deliberately not part of the probe: a task-scoped
        commit that needs it fails closed in :meth:`acommit` with a named
        error, while state-only commits (the common case) still work.
        """
        return all(
            callable(getattr(self._saver, name, None))
            for name in ("aget_tuple", "aput")
        )

    # -- config plumbing ---------------------------------------------------

    @staticmethod
    def resolve_thread_id(config: Mapping[str, Any] | None) -> str | None:
        """Return the effective thread id for a config."""
        thread_id = _configurable(config).get("thread_id")
        return str(thread_id) if thread_id else None

    def require_thread_id(self, config: Mapping[str, Any] | None) -> str:
        """Return the thread id, raising when it is absent."""
        thread_id = self.resolve_thread_id(config)
        if thread_id is None:
            raise ValueError(THREAD_ID_REQUIRED_MESSAGE)
        return thread_id

    @staticmethod
    def _with_configurable(config: Mapping[str, Any] | None, **values: Any) -> dict[str, Any]:
        # Shallow copy on purpose: a RunnableConfig legitimately carries values
        # that cannot be deep-copied (locks, open handles, live host objects).
        # ``deepcopy`` raised ``cannot pickle '_thread.RLock' object`` here and
        # aborted the whole run before a single node executed. Only the
        # ``configurable`` mapping is rewritten; every other top-level key is
        # passed through by reference, which is what the caller's config expects.
        out = dict(config) if isinstance(config, Mapping) else {}
        configurable = dict(_configurable(config))
        for key, value in values.items():
            if value is None:
                configurable.pop(key, None)
            else:
                configurable[key] = value
        out["configurable"] = configurable
        return out

    def _read_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        """Config pinned to an explicit thread/namespace for a head read.

        The caller's own ``checkpoint_id`` (if any) is preserved so a caller can
        address a specific historical checkpoint, matching ``get_tuple``.
        """
        configurable = _configurable(config)
        values: dict[str, Any] = {
            "thread_id": self.require_thread_id(config),
            "checkpoint_ns": str(configurable.get("checkpoint_ns") or ""),
        }
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id:
            values["checkpoint_id"] = checkpoint_id
        return self._with_configurable(config, **values)

    # -- reads -------------------------------------------------------------

    def get_head(self, config: Mapping[str, Any] | None) -> _Head | None:
        """Load the addressed checkpoint, or ``None`` for an unseen thread."""
        read_config = self._read_config(config)
        record = self._saver.get_tuple(read_config)
        return self._to_head(record, read_config)

    async def aget_head(self, config: Mapping[str, Any] | None) -> _Head | None:
        """Async twin of :meth:`head`."""
        if not self._supports_async():
            # A sync-only saver is still drivable from async callers; the graph
            # body already runs in an executor thread.
            return self.get_head(config)
        read_config = self._read_config(config)
        record = await self._saver.aget_tuple(read_config)
        return self._to_head(record, read_config)

    @staticmethod
    def _to_head(record: Any, read_config: dict[str, Any]) -> _Head | None:
        if record is None:
            return None
        checkpoint = getattr(record, "checkpoint", None)
        if not isinstance(checkpoint, Mapping):
            raise CheckpointStateError(
                f"checkpointer returned {type(record).__name__} without a checkpoint "
                "mapping; cannot materialize thread state."
            )
        config = getattr(record, "config", None) or read_config
        configurable = _configurable(config)
        return _Head(
            config=dict(config),
            checkpoint=copy.deepcopy(dict(checkpoint)),
            metadata=dict(getattr(record, "metadata", None) or {}),
            parent_config=getattr(record, "parent_config", None),
            pending_writes=list(getattr(record, "pending_writes", None) or ()),
            thread_id=str(
                configurable.get("thread_id")
                or _configurable(read_config).get("thread_id")
                or ""
            ),
            checkpoint_ns=str(configurable.get("checkpoint_ns") or ""),
        )

    def snapshot(self, head: _Head | None, config: Mapping[str, Any] | None) -> StateSnapshot:
        """Materialize a stored head as a ``StateSnapshot``.

        An unseen thread yields an empty snapshot carrying the caller's config
        and ``metadata=None``, exactly as upstream does — callers probe with
        ``get_state`` before a thread exists.
        """
        if head is None:
            return StateSnapshot(
                values={},
                next=(),
                config=_config_snapshot(config),
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )
        return StateSnapshot(
            values=self.load_values(head),
            next=(),
            config=_config_snapshot(head.config),
            metadata=dict(head.metadata),
            created_at=head.checkpoint.get("ts"),
            parent_config=(
                _config_snapshot(head.parent_config)
                if head.parent_config is not None
                else None
            ),
            tasks=(),
            interrupts=(),
        )


    async def asnapshot(
        self, head: _Head | None, config: Mapping[str, Any] | None
    ) -> StateSnapshot:
        """Async twin of :meth:`snapshot`.

        A saver with an async delta-history override (``AsyncSqliteSaver``)
        rejects the synchronous walk from the event loop thread, so the async
        read path must never fall back to :meth:`load_values`.
        """
        if head is None:
            return StateSnapshot(
                values={},
                next=(),
                config=_config_snapshot(config),
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )
        return StateSnapshot(
            values=await self.aload_values(head),
            next=(),
            config=_config_snapshot(head.config),
            metadata=dict(head.metadata),
            created_at=head.checkpoint.get("ts"),
            parent_config=(
                _config_snapshot(head.parent_config)
                if head.parent_config is not None
                else None
            ),
            tasks=(),
            interrupts=(),
        )

    def history(
        self,
        config: Mapping[str, Any] | None,
        *,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> Iterator[StateSnapshot]:
        """Iterate checkpoint history, newest first."""
        lister = getattr(self._saver, "list", None)
        if not callable(lister):
            raise CheckpointStateError(
                f"checkpointer {type(self._saver).__name__!r} cannot list checkpoints, "
                "so get_state_history() cannot be honoured. Refusing to return a "
                "partial history."
            )
        read_config = self._read_config(config)
        kwargs: dict[str, Any] = {"limit": limit}
        if before is not None:
            kwargs["before"] = self._with_configurable(
                before,
                thread_id=self.require_thread_id(config),
                checkpoint_ns=str(_configurable(config).get("checkpoint_ns") or ""),
            )
        if filter is not None:
            kwargs["filter"] = dict(filter)
        # Drain before yielding, exactly as upstream does ("eagerly consume
        # list() to avoid holding up the db cursor"): a consumer that stops
        # early must not leave the saver's cursor open, and materializing the
        # snapshots up front keeps a mid-iteration saver error from surfacing
        # halfway through a caller's write.
        records = list(lister(read_config, **kwargs))
        for record in records:
            head = self._to_head(record, read_config)
            if head is not None:
                yield self.snapshot(head, config)

    async def ahistory(
        self,
        config: Mapping[str, Any] | None,
        *,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        """Async twin of :meth:`history`."""
        if not self._supports_async() or not callable(getattr(self._saver, "alist", None)):
            for snapshot in self.history(config, before=before, limit=limit, filter=filter):
                yield snapshot
            return
        read_config = self._read_config(config)
        kwargs: dict[str, Any] = {"limit": limit}
        if before is not None:
            kwargs["before"] = self._with_configurable(
                before,
                thread_id=self.require_thread_id(config),
                checkpoint_ns=str(_configurable(config).get("checkpoint_ns") or ""),
            )
        if filter is not None:
            kwargs["filter"] = dict(filter)
        # Same eager drain as the sync path: ``AsyncSqliteSaver.alist`` closes
        # its cursor in an async-with, and a consumer ``break`` would otherwise
        # leave it open (``ValueError: Connection closed``).
        records = [record async for record in self._saver.alist(read_config, **kwargs)]
        for record in records:
            head = self._to_head(record, read_config)
            if head is not None:
                yield await self.asnapshot(head, config)

    # -- writes ------------------------------------------------------------

    def _changed_channels(
        self, head: _Head | None, values: Mapping[str, Any], written: Sequence[str]
    ) -> list[str]:
        """Every channel whose stored value actually changed in this commit.

        The caller names the channels its input touched, but middleware and the
        graph body can write others (``note``, ``summary_text``, ``jump_to``).
        Savers key channel blobs by version — LangGraph's ``InMemorySaver`` uses
        ``(thread, ns, channel, version)`` — so a changed value that keeps its
        parent's version is filed under the parent's key and rewrites history in
        place. Compare values to catch those writes; when a value cannot be
        compared safely, assume it changed rather than risk a silent collision.
        """
        stored = head.checkpoint.get("channel_values") if head is not None else None
        stored = stored if isinstance(stored, Mapping) else {}
        touched = list(dict.fromkeys(written))
        seen = set(touched)
        for channel, value in values.items():
            if channel in seen:
                continue
            if channel not in stored:
                # A DeltaChannel is deliberately absent from a non-snapshot
                # checkpoint; ``values`` may still contain its replayed value.
                # That is reconstructed state, not a write in this superstep.
                # Upstream likewise derives ``updated_channels`` from task
                # writes, so counting the replay as a change would spuriously
                # bump its version and cadence on every read-modify-commit.
                if not isinstance(self._channels.get(channel), DeltaChannel):
                    touched.append(channel)
                    seen.add(channel)
                continue
            previous = stored[channel]
            if previous is value:
                continue
            try:
                unchanged = bool(previous == value)
            except Exception:  # noqa: BLE001 - an uncomparable value is a change
                unchanged = False
            if not unchanged:
                touched.append(channel)
                seen.add(channel)
        return touched

    def _next_versions(self, head: _Head | None, written: Sequence[str]) -> dict[str, Any]:
        current = dict(head.checkpoint.get("channel_versions") or {}) if head is not None else {}
        next_version = getattr(self._saver, "get_next_version", None)
        versions: dict[str, Any] = {}
        for channel in written:
            if callable(next_version):
                versions[channel] = next_version(current.get(channel), channel)
            else:
                base = current.get(channel)
                versions[channel] = (int(base) if isinstance(base, int) else 0) + 1
        return versions

    def _checkpoint_for(
        self,
        head: _Head | None,
        values: Mapping[str, Any],
        written: Sequence[str],
        versions: Mapping[str, Any],
        *,
        channels_to_snapshot: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build the snapshot to store.

        ``versions`` is passed in rather than derived here: savers that hash the
        version into a blob key (LangGraph's ``InMemorySaver`` returns
        ``"<counter>.<random>"``) must receive *the same* mapping that lands in
        ``channel_versions``, or the stored blob is filed under a key the
        checkpoint never references and reads back empty.
        """
        checkpoint = (
            copy.deepcopy(head.checkpoint)
            if head is not None
            else _empty_checkpoint(new_checkpoint_id())
        )
        # ``id``/``ts`` describe *this* snapshot, never the parent's.
        checkpoint["id"] = new_checkpoint_id()
        checkpoint["ts"] = datetime.now(timezone.utc).isoformat()
        checkpoint["v"] = checkpoint.get("v") or LATEST_CHECKPOINT_VERSION
        channel_values = copy.deepcopy(dict(values))
        if channels_to_snapshot is not None:
            for name, channel in self._channels.items():
                if not isinstance(channel, DeltaChannel):
                    continue
                if name in channels_to_snapshot:
                    # Host savers serialise through their own serde, and host
                    # DeltaChannel.from_checkpoint recognises only its own
                    # namedtuple blob.
                    channel_values[name] = _host_delta_snapshot(values[name])
                    continue
                # Non-snapshot delta writes live on the parent checkpoint's
                # pending writes; the ancestor walk reconstructs the value.
                channel_values.pop(name, None)
        checkpoint["channel_values"] = channel_values
        channel_versions = dict(checkpoint.get("channel_versions") or {})
        channel_versions.update(versions)
        checkpoint["channel_versions"] = channel_versions
        checkpoint["updated_channels"] = list(written)
        checkpoint.setdefault("versions_seen", {})
        checkpoint.setdefault("pending_sends", [])
        return checkpoint

    def _delta_plan(
        self,
        head: _Head | None,
        values: Mapping[str, Any],
        written: Sequence[str],
        delta_writes: Mapping[str, Any] | None,
    ) -> tuple[set[str], dict[str, Any], dict[str, Any], bool]:
        """Plan DeltaChannel snapshot/counter state for one state mutation.

        Returns ``(channels_to_snapshot, metadata_extra, parent_writes,
        plan_delta)``.
        Only callers that provide the *raw* writes can use the non-snapshot
        path: a folded full value cannot be replayed through a reducer without
        duplicating it.
        """
        if delta_writes is None:
            # ``None`` means the caller has only a folded value and therefore
            # cannot use ancestor replay; it must keep full snapshot semantics.
            # An empty mapping is still a plan: the caller performed a mutation
            # that touched no delta channel, so omission is safe.
            return set(), {}, {}, False
        delta_channels = {
            name: channel
            for name, channel in self._channels.items()
            if isinstance(channel, DeltaChannel)
        }
        if not delta_channels:
            return set(), {}, {}, True
        touched = set(written)
        raw = {
            name: delta_writes[name]
            for name in touched
            if name in delta_channels and name in delta_writes
        }
        if head is None:
            head_snapshots = {
                name for name in delta_channels if name in values or name in raw
            }
            # A fresh thread has no parent to carry pending writes; upstream
            # therefore snapshots the folded value into this first checkpoint.
            return head_snapshots, {}, {}, True
        previous = dict(head.metadata.get("counters_since_delta_snapshot") or {})
        snapshots, counters = self._advance_delta_counters(
            delta_channels, previous, touched, values
        )
        metadata = {"counters_since_delta_snapshot": counters} if counters else {}
        return snapshots, metadata, raw, True

    @staticmethod
    def _advance_delta_counters(
        delta_channels: Mapping[str, DeltaChannel],
        previous: Mapping[str, Any],
        touched: set[str],
        values: Mapping[str, Any],
    ) -> tuple[set[str], dict[str, Any]]:
        """Advance per-channel counters and pick which channels to snapshot.

        Upstream advances counters for every delta channel but snapshots only
        channels whose current value is available. A never-written channel can
        reach its cadence while still absent from ``values``; adding it here
        would make the checkpoint builder index a missing value. Keep its
        counters so a later real write snapshots promptly.
        """
        snapshots: set[str] = set()
        counters: dict[str, Any] = {}
        for name, channel in delta_channels.items():
            current = previous.get(name, (0, 0))
            updates, supersteps = int(current[0]), int(current[1])
            supersteps += 1
            if name in touched:
                updates += 1
            due = (
                updates >= channel.snapshot_frequency
                or supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
            )
            if due and name in values:
                snapshots.add(name)
                counters[name] = (0, 0)
            else:
                counters[name] = (updates, supersteps)
        counters = {name: value for name, value in counters.items() if value != (0, 0)}
        return snapshots, counters

    @staticmethod
    def _delta_write_task_id(checkpoint_id: str) -> str:
        try:
            return str(uuid.uuid5(uuid.UUID(checkpoint_id), "update_state"))
        except (ValueError, AttributeError):
            return f"update_state:{checkpoint_id}"

    def commit(
        self,
        config: Mapping[str, Any] | None,
        head: _Head | None,
        values: Mapping[str, Any],
        *,
        written: Sequence[str],
        metadata: Mapping[str, Any],
        task_id: str | None = None,
        delta_writes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist *values* as a new checkpoint and return the new config."""
        thread_id = self.require_thread_id(config)
        checkpoint_ns = str(_configurable(config).get("checkpoint_ns") or "")
        parent_config = head.config if head is not None else None
        parent_id = _configurable(parent_config).get("checkpoint_id") if parent_config else None
        written = self._changed_channels(head, values, written)
        versions = self._next_versions(head, written)
        channels_to_snapshot, delta_metadata, parent_writes, plan_delta = self._delta_plan(
            head, values, written, delta_writes
        )
        checkpoint = self._checkpoint_for(
            head,
            values,
            written,
            versions,
            channels_to_snapshot=channels_to_snapshot if plan_delta else None,
        )
        write_config = self._with_configurable(
            config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=parent_id,
        )
        if parent_writes:
            put_writes = getattr(self._saver, "put_writes", None)
            if not callable(put_writes):
                raise CheckpointStateError(
                    f"checkpointer {type(self._saver).__name__!r} cannot record "
                    "pending delta writes, so a non-snapshot update cannot be "
                    "reconstructed. Refusing to commit state that would be lost."
                )
            put_writes(
                write_config,
                list(parent_writes.items()),
                task_id or self._delta_write_task_id(checkpoint["id"]),
            )
        stored = self._saver.put(
            write_config,
            checkpoint,
            _merge_config_metadata(config, {**metadata, **delta_metadata}),
            versions,
        )
        stored_config = dict(stored) if isinstance(stored, Mapping) else write_config
        new_id = _configurable(stored_config).get("checkpoint_id") or checkpoint["id"]
        result = self._with_configurable(
            stored_config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=new_id,
        )
        if task_id:
            # Pending writes are the durability signal DeerFlow reads to decide
            # whether a turn is still in flight; record the same channel values
            # the checkpoint committed so the pair never disagrees.
            put_writes = getattr(self._saver, "put_writes", None)
            if not callable(put_writes):
                raise CheckpointStateError(
                    f"checkpointer {type(self._saver).__name__!r} cannot record "
                    "pending writes, so a task-scoped commit cannot be honoured. "
                    "Refusing to commit a checkpoint whose durability signal "
                    "would be missing."
                )
            put_writes(
                result,
                [(channel, values[channel]) for channel in written if channel in values],
                task_id,
            )
        return result

    async def acommit(
        self,
        config: Mapping[str, Any] | None,
        head: _Head | None,
        values: Mapping[str, Any],
        *,
        written: Sequence[str],
        metadata: Mapping[str, Any],
        task_id: str | None = None,
        delta_writes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._supports_async():
            return self.commit(
                config,
                head,
                values,
                written=written,
                metadata=metadata,
                task_id=task_id,
                delta_writes=delta_writes,
            )
        thread_id = self.require_thread_id(config)
        checkpoint_ns = str(_configurable(config).get("checkpoint_ns") or "")
        parent_config = head.config if head is not None else None
        parent_id = _configurable(parent_config).get("checkpoint_id") if parent_config else None
        written = self._changed_channels(head, values, written)
        versions = self._next_versions(head, written)
        channels_to_snapshot, delta_metadata, parent_writes, plan_delta = self._delta_plan(
            head, values, written, delta_writes
        )
        checkpoint = self._checkpoint_for(
            head,
            values,
            written,
            versions,
            channels_to_snapshot=channels_to_snapshot if plan_delta else None,
        )
        write_config = self._with_configurable(
            config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=parent_id,
        )
        if parent_writes:
            aput_writes = getattr(self._saver, "aput_writes", None)
            if not callable(aput_writes):
                raise CheckpointStateError(
                    f"checkpointer {type(self._saver).__name__!r} cannot record "
                    "pending delta writes, so a non-snapshot update cannot be "
                    "reconstructed. Refusing to commit state that would be lost."
                )
            await aput_writes(
                write_config,
                list(parent_writes.items()),
                task_id or self._delta_write_task_id(checkpoint["id"]),
            )
        stored = await self._saver.aput(
            write_config,
            checkpoint,
            _merge_config_metadata(config, {**metadata, **delta_metadata}),
            versions,
        )
        stored_config = dict(stored) if isinstance(stored, Mapping) else write_config
        new_id = _configurable(stored_config).get("checkpoint_id") or checkpoint["id"]
        result = self._with_configurable(
            stored_config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=new_id,
        )
        if task_id:
            aput_writes = getattr(self._saver, "aput_writes", None)
            if not callable(aput_writes):
                raise CheckpointStateError(
                    f"checkpointer {type(self._saver).__name__!r} cannot record "
                    "pending writes, so a task-scoped commit cannot be honoured. "
                    "Refusing to commit a checkpoint whose durability signal "
                    "would be missing."
                )
            await aput_writes(
                result,
                [(channel, values[channel]) for channel in written if channel in values],
                task_id,
            )
        return result

    # -- channel folding ---------------------------------------------------

    def load_values(self, head: _Head | None) -> dict[str, Any]:
        """Materialize every channel of *head*, replaying delta channels.

        A delta channel keeps its value out of ``channel_values`` except on the
        snapshot cadence; between snapshots the value exists only as
        ``pending_writes`` on the ancestors that produced it. Upstream's
        ``channels_from_checkpoint`` therefore replays the ancestor history for
        exactly those channels, and reading ``channel_values`` alone would
        silently report an empty transcript.
        """
        if head is None:
            return {}
        values = head.checkpoint.get("channel_values")
        loaded = copy.deepcopy(dict(values)) if isinstance(values, Mapping) else {}
        loaded.update(self._replay_delta_values(head, loaded))
        return loaded

    async def aload_values(self, head: _Head | None) -> dict[str, Any]:
        """Async twin of :meth:`load_values`."""
        if head is None:
            return {}
        values = head.checkpoint.get("channel_values")
        loaded = copy.deepcopy(dict(values)) if isinstance(values, Mapping) else {}
        loaded.update(await self._areplay_delta_values(head, loaded))
        return loaded

    # -- delta reconstruction ----------------------------------------------

    def _replay_delta_values(
        self, head: _Head, stored: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Rebuild delta channels whose stored value needs an ancestor replay.

        Mirrors upstream ``_needs_replay``: a ``DeltaChannel`` whose
        ``channel_values`` entry is absent needs the ancestor walk. A stored
        snapshot blob is decoded through the channel contract, while a legacy
        plain value restores directly.
        """
        replayed: dict[str, Any] = {}
        pending: list[str] = []
        for name, spec in self._channels.items():
            if not isinstance(spec, DeltaChannel):
                continue
            if name not in stored:
                pending.append(name)
                continue
            channel = spec.from_checkpoint(stored[name])
            if channel.is_available():
                replayed[name] = channel.get()
        if not pending:
            return replayed
        histories = self._delta_histories(head, pending)
        for name in pending:
            history = histories.get(name)
            if not isinstance(history, Mapping):
                continue
            channel = self._channels[name].from_checkpoint(
                history.get("seed", _CHANNEL_EMPTY)
            )
            channel.replay_writes(_normalize_delta_writes(history.get("writes")))
            if channel.is_available():
                replayed[name] = channel.get()
        return replayed

    async def _areplay_delta_values(
        self, head: _Head, stored: Mapping[str, Any]
    ) -> dict[str, Any]:
        replayed: dict[str, Any] = {}
        pending: list[str] = []
        for name, spec in self._channels.items():
            if not isinstance(spec, DeltaChannel):
                continue
            if name not in stored:
                pending.append(name)
                continue
            channel = spec.from_checkpoint(stored[name])
            if channel.is_available():
                replayed[name] = channel.get()
        if not pending:
            return replayed
        histories = await self._adelta_histories(head, pending)
        for name in pending:
            history = histories.get(name)
            if not isinstance(history, Mapping):
                continue
            channel = self._channels[name].from_checkpoint(
                history.get("seed", _CHANNEL_EMPTY)
            )
            channel.replay_writes(_normalize_delta_writes(history.get("writes")))
            if channel.is_available():
                replayed[name] = channel.get()
        return replayed

    async def _adelta_histories(
        self, head: _Head, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        getter = getattr(self._saver, "aget_delta_channel_history", None)
        if callable(getter):
            histories = await getter(config=head.config, channels=list(channels))
            return histories if isinstance(histories, Mapping) else {}
        sync_getter = getattr(self._saver, "get_delta_channel_history", None)
        if callable(sync_getter) and not self._supports_async():
            histories = sync_getter(config=head.config, channels=list(channels))
            return histories if isinstance(histories, Mapping) else {}
        return await self._awalk_delta_history(head.config, channels)

    async def _awalk_delta_history(
        self, config: Mapping[str, Any], channels: Sequence[str]
    ) -> dict[str, Any]:
        collected: dict[str, list[tuple[Any, Any, Any]]] = {name: [] for name in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)
        target = await self._saver.aget_tuple(config)
        cursor = getattr(target, "parent_config", None) if target is not None else None
        while cursor is not None and remaining:
            record = await self._saver.aget_tuple(cursor)
            if record is None:
                break
            for write in reversed(getattr(record, "pending_writes", None) or ()):
                if len(write) < 2 or write[1] not in remaining:
                    continue
                collected[write[1]].append(_as_delta_write(write))
            values = getattr(record, "checkpoint", None)
            values = values.get("channel_values") if isinstance(values, Mapping) else None
            values = values if isinstance(values, Mapping) else {}
            for name in list(remaining):
                if name in values:
                    seeds[name] = values[name]
                    remaining.discard(name)
            cursor = getattr(record, "parent_config", None)
        result: dict[str, Any] = {}
        for name in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[name]))}
            if name in seeds:
                entry["seed"] = seeds[name]
            result[name] = entry
        return result

    def _delta_histories(
        self, head: _Head, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        """One batched ancestor walk, preferring the saver's own override.

        ``BaseCheckpointSaver.get_delta_channel_history`` is the upstream
        contract; savers with direct storage access override it for speed. A
        duck-typed saver that predates the method still gets the documented
        parent-chain walk rather than a silently empty value.
        """
        getter = getattr(self._saver, "get_delta_channel_history", None)
        if callable(getter):
            histories = getter(config=head.config, channels=list(channels))
            return histories if isinstance(histories, Mapping) else {}
        return self._walk_delta_history(head.config, channels)

    def _walk_delta_history(
        self, config: Mapping[str, Any], channels: Sequence[str]
    ) -> dict[str, Any]:
        """Upstream's default walk: collect ancestor writes and the nearest seed.

        The target checkpoint's own writes belong to its child, so the walk
        starts at ``parent_config``; it terminates per channel at the nearest
        ancestor that stored a blob for it.
        """
        collected: dict[str, list[tuple[Any, Any, Any]]] = {name: [] for name in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)
        target = self._saver.get_tuple(config)
        cursor = getattr(target, "parent_config", None) if target is not None else None
        while cursor is not None and remaining:
            record = self._saver.get_tuple(cursor)
            if record is None:
                break
            for write in reversed(getattr(record, "pending_writes", None) or ()):
                if len(write) < 2 or write[1] not in remaining:
                    continue
                collected[write[1]].append(_as_delta_write(write))
            values = getattr(record, "checkpoint", None)
            values = values.get("channel_values") if isinstance(values, Mapping) else None
            values = values if isinstance(values, Mapping) else {}
            for name in list(remaining):
                if name in values:
                    seeds[name] = values[name]
                    remaining.discard(name)
            cursor = getattr(record, "parent_config", None)
        result: dict[str, Any] = {}
        for name in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[name]))}
            if name in seeds:
                entry["seed"] = seeds[name]
            result[name] = entry
        return result

    def fold_input(
        self, values: Mapping[str, Any], update: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Fold a caller input through the channel table.

        Returns the folded values and the channel names that changed. A reducer
        channel accumulates (``messages`` appends); a last-value channel is
        replaced. The write is handed to the channel's own ``update`` hook, so
        an ``Overwrite`` replacement and every subclass normalisation behave
        exactly as they do inside a graph run.
        """
        current = dict(values)
        written: list[str] = []
        for key, write in update.items():
            present = key in current
            current[key] = self._apply_write(
                key, current.get(key), write, present=present
            )
            written.append(key)
        return current, written

    def _apply_write(
        self, key: str, current: Any, write: Any, *, present: bool
    ) -> Any:
        """Fold one write through the channel's own ``update`` contract.

        Upstream's runtime restores the channel from its stored value and then
        calls ``channel.update([write])`` — the write is never unwrapped first,
        because a reducer channel is expected to interpret ``Overwrite``
        itself. That hook is part of the contract: DeerFlow's
        ``TaskNotesChannel`` re-validates and re-canonicalizes *every* write
        there, including first writes and replacements. Calling the raw
        ``operator`` (or unwrapping before the call) bypasses that validation
        and persists caller-supplied values verbatim.

        Last-value channels keep replacement semantics, and a missing channel
        table entry fails open to replacement as before.
        """
        channel = self._channels.get(key)
        if channel is None or channel_reducer(channel) is None:
            return write
        restored = channel.from_checkpoint(current if present else _CHANNEL_EMPTY)
        restored.update([write])
        if not restored.is_available():
            # ``add_messages`` rejects a ``None`` left operand; upstream leaves
            # the channel empty rather than storing a sentinel value.
            return write
        return restored.get()

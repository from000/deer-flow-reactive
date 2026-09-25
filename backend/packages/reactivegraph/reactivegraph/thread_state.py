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
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NamedTuple

from reactivegraph.channels import channel_reducer
from reactivegraph.types import Overwrite

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


def _configurable(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    value = config.get("configurable")
    return dict(value) if isinstance(value, Mapping) else {}


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
        """
        missing = [
            name
            for name in ("get_tuple", "put", "put_writes")
            if not callable(getattr(self._saver, name, None))
        ]
        if missing:
            raise TypeError(
                "Invalid checkpointer provided. Expected an instance of "
                "`BaseCheckpointSaver`, `True`, `False`, or `None`. "
                f"Received {type(self._saver).__name__}. "
                "Pass a proper saver (e.g., InMemorySaver, AsyncPostgresSaver)."
            )

    def _supports_async(self) -> bool:
        return all(
            callable(getattr(self._saver, name, None))
            for name in ("aget_tuple", "aput", "aput_writes")
        )

    # -- config plumbing ---------------------------------------------------

    @staticmethod
    def resolve_thread_id(config: Mapping[str, Any] | None) -> str | None:
        thread_id = _configurable(config).get("thread_id")
        return str(thread_id) if thread_id else None

    def require_thread_id(self, config: Mapping[str, Any] | None) -> str:
        thread_id = self.resolve_thread_id(config)
        if thread_id is None:
            raise ValueError(THREAD_ID_REQUIRED_MESSAGE)
        return thread_id

    @staticmethod
    def _with_configurable(config: Mapping[str, Any] | None, **values: Any) -> dict[str, Any]:
        out = copy.deepcopy(dict(config)) if isinstance(config, Mapping) else {}
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
                config=copy.deepcopy(dict(config)) if isinstance(config, Mapping) else {},
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )
        values = head.checkpoint.get("channel_values")
        return StateSnapshot(
            values=dict(values) if isinstance(values, Mapping) else {},
            next=(),
            config=copy.deepcopy(head.config),
            metadata=dict(head.metadata),
            created_at=head.checkpoint.get("ts"),
            parent_config=copy.deepcopy(head.parent_config),
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
                yield self.snapshot(head, config)

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
        checkpoint["channel_values"] = copy.deepcopy(dict(values))
        channel_versions = dict(checkpoint.get("channel_versions") or {})
        channel_versions.update(versions)
        checkpoint["channel_versions"] = channel_versions
        checkpoint["updated_channels"] = list(written)
        checkpoint.setdefault("versions_seen", {})
        checkpoint.setdefault("pending_sends", [])
        return checkpoint

    def commit(
        self,
        config: Mapping[str, Any] | None,
        head: _Head | None,
        values: Mapping[str, Any],
        *,
        written: Sequence[str],
        metadata: Mapping[str, Any],
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist *values* as a new checkpoint and return the new config."""
        thread_id = self.require_thread_id(config)
        checkpoint_ns = str(_configurable(config).get("checkpoint_ns") or "")
        parent_config = head.config if head is not None else None
        parent_id = _configurable(parent_config).get("checkpoint_id") if parent_config else None
        written = self._changed_channels(head, values, written)
        versions = self._next_versions(head, written)
        checkpoint = self._checkpoint_for(head, values, written, versions)
        write_config = self._with_configurable(
            config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=parent_id,
        )
        stored = self._saver.put(
            write_config,
            checkpoint,
            _merge_config_metadata(config, metadata),
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
            self._saver.put_writes(
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
    ) -> dict[str, Any]:
        if not self._supports_async():
            return self.commit(
                config, head, values, written=written, metadata=metadata, task_id=task_id
            )
        thread_id = self.require_thread_id(config)
        checkpoint_ns = str(_configurable(config).get("checkpoint_ns") or "")
        parent_config = head.config if head is not None else None
        parent_id = _configurable(parent_config).get("checkpoint_id") if parent_config else None
        written = self._changed_channels(head, values, written)
        versions = self._next_versions(head, written)
        checkpoint = self._checkpoint_for(head, values, written, versions)
        write_config = self._with_configurable(
            config,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=parent_id,
        )
        stored = await self._saver.aput(
            write_config,
            checkpoint,
            _merge_config_metadata(config, metadata),
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
            await self._saver.aput_writes(
                result,
                [(channel, values[channel]) for channel in written if channel in values],
                task_id,
            )
        return result

    # -- channel folding ---------------------------------------------------

    def load_values(self, head: _Head | None) -> dict[str, Any]:
        if head is None:
            return {}
        values = head.checkpoint.get("channel_values")
        return copy.deepcopy(dict(values)) if isinstance(values, Mapping) else {}

    def fold_input(
        self, values: Mapping[str, Any], update: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Fold a caller input through the channel table.

        Returns the folded values and the channel names that changed. A reducer
        channel accumulates (``messages`` appends); a last-value channel is
        replaced. ``Overwrite`` bypasses the reducer, which is how DeerFlow's
        compaction and rollback restore a transcript instead of appending to it.
        """
        current = dict(values)
        written: list[str] = []
        for key, write in update.items():
            is_overwrite, raw = Overwrite.is_overwrite(write)
            if is_overwrite:
                current[key] = raw
                written.append(key)
                continue
            channel = self._channels.get(key)
            reducer = channel_reducer(channel) if channel is not None else None
            if reducer is None or key not in current:
                # First write initializes the channel; ``add_messages`` rejects
                # a ``None`` left operand, and upstream behaves the same way.
                current[key] = write
            else:
                current[key] = reducer(current[key], write)
            written.append(key)
        return current, written

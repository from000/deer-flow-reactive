"""Checkpoint saver contract tests.

The saver is LangGraph-*shaped* but must not import LangGraph: the contract is
reproduced natively so a host (DeerFlow) can adapt it to whatever interface its
engine demands. These tests pin the storage semantics only.
"""

from __future__ import annotations

import pytest
from reactivegraph.checkpoint import (
    CheckpointStore,
    MemoryCheckpointSaver,
    SqliteCheckpointSaver,
    get_checkpointer,
    reset_checkpointer,
)


def _cp(checkpoint_id: str, values: dict, ts: str = "2026-01-01T00:00:00+00:00") -> dict:
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": ts,
        "channel_values": values,
        "channel_versions": {k: 1 for k in values},
        "versions_seen": {},
    }


def _cfg(thread_id: str, checkpoint_id: str | None = None, checkpoint_ns: str = "") -> dict:
    configurable: dict = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class TestStoreContract:
    def test_put_and_get_by_checkpoint_id(self) -> None:
        store = CheckpointStore()
        cfg = _cfg("t1", "cp-1")
        store.put(cfg, _cp("cp-1", {"n": 1}), {"source": "loop"}, {"n": 1})
        tup = store.get_tuple(cfg)
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"n": 1}
        assert tup.metadata == {"source": "loop"}
        assert tup.config["configurable"]["checkpoint_id"] == "cp-1"

    def test_get_tuple_without_id_returns_latest(self) -> None:
        store = CheckpointStore()
        store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}, "2026-01-01T00:00:00+00:00"), {}, {})
        store.put(_cfg("t1", "cp-2"), _cp("cp-2", {"n": 2}, "2026-01-01T00:00:01+00:00"), {}, {})
        tup = store.get_tuple(_cfg("t1"))
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"n": 2}

    def test_get_tuple_unknown_returns_none(self) -> None:
        store = CheckpointStore()
        assert store.get_tuple(_cfg("missing")) is None
        assert store.get_tuple(_cfg("t1", "nope")) is None
        assert store.get_tuple(None) is None

    def test_isolates_threads_and_namespaces(self) -> None:
        store = CheckpointStore()
        store.put(_cfg("a", "cp-1"), _cp("cp-1", {"n": "a"}), {}, {})
        store.put(_cfg("b", "cp-1"), _cp("cp-1", {"n": "b"}), {}, {})
        store.put(_cfg("a", "cp-1", "ns"), _cp("cp-1", {"n": "ns"}), {}, {})
        assert store.get_tuple(_cfg("a", "cp-1")).checkpoint["channel_values"] == {"n": "a"}
        assert store.get_tuple(_cfg("b", "cp-1")).checkpoint["channel_values"] == {"n": "b"}
        assert store.get_tuple(_cfg("a", "cp-1", "ns")).checkpoint["channel_values"] == {"n": "ns"}
        # A namespace-less lookup resolves the root namespace, not the child ns.
        assert store.get_tuple(_cfg("a")).checkpoint["channel_values"] == {"n": "a"}

    def test_records_parent_chain_and_lists_newest_first(self) -> None:
        store = CheckpointStore()
        first = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}, "2026-01-01T00:00:00+00:00"), {}, {})
        second = store.put(first, _cp("cp-2", {"n": 2}, "2026-01-01T00:00:01+00:00"), {}, {})
        store.put(second, _cp("cp-3", {"n": 3}, "2026-01-01T00:00:02+00:00"), {}, {})

        history = list(store.list(_cfg("t1")))
        assert [t.checkpoint["id"] for t in history] == ["cp-3", "cp-2", "cp-1"]
        assert history[0].parent_config["configurable"]["checkpoint_id"] == "cp-2"
        assert history[-1].parent_config is None

    def test_list_supports_limit_and_before_cursor(self) -> None:
        store = CheckpointStore()
        cfg = _cfg("t1", "cp-1")
        for i in range(1, 5):
            cfg = store.put(cfg, _cp(f"cp-{i}", {"n": i}, f"2026-01-01T00:00:0{i}+00:00"), {}, {})

        assert [t.checkpoint["id"] for t in store.list(_cfg("t1"), limit=2)] == ["cp-4", "cp-3"]
        rest = list(store.list(_cfg("t1"), before=_cfg("t1", "cp-3")))
        assert [t.checkpoint["id"] for t in rest] == ["cp-2", "cp-1"]

    def test_list_without_namespace_spans_the_thread_like_upstream(self) -> None:
        store = CheckpointStore()
        store.put(
            _cfg("t1", "root", ""),
            _cp("root", {"scope": "root"}, "2026-01-01T00:00:00+00:00"),
            {},
            {},
        )
        store.put(
            _cfg("t1", "child", "subgraph"),
            _cp("child", {"scope": "child"}, "2026-01-01T00:00:01+00:00"),
            {},
            {},
        )

        thread_wide = {"configurable": {"thread_id": "t1"}}
        assert {item.checkpoint["id"] for item in store.list(thread_wide)} == {"child", "root"}
        assert [item.checkpoint["id"] for item in store.list(_cfg("t1"))] == ["root"]
        assert [item.checkpoint["id"] for item in store.list(_cfg("t1", checkpoint_ns="subgraph"))] == ["child"]

    def test_list_filters_by_metadata(self) -> None:
        store = CheckpointStore()
        store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {"source": "loop"}, {})
        store.put(_cfg("t1", "cp-2"), _cp("cp-2", {"n": 2}), {"source": "input"}, {})
        listed = [t.checkpoint["id"] for t in store.list(_cfg("t1"), filter={"source": "loop"})]
        assert listed == ["cp-1"]

    def test_pending_writes_are_attached_to_checkpoint(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "messages", ["m"])], "task-a", "path")

        tup = store.get_tuple(cfg)
        assert tup is not None
        assert tup.pending_writes == [("task-a", "messages", ["m"])]

    def test_writes_may_precede_checkpoint_like_upstream(self) -> None:
        """Pregel commits writes before the checkpoint row is visible."""
        store = CheckpointStore()
        cfg = _cfg("t1", "cp-1")
        store.put_writes(cfg, [("task-a", "messages", ["m"])], "task-a")

        assert store.get_tuple(cfg) is None

        store.put(cfg, _cp("cp-1", {"n": 1}), {}, {})
        tup = store.get_tuple(cfg)
        assert tup is not None
        assert tup.pending_writes == [("task-a", "messages", ["m"])]

    def test_delete_thread_purges_only_that_thread(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "messages", ["m"])], "task-a")
        store.put(_cfg("t2", "cp-1"), _cp("cp-1", {"n": 2}), {}, {})

        store.delete_thread("t1")

        assert store.get_tuple(_cfg("t1")) is None
        assert list(store.list(_cfg("t1"))) == []
        assert store.get_tuple(_cfg("t2")) is not None

    def test_put_is_idempotent_for_same_checkpoint_id(self) -> None:
        store = CheckpointStore()
        cfg = _cfg("t1", "cp-1")
        store.put(cfg, _cp("cp-1", {"n": 1}), {}, {})
        store.put(cfg, _cp("cp-1", {"n": 999}), {}, {})
        history = list(store.list(_cfg("t1")))
        assert [t.checkpoint["id"] for t in history] == ["cp-1"]
        assert history[0].checkpoint["channel_values"] == {"n": 1}

    def test_rejects_checkpoint_without_id(self) -> None:
        store = CheckpointStore()
        with pytest.raises(ValueError, match="id"):
            store.put(_cfg("t1"), {"channel_values": {}}, {}, {})

    def test_requires_thread_id(self) -> None:
        store = CheckpointStore()
        with pytest.raises(ValueError, match="thread_id"):
            store.put({"configurable": {}}, _cp("cp-1", {"n": 1}), {}, {})

    def test_stored_state_is_deep_copied_in_and_out(self) -> None:
        store = CheckpointStore()
        checkpoint = _cp("cp-1", {"n": 1})
        store.put(_cfg("t1", "cp-1"), checkpoint, {"source": "loop"}, {})
        # Mutating the caller's object must not reach the store...
        checkpoint["channel_values"]["n"] = 999
        assert store.get_tuple(_cfg("t1")).checkpoint["channel_values"] == {"n": 1}
        # ...and mutating a read result must not corrupt the store either.
        read = store.get_tuple(_cfg("t1"))
        read.checkpoint["channel_values"]["n"] = 999
        assert store.get_tuple(_cfg("t1")).checkpoint["channel_values"] == {"n": 1}

    def test_get_next_version_increments(self) -> None:
        store = CheckpointStore()
        assert store.get_next_version(None, "messages") == 1
        assert store.get_next_version(1, "messages") == 2
        assert store.get_next_version("3", "messages") == 4

    def test_new_checkpoint_id_is_unique_and_monotonic(self) -> None:
        store = CheckpointStore()
        ids = [store.new_checkpoint_id() for _ in range(50)]
        assert len(set(ids)) == 50
        assert ids == sorted(ids)

    def test_list_all_threads_when_config_is_none(self) -> None:
        store = CheckpointStore()
        store.put(_cfg("a", "cp-1"), _cp("cp-1", {"n": "a"}), {}, {})
        store.put(_cfg("b", "cp-1"), _cp("cp-1", {"n": "b"}), {}, {})
        listed = sorted(t.config["configurable"]["thread_id"] for t in store.list(None))
        assert listed == ["a", "b"]


class TestWriteIndexSemantics:
    """Pregel retries a task by replaying its writes.

    The storage contract therefore keys writes by ``(task_id, index)`` where the
    index is the position inside the batch, or a fixed negative constant for the
    reserved channels. Regular writes are first-write-wins so a retry cannot
    duplicate them; reserved channels overwrite so the latest interrupt/error
    wins. Getting this wrong corrupts channel history only under retry, which is
    why it is pinned explicitly.
    """

    def test_regular_writes_are_first_write_wins_per_position(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "log", "first")], "task-a")
        store.put_writes(cfg, [("task-a", "log", "second")], "task-a")

        assert store.get_tuple(cfg).pending_writes == [("task-a", "log", "first")]

    def test_writes_in_one_batch_use_distinct_positions(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "a", 1), ("task-a", "b", 2)], "task-a")

        assert store.get_tuple(cfg).pending_writes == [
            ("task-a", "a", 1),
            ("task-a", "b", 2),
        ]

    def test_different_tasks_do_not_collide(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "log", "a")], "task-a")
        store.put_writes(cfg, [("task-b", "log", "b")], "task-b")

        assert store.get_tuple(cfg).pending_writes == [
            ("task-a", "log", "a"),
            ("task-b", "log", "b"),
        ]

    def test_reserved_channel_overwrites_and_keeps_position(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "log", "regular")], "task-a")
        store.put_writes(cfg, [("task-a", "__interrupt__", "first")], "task-a")
        store.put_writes(cfg, [("task-a", "__interrupt__", "second")], "task-a")

        assert store.get_tuple(cfg).pending_writes == [
            ("task-a", "log", "regular"),
            ("task-a", "__interrupt__", "second"),
        ]

    def test_writes_are_scoped_per_checkpoint(self) -> None:
        store = CheckpointStore()
        first = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        second = store.put(first, _cp("cp-2", {"n": 2}), {}, {})
        store.put_writes(first, [("task-a", "log", "one")], "task-a")
        store.put_writes(second, [("task-a", "log", "two")], "task-a")

        assert store.get_tuple(first).pending_writes == [("task-a", "log", "one")]
        assert store.get_tuple(second).pending_writes == [("task-a", "log", "two")]

    def test_delete_thread_purges_writes_too(self) -> None:
        store = CheckpointStore()
        cfg = store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        store.put_writes(cfg, [("task-a", "log", "one")], "task-a")
        store.delete_thread("t1")
        # Re-creating the same checkpoint id must not resurrect stale writes.
        store.put(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        assert store.get_tuple(cfg).pending_writes == []


class TestAsyncSurface:
    @pytest.mark.anyio
    async def test_async_surface_matches_sync(self) -> None:
        store = CheckpointStore()
        cfg = await store.aput(_cfg("t1", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        await store.aput_writes(cfg, [("task-a", "messages", ["m"])], "task-a")

        tup = await store.aget_tuple(cfg)
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"n": 1}
        assert tup.pending_writes == [("task-a", "messages", ["m"])]

        listed = [t async for t in store.alist(_cfg("t1"))]
        assert [t.checkpoint["id"] for t in listed] == ["cp-1"]

        await store.adelete_thread("t1")
        assert await store.aget_tuple(_cfg("t1")) is None


class TestProvider:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_checkpointer()
        yield
        reset_checkpointer()

    def test_defaults_to_memory_saver(self) -> None:
        assert isinstance(get_checkpointer(), MemoryCheckpointSaver)

    def test_memory_singleton_and_reset(self) -> None:
        first = get_checkpointer()
        assert get_checkpointer() is first
        reset_checkpointer()
        assert get_checkpointer() is not first

    def test_explicit_memory_config(self) -> None:
        assert isinstance(get_checkpointer({"type": "memory"}), MemoryCheckpointSaver)

    def test_unknown_backend_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            get_checkpointer({"type": "unknown"})

    def test_postgres_without_connection_string_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="connection_string is required"):
            get_checkpointer({"type": "postgres"})

    def test_sqlite_backend_is_durable(self, tmp_path) -> None:
        config = {
            "type": "sqlite",
            "connection_string": str(tmp_path / "checkpoints.sqlite"),
        }
        saver = get_checkpointer(config)
        assert isinstance(saver, SqliteCheckpointSaver)


def test_with_checkpoint_id_does_not_copy_runtime_execution_state() -> None:
    """Checkpoint identity must not deep-copy locks/callbacks from runtime config."""
    import threading

    runtime_config = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_ns": "subgraph",
            "runtime_lock": threading.RLock(),
        },
        "callbacks": [lambda: None],
    }

    stored = CheckpointStore._with_checkpoint_id(runtime_config, "cp-1")

    assert stored == {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_ns": "subgraph",
            "checkpoint_id": "cp-1",
        }
    }


def test_put_writes_accepts_upstream_two_tuple_writes() -> None:
    """Hosts call ``put_writes`` with ``(channel, value)`` pairs.

    LangGraph's ``BaseCheckpointSaver.put_writes`` documents
    ``Sequence[tuple[str, Any]]``; the engine's store used to unpack a
    three-tuple ``(task_id, channel, value)``, so a host graph driving the
    engine's saver raised ``IndexError`` mid-run. Both shapes are accepted, and
    reads must always report the canonical three-tuple form.
    """
    store = CheckpointStore()
    config = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    stored = store.put(config, _cp("c1", {}), {"step": 0}, {})
    store.put_writes(stored, [("messages", ["hi"]), ("title", "t")], "task-1")

    tuple_ = store.get_tuple(stored)
    assert sorted(write[1] for write in tuple_.pending_writes) == ["messages", "title"]
    assert all(len(write) == 3 for write in tuple_.pending_writes)


def test_put_writes_still_accepts_the_engine_three_tuple_form() -> None:
    """The engine's own ``(task_id, channel, value)`` entries keep working."""
    store = CheckpointStore()
    config = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    stored = store.put(config, _cp("c1", {}), {"step": 0}, {})
    store.put_writes(stored, [("task-9", "messages", ["hi"])], "task-1")

    tuple_ = store.get_tuple(stored)
    assert [write[1:] for write in tuple_.pending_writes] == [("messages", ["hi"])]


@pytest.mark.anyio
async def test_memory_maintenance_stats_and_delete_capabilities() -> None:
    store = CheckpointStore()
    first = store.put(
        _cfg("maint", "cp-1"),
        _cp("cp-1", {"n": 1, "blob": "x" * 32}),
        {"source": "loop"},
        {},
    )
    second = store.put(
        first,
        _cp("cp-2", {"n": 2, "blob": "y" * 48}),
        {"source": "loop"},
        {},
    )
    store.put_writes(second, [("messages", ["pending"])], "task-1")

    stats = await store.astorage_stats("maint")
    assert stats["checkpoint_rows"] == 2
    assert stats["write_rows"] == 1
    assert stats["checkpoint_bytes"] > 0
    assert stats["write_bytes"] > 0
    assert await store.acheckpoint_ids_with_writes("maint") == {("", "cp-2")}

    await store.adelete_checkpoint("maint", ("", "cp-1"))
    assert store.get_tuple(_cfg("maint", "cp-1")) is None
    assert store.get_tuple(_cfg("maint", "cp-2")) is not None
    after = await store.astorage_stats("maint")
    assert after["checkpoint_rows"] == 1

    # Deleting the current head must move the latest pointer back to the
    # surviving parent instead of leaving a dangling checkpoint id.
    await store.adelete_checkpoint("maint", ("", "cp-2"))
    assert store.get_tuple(_cfg("maint")) is None
    assert await store.astorage_stats("maint") == {
        "logical_checkpoint_bytes": 0,
        "logical_write_bytes": 0,
        "checkpoint_rows": 0,
        "checkpoint_bytes": 0,
        "blob_rows": 0,
        "blob_bytes": 0,
        "write_rows": 0,
        "write_bytes": 0,
    }


@pytest.mark.anyio
async def test_memory_blob_gc_is_a_noop_for_inline_checkpoints() -> None:
    store = CheckpointStore()
    store.put(_cfg("gc", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
    store.put(_cfg("gc", "cp-2"), _cp("cp-2", {"n": 2}), {}, {})

    before = await store.astorage_stats("gc")
    await store.adelete_unreachable_blobs("gc", {"cp-1"})
    after = await store.astorage_stats("gc")

    # Native memory/SQLite checkpoints are inline; there is no independent
    # blob table for GC to touch. The call remains a safe no-op.
    assert before["blob_rows"] == after["blob_rows"] == 0
    assert before["blob_bytes"] == after["blob_bytes"] == 0
    assert store.get_tuple(_cfg("gc", "cp-1")) is not None
    assert store.get_tuple(_cfg("gc", "cp-2")) is not None

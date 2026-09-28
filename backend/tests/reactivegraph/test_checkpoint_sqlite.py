"""SQLite checkpoint contract.

The durable saver must keep the same observable lineage/write semantics as the
in-memory saver, survive reopening the file, and reject types it cannot safely
round-trip. These tests intentionally exercise only the public API so the
implementation may change internally.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from reactivegraph.checkpoint import (
    CheckpointStore,
    MemoryCheckpointSaver,
    SqliteCheckpointSaver,
    get_checkpointer,
    reset_checkpointer,
)


class Payload(BaseModel):
    name: str
    count: int


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


def test_sqlite_durable_round_trip_and_reopen(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite"
    saver = SqliteCheckpointSaver(path)
    first = saver.put(
        _cfg("thread-1", "cp-1"),
        _cp("cp-1", {"n": 1, "payload": b"\x00\xff"}),
        {"source": "loop"},
        {"n": 1},
    )
    saver.put_writes(first, [("n", 2)], "task-1")
    saver.close()

    reopened = SqliteCheckpointSaver(path)
    try:
        tup = reopened.get_tuple(_cfg("thread-1", "cp-1"))
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"n": 1, "payload": b"\x00\xff"}
        assert tup.metadata == {"source": "loop"}
        assert tup.pending_writes == [("task-1", "n", 2)]
    finally:
        reopened.close()


def test_sqlite_recovers_after_writer_process_exit(tmp_path) -> None:
    """A later process must read and continue the durable checkpoint file."""
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "cross-process.sqlite"
    script = """
import sys
from reactivegraph.checkpoint import SqliteCheckpointSaver
config = {"configurable": {"thread_id": "cross-process", "checkpoint_id": "cp-1"}}
saver = SqliteCheckpointSaver(sys.argv[1])
saver.put(
    config,
    {
        "v": 1,
        "id": "cp-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"n": 7},
        "channel_versions": {"n": 1},
        "versions_seen": {},
    },
    {"source": "loop"},
    {"n": 1},
)
saver.close()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )

    reopened = SqliteCheckpointSaver(path)
    try:
        tup = reopened.get_tuple(_cfg("cross-process", "cp-1"))
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"n": 7}
        assert tup.metadata == {"source": "loop"}
    finally:
        reopened.close()


def test_sqlite_existing_checkpoint_id_is_immutable(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "immutable.sqlite")
    try:
        config = _cfg("immutable", "cp-1")
        first = _cp("cp-1", {"n": 1})
        second = _cp("cp-1", {"n": 999})
        saver.put(config, first, {"source": "first"}, {"n": 1})
        saver.put(config, second, {"source": "second"}, {"n": 2})

        stored = saver.get_tuple(config)
        assert stored is not None
        assert stored.checkpoint["channel_values"] == {"n": 1}
        assert stored.metadata == {"source": "first"}
    finally:
        saver.close()


def test_sqlite_lineage_listing_and_metadata_filter(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        config = _cfg("thread-1", "cp-1")
        for index in range(1, 5):
            config = saver.put(
                config,
                _cp(f"cp-{index}", {"n": index}, f"2026-01-01T00:00:0{index}+00:00"),
                {"source": "loop" if index % 2 else "input"},
                {"n": index},
            )

        history = list(saver.list(_cfg("thread-1")))
        assert [t.checkpoint["id"] for t in history] == ["cp-4", "cp-3", "cp-2", "cp-1"]
        assert history[0].parent_config["configurable"]["checkpoint_id"] == "cp-3"
        assert history[-1].parent_config["configurable"]["checkpoint_id"] == "cp-1"
        assert [
            t.checkpoint["id"]
            for t in saver.list(_cfg("thread-1"), filter={"source": "loop"}, limit=1)
        ] == ["cp-3"]
        assert [
            t.checkpoint["id"]
            for t in saver.list(_cfg("thread-1"), before=_cfg("thread-1", "cp-3"))
        ] == ["cp-2", "cp-1"]
    finally:
        saver.close()


def test_sqlite_write_idempotency_and_reserved_overwrite(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        config = saver.put(_cfg("t", "cp-1"), _cp("cp-1", {}), {}, {})
        saver.put_writes(config, [("messages", ["first"])], "task-1")
        saver.put_writes(config, [("messages", ["second"])], "task-1")
        saver.put_writes(config, [("__error__", "first")], "task-1")
        saver.put_writes(config, [("__error__", "second")], "task-1")

        writes = saver.get_tuple(config).pending_writes
        assert writes == [("task-1", "__error__", "second"), ("task-1", "messages", ["first"])]
    finally:
        saver.close()


def test_sqlite_preserves_host_delta_snapshot_identity(tmp_path) -> None:
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.serde.types import _DeltaSnapshot as HostDeltaSnapshot

    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        original = HostDeltaSnapshot([1, 2, 3])
        cfg = saver.put(
            _cfg("delta", "cp-1"),
            _cp("cp-1", {"messages": original}),
            {},
            {},
        )

        restored = saver.get_tuple(cfg).checkpoint["channel_values"]["messages"]
        assert isinstance(restored, HostDeltaSnapshot)
        assert restored.value == [1, 2, 3]
    finally:
        saver.close()


def test_sqlite_typed_values_round_trip(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        value = {
            "when": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "pair": ("a", 1),
            "model": Payload(name="x", count=2),
            "nested": {"items": [1, None, True]},
        }
        config = saver.put(_cfg("typed", "cp-1"), _cp("cp-1", value), {}, {})
        loaded = saver.get_tuple(config).checkpoint["channel_values"]
        assert loaded["when"] == value["when"]
        assert loaded["pair"] == value["pair"]
        assert loaded["model"] == value["model"]
        assert loaded["nested"] == value["nested"]
    finally:
        saver.close()


def test_sqlite_unknown_python_object_fails_closed(tmp_path) -> None:
    class Unknown:
        pass

    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        with pytest.raises(TypeError, match="serialize"):
            saver.put(_cfg("typed", "cp-1"), _cp("cp-1", {"x": Unknown()}), {}, {})
    finally:
        saver.close()


def test_sqlite_writes_may_precede_checkpoint_like_upstream(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        cfg = _cfg("t", "cp-1")
        saver.put_writes(cfg, [("messages", ["m"])], "task-a")
        assert saver.get_tuple(cfg) is None

        saver.put(cfg, _cp("cp-1", {"n": 1}), {}, {})
        tup = saver.get_tuple(cfg)
        assert tup is not None
        assert tup.pending_writes == [("task-a", "messages", ["m"])]
    finally:
        saver.close()


def test_sqlite_list_without_namespace_spans_the_thread(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        saver.put(
            _cfg("t", "root", ""),
            _cp("root", {"scope": "root"}, "2026-01-01T00:00:00+00:00"),
            {},
            {},
        )
        saver.put(
            _cfg("t", "child", "subgraph"),
            _cp("child", {"scope": "child"}, "2026-01-01T00:00:01+00:00"),
            {},
            {},
        )

        thread_wide = {"configurable": {"thread_id": "t"}}
        assert {
            item.checkpoint["id"] for item in saver.list(thread_wide)
        } == {"child", "root"}
        assert [item.checkpoint["id"] for item in saver.list(_cfg("t"))] == ["root"]
        assert [
            item.checkpoint["id"]
            for item in saver.list(_cfg("t", checkpoint_ns="subgraph"))
        ] == ["child"]
    finally:
        saver.close()


def test_sqlite_delete_checkpoint_rolls_back_when_write_delete_fails(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        cfg = saver.put(_cfg("t", "cp-1"), _cp("cp-1", {}), {}, {})
        saver.put_writes(cfg, [("messages", ["m"])], "task-a")
        real_execute = saver._execute

        def fail_write_delete(sql, params=()):
            if sql.strip().startswith("DELETE FROM writes"):
                raise sqlite3.OperationalError("injected write-delete failure")
            return real_execute(sql, params)

        saver._execute = fail_write_delete
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            saver.delete_checkpoint("t", ("", "cp-1"))
        saver._execute = real_execute

        tup = saver.get_tuple(cfg)
        assert tup is not None
        assert tup.pending_writes == [("task-a", "messages", ["m"])]
    finally:
        saver.close()


def test_sqlite_delete_thread_removes_checkpoints_and_writes(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        config = saver.put(_cfg("t", "cp-1"), _cp("cp-1", {}), {}, {})
        saver.put_writes(config, [("x", 1)], "task")
        saver.delete_thread("t")
        assert saver.get_tuple(config) is None
        assert list(saver.list(_cfg("t"))) == []
    finally:
        saver.close()


@pytest.mark.anyio
async def test_sqlite_async_mirror(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    try:
        config = await saver.aput(_cfg("t", "cp-1"), _cp("cp-1", {"n": 1}), {}, {})
        await saver.aput_writes(config, [("n", 2)], "task")
        tup = await saver.aget_tuple(config)
        assert tup is not None and tup.checkpoint["channel_values"] == {"n": 1}
        assert [t.checkpoint["id"] async for t in saver.alist(_cfg("t"))] == ["cp-1"]
        await saver.adelete_thread("t")
        assert await saver.aget_tuple(config) is None
    finally:
        saver.close()


def test_sqlite_is_a_checkpoint_store_and_uses_compatible_schema(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite"
    saver = SqliteCheckpointSaver(path)
    try:
        assert isinstance(saver, CheckpointStore)
        saver.put(_cfg("t", "cp-1"), _cp("cp-1", {}), {}, {})
    finally:
        saver.close()

    conn = sqlite3.connect(path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(checkpoints)")}
        assert {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "parent_checkpoint_id",
            "type",
            "checkpoint",
            "metadata",
        } <= columns
        columns = {row[1] for row in conn.execute("PRAGMA table_info(writes)")}
        assert {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "channel",
            "type",
            "value",
        } <= columns
    finally:
        conn.close()


def test_get_checkpointer_sqlite_does_not_degrade_to_memory(tmp_path) -> None:
    reset_checkpointer()
    try:
        saver = get_checkpointer(
            {"type": "sqlite", "connection_string": str(tmp_path / "cp.sqlite")}
        )
        assert isinstance(saver, SqliteCheckpointSaver)
        assert not isinstance(saver, MemoryCheckpointSaver)
    finally:
        reset_checkpointer()



@pytest.mark.anyio
async def test_sqlite_maintenance_stats_delete_and_gc(tmp_path) -> None:
    saver = SqliteCheckpointSaver(tmp_path / "maintenance.sqlite")
    try:
        first = saver.put(
            _cfg("maint", "cp-1"),
            _cp("cp-1", {"payload": "x" * 32}),
            {"source": "loop"},
            {},
        )
        second = saver.put(
            first,
            _cp("cp-2", {"payload": "y" * 48}),
            {"source": "loop"},
            {},
        )
        saver.put_writes(second, [("messages", ["pending"])], "task-1")

        stats = await saver.astorage_stats("maint")
        assert stats["checkpoint_rows"] == 2
        assert stats["write_rows"] == 1
        assert stats["checkpoint_bytes"] > 0
        assert stats["write_bytes"] > 0
        assert stats["blob_rows"] == stats["blob_bytes"] == 0
        assert await saver.acheckpoint_ids_with_writes("maint") == {("", "cp-2")}

        await saver.adelete_checkpoint("maint", ("", "cp-1"))
        assert saver.get_tuple(_cfg("maint", "cp-1")) is None
        assert saver.get_tuple(_cfg("maint", "cp-2")) is not None

        await saver.adelete_unreachable_blobs("maint", {"cp-2"})
        after = await saver.astorage_stats("maint")
        assert after["checkpoint_rows"] == 1
        assert after["write_rows"] == 1
        assert after["blob_rows"] == after["blob_bytes"] == 0
    finally:
        saver.close()

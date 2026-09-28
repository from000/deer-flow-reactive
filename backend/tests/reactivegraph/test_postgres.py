"""PostgreSQL checkpoint/store integration contracts.

These tests run only when ``REACTIVEGRAPH_TEST_POSTGRES_DSN`` is set. They use a
real PostgreSQL server and verify the native ReactiveGraph durable backends.
"""

from __future__ import annotations

import os
import threading
import time
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Core package tests are copied from the ReactiveGraph repository; add the
# source package explicitly because the backend's editable workspace paths do
# not include this standalone package directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "reactivegraph"))

import pytest
from pydantic import BaseModel

from reactivegraph.checkpoint import (
    PostgresCheckpointSaver,
    get_checkpointer,
    reset_checkpointer,
)
class Payload(BaseModel):
    name: str
    count: int


try:
    from reactivegraph.store import PostgresStore
except ImportError:
    PostgresStore = None
from reactivegraph.store import get_store, reset_store

DSN = os.environ.get("REACTIVEGRAPH_TEST_POSTGRES_DSN")


def _fresh_schema() -> str:
    return f"rg_test_{uuid.uuid4().hex}"


def _drop_schema(dsn: str, schema: str) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_postgres_backends_are_package_exports() -> None:
    import reactivegraph

    assert reactivegraph.PostgresCheckpointSaver is PostgresCheckpointSaver
    assert reactivegraph.PostgresStore is PostgresStore


@pytest.fixture
def schema() -> str:
    return _fresh_schema()


@pytest.fixture
def dsn() -> str:
    return DSN


def test_postgres_checkpoint_round_trip_reopen_and_maintenance(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    try:
        first = PostgresCheckpointSaver(dsn, schema=schema)
        config = first.put(
            {"configurable": {"thread_id": "pg-thread", "checkpoint_id": "cp-1"}},
            {
                "v": 1,
                "id": "cp-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {"n": 7, "text": "value"},
                "channel_versions": {"n": 1, "text": 1},
                "versions_seen": {},
            },
            {"source": "loop"},
            {"n": 1},
        )
        first.put_writes(config, [("n", 8), ("__error__", "first"), ("__error__", "second")], "task-1")
        assert first.storage_stats("pg-thread")["checkpoint_rows"] == 1
        assert first.checkpoint_ids_with_writes("pg-thread") == {("", "cp-1")}
        first.close()

        reopened = PostgresCheckpointSaver(dsn, schema=schema)
        try:
            tup = reopened.get_tuple({"configurable": {"thread_id": "pg-thread"}})
            assert tup is not None
            assert tup.checkpoint["channel_values"]["n"] == 7
            assert tup.metadata == {"source": "loop"}
            assert tup.pending_writes == [
                ("task-1", "__error__", "second"),
                ("task-1", "n", 8),
            ]
            assert [t.checkpoint["id"] for t in reopened.list({"configurable": {"thread_id": "pg-thread"}})] == ["cp-1"]
            reopened.delete_checkpoint("pg-thread", ("", "cp-1"))
            assert reopened.get_tuple({"configurable": {"thread_id": "pg-thread"}}) is None
        finally:
            reopened.close()
    finally:
        _drop_schema(dsn, schema)


def test_postgres_checkpoint_put_is_immutable_for_existing_id(
    dsn: str, schema: str
) -> None:
    """Reusing a checkpoint id must not rewrite its immutable snapshot."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")

    saver = PostgresCheckpointSaver(dsn, schema=schema)
    try:
        config = {
            "configurable": {
                "thread_id": "pg-immutable",
                "checkpoint_id": "cp-1",
            }
        }
        first = {
            "v": 1,
            "id": "cp-1",
            "ts": "2026-01-01T00:00:00+00:00",
            "channel_values": {"n": 1},
            "channel_versions": {"n": 1},
            "versions_seen": {},
        }
        second = {
            **first,
            "channel_values": {"n": 999},
            "channel_versions": {"n": 2},
        }
        saver.put(config, first, {"source": "first"}, {"n": 1})
        saver.put(config, second, {"source": "second"}, {"n": 2})

        stored = saver.get_tuple(config)
        assert stored is not None
        assert stored.checkpoint["channel_values"] == {"n": 1}
        assert stored.metadata == {"source": "first"}
    finally:
        saver.close()
        _drop_schema(dsn, schema)


def test_postgres_checkpoint_lineage_listing_and_metadata_filter(
    dsn: str, schema: str
) -> None:
    """PostgreSQL must preserve checkpoint lineage and listing semantics."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")

    def cfg(thread_id: str, checkpoint_id: str | None = None, checkpoint_ns: str = "") -> dict:
        configurable: dict = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id is not None:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    def cp(checkpoint_id: str, value: int, ts: str) -> dict:
        return {
            "v": 1,
            "id": checkpoint_id,
            "ts": ts,
            "channel_values": {"n": value},
            "channel_versions": {"n": 1},
            "versions_seen": {},
        }

    saver = PostgresCheckpointSaver(dsn, schema=schema)
    try:
        config = cfg("pg-lineage", "cp-1")
        for index in range(1, 5):
            config = saver.put(
                config,
                cp(
                    f"cp-{index}",
                    index,
                    f"2026-01-01T00:00:0{index}+00:00",
                ),
                {"source": "loop" if index % 2 else "input"},
                {"n": index},
            )

        history = list(saver.list(cfg("pg-lineage")))
        assert [item.checkpoint["id"] for item in history] == [
            "cp-4",
            "cp-3",
            "cp-2",
            "cp-1",
        ]
        assert history[0].parent_config["configurable"]["checkpoint_id"] == "cp-3"
        assert history[-1].parent_config["configurable"]["checkpoint_id"] == "cp-1"
        assert [
            item.checkpoint["id"]
            for item in saver.list(
                cfg("pg-lineage"), filter={"source": "loop"}, limit=1
            )
        ] == ["cp-3"]
        assert [
            item.checkpoint["id"]
            for item in saver.list(cfg("pg-lineage"), before=cfg("pg-lineage", "cp-3"))
        ] == ["cp-2", "cp-1"]

        # Without an explicit checkpoint_ns, listing spans the whole thread.
        saver.put(
            cfg("pg-lineage", "child", "subgraph"),
            cp("child", 5, "2026-01-01T00:00:05+00:00"),
            {},
            {},
        )
        assert {
            item.checkpoint["id"]
            for item in saver.list({"configurable": {"thread_id": "pg-lineage"}})
        } == {"cp-1", "cp-2", "cp-3", "cp-4", "child"}
    finally:
        saver.close()
        _drop_schema(dsn, schema)


def test_postgres_checkpoint_write_idempotency_and_typed_round_trip(
    dsn: str, schema: str
) -> None:
    """Regular writes are idempotent, reserved writes overwrite, and values round-trip."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")

    saver = PostgresCheckpointSaver(dsn, schema=schema)
    try:
        config = saver.put(
            {"configurable": {"thread_id": "pg-writes", "checkpoint_id": "cp-1"}},
            {
                "v": 1,
                "id": "cp-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {
                    "when": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
                    "pair": ("a", 1),
                    "model": Payload(name="x", count=2),
                    "nested": {"items": [1, None, True]},
                },
                "channel_versions": {"n": 1},
                "versions_seen": {},
            },
            {},
            {},
        )
        saver.put_writes(config, [("messages", ["first"])], "task-1")
        saver.put_writes(config, [("messages", ["second"])], "task-1")
        saver.put_writes(config, [("__error__", "first")], "task-1")
        saver.put_writes(config, [("__error__", "second")], "task-1")

        tuple_ = saver.get_tuple(config)
        assert tuple_ is not None
        assert tuple_.pending_writes == [
            ("task-1", "__error__", "second"),
            ("task-1", "messages", ["first"]),
        ]
        loaded = tuple_.checkpoint["channel_values"]
        assert loaded["when"] == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert loaded["pair"] == ("a", 1)
        assert loaded["model"] == Payload(name="x", count=2)
        assert loaded["nested"] == {"items": [1, None, True]}
    finally:
        saver.close()
        _drop_schema(dsn, schema)


def test_postgres_writes_may_precede_checkpoint(dsn: str, schema: str) -> None:
    """Pregel may persist task writes before the checkpoint row exists."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")

    saver = PostgresCheckpointSaver(dsn, schema=schema)
    try:
        cfg = {
            "configurable": {
                "thread_id": "pg-preceding-writes",
                "checkpoint_id": "cp-1",
            }
        }
        saver.put_writes(cfg, [("messages", ["m"])], "task-a")
        assert saver.get_tuple(cfg) is None

        saver.put(
            cfg,
            {
                "v": 1,
                "id": "cp-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {"n": 1},
                "channel_versions": {"n": 1},
                "versions_seen": {},
            },
            {},
            {},
        )
        tuple_ = saver.get_tuple(cfg)
        assert tuple_ is not None
        assert tuple_.pending_writes == [("task-a", "messages", ["m"])]
    finally:
        saver.close()
        _drop_schema(dsn, schema)


def test_postgres_store_list_namespaces_excludes_expired_rows(
    dsn: str, schema: str
) -> None:
    """A namespace visible only through expired items must be hidden."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")

    store = PostgresStore(dsn, schema=schema)
    try:
        store.put(("live",), "key", {"value": 1})
        store.put(("expired",), "key", {"value": 2}, ttl=-1)
        store.put(("mixed",), "live", {"value": 3})
        store.put(("mixed",), "expired", {"value": 4}, ttl=-1)

        assert store.list_namespaces() == [("live",), ("mixed",)]
    finally:
        store.close()
        _drop_schema(dsn, schema)


def test_get_checkpointer_postgres_native(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    reset_checkpointer()
    try:
        saver = get_checkpointer({"type": "postgres", "connection_string": dsn, "schema": schema})
        assert isinstance(saver, PostgresCheckpointSaver)
    finally:
        reset_checkpointer()
        _drop_schema(dsn, schema)


def test_postgres_store_round_trip_search_and_reopen(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")
    try:
        first = PostgresStore(dsn, schema=schema)
        first.put(("docs",), "a", {"status": "open", "n": 1})
        first.put(("docs",), "b", {"status": "closed", "n": 2})
        first.close()

        reopened = PostgresStore(dsn, schema=schema)
        try:
            item = reopened.get(("docs",), "a")
            assert item is not None
            assert item.value == {"status": "open", "n": 1}
            assert [x.key for x in reopened.search(("docs",), filter={"status": "open"})] == ["a"]
            assert reopened.list_namespaces(prefix=("docs",)) == [("docs",)]
            reopened.delete(("docs",), "a")
            assert reopened.get(("docs",), "a") is None
        finally:
            reopened.close()
    finally:
        _drop_schema(dsn, schema)


def test_postgres_store_put_resets_created_and_updated_at(
    dsn: str, schema: str
) -> None:
    """A repeated put replaces the item and resets both timestamps."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")

    store = PostgresStore(dsn, schema=schema)
    try:
        store.put(("timestamps",), "key", {"value": 1})
        first = store.get(("timestamps",), "key")
        assert first is not None
        time.sleep(0.01)
        store.put(("timestamps",), "key", {"value": 2})
        second = store.get(("timestamps",), "key")
        assert second is not None
        assert second.value == {"value": 2}
        assert second.created_at > first.created_at
        assert second.updated_at > first.updated_at
    finally:
        store.close()
        _drop_schema(dsn, schema)


def test_get_store_postgres_native(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")
    reset_store()
    try:
        store = get_store({"type": "postgres", "connection_string": dsn, "schema": schema})
        assert isinstance(store, PostgresStore)
    finally:
        reset_store()
        _drop_schema(dsn, schema)


@pytest.mark.anyio
async def test_postgres_async_mirror_delete_and_lifecycle(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    saver = PostgresCheckpointSaver(dsn, schema=schema)
    try:
        config = await saver.aput(
            {"configurable": {"thread_id": "async-thread"}},
            {
                "v": 2,
                "id": "cp-async",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {"value": "native"},
                "channel_versions": {"value": 1},
                "versions_seen": {},
            },
            {"source": "loop"},
            {},
        )
        await saver.aput_writes(config, [("value", "queued")], "task-async")
        tuple_ = await saver.aget_tuple(config)
        assert tuple_ is not None
        assert tuple_.pending_writes == [("task-async", "value", "queued")]
        assert [item.checkpoint["id"] async for item in saver.alist(config)] == [
            "cp-async"
        ]
        assert await saver.acheckpoint_ids_with_writes("async-thread") == {
            ("", "cp-async")
        }
        assert (await saver.astorage_stats("async-thread"))["checkpoint_rows"] == 1

        await saver.adelete_checkpoint("async-thread", ("", "cp-async"))
        assert await saver.aget_tuple(config) is None
        await saver.adelete_thread("async-thread")
        assert await saver.astorage_stats("async-thread") == {
            "logical_checkpoint_bytes": 0,
            "logical_write_bytes": 0,
            "checkpoint_rows": 0,
            "checkpoint_bytes": 0,
            "blob_rows": 0,
            "blob_bytes": 0,
            "write_rows": 0,
            "write_bytes": 0,
        }
    finally:
        saver.close()
        _drop_schema(dsn, schema)


@pytest.mark.anyio
async def test_postgres_store_async_ttl_search_and_lifecycle(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")
    store = PostgresStore(dsn, schema=schema)
    try:
        await store.aput(("docs", "one"), "open", {"status": "open", "n": 1})
        await store.aput(("docs", "one"), "closed", {"status": "closed", "n": 2})
        await store.aput(("docs", "two"), "nested", {"status": "open", "n": 3})
        await store.aput(("docs", "expired"), "gone", {"status": "open"}, ttl=-1)

        item = await store.aget(("docs", "one"), "open")
        assert item is not None and item.value == {"status": "open", "n": 1}
        assert [item.key for item in await store.asearch(("docs",), filter={"status": "open"})] == [
            "nested",
            "open",
        ]
        assert await store.alist_namespaces(prefix=("docs", "one")) == [("docs", "one")]
        assert await store.alist_namespaces(suffix=("two",)) == [("docs", "two")]
        assert await store.aget(("docs", "expired"), "gone") is None
    finally:
        store.close()
        _drop_schema(dsn, schema)


def test_postgres_operations_fail_closed_after_close(dsn: str, schema: str) -> None:
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")
    saver = PostgresCheckpointSaver(dsn, schema=schema)
    store = PostgresStore(dsn, schema=schema)
    try:
        saver.close()
        store.close()
        with pytest.raises(RuntimeError, match="checkpoint saver is closed"):
            saver.storage_stats("closed")
        with pytest.raises(RuntimeError, match="store is closed"):
            store.get(("closed",), "key")
    finally:
        _drop_schema(dsn, schema)

def test_postgres_concurrent_instances_bootstrap_and_preserve_writes(
    dsn: str, schema: str
) -> None:
    """Separate clients can bootstrap one schema and write concurrently."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")

    worker_count = 6
    start = threading.Barrier(worker_count)
    thread_id = "pg-concurrent"

    def write(worker: int) -> None:
        start.wait(timeout=20)
        saver = PostgresCheckpointSaver(dsn, schema=schema)
        store = PostgresStore(dsn, schema=schema)
        try:
            config = saver.put(
                {"configurable": {"thread_id": thread_id}},
                {
                    "v": 2,
                    "id": f"cp-{worker}",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "channel_values": {"worker": worker},
                    "channel_versions": {"worker": 1},
                    "versions_seen": {},
                },
                {"source": "concurrent", "worker": worker},
                {},
            )
            saver.put_writes(config, [("worker", worker)], f"task-{worker}")
            store.put(("concurrent",), f"key-{worker}", {"worker": worker})
        finally:
            saver.close()
            store.close()

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(write, worker) for worker in range(worker_count)]
            for future in futures:
                future.result(timeout=30)

        saver = PostgresCheckpointSaver(dsn, schema=schema)
        store = PostgresStore(dsn, schema=schema)
        try:
            assert saver.storage_stats(thread_id)["checkpoint_rows"] == worker_count
            assert saver.checkpoint_ids_with_writes(thread_id) == {
                ("", f"cp-{worker}") for worker in range(worker_count)
            }
            for worker in range(worker_count):
                tuple_ = saver.get_tuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": f"cp-{worker}",
                        }
                    }
                )
                assert tuple_ is not None
                assert tuple_.checkpoint["channel_values"] == {"worker": worker}
                assert tuple_.pending_writes == [
                    (f"task-{worker}", "worker", worker)
                ]
                item = store.get(("concurrent",), f"key-{worker}")
                assert item is not None
                assert item.value == {"worker": worker}
        finally:
            saver.close()
            store.close()
    finally:
        _drop_schema(dsn, schema)


def test_postgres_concurrent_same_key_upsert_keeps_complete_value(
    dsn: str, schema: str
) -> None:
    """Concurrent writers to one key leave one complete committed value."""
    if not dsn:
        pytest.skip("set REACTIVEGRAPH_TEST_POSTGRES_DSN")
    if PostgresStore is None:
        pytest.skip("PostgresStore not implemented yet")

    worker_count = 8
    rounds = 12
    start = threading.Barrier(worker_count)

    def write(worker: int) -> None:
        store = PostgresStore(dsn, schema=schema)
        try:
            start.wait(timeout=20)
            for round_ in range(rounds):
                store.put(
                    ("concurrent",),
                    "shared",
                    {"worker": worker, "round": round_, "payload": f"{worker}:{round_}"},
                )
        finally:
            store.close()

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(write, worker) for worker in range(worker_count)]
            for future in futures:
                future.result(timeout=30)

        store = PostgresStore(dsn, schema=schema)
        try:
            item = store.get(("concurrent",), "shared")
            assert item is not None
            assert (
                item.value["worker"],
                item.value["round"],
                item.value["payload"],
            ) in {
                (worker, round_, f"{worker}:{round_}")
                for worker in range(worker_count)
                for round_ in range(rounds)
            }
        finally:
            store.close()
    finally:
        _drop_schema(dsn, schema)

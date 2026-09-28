"""SQLite long-term store contract."""

from __future__ import annotations

import sqlite3

import pytest

from reactivegraph.store import (
    InvalidNamespaceError,
    MemoryStore,
    SearchItem,
    SqliteStore,
    get_store,
    reset_store,
)


def test_sqlite_store_reopens_and_round_trips(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    store = SqliteStore(path)
    store.put(("users", "u1"), "prefs", {"theme": "dark", "nested": {"n": 1}})
    store.close()

    reopened = SqliteStore(path)
    try:
        item = reopened.get(("users", "u1"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark", "nested": {"n": 1}}
        assert item.namespace == ("users", "u1")
        assert item.created_at.tzinfo is not None
        assert item.updated_at.tzinfo is not None
    finally:
        reopened.close()


def test_sqlite_store_recovers_after_writer_process_exit(tmp_path) -> None:
    """A later process must read and continue durable store state."""
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "cross-process-store.sqlite"
    script = """
import sys
from reactivegraph.store import SqliteStore
store = SqliteStore(sys.argv[1])
store.put(("users", "u1"), "prefs", {"theme": "dark", "restart": True})
store.close()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )

    reopened = SqliteStore(path)
    try:
        item = reopened.get(("users", "u1"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark", "restart": True}
        reopened.put(("users", "u1"), "prefs", {"theme": "light", "restart": False})
        assert reopened.get(("users", "u1"), "prefs").value == {
            "theme": "light",
            "restart": False,
        }
    finally:
        reopened.close()


def test_sqlite_store_crud_overwrite_and_delete(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        store.put(("n",), "k", {"v": 1})
        first = store.get(("n",), "k")
        store.put(("n",), "k", {"v": 2})
        second = store.get(("n",), "k")
        assert first.value == {"v": 1}
        assert second.value == {"v": 2}
        store.delete(("n",), "k")
        assert store.get(("n",), "k") is None
    finally:
        store.close()


def test_sqlite_store_search_filters_operators_and_paging(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        store.put(("docs",), "a", {"status": "open", "score": 1})
        store.put(("docs",), "b", {"status": "closed", "score": 2})
        store.put(("docs",), "c", {"status": "open", "score": 3})
        assert [i.key for i in store.search(("docs",), filter={"status": "open"})] == ["c", "a"]
        assert all(isinstance(item, SearchItem) for item in store.search(("docs",)))
        assert [i.key for i in store.search(("docs",), filter={"score": {"$gte": 2}})] == ["c", "b"]
        assert [
            i.key
            for i in store.search(
                ("docs",), filter={"score": {"$gt": 1}}, limit=1, offset=1
            )
        ] == ["b"]
    finally:
        store.close()


def test_sqlite_store_search_orders_by_updated_at_desc(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        store.put(("docs",), "old", {"v": 1})
        store.put(("docs",), "new", {"v": 2})
        assert [item.key for item in store.search(("docs",))] == ["new", "old"]
    finally:
        store.close()


def test_sqlite_store_batch_reads_precede_writes(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        from reactivegraph.store import GetOp, PutOp, SearchOp

        store.put(("docs",), "k", {"v": 1})
        results = store.batch(
            [
                SearchOp(("docs",), limit=10),
                PutOp(("docs",), "k", {"v": 2}, None, None),
                GetOp(("docs",), "k", True),
            ]
        )
        assert [item.value for item in results[0]] == [{"v": 1}]
        assert results[2].value == {"v": 1}
    finally:
        store.close()


def test_sqlite_store_list_namespaces_prefix_suffix_and_wildcard(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        store.put(("a", "b"), "1", {"v": 1})
        store.put(("a", "c"), "1", {"v": 1})
        store.put(("x", "b"), "1", {"v": 1})
        assert store.list_namespaces(prefix=("a",)) == [("a", "b"), ("a", "c")]
        assert store.list_namespaces(suffix=("b",)) == [("a", "b"), ("x", "b")]
        assert store.list_namespaces(prefix=("a", "*")) == [("a", "b"), ("a", "c")]
        assert store.list_namespaces(max_depth=1) == [("a",), ("x",)]
    finally:
        store.close()


def test_sqlite_store_namespace_validation(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        with pytest.raises(InvalidNamespaceError, match="Namespace cannot be empty"):
            store.put((), "k", {})
        with pytest.raises(InvalidNamespaceError, match="cannot contain periods"):
            store.put(("a.b",), "k", {})
    finally:
        store.close()


def test_sqlite_store_ttl_is_honored(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        store.put(("n",), "k", {"v": 1}, ttl=-1)
        assert store.get(("n",), "k") is None
    finally:
        store.close()


def test_sqlite_store_ttl_is_stored_as_langgraph_compatible_timestamp(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    store = SqliteStore(path)
    try:
        store.put(("n",), "k", {"v": 1}, ttl=5)
    finally:
        store.close()
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT expires_at, ttl_minutes, typeof(expires_at) FROM store"
        ).fetchone()
        assert row is not None
        assert row[1] == 5
        assert row[2] == "text"
        assert "+00:00" in row[0]
        assert " " in row[0]
    finally:
        conn.close()


@pytest.mark.anyio
async def test_sqlite_store_async_mirror(tmp_path) -> None:
    store = SqliteStore(tmp_path / "store.sqlite")
    try:
        await store.aput(("n",), "k", {"v": 1})
        item = await store.aget(("n",), "k")
        assert item is not None and item.value == {"v": 1}
        assert [i.key for i in await store.asearch(("n",))] == ["k"]
        assert await store.alist_namespaces() == [("n",)]
        await store.adelete(("n",), "k")
        assert await store.aget(("n",), "k") is None
    finally:
        store.close()


def test_sqlite_store_is_not_memory_store(tmp_path) -> None:
    reset_store()
    store = get_store({"type": "sqlite", "connection_string": str(tmp_path / "store.sqlite")})
    try:
        assert isinstance(store, SqliteStore)
        assert not isinstance(store, MemoryStore)
    finally:
        store.close()
        reset_store()



def test_unknown_store_backend_fails_closed() -> None:
    reset_store()
    try:
        with pytest.raises(NotImplementedError, match="Refusing to silently fall back"):
            get_store({"type": "redis", "connection_string": "redis://example"})
    finally:
        reset_store()


def test_sqlite_store_schema(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    store = SqliteStore(path)
    try:
        store.put(("n",), "k", {"v": 1})
    finally:
        store.close()

    conn = sqlite3.connect(path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(store)")}
        assert {
            "prefix",
            "key",
            "value",
            "created_at",
            "updated_at",
            "expires_at",
            "ttl_minutes",
        } <= columns
    finally:
        conn.close()


def test_sqlite_store_migrates_pre_ttl_schema_before_creating_indexes(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE store (
                prefix TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (prefix, key)
            );
            CREATE TABLE store_migrations (v INTEGER PRIMARY KEY);
            INSERT INTO store_migrations (v) VALUES (0), (1), (2), (3);
            INSERT INTO store (prefix, key, value)
            VALUES ('legacy', 'k', '{"v": 1}');
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteStore(path)
    try:
        item = store.get(("legacy",), "k")
        assert item is not None
        assert item.value == {"v": 1}
    finally:
        store.close()

    conn = sqlite3.connect(path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(store)")}
        assert {"expires_at", "ttl_minutes"} <= columns
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(store)")}
        assert {"store_prefix_idx", "idx_store_expires_at"} <= indexes
        versions = {row[0] for row in conn.execute("SELECT v FROM store_migrations")}
        assert versions == {0, 1, 2, 3, 4}
    finally:
        conn.close()

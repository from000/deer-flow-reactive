"""Optional live PostgreSQL schema integration tests for issue #3380."""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from deerflow.config.database_config import DatabaseConfig
from deerflow.persistence.engine import close_engine, get_engine, init_engine_from_config
from deerflow.runtime.checkpointer.async_provider import make_checkpointer
from deerflow.runtime.checkpointer.provider import _resolve_checkpointer_config, _sync_checkpointer_cm
from deerflow.runtime.store.async_provider import make_store
from deerflow.runtime.store.provider import _resolve_store_config, _sync_store_cm

POSTGRES_URL = os.getenv("DEERFLOW_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set DEERFLOW_TEST_POSTGRES_URL to run live PostgreSQL schema integration tests",
)


@pytest.mark.anyio
async def test_postgres_schema_places_orm_checkpointer_and_store_tables_together():
    """Verify a real PostgreSQL backend places all persistence tables in one schema."""
    schema = f"deerflow_test_{uuid.uuid4().hex[:12]}"
    db_config = DatabaseConfig(backend="postgres", postgres_url=POSTGRES_URL or "", postgres_schema=schema)
    app_config = SimpleNamespace(checkpointer=None, database=db_config)

    def _public_tables(rows):
        return {
            row.table_name
            for row in rows
            if row.table_schema == "public"
            and ("checkpoint" in row.table_name or row.table_name == "store")
        }

    await init_engine_from_config(db_config)
    engine = get_engine()
    assert engine is not None

    try:
        async with engine.connect() as conn:
            public_before = _public_tables(
                (
                    await conn.execute(
                        text(
                            """
                            SELECT table_schema, table_name
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                            """
                        )
                    )
                ).all()
            )
        async with make_checkpointer(app_config) as checkpointer:
            assert checkpointer is not None
        async with make_store(app_config) as store:
            assert store is not None

        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        """
                        SELECT table_schema, table_name
                        FROM information_schema.tables
                        WHERE table_schema IN (:schema, 'public')
                        ORDER BY table_schema, table_name
                        """
                    ),
                    {"schema": schema},
                )
            ).all()

        by_schema = {(row.table_schema, row.table_name) for row in rows}
        orm_tables = {"runs", "run_events", "threads_meta", "feedback", "users"}
        assert {(schema, table) for table in orm_tables}.issubset(by_schema)
        assert any(table_schema == schema and "checkpoint" in table_name for table_schema, table_name in by_schema)
        assert any(table_schema == schema and ("store" in table_name or "migration" in table_name) for table_schema, table_name in by_schema)
        assert _public_tables(rows) == public_before
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await close_engine()


def test_sync_postgres_schema_places_checkpointer_and_store_tables_together():
    """Verify the sync (psycopg) path honours search_path via the DSN encoding.

    This exercises ``dsn_with_search_path`` against a real psycopg connection,
    guarding against regression of the ``%20`` vs ``+`` libpq encoding bug.
    """
    import psycopg

    schema = f"deerflow_test_{uuid.uuid4().hex[:12]}"
    db_config = DatabaseConfig(
        backend="postgres",
        postgres_url=POSTGRES_URL or "",
        postgres_schema=schema,
    )
    checkpointer_config = _resolve_checkpointer_config(SimpleNamespace(checkpointer=None, database=db_config))
    store_config = _resolve_store_config(SimpleNamespace(checkpointer=None, database=db_config))

    try:
        with psycopg.connect(POSTGRES_URL or "", autocommit=True) as conn:
            public_before = {
                table_name
                for (table_name,) in conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    """
                ).fetchall()
                if "checkpoint" in table_name or table_name == "store"
            }
        with _sync_checkpointer_cm(checkpointer_config) as checkpointer:
            assert checkpointer is not None
        with _sync_store_cm(store_config) as store:
            assert store is not None

        with psycopg.connect(POSTGRES_URL or "", autocommit=True) as conn:
            rows = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema IN (%s, 'public')
                ORDER BY table_schema, table_name
                """,
                (schema,),
            ).fetchall()

        by_schema = {(table_schema, table_name) for table_schema, table_name in rows}
        assert any(table_schema == schema and "checkpoint" in table_name for table_schema, table_name in by_schema)
        assert any(table_schema == schema and ("store" in table_name or "migration" in table_name) for table_schema, table_name in by_schema)
        # The run must not leak persistence tables into public. Existing tables
        # are tolerated because the local developer database may be shared.
        public_after = {
            table_name
            for table_schema, table_name in by_schema
            if table_schema == "public"
            and ("checkpoint" in table_name or table_name == "store")
        }
        assert public_after == public_before
    finally:
        with psycopg.connect(POSTGRES_URL or "", autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

@pytest.mark.anyio
async def test_async_postgres_provider_uses_reactivegraph_native_backends():
    """The async provider must not fall back to LangGraph's PostgreSQL classes."""

    from reactivegraph.checkpoint import PostgresCheckpointSaver
    from reactivegraph.store import PostgresStore

    from deerflow.config.checkpointer_config import CheckpointerConfig
    from deerflow.runtime.checkpointer.async_provider import _async_checkpointer
    from deerflow.runtime.store.async_provider import _async_store

    schema = f"deerflow_test_{uuid.uuid4().hex[:12]}"
    config = CheckpointerConfig(
        type="postgres",
        connection_string=POSTGRES_URL or "",
        postgres_schema=schema,
    )
    try:
        async with _async_checkpointer(config) as checkpointer:
            from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

            assert isinstance(checkpointer, ReactiveGraphSaver)
            assert isinstance(checkpointer.store, PostgresCheckpointSaver)
        async with _async_store(config) as store:
            assert isinstance(store, PostgresStore)
    finally:
        import psycopg

        with psycopg.connect(POSTGRES_URL or "", autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.mark.anyio
async def test_unified_database_postgres_uses_reactivegraph_native_checkpointer():
    """The unified database config must use the native PostgreSQL checkpoint store."""

    from reactivegraph.checkpoint import PostgresCheckpointSaver

    from deerflow.runtime.checkpointer.async_provider import _async_checkpointer_from_database
    from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

    schema = f"deerflow_test_{uuid.uuid4().hex[:12]}"
    db_config = DatabaseConfig(
        backend="postgres",
        postgres_url=POSTGRES_URL or "",
        postgres_schema=schema,
    )
    try:
        async with _async_checkpointer_from_database(db_config) as checkpointer:
            assert isinstance(checkpointer, ReactiveGraphSaver)
            assert isinstance(checkpointer.store, PostgresCheckpointSaver)
    finally:
        import psycopg

        with psycopg.connect(POSTGRES_URL or "", autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

"""Async checkpointer factory.

Provides an **async context manager** for long-running async servers that need
proper resource cleanup.

Supported backends: memory, sqlite, postgres.

Usage (e.g. FastAPI lifespan)::

    from deerflow.runtime.checkpointer.async_provider import make_checkpointer

    async with make_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer  # ReactiveGraphSaver if not configured

For sync usage see :mod:`deerflow.runtime.checkpointer.provider`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from reactivegraph import Checkpointer

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.persistence.postgres_schema import normalize_libpq_dsn
from deerflow.runtime.checkpointer.provider import POSTGRES_CONN_REQUIRED
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)


def _prepare_sqlite_checkpointer_path(raw: str) -> str:
    conn_str = resolve_sqlite_conn_str(raw)
    ensure_sqlite_parent_dir(conn_str)
    return conn_str


def _prepare_database_sqlite_checkpointer_path(db_config) -> str:
    conn_str = db_config.checkpointer_sqlite_path
    ensure_sqlite_parent_dir(conn_str)
    return conn_str


# ---------------------------------------------------------------------------
# Async factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer(config) -> AsyncIterator[Checkpointer]:
    """Async context manager that constructs and tears down a checkpointer."""
    if config.type == "memory":
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        yield ReactiveGraphSaver()
        return

    if config.type == "sqlite":
        from reactivegraph.checkpoint import SqliteCheckpointSaver

        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        conn_str = await asyncio.to_thread(
            _prepare_sqlite_checkpointer_path,
            config.connection_string or "store.db",
        )
        store = await asyncio.to_thread(SqliteCheckpointSaver, conn_str)
        saver = ReactiveGraphSaver(store=store)
        try:
            await saver.asetup()
            yield saver
        finally:
            await saver.aclose()
        return

    if config.type == "postgres":
        from reactivegraph.checkpoint import PostgresCheckpointSaver

        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        store = await asyncio.to_thread(
            PostgresCheckpointSaver,
            config.connection_string,
            schema=config.postgres_schema or "public",
        )
        saver = ReactiveGraphSaver(store=store)
        try:
            await saver.asetup()
            yield saver
        finally:
            await saver.aclose()
        return

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ---------------------------------------------------------------------------
# Public async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer_from_database(db_config) -> AsyncIterator[Checkpointer]:
    """Async context manager that constructs a checkpointer from unified DatabaseConfig."""
    if db_config.backend == "memory":
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        yield ReactiveGraphSaver()
        return

    if db_config.backend == "sqlite":
        from reactivegraph.checkpoint import SqliteCheckpointSaver

        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        conn_str = await asyncio.to_thread(
            _prepare_database_sqlite_checkpointer_path, db_config
        )
        store = await asyncio.to_thread(SqliteCheckpointSaver, conn_str)
        saver = ReactiveGraphSaver(store=store)
        try:
            await saver.asetup()
            yield saver
        finally:
            await saver.aclose()
        return

    if db_config.backend == "postgres":
        if not db_config.postgres_url:
            raise ValueError("database.postgres_url is required for the postgres backend")

        from reactivegraph.checkpoint import PostgresCheckpointSaver

        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        store = await asyncio.to_thread(
            PostgresCheckpointSaver,
            normalize_libpq_dsn(db_config.postgres_url),
            schema=db_config.postgres_schema or "public",
        )
        saver = ReactiveGraphSaver(store=store)
        try:
            await saver.asetup()
            yield saver
        finally:
            await saver.aclose()
        return

    raise ValueError(f"Unknown database backend: {db_config.backend!r}")


@contextlib.asynccontextmanager
async def _select_inner_checkpointer(app_config: AppConfig) -> AsyncIterator[Checkpointer]:
    """Yield the raw checkpointer selected by *app_config* (no delta-cache wrapping).

    Priority:
    1. Legacy ``checkpointer:`` config section (backward compatible)
    2. Unified ``database:`` config section
    3. Default ReactiveGraphSaver
    """
    # Legacy: standalone checkpointer config takes precedence
    if app_config.checkpointer is not None:
        async with _async_checkpointer(app_config.checkpointer) as saver:
            yield saver
            return

    # Unified database config
    db_config = getattr(app_config, "database", None)
    if db_config is not None and db_config.backend != "memory":
        async with _async_checkpointer_from_database(db_config) as saver:
            yield saver
            return

    # Default: in-memory
    from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

    yield ReactiveGraphSaver()


@contextlib.asynccontextmanager
async def make_checkpointer(app_config: AppConfig | None = None) -> AsyncIterator[Checkpointer]:
    """Async context manager that yields a checkpointer for the caller's lifetime.
    Resources are opened on enter and closed on exit -- no global state::

        async with make_checkpointer(app_config) as checkpointer:
            app.state.checkpointer = checkpointer

    Yields a ``ReactiveGraphSaver`` when no checkpointer is configured in *config.yaml*.

    Backend selection priority:
    1. Legacy ``checkpointer:`` config section (backward compatible)
    2. Unified ``database:`` config section
    3. Default ReactiveGraphSaver

    When the effective checkpoint channel mode is ``delta`` (the process-frozen
    mode wins, falling back to ``database.checkpoint_channel_mode``), the raw
    saver is wrapped in a :class:`CachedHistorySaver` backed by a history cache
    whose lifetime equals this context manager's.
    """
    from deerflow.runtime.checkpoint_mode import frozen_checkpoint_channel_mode

    if app_config is None:
        app_config = get_app_config()

    async with _select_inner_checkpointer(app_config) as saver:
        db_config = getattr(app_config, "database", None)
        mode = frozen_checkpoint_channel_mode() or (db_config.checkpoint_channel_mode if db_config is not None else "full")
        if mode == "delta":
            from deerflow.runtime.checkpoint_cache.provider import (
                checkpoint_cache_key_prefix,
                make_checkpoint_cache,
            )
            from deerflow.runtime.checkpointer.cached_saver import CachedHistorySaver

            async with make_checkpoint_cache(app_config, serde=saver.serde) as cache:
                yield CachedHistorySaver(saver, cache, key_prefix=checkpoint_cache_key_prefix(app_config))
        else:
            yield saver

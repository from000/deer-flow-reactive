"""Async Store factory — backend mirrors runtime persistence configuration.

The deprecated ``checkpointer`` section takes precedence when present;
otherwise Store follows the unified ``database`` section in *config.yaml*:

- ``memory``   → :class:`reactivegraph.store.MemoryStore`
- ``sqlite``   → :class:`reactivegraph.store.SqliteStore`
- ``postgres`` → :class:`reactivegraph.store.PostgresStore`

Usage (e.g. FastAPI lifespan)::

    from deerflow.runtime.store import make_store

    async with make_store() as store:
        app.state.store = store
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from reactivegraph import BaseStore

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.runtime.store.provider import (
    POSTGRES_CONN_REQUIRED,
    _resolve_store_config,
    ensure_sqlite_parent_dir,
    resolve_sqlite_conn_str,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Internal backend factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_store(config) -> AsyncIterator[BaseStore]:
    """Async context manager that constructs and tears down a Store.

    The ``config`` argument is a :class:`deerflow.config.checkpointer_config.CheckpointerConfig`
    instance — the same object used by the checkpointer factory.
    """
    if config.type == "memory":
        from reactivegraph.store import MemoryStore

        logger.info("Store: using ReactiveGraph MemoryStore (in-process, not persistent)")
        yield MemoryStore()
        return

    if config.type == "sqlite":
        from reactivegraph.store import SqliteStore

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        await asyncio.to_thread(ensure_sqlite_parent_dir, conn_str)
        store = await asyncio.to_thread(SqliteStore, conn_str)
        try:
            await asyncio.to_thread(store.setup)
            logger.info("Store: using ReactiveGraph SqliteStore (%s)", conn_str)
            yield store
        finally:
            await asyncio.to_thread(store.close)
        return

    if config.type == "postgres":
        from reactivegraph.store import PostgresStore

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        store = await asyncio.to_thread(
            PostgresStore,
            config.connection_string,
            schema=config.postgres_schema or "public",
        )
        try:
            await asyncio.to_thread(store.setup)
            logger.info("Store: using ReactiveGraph PostgresStore")
            yield store
        finally:
            await asyncio.to_thread(store.close)
        return

    raise ValueError(f"Unknown store backend type: {config.type!r}")


# ---------------------------------------------------------------------------
# Public async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def make_store(app_config: AppConfig | None = None) -> AsyncIterator[BaseStore]:
    """Yield a Store selected from legacy or unified persistence config.

    The legacy ``checkpointer`` section takes precedence when configured;
    otherwise the unified ``database`` section selects the backend, matching
    :func:`deerflow.runtime.checkpointer.async_provider.make_checkpointer`::

        async with make_store(app_config) as store:
            app.state.store = store

    A :class:`~reactivegraph.store.MemoryStore` is returned only when the
    resolved backend is explicitly ``memory``.
    """
    if app_config is None:
        app_config = get_app_config()

    config = _resolve_store_config(app_config)
    async with _async_store(config) as store:
        yield store

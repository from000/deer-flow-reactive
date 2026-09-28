"""Regression test: sqlite path setup must run off the event loop.

Anchors the production offload from
`runtime/checkpointer/async_provider.py:_async_checkpointer`, where SQLite
path resolution and `ensure_sqlite_parent_dir` are dispatched via
`await asyncio.to_thread(...)`.
That fix addressed #1912, where the sync `Path.mkdir` / `os.mkdir` inside
`ensure_sqlite_parent_dir` ran on the FastAPI lifespan event loop thread
and blocked startup.

This test invokes the production `_async_checkpointer()` path under the
strict Blockbuster context. The target path's parent does not yet exist, so
the underlying path resolution and `os.mkdir` both execute. ReactiveGraph's
SQLite saver creates its schema during construction as well, so the whole
construction must stay off the event loop. If any step regresses to run
directly on the loop, Blockbuster raises `BlockingError` and this test fails.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from reactivegraph.checkpoint import SqliteCheckpointSaver

pytestmark = pytest.mark.asyncio


async def test_async_checkpointer_sqlite_setup_does_not_block_event_loop(tmp_path: Path) -> None:
    from deerflow.config.checkpointer_config import CheckpointerConfig
    from deerflow.runtime.checkpointer.async_provider import _async_checkpointer

    db_file = tmp_path / "subdir" / "store.db"
    loop_thread_id = threading.get_ident()
    construction_thread_ids: list[int] = []
    setup_calls = 0
    close_calls = 0

    class TrackingSqliteCheckpointSaver(SqliteCheckpointSaver):
        def __init__(self, conn_string: str) -> None:
            construction_thread_ids.append(threading.get_ident())
            super().__init__(conn_string)

        def setup(self) -> None:
            nonlocal setup_calls
            setup_calls += 1
            super().setup()

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            super().close()

    with patch("reactivegraph.checkpoint.SqliteCheckpointSaver", TrackingSqliteCheckpointSaver):
        async with _async_checkpointer(CheckpointerConfig(type="sqlite", connection_string=str(db_file))) as saver:
            assert isinstance(saver.store, TrackingSqliteCheckpointSaver)

    assert db_file.parent.exists()
    assert db_file.exists()
    assert len(construction_thread_ids) == 1
    assert construction_thread_ids[0] != loop_thread_id
    # Construction performs the first schema setup; ``asetup()`` is deliberately
    # idempotent because hosts may call it again after opening the saver.
    assert setup_calls == 2
    assert close_calls == 1

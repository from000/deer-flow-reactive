"""Unit tests for checkpointer config, packaging metadata, and factories."""

import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deerflow.config.app_config as app_config_module
from deerflow.config.checkpointer_config import (
    CheckpointerConfig,
    ensure_config_loaded,
    get_checkpointer_config,
    load_checkpointer_config_from_dict,
    set_checkpointer_config,
)
from deerflow.config.database_config import DatabaseConfig
from deerflow.runtime.checkpointer import get_checkpointer, reset_checkpointer
from deerflow.runtime.checkpointer.provider import POSTGRES_INSTALL
from deerflow.runtime.store import get_store, reset_store
from deerflow.runtime.store.provider import POSTGRES_STORE_INSTALL


@pytest.fixture(autouse=True)
def reset_state():
    """Reset singleton state before each test."""
    app_config_module._app_config = None
    set_checkpointer_config(None)
    reset_checkpointer()
    reset_store()
    yield
    app_config_module._app_config = None
    set_checkpointer_config(None)
    reset_checkpointer()
    reset_store()


class _BlockingSingletonContext:
    def __init__(self, value: object, entered: Event, release: Event, stats: dict[str, object]):
        self._value = value
        self._entered = entered
        self._release = release
        self._stats = stats

    def __enter__(self):
        with self._stats["lock"]:
            self._stats["enters"] += 1
            self._entered.set()
        assert self._release.wait(timeout=3), "timed out waiting to release singleton initialization"
        return self._value

    def __exit__(self, exc_type, exc, tb):
        with self._stats["lock"]:
            self._stats["exits"] += 1
        return False


class _BlockingSingletonFactory:
    def __init__(self):
        self.value = object()
        self.entered = Event()
        self.release = Event()
        self.stats = {"enters": 0, "exits": 0, "lock": Lock()}

    def context_manager(self, _config):
        return _BlockingSingletonContext(self.value, self.entered, self.release, self.stats)

    def enter_count(self) -> int:
        with self.stats["lock"]:
            return self.stats["enters"]

    def exit_count(self) -> int:
        with self.stats["lock"]:
            return self.stats["exits"]


class _TrackingLock:
    def __init__(self):
        self._lock = Lock()
        self.acquired = Event()

    def acquire(self, *args, **kwargs):
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired:
            self.acquired.set()
        return acquired

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def locked(self) -> bool:
        return self._lock.locked()


def _call_getter_concurrently(getter, workers: int = 8) -> list[object]:
    ready = Barrier(workers + 1)

    def worker():
        ready.wait(timeout=3)
        return getter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker) for _ in range(workers)]
        ready.wait(timeout=3)
        return [future.result(timeout=3) for future in futures]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestCheckpointerConfig:
    def test_load_memory_config(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        config = get_checkpointer_config()
        assert config is not None
        assert config.type == "memory"
        assert config.connection_string is None

    def test_load_sqlite_config(self):
        load_checkpointer_config_from_dict({"type": "sqlite", "connection_string": "/tmp/test.db"})
        config = get_checkpointer_config()
        assert config is not None
        assert config.type == "sqlite"
        assert config.connection_string == "/tmp/test.db"

    def test_load_postgres_config(self):
        load_checkpointer_config_from_dict({"type": "postgres", "connection_string": "postgresql://localhost/db"})
        config = get_checkpointer_config()
        assert config is not None
        assert config.type == "postgres"
        assert config.connection_string == "postgresql://localhost/db"
        assert config.postgres_schema == ""

    def test_postgres_schema_accepts_valid_identifier(self):
        config = CheckpointerConfig(type="postgres", connection_string="postgresql://localhost/db", postgres_schema="deerflow")
        assert config.postgres_schema == "deerflow"

    @pytest.mark.parametrize("schema", ["1abc", "a b", "a;b", "a-b", "a" * 64, 'a"b', "MySchema", "Orders", "Public"])
    def test_postgres_schema_rejects_invalid_identifier(self, schema):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CheckpointerConfig(type="postgres", connection_string="postgresql://localhost/db", postgres_schema=schema)

    def test_default_connection_string_is_none(self):
        config = CheckpointerConfig(type="memory")
        assert config.connection_string is None

    def test_set_config_to_none(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        set_checkpointer_config(None)
        assert get_checkpointer_config() is None

    def test_ensure_config_loaded_loads_app_config_when_uninitialized(self):
        def fake_get_app_config():
            load_checkpointer_config_from_dict({"type": "memory"})

        with patch("deerflow.config.app_config.get_app_config", side_effect=fake_get_app_config) as mock_get_app_config:
            ensure_config_loaded()

        mock_get_app_config.assert_called_once()
        config = get_checkpointer_config()
        assert config is not None
        assert config.type == "memory"

    def test_ensure_config_loaded_skips_explicit_config(self):
        load_checkpointer_config_from_dict({"type": "memory"})

        with patch("deerflow.config.app_config.get_app_config") as mock_get_app_config:
            ensure_config_loaded()

        mock_get_app_config.assert_not_called()

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            load_checkpointer_config_from_dict({"type": "unknown"})

    def test_connection_string_description_matches_runtime_defaults(self):
        description = CheckpointerConfig.model_fields["connection_string"].description

        assert description is not None
        assert "Optional for sqlite" in description
        assert "defaults to 'store.db'" in description
        assert "Required for postgres" in description


class TestHarnessPackaging:
    def test_pyproject_declares_postgres_extra(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "packages" / "harness" / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text())

        optional_dependencies = data["project"]["optional-dependencies"]
        assert "postgres" in optional_dependencies
        assert optional_dependencies["postgres"] == [
            "asyncpg>=0.29",
            "langgraph-checkpoint-postgres>=3.1.2,<3.2",
            "psycopg[binary]>=3.3.3",
            "psycopg-pool>=3.3.0",
        ]

    def test_workspace_pyproject_forwards_postgres_extra_to_harness(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text())

        optional_dependencies = data["project"]["optional-dependencies"]
        assert optional_dependencies["postgres"] == ["deerflow-harness[postgres]"]

    def test_postgres_missing_dependency_messages_recommend_package_extra(self):
        assert "deerflow-harness[postgres]" in POSTGRES_INSTALL
        assert "deerflow-harness[postgres]" in POSTGRES_STORE_INSTALL
        assert "uv sync --all-packages --extra postgres" in POSTGRES_INSTALL
        assert "uv sync --all-packages --extra postgres" in POSTGRES_STORE_INSTALL


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestGetCheckpointer:
    def test_returns_in_memory_saver_when_not_configured(self):
        """get_checkpointer should return the ReactiveGraph memory saver when not configured."""
        from langgraph.checkpoint.base import BaseCheckpointSaver

        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        with patch("deerflow.runtime.checkpointer.provider.get_app_config", side_effect=FileNotFoundError):
            cp = get_checkpointer()
        assert cp is not None
        assert isinstance(cp, ReactiveGraphSaver)
        # LangGraph's own gate must still accept it.
        assert isinstance(cp, BaseCheckpointSaver)

    def test_memory_returns_in_memory_saver(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        cp = get_checkpointer()
        assert isinstance(cp, ReactiveGraphSaver)

    def test_memory_singleton(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is cp2

    def test_reset_clears_singleton(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        cp1 = get_checkpointer()
        reset_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is not cp2

    def test_sqlite_uses_native_saver(self):
        load_checkpointer_config_from_dict({"type": "sqlite", "connection_string": "/tmp/test.db"})

        with patch("deerflow.runtime.checkpointer.provider.ensure_sqlite_parent_dir"):
            reset_checkpointer()
            cp = get_checkpointer()

        from reactivegraph.checkpoint import SqliteCheckpointSaver
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        assert isinstance(cp, ReactiveGraphSaver)
        assert isinstance(cp.store, SqliteCheckpointSaver)

    def test_postgres_uses_native_saver(self):
        load_checkpointer_config_from_dict(
            {"type": "postgres", "connection_string": "postgresql://localhost/db"}
        )
        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch(
                "reactivegraph.checkpoint.PostgresCheckpointSaver",
                mock_store,
            ),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
        ):
            reset_checkpointer()
            cp = get_checkpointer()

        assert cp is mock_saver
        mock_store.assert_called_once_with(
            "postgresql://localhost/db", schema="public"
        )
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.setup.assert_called_once_with()

    def test_postgres_raises_when_connection_string_missing(self):
        load_checkpointer_config_from_dict({"type": "postgres"})
        mock_saver = MagicMock()
        mock_module = MagicMock()
        mock_module.PostgresSaver = mock_saver
        with patch.dict(sys.modules, {"langgraph.checkpoint.postgres": mock_module}):
            reset_checkpointer()
            with pytest.raises(ValueError, match="connection_string is required"):
                get_checkpointer()

    def test_sqlite_creates_saver(self):
        """SQLite checkpointer is created with the resolved native backend."""
        load_checkpointer_config_from_dict({"type": "sqlite", "connection_string": "/tmp/test.db"})

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("reactivegraph.checkpoint.SqliteCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
        ):
            reset_checkpointer()
            cp = get_checkpointer()

        assert cp is mock_saver
        mock_store.assert_called_once_with(str(Path("/tmp/test.db").resolve()))
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.setup.assert_called_once_with()

    def test_sqlite_creates_parent_dir(self):
        """Sync SQLite checkpointer should call ensure_sqlite_parent_dir before connecting.

        This mirrors the async checkpointer's behaviour and prevents
        'sqlite3.OperationalError: unable to open database file' when the
        parent directory for the database file does not yet exist (e.g. when
        using the harness package from an external virtualenv where the
        .deer-flow directory has not been created).
        """
        load_checkpointer_config_from_dict({"type": "sqlite", "connection_string": "relative/test.db"})

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("reactivegraph.checkpoint.SqliteCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
            patch("deerflow.runtime.checkpointer.provider.ensure_sqlite_parent_dir") as mock_ensure,
            patch(
                "deerflow.runtime.checkpointer.provider.resolve_sqlite_conn_str",
                return_value="/tmp/resolved/relative/test.db",
            ),
        ):
            reset_checkpointer()
            cp = get_checkpointer()

        assert cp is mock_saver
        mock_ensure.assert_called_once_with("/tmp/resolved/relative/test.db")
        mock_store.assert_called_once_with("/tmp/resolved/relative/test.db")

    def test_sqlite_ensure_parent_dir_before_connect(self):
        """ensure_sqlite_parent_dir must be called before from_conn_string."""
        load_checkpointer_config_from_dict({"type": "sqlite", "connection_string": "relative/test.db"})

        call_order = []

        def record_connect(*args, **kwargs):
            call_order.append("connect")
            return MagicMock()

        def record_ensure(*a, **kw):
            call_order.append("ensure")

        with (
            patch(
                "reactivegraph.checkpoint.SqliteCheckpointSaver",
                side_effect=record_connect,
            ),
            patch(
                "deerflow.runtime.checkpointer.provider.ensure_sqlite_parent_dir",
                side_effect=record_ensure,
            ),
            patch(
                "deerflow.runtime.checkpointer.provider.resolve_sqlite_conn_str",
                return_value="/tmp/resolved/relative/test.db",
            ),
        ):
            reset_checkpointer()
            get_checkpointer()

        assert call_order == ["ensure", "connect"]

    def test_postgres_creates_saver(self):
        """Postgres checkpointer is created with the native backend."""
        load_checkpointer_config_from_dict({"type": "postgres", "connection_string": "postgresql://localhost/db"})

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch(
                "reactivegraph.checkpoint.PostgresCheckpointSaver",
                mock_store,
            ),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
        ):
            reset_checkpointer()
            cp = get_checkpointer()

        assert cp is mock_saver
        mock_store.assert_called_once_with(
            "postgresql://localhost/db", schema="public"
        )
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.setup.assert_called_once_with()

    def test_postgres_schema_creates_schema_and_sets_search_path(self):
        """Sync Postgres checkpointer should create schema before setup."""
        load_checkpointer_config_from_dict({"type": "postgres", "connection_string": "postgresql://localhost/db", "postgres_schema": "deerflow"})

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch(
                "reactivegraph.checkpoint.PostgresCheckpointSaver",
                mock_store,
            ),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
        ):
            reset_checkpointer()
            cp = get_checkpointer()

        assert cp is mock_saver
        mock_store.assert_called_once_with(
            "postgresql://localhost/db", schema="deerflow"
        )
        mock_saver.setup.assert_called_once_with()

    def test_store_postgres_schema_creates_schema_and_sets_search_path(self):
        """Sync Postgres store should use the legacy checkpointer schema."""
        load_checkpointer_config_from_dict({"type": "postgres", "connection_string": "postgresql://localhost/db", "postgres_schema": "deerflow"})

        mock_store_cls = MagicMock()
        mock_store = mock_store_cls.return_value

        with patch("reactivegraph.store.PostgresStore", mock_store_cls):
            reset_store()
            store = get_store()

        assert store is mock_store
        mock_store_cls.assert_called_once_with(
            "postgresql://localhost/db", schema="deerflow"
        )
        mock_store.setup.assert_called_once_with()


class TestSyncSingletonThreadSafety:
    def test_store_reset_clears_singleton(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        store1 = get_store()
        reset_store()
        store2 = get_store()
        assert store1 is not store2

    def test_concurrent_checkpointer_getter_creates_one_instance(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        factory = _BlockingSingletonFactory()

        with patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", side_effect=factory.context_manager):
            futures_started = ThreadPoolExecutor(max_workers=1)
            try:
                result_future = futures_started.submit(_call_getter_concurrently, get_checkpointer)
                assert factory.entered.wait(timeout=3)
                factory.release.wait(timeout=0.05)
                factory.release.set()
                results = result_future.result(timeout=3)
            finally:
                futures_started.shutdown(wait=True)

        assert all(result is factory.value for result in results)
        assert factory.enter_count() == 1

    def test_concurrent_store_getter_creates_one_instance(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        factory = _BlockingSingletonFactory()

        with patch("deerflow.runtime.store.provider._sync_store_cm", side_effect=factory.context_manager):
            futures_started = ThreadPoolExecutor(max_workers=1)
            try:
                result_future = futures_started.submit(_call_getter_concurrently, get_store)
                assert factory.entered.wait(timeout=3)
                factory.release.wait(timeout=0.05)
                factory.release.set()
                results = result_future.result(timeout=3)
            finally:
                futures_started.shutdown(wait=True)

        assert all(result is factory.value for result in results)
        assert factory.enter_count() == 1

    def test_checkpointer_loads_config_outside_singleton_lock(self):
        tracking_lock = _TrackingLock()

        def fake_ensure_config_loaded():
            assert not tracking_lock.locked()
            load_checkpointer_config_from_dict({"type": "memory"})

        with (
            patch("deerflow.runtime.checkpointer.provider._checkpointer_lock", tracking_lock),
            patch("deerflow.runtime.checkpointer.provider.ensure_config_loaded", side_effect=fake_ensure_config_loaded),
        ):
            checkpointer = get_checkpointer()

        assert checkpointer is not None
        assert tracking_lock.acquired.is_set()

    def test_store_loads_config_outside_singleton_lock(self):
        tracking_lock = _TrackingLock()

        def fake_ensure_config_loaded():
            assert not tracking_lock.locked()
            load_checkpointer_config_from_dict({"type": "memory"})

        with (
            patch("deerflow.runtime.store.provider._store_lock", tracking_lock),
            patch("deerflow.runtime.store.provider.ensure_config_loaded", side_effect=fake_ensure_config_loaded),
        ):
            store = get_store()

        assert store is not None
        assert tracking_lock.acquired.is_set()

    def test_checkpointer_reset_waits_for_initialization(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        factory = _BlockingSingletonFactory()

        with (
            patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", side_effect=factory.context_manager),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            get_future = executor.submit(get_checkpointer)
            assert factory.entered.wait(timeout=3)

            reset_started = Event()

            def reset_worker():
                reset_started.set()
                reset_checkpointer()

            reset_future = executor.submit(reset_worker)
            assert reset_started.wait(timeout=3)
            factory.release.wait(timeout=0.05)

            assert not reset_future.done()
            assert factory.exit_count() == 0

            factory.release.set()
            assert get_future.result(timeout=3) is factory.value
            reset_future.result(timeout=3)

        assert factory.exit_count() == 1

    def test_store_reset_waits_for_initialization(self):
        load_checkpointer_config_from_dict({"type": "memory"})
        factory = _BlockingSingletonFactory()

        with (
            patch("deerflow.runtime.store.provider._sync_store_cm", side_effect=factory.context_manager),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            get_future = executor.submit(get_store)
            assert factory.entered.wait(timeout=3)

            reset_started = Event()

            def reset_worker():
                reset_started.set()
                reset_store()

            reset_future = executor.submit(reset_worker)
            assert reset_started.wait(timeout=3)
            factory.release.wait(timeout=0.05)

            assert not reset_future.done()
            assert factory.exit_count() == 0

            factory.release.set()
            assert get_future.result(timeout=3) is factory.value
            reset_future.result(timeout=3)

        assert factory.exit_count() == 1


class TestAsyncCheckpointer:
    @pytest.mark.anyio
    async def test_sqlite_creates_parent_dir_via_to_thread(self):
        """Async SQLite setup should move mkdir off the event loop."""
        from deerflow.runtime.checkpointer.async_provider import _prepare_sqlite_checkpointer_path, make_checkpointer

        mock_config = MagicMock()
        mock_config.checkpointer = CheckpointerConfig(type="sqlite", connection_string="relative/test.db")

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()
        mock_saver.aclose = AsyncMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("deerflow.runtime.checkpointer.async_provider.get_app_config", return_value=mock_config),
            patch("reactivegraph.checkpoint.SqliteCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
            patch(
                "deerflow.runtime.checkpointer.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=["/tmp/resolved/test.db", mock_store.return_value],
            ) as mock_to_thread,
        ):
            async with make_checkpointer() as saver:
                assert saver is mock_saver

        assert mock_to_thread.await_count == 2
        first_fn, first_path = mock_to_thread.await_args_list[0].args
        second_fn, second_path = mock_to_thread.await_args_list[1].args
        assert first_fn is _prepare_sqlite_checkpointer_path
        assert first_path == "relative/test.db"
        assert second_fn is mock_store
        assert second_path == "/tmp/resolved/test.db"
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.asetup.assert_awaited_once()
        mock_saver.aclose.assert_awaited_once()

    @pytest.mark.anyio
    async def test_postgres_uses_native_connection(self):
        """Async postgres checkpointer should use the native saver."""
        from deerflow.runtime.checkpointer.async_provider import make_checkpointer

        mock_config = MagicMock()
        mock_config.checkpointer = CheckpointerConfig(type="postgres", connection_string="postgresql://localhost/db", postgres_schema="deerflow")

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()
        mock_saver.aclose = AsyncMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("deerflow.runtime.checkpointer.async_provider.get_app_config", return_value=mock_config),
            patch("reactivegraph.checkpoint.PostgresCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
            patch(
                "deerflow.runtime.checkpointer.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store.return_value,
            ) as mock_to_thread,
        ):
            async with make_checkpointer() as saver:
                assert saver is mock_saver

        mock_to_thread.assert_awaited_once_with(
            mock_store,
            "postgresql://localhost/db",
            schema="deerflow",
        )
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.asetup.assert_awaited_once()
        mock_saver.aclose.assert_awaited_once()

    @pytest.mark.anyio
    async def test_database_postgres_uses_native_connection(self):
        """Unified database postgres path should use the native saver."""
        from deerflow.config.database_config import DatabaseConfig
        from deerflow.runtime.checkpointer.async_provider import make_checkpointer

        db_config = DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db", postgres_schema="deerflow")
        mock_config = MagicMock()
        mock_config.checkpointer = None
        mock_config.database = db_config

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()
        mock_saver.aclose = AsyncMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("deerflow.runtime.checkpointer.async_provider.get_app_config", return_value=mock_config),
            patch("reactivegraph.checkpoint.PostgresCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
            patch(
                "deerflow.runtime.checkpointer.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store.return_value,
            ) as mock_to_thread,
        ):
            async with make_checkpointer() as saver:
                assert saver is mock_saver

        mock_to_thread.assert_awaited_once_with(
            mock_store,
            "postgresql://localhost/db",
            schema="deerflow",
        )
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.asetup.assert_awaited_once()
        mock_saver.aclose.assert_awaited_once()

    @pytest.mark.anyio
    async def test_database_sqlite_creates_parent_dir_via_to_thread(self):
        """Unified database SQLite setup should also move path IO off the event loop."""
        from deerflow.config.database_config import DatabaseConfig
        from deerflow.runtime.checkpointer.async_provider import _prepare_database_sqlite_checkpointer_path, make_checkpointer

        db_config = DatabaseConfig(backend="sqlite", sqlite_dir="relative-data")
        mock_config = MagicMock()
        mock_config.checkpointer = None
        mock_config.database = db_config

        mock_store = MagicMock()
        mock_saver = MagicMock()
        mock_saver.asetup = AsyncMock()
        mock_saver.aclose = AsyncMock()
        mock_saver_cls = MagicMock(return_value=mock_saver)

        with (
            patch("deerflow.runtime.checkpointer.async_provider.get_app_config", return_value=mock_config),
            patch("reactivegraph.checkpoint.SqliteCheckpointSaver", mock_store),
            patch(
                "deerflow.runtime.checkpointer.reactive_saver.ReactiveGraphSaver",
                mock_saver_cls,
            ),
            patch(
                "deerflow.runtime.checkpointer.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=["/tmp/data/deerflow.db", mock_store.return_value],
            ) as mock_to_thread,
        ):
            async with make_checkpointer() as saver:
                assert saver is mock_saver

        assert mock_to_thread.await_count == 2
        first_fn, first_db_config = mock_to_thread.await_args_list[0].args
        second_fn, second_path = mock_to_thread.await_args_list[1].args
        assert first_fn is _prepare_database_sqlite_checkpointer_path
        assert first_db_config is db_config
        assert second_fn is mock_store
        assert second_path == "/tmp/data/deerflow.db"
        mock_saver_cls.assert_called_once_with(store=mock_store.return_value)
        mock_saver.asetup.assert_awaited_once()
        mock_saver.aclose.assert_awaited_once()


class TestAsyncStore:
    @pytest.mark.anyio
    async def test_postgres_schema_creates_schema_and_sets_search_path(self):
        """Async Postgres store should use the legacy checkpointer schema."""
        from deerflow.runtime.store.async_provider import make_store

        mock_config = MagicMock()
        mock_config.checkpointer = CheckpointerConfig(type="postgres", connection_string="postgresql://localhost/db", postgres_schema="deerflow")

        mock_store = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store)

        with (
            patch("reactivegraph.store.PostgresStore", mock_store_cls),
            patch(
                "deerflow.runtime.store.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store,
            ) as mock_to_thread,
        ):
            async with make_store(mock_config) as store:
                assert store is mock_store

        assert mock_to_thread.await_count == 3
        constructor_call = mock_to_thread.await_args_list[0]
        assert constructor_call.args == (
            mock_store_cls,
            "postgresql://localhost/db",
        )
        assert constructor_call.kwargs == {"schema": "deerflow"}
        assert mock_to_thread.await_args_list[1].args == (mock_store.setup,)
        assert mock_to_thread.await_args_list[2].args == (mock_store.close,)

    @pytest.mark.anyio
    async def test_database_postgres_schema_creates_schema_and_sets_search_path(self):
        """Unified database postgres store should use database.postgres_schema."""
        from deerflow.config.database_config import DatabaseConfig
        from deerflow.runtime.store.async_provider import make_store

        mock_config = MagicMock()
        mock_config.checkpointer = None
        mock_config.database = DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db", postgres_schema="deerflow")

        mock_store = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store)

        with (
            patch("reactivegraph.store.PostgresStore", mock_store_cls),
            patch(
                "deerflow.runtime.store.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store,
            ) as mock_to_thread,
        ):
            async with make_store(mock_config) as store:
                assert store is mock_store

        assert mock_to_thread.await_count == 3
        constructor_call = mock_to_thread.await_args_list[0]
        assert constructor_call.args == (
            mock_store_cls,
            "postgresql://localhost/db",
        )
        assert constructor_call.kwargs == {"schema": "deerflow"}
        assert mock_to_thread.await_args_list[1].args == (mock_store.setup,)
        assert mock_to_thread.await_args_list[2].args == (mock_store.close,)


class TestCheckpointerDatabaseConfig:
    """The sync checkpointer must follow the unified ``database`` section when no
    legacy ``checkpointer`` section is configured — matching the async
    ``make_checkpointer`` factory and the sync Store provider.

    Regression: ``get_checkpointer`` / ``checkpointer_context`` previously read
    only the legacy ``checkpointer`` section and fell back to ``InMemorySaver``,
    silently ignoring ``database``. Embedded callers (``DeerFlowClient``) and the
    TUI then persisted Store rows to sqlite/postgres while checkpoints went to an
    in-memory saver and were lost on exit.
    """

    def test_sync_checkpointer_context_uses_database_config(self):
        """The one-shot sync checkpointer factory must follow unified database config."""
        from deerflow.runtime.checkpointer.provider import checkpointer_context

        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected = object()
        factory = MagicMock(return_value=nullcontext(expected))

        with (
            patch("deerflow.runtime.checkpointer.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", factory),
            checkpointer_context() as cp,
        ):
            assert cp is expected

        resolved = factory.call_args.args[0]
        assert resolved.type == "postgres"
        assert resolved.connection_string == "postgresql://localhost/db"

    def test_sync_checkpointer_context_uses_sqlite_database_config(self, tmp_path):
        """The one-shot sync checkpointer factory must resolve the sqlite branch too, not just postgres."""
        from deerflow.runtime.checkpointer.provider import checkpointer_context

        db_config = DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path))
        app_config = SimpleNamespace(checkpointer=None, database=db_config)
        expected = object()
        factory = MagicMock(return_value=nullcontext(expected))

        with (
            patch("deerflow.runtime.checkpointer.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", factory),
            checkpointer_context() as cp,
        ):
            assert cp is expected

        resolved = factory.call_args.args[0]
        assert resolved.type == "sqlite"
        assert resolved.connection_string == db_config.checkpointer_sqlite_path

    def test_sync_checkpointer_singleton_uses_database_config(self):
        """The cached sync checkpointer factory must resolve database config before locking."""
        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected = object()
        factory = MagicMock(return_value=nullcontext(expected))

        with (
            patch("deerflow.runtime.checkpointer.provider.ensure_config_loaded"),
            patch("deerflow.runtime.checkpointer.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", factory),
        ):
            assert get_checkpointer() is expected

        resolved = factory.call_args.args[0]
        assert resolved.type == "postgres"
        assert resolved.connection_string == "postgresql://localhost/db"

    def test_sync_checkpointer_falls_back_to_memory_when_config_file_is_missing(self):
        """The sync checkpointer keeps its no-config fallback for embedded callers."""
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        with (
            patch("deerflow.runtime.checkpointer.provider.ensure_config_loaded"),
            patch("deerflow.runtime.checkpointer.provider.get_checkpointer_config", return_value=None),
            patch("deerflow.runtime.checkpointer.provider.get_app_config", side_effect=FileNotFoundError),
        ):
            assert isinstance(get_checkpointer(), ReactiveGraphSaver)

    def test_legacy_checkpointer_config_takes_precedence(self):
        """Backward-compatible checkpointer config must override database."""
        from deerflow.runtime.checkpointer.provider import checkpointer_context

        app_config = SimpleNamespace(
            checkpointer=CheckpointerConfig(type="memory"),
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected = object()
        factory = MagicMock(return_value=nullcontext(expected))

        with (
            patch("deerflow.runtime.checkpointer.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.checkpointer.provider._sync_checkpointer_cm", factory),
            checkpointer_context() as cp,
        ):
            assert cp is expected

        resolved = factory.call_args.args[0]
        assert resolved.type == "memory"
        assert resolved.connection_string is None

    def test_explicit_memory_database_uses_in_memory_saver(self):
        """Explicit memory mode remains an intentional non-persistent checkpointer."""
        from deerflow.runtime.checkpointer.provider import checkpointer_context
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        app_config = SimpleNamespace(checkpointer=None, database=DatabaseConfig(backend="memory"))

        with (
            patch("deerflow.runtime.checkpointer.provider.get_app_config", return_value=app_config),
            checkpointer_context() as cp,
        ):
            assert isinstance(cp, ReactiveGraphSaver)


class TestStoreDatabaseConfig:
    def test_sync_store_falls_back_to_memory_when_config_file_is_missing(self):
        """The sync Store keeps its no-config fallback for embedded callers.

        The memory backend is served by ReactiveGraph's ``MemoryStore``; the
        intent of this test (never None, always a usable in-process store) is
        unchanged.
        """
        from reactivegraph.store import MemoryStore

        with (
            patch("deerflow.runtime.store.provider.ensure_config_loaded"),
            patch("deerflow.runtime.store.provider.get_checkpointer_config", return_value=None),
            patch("deerflow.runtime.store.provider.get_app_config", side_effect=FileNotFoundError),
        ):
            assert isinstance(get_store(), MemoryStore)

    @pytest.mark.anyio
    async def test_async_postgres_store_uses_database_config(self, caplog):
        """Unified database postgres config must not fall back to the memory store."""
        from deerflow.runtime.store.async_provider import make_store

        caplog.set_level("WARNING", logger="deerflow.runtime.store.async_provider")
        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        mock_store = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store)

        with (
            patch("reactivegraph.store.PostgresStore", mock_store_cls),
            patch(
                "deerflow.runtime.store.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store,
            ) as mock_to_thread,
        ):
            async with make_store(app_config) as store:
                assert store is mock_store

        assert mock_to_thread.await_count == 3
        constructor_call = mock_to_thread.await_args_list[0]
        assert constructor_call.args == (
            mock_store_cls,
            "postgresql://localhost/db",
        )
        assert constructor_call.kwargs == {"schema": "public"}
        assert mock_to_thread.await_args_list[1].args == (mock_store.setup,)
        assert mock_to_thread.await_args_list[2].args == (mock_store.close,)
        assert "No 'checkpointer' section" not in caplog.text

    @pytest.mark.anyio
    async def test_async_sqlite_store_uses_unified_database_path(self, tmp_path):
        """Unified database SQLite config must use the shared deerflow.db path."""
        from deerflow.runtime.store.async_provider import make_store
        from deerflow.runtime.store.provider import ensure_sqlite_parent_dir

        db_config = DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path))
        app_config = SimpleNamespace(checkpointer=None, database=db_config)
        mock_store = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store)

        with (
            patch("reactivegraph.store.SqliteStore", mock_store_cls),
            patch(
                "deerflow.runtime.store.async_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_store,
            ) as mock_to_thread,
        ):
            async with make_store(app_config) as store:
                assert store is mock_store

        assert mock_to_thread.await_count == 4
        called_fn, called_path = mock_to_thread.await_args_list[0].args
        assert called_fn is ensure_sqlite_parent_dir
        assert called_path == db_config.checkpointer_sqlite_path
        assert mock_to_thread.await_args_list[1].args == (
            mock_store_cls,
            db_config.checkpointer_sqlite_path,
        )
        assert mock_to_thread.await_args_list[2].args == (mock_store.setup,)
        assert mock_to_thread.await_args_list[3].args == (mock_store.close,)

    def test_sync_store_context_uses_database_config(self):
        """The one-shot sync Store factory must follow unified database config."""
        from deerflow.runtime.store.provider import store_context

        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected_store = object()
        factory = MagicMock(return_value=nullcontext(expected_store))

        with (
            patch("deerflow.runtime.store.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.store.provider._sync_store_cm", factory),
            store_context() as store,
        ):
            assert store is expected_store

        resolved = factory.call_args.args[0]
        assert resolved.type == "postgres"
        assert resolved.connection_string == "postgresql://localhost/db"

    def test_sync_store_singleton_uses_database_config(self):
        """The cached sync Store factory must resolve database config before locking."""
        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected_store = object()
        factory = MagicMock(return_value=nullcontext(expected_store))

        with (
            patch("deerflow.runtime.store.provider.ensure_config_loaded"),
            patch("deerflow.runtime.store.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.store.provider._sync_store_cm", factory),
        ):
            assert get_store() is expected_store

        resolved = factory.call_args.args[0]
        assert resolved.type == "postgres"
        assert resolved.connection_string == "postgresql://localhost/db"

    def test_legacy_checkpointer_config_takes_precedence_for_store(self):
        """Backward-compatible checkpointer config must override database for Store."""
        from deerflow.runtime.store.provider import store_context

        app_config = SimpleNamespace(
            checkpointer=CheckpointerConfig(type="memory"),
            database=DatabaseConfig(backend="postgres", postgres_url="postgresql://localhost/db"),
        )
        expected_store = object()
        factory = MagicMock(return_value=nullcontext(expected_store))

        with (
            patch("deerflow.runtime.store.provider.get_app_config", return_value=app_config),
            patch("deerflow.runtime.store.provider._sync_store_cm", factory),
            store_context() as store,
        ):
            assert store is expected_store

        resolved = factory.call_args.args[0]
        assert resolved.type == "memory"
        assert resolved.connection_string is None

    def test_explicit_memory_database_uses_in_memory_store(self):
        """Explicit memory mode remains an intentional non-persistent Store."""
        from reactivegraph.store import MemoryStore

        from deerflow.runtime.store.provider import store_context

        app_config = SimpleNamespace(
            checkpointer=None,
            database=DatabaseConfig(backend="memory"),
        )
        with patch("deerflow.runtime.store.provider.get_app_config", return_value=app_config):
            with store_context() as store:
                assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# app_config.py integration
# ---------------------------------------------------------------------------


class TestAppConfigLoadsCheckpointer:
    def test_load_checkpointer_section(self):
        """load_checkpointer_config_from_dict populates the global config."""
        set_checkpointer_config(None)
        load_checkpointer_config_from_dict({"type": "memory"})
        cfg = get_checkpointer_config()
        assert cfg is not None
        assert cfg.type == "memory"


# ---------------------------------------------------------------------------
# DeerFlowClient falls back to config checkpointer
# ---------------------------------------------------------------------------


class TestClientCheckpointerFallback:
    def test_client_uses_config_checkpointer_when_none_provided(self):
        """DeerFlowClient._ensure_agent falls back to get_checkpointer() when checkpointer=None."""
        from deerflow.client import DeerFlowClient
        from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver

        load_checkpointer_config_from_dict({"type": "memory"})

        captured_kwargs = {}

        def fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        model_mock = MagicMock()
        config_mock = MagicMock()
        config_mock.models = [model_mock]
        config_mock.database.checkpoint_delta.snapshot_frequency = 10
        config_mock.get_model_config.return_value = MagicMock(supports_vision=False)
        config_mock.checkpointer = None

        config_mock.skills.deferred_discovery = False
        config_mock.skills.container_path = "/mnt/skills"
        config_mock.tool_search.enabled = False

        with (
            patch("deerflow.client.get_app_config", return_value=config_mock),
            patch("deerflow.client.create_agent", side_effect=fake_create_agent),
            patch("deerflow.client.create_chat_model", return_value=MagicMock()),
            patch("deerflow.client.build_middlewares", return_value=[]),
            patch("deerflow.client.apply_prompt_template", return_value=""),
            patch("deerflow.client.get_enabled_skills_for_config", return_value=[]),
            patch("deerflow.client.DeerFlowClient._get_tools", return_value=[]),
        ):
            client = DeerFlowClient(checkpointer=None)
            config = client._get_runnable_config("test-thread")
            client._ensure_agent(config)

        assert "checkpointer" in captured_kwargs
        assert isinstance(captured_kwargs["checkpointer"], ReactiveGraphSaver)

    def test_client_explicit_checkpointer_takes_precedence(self):
        """An explicitly provided checkpointer is used even when config checkpointer is set."""
        from deerflow.client import DeerFlowClient

        load_checkpointer_config_from_dict({"type": "memory"})

        explicit_cp = MagicMock()
        captured_kwargs = {}

        def fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        model_mock = MagicMock()
        config_mock = MagicMock()
        config_mock.models = [model_mock]
        config_mock.database.checkpoint_delta.snapshot_frequency = 10
        config_mock.get_model_config.return_value = MagicMock(supports_vision=False)
        config_mock.checkpointer = None

        config_mock.skills.deferred_discovery = False
        config_mock.skills.container_path = "/mnt/skills"
        config_mock.tool_search.enabled = False

        with (
            patch("deerflow.client.get_app_config", return_value=config_mock),
            patch("deerflow.client.create_agent", side_effect=fake_create_agent),
            patch("deerflow.client.create_chat_model", return_value=MagicMock()),
            patch("deerflow.client.build_middlewares", return_value=[]),
            patch("deerflow.client.apply_prompt_template", return_value=""),
            patch("deerflow.client.get_enabled_skills_for_config", return_value=[]),
            patch("deerflow.client.DeerFlowClient._get_tools", return_value=[]),
        ):
            client = DeerFlowClient(checkpointer=explicit_cp)
            config = client._get_runnable_config("test-thread")
            client._ensure_agent(config)

        assert captured_kwargs["checkpointer"] is explicit_cp

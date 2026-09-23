"""Integration: DeerFlow's Store factory serves ReactiveGraph's MemoryStore.

The Store is DeerFlow's long-term memory surface (thread metadata, agent
memories). These tests pin the in-place swap of the ``memory`` backend: the
factory must return our store, and every operation DeerFlow actually performs
against it (through ``MemoryThreadMetaStore`` and directly) must behave the
same as it did on ``langgraph.store.memory.InMemoryStore``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.store.base import BaseStore

from deerflow.config.database_config import DatabaseConfig
from deerflow.runtime.store.async_provider import make_store
from deerflow.runtime.store.provider import get_store, reset_store, store_context
from reactivegraph.store import MemoryStore as ReactiveMemoryStore


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_store()
    yield
    reset_store()


def _app_config(backend: str):
    return SimpleNamespace(
        checkpointer=None,
        database=DatabaseConfig(backend=backend),
    )


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_sync_memory_store_is_reactivegraph() -> None:
    """The sync factory's memory backend is ours, not LangGraph's."""
    with patch(
        "deerflow.runtime.store.provider.get_app_config",
        return_value=_app_config("memory"),
    ):
        with store_context() as store:
            assert isinstance(store, ReactiveMemoryStore)


def test_sync_singleton_is_reactivegraph() -> None:
    with patch(
        "deerflow.runtime.store.provider.ensure_config_loaded",
    ), patch(
        "deerflow.runtime.store.provider.get_checkpointer_config",
        return_value=None,
    ), patch(
        "deerflow.runtime.store.provider.get_app_config",
        side_effect=FileNotFoundError,
    ):
        assert isinstance(get_store(), ReactiveMemoryStore)


@pytest.mark.anyio
async def test_async_memory_store_is_reactivegraph() -> None:
    async with make_store(_app_config("memory")) as store:
        assert isinstance(store, ReactiveMemoryStore)


def test_memory_store_satisfies_host_protocol() -> None:
    """Host code passes the store where a BaseStore is expected."""
    with patch(
        "deerflow.runtime.store.provider.get_app_config",
        return_value=_app_config("memory"),
    ):
        with store_context() as store:
            # Structural, not nominal: our store is not a BaseStore subclass,
            # and host code must not require it to be.
            assert not isinstance(store, BaseStore)
            for method in (
                "get", "put", "delete", "search", "list_namespaces", "batch",
                "aget", "aput", "adelete", "asearch", "alist_namespaces", "abatch",
            ):
                assert callable(getattr(store, method))


# ---------------------------------------------------------------------------
# Thread-metadata workload (the real consumer)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_thread_meta_store_roundtrip_on_reactive_store() -> None:
    """The memory thread-metadata backend works unchanged on our store."""
    from deerflow.persistence.thread_meta.memory import MemoryThreadMetaStore

    store = ReactiveMemoryStore()
    threads = MemoryThreadMetaStore(store)

    created = await threads.create(
        "t-1", assistant_id="lead", display_name="First", user_id="u1"
    )
    assert created["thread_id"] == "t-1"

    fetched = await threads.get("t-1", user_id="u1")
    assert fetched is not None
    assert fetched["display_name"] == "First"

    listing = await threads.search(user_id="u1", limit=10)
    assert [record["thread_id"] for record in listing] == ["t-1"]
    # Ownership scoping still filters other users out.
    assert await threads.search(user_id="u2", limit=10) == []

    assert await threads.check_access("t-1", "u1") is True
    assert await threads.check_access("t-1", "u2") is False


@pytest.mark.anyio
async def test_thread_meta_search_filters_by_status_on_reactive_store() -> None:
    from deerflow.persistence.thread_meta.memory import MemoryThreadMetaStore

    threads = MemoryThreadMetaStore(ReactiveMemoryStore())
    await threads.create("idle-1", user_id="u1")
    await threads.create("closed-1", user_id="u1")
    await threads.update_status("closed-1", "closed", user_id="u1")

    # New threads start "idle" (upstream default), so the status filter must
    # partition them exactly as it does on LangGraph's store.
    idle_ids = {
        r["thread_id"] for r in await threads.search(status="idle", user_id="u1", limit=10)
    }
    closed_ids = {
        r["thread_id"] for r in await threads.search(status="closed", user_id="u1", limit=10)
    }
    assert idle_ids == {"idle-1"}
    assert closed_ids == {"closed-1"}


# ---------------------------------------------------------------------------
# Direct store behaviour through the factory
# ---------------------------------------------------------------------------


def test_store_operations_through_factory() -> None:
    with patch(
        "deerflow.runtime.store.provider.get_app_config",
        return_value=_app_config("memory"),
    ):
        with store_context() as store:
            store.put(("memories", "user-1"), "m1", {"text": "likes tea"})
            item = store.get(("memories", "user-1"), "m1")
            assert item is not None and item.value == {"text": "likes tea"}

            found = store.search(("memories",), filter={"text": "likes tea"})
            assert [i.key for i in found] == ["m1"]

            store.delete(("memories", "user-1"), "m1")
            assert store.get(("memories", "user-1"), "m1") is None


@pytest.mark.anyio
async def test_store_async_operations_through_factory() -> None:
    async with make_store(_app_config("memory")) as store:
        await store.aput(("memories",), "m1", {"text": "async"})
        assert (await store.aget(("memories",), "m1")).value == {"text": "async"}
        assert [i.key for i in await store.asearch(("memories",))] == ["m1"]


def test_unknown_backend_fails_closed() -> None:
    """A backend we have not ported must raise, never silently degrade."""
    from deerflow.runtime.store.provider import _sync_store_cm

    with pytest.raises(ValueError, match="Unknown store backend type"):
        with _sync_store_cm(SimpleNamespace(type="nonexistent", connection_string=None)):
            pass

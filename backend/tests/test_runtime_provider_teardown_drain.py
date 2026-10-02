"""Provider teardown must settle before its owner is considered finished.

Context for the invariant: `backend/packages/harness/deerflow/runtime/AGENTS.md`
("Provider contexts drain owned cache/bridge/checkpointer/Store teardown across
cancellation").

The bug these tests pin: the providers used to release their backend with

    try:
        yield resource
    finally:
        await asyncio.to_thread(resource.close)

`asyncio.to_thread` is not shielded. When the caller is cancelled while the
close is in flight, the `await` raises immediately and the close keeps running
in a detached thread — so the provider looks finished while the database handle
is still open. At shutdown the OS can kill that orphaned close mid-flight.

The tests therefore assert on the resource state *at the moment the owner task
finishes*, not after an extra sleep: sleeping first hides exactly the bug.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest

from deerflow.runtime.cancellation import drained_async_context, drained_resource


class _BlockingClose:
    """A resource whose (blocking) close takes long enough to be interrupted."""

    def __init__(self, *, close_delay: float = 0.1, fail_setup: bool = False) -> None:
        self.closed = False
        self.close_started = False
        self.setup_called = False
        self._close_delay = close_delay
        self._fail_setup = fail_setup

    # `setup`/`close` are called two different ways in the codebase: the
    # checkpointer provider awaits them directly, the store provider bridges
    # the synchronous backend through `asyncio.to_thread`. Behave like the real
    # backends by exposing both a sync and an awaitable form.
    def setup(self) -> None:
        self.setup_called = True
        if self._fail_setup:
            raise RuntimeError("setup boom")

    def close(self) -> None:
        self.close_started = True
        time.sleep(self._close_delay)
        self.closed = True

    async def asetup(self) -> None:
        self.setup()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _close_blocking(self) -> None:
        self.close_started = True
        time.sleep(self._close_delay)
        self.closed = True


async def _cancel_repeatedly(task: asyncio.Task[None], *, rounds: int = 2) -> None:
    """Cancel *task* while its teardown is running, then await it."""
    task.cancel()
    for _ in range(rounds):
        await asyncio.sleep(0.02)
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_drained_resource_settles_close_before_owner_exits():
    resource = _BlockingClose()

    @asynccontextmanager
    async def provider():
        async with drained_resource(resource, resource.aclose):
            yield resource

    async def owner() -> None:
        async with provider():
            await asyncio.sleep(10)

    task = asyncio.create_task(owner())
    await asyncio.sleep(0.02)
    await _cancel_repeatedly(task)

    # No sleep here on purpose: the close must already be finished.
    assert resource.close_started is True
    assert resource.closed is True, "provider returned before the backend close settled; the handle is left open in a detached thread"


@pytest.mark.anyio
async def test_drained_resource_releases_on_setup_failure():
    """A failing setup must still close the resource it already opened."""
    resource = _BlockingClose(fail_setup=True, close_delay=0.0)

    @asynccontextmanager
    async def provider():
        async with drained_resource(resource, resource.aclose):
            await resource.asetup()
            yield resource

    with pytest.raises(RuntimeError, match="setup boom"):
        async with provider():
            pass

    assert resource.setup_called is True
    assert resource.closed is True


@pytest.mark.anyio
async def test_drained_resource_propagates_close_failure():
    """A broken close is reported instead of being masked by cancellation."""

    class _BrokenClose:
        async def close(self) -> None:
            raise OSError("close boom")

    resource = _BrokenClose()

    async def owner() -> None:
        async with drained_resource(resource, resource.close):
            await asyncio.sleep(10)

    task = asyncio.create_task(owner())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(OSError, match="close boom"):
        await task


@pytest.mark.anyio
async def test_drained_async_context_settles_aexit_before_owner_exits():
    exit_finished = asyncio.Event()
    exit_started = asyncio.Event()
    allow_exit = asyncio.Event()

    class _SlowExit:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            exit_started.set()
            await allow_exit.wait()
            exit_finished.set()
            return False

    async def owner() -> None:
        async with drained_async_context(_SlowExit()):
            await asyncio.sleep(10)

    task = asyncio.create_task(owner())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.wait_for(exit_started.wait(), timeout=1)
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done(), "owner returned before the context exit settled"

    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done(), "repeated cancellation interrupted the context exit"

    allow_exit.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert exit_finished.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["checkpointer", "store"])
async def test_real_provider_releases_backend_across_cancellation(kind: str, tmp_path, monkeypatch) -> None:
    """Drive the real provider with a probe backend and watch the handle.

    The unit tests above pin `drained_resource`; this one pins that the
    providers actually use it. The providers import their backend class lazily
    *inside* the function, so the probe replaces the class on the source module
    (`reactivegraph.checkpoint` / `reactivegraph.store`) rather than on the
    provider module.
    """
    import importlib

    backend = _BlockingClose(close_delay=0.15)

    if kind == "checkpointer":
        module = importlib.import_module("deerflow.runtime.checkpointer.async_provider")
        source = importlib.import_module("reactivegraph.checkpoint")
        monkeypatch.setattr(source, "SqliteCheckpointSaver", lambda *a, **k: backend)
        monkeypatch.setattr(module, "_prepare_sqlite_checkpointer_path", lambda raw: raw)
        config = type("Cfg", (), {"type": "sqlite", "connection_string": str(tmp_path / "x.db")})()

        def factory():
            return module._async_checkpointer(config)
    else:
        module = importlib.import_module("deerflow.runtime.store.async_provider")
        source = importlib.import_module("reactivegraph.store")
        monkeypatch.setattr(source, "SqliteStore", lambda *a, **k: backend)
        monkeypatch.setattr(module, "resolve_sqlite_conn_str", lambda s: s)
        monkeypatch.setattr(module, "ensure_sqlite_parent_dir", lambda s: None)
        config = type("Cfg", (), {"type": "sqlite", "connection_string": str(tmp_path / "x.db")})()

        def factory():
            return module._async_store(config)

    holder: dict[str, object] = {}

    async def owner() -> None:
        async with factory() as resource:
            holder["resource"] = resource
            await asyncio.sleep(10)

    task = asyncio.create_task(owner())
    await asyncio.sleep(0.05)
    assert holder, "provider never yielded a resource"
    await _cancel_repeatedly(task)

    assert backend.setup_called is True, "probe backend was never set up"
    assert backend.closed is True, f"{kind} provider returned before the backend close settled"

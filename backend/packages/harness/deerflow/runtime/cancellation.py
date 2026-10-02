from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TypeVar

T = TypeVar("T")


async def wait_for_task_until(  # noqa: UP047
    task: asyncio.Future[T], *, deadline: float
) -> bool:
    """Wait through repeated caller cancellation without cancelling task."""
    loop = asyncio.get_running_loop()
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            continue
        if task in done:
            return True
    return True


async def _drain_teardown(awaitable: Awaitable[None]) -> None:
    """Run *awaitable* to completion, absorbing repeated caller cancellation.

    A teardown that is abandoned half-way leaves the resource owned by a task
    that has already finished. For a database handle that means the connection
    is never returned, and at process shutdown the close can be killed
    mid-flight. So cancellation is deferred until the teardown settles, then
    re-raised — matching normal context-manager semantics.

    A failure raised by the teardown wins over the cancellation, so a broken
    close is reported rather than masked by the ``CancelledError``.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                continue
        task.result()  # surface a teardown failure in place of the cancellation
        raise


@asynccontextmanager
async def drained_async_context[T](  # noqa: UP047
    manager: AbstractAsyncContextManager[T],
) -> AsyncIterator[T]:
    """Keep an entered async context owned until its exit fully settles."""
    value = await manager.__aenter__()
    try:
        yield value
    except BaseException as exc:
        task = asyncio.ensure_future(manager.__aexit__(type(exc), exc, exc.__traceback__))
        try:
            suppressed = await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    continue
            task.result()
            raise
        if not suppressed:
            raise
    else:
        await _drain_teardown(manager.__aexit__(None, None, None))


@asynccontextmanager
async def drained_resource[T](  # noqa: UP047
    resource: T,
    close: Callable[[], Awaitable[None]],
) -> AsyncIterator[T]:
    """Yield *resource*, then drain ``close()`` across repeated cancellation.

    The providers in this package own a plain object exposing a synchronous
    ``close`` (bridged through ``asyncio.to_thread``) rather than an async
    context manager, so they cannot use :func:`drained_async_context`. Without
    draining, ``await asyncio.to_thread(store.close)`` returns as soon as the
    cancellation is delivered while the close keeps running in a detached
    thread — the owner looks finished while the connection is still open.
    """
    try:
        yield resource
    finally:
        await _drain_teardown(close())

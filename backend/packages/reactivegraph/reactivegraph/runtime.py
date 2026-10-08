"""Run-scoped runtime context for tools, middleware and node functions.

A node rarely receives the run's config as an argument: it is called deep
inside a tool or middleware that only knows about its own parameters. The
engine therefore publishes the active config and runtime on a
:class:`~contextvars.ContextVar`, and these helpers read it back.

The value is scoped to the *task* that set it, so concurrent runs in one
process never observe each other's context.

Portions of this module are adapted from upstream `langgraph`
(https://github.com/langchain-ai/langgraph) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Generator, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Generic

from typing_extensions import TypeVar

__all__ = (
    "Runtime",
    "get_config",
    "get_runtime",
    "get_stream_writer",
    "runtime_context",
)

StreamWriter = Callable[[Any], None]
ContextT = TypeVar("ContextT", default=None)
T = TypeVar("T")


class _PinnedIterator(Generic[T]):
    """An iterator whose generator always advances in one captured Context.

    A synchronous generator may be advanced by different executor threads
    (``asyncio.to_thread(next, iterator)``). A ContextVar token set before one
    ``yield`` cannot be reset after a later ``yield`` in another thread's
    Context, and values set before the first yield would otherwise disappear
    on the next advance. Pinning every generator operation to the Context
    captured at construction keeps the whole run in one Context, matching the
    way LangGraph supports the same thread-hopping iteration pattern.
    """

    def __init__(self, generator: Generator[T, Any, Any]) -> None:
        self._generator = generator
        self._context = contextvars.copy_context()

    def __iter__(self) -> _PinnedIterator[T]:
        return self

    def __next__(self) -> T:
        return self._context.run(next, self._generator)

    def send(self, value: Any) -> T:
        """Send a value into the channel."""
        return self._context.run(self._generator.send, value)

    def throw(self, *args: Any) -> T:
        """Raise an exception inside the channel."""
        return self._context.run(self._generator.throw, *args)

    def close(self) -> None:
        """Close the channel / backing resource."""
        self._context.run(self._generator.close)

    def __del__(self) -> None:
        # A caller may abandon an iterator without exhausting or closing it.
        # CPython then closes the raw generator directly in the GC thread,
        # bypassing this wrapper and the pinned Context. Explicitly close it
        # here so GeneratorExit always reaches the run scope in that Context.
        close = getattr(self, "close", None)
        if close is not None:
            close()


def _no_op_stream_writer(_chunk: Any) -> None:
    """Default writer for a runtime that is not attached to a stream."""


def _no_op_heartbeat() -> None:
    """Default progress signal for a runtime with no idle timeout to refresh."""


@dataclass
class Runtime(Generic[ContextT]):
    """Run-scoped context and utilities handed to nodes and middleware.

    ``config`` is deliberately *not* a field here: read it with
    :func:`get_config`, which returns the whole active config including
    ``configurable`` and tracing metadata.
    """

    context: ContextT | None = None
    store: Any = None
    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    previous: Any = None
    execution_info: Any = None
    server_info: Any = None
    control: Any = None

    def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
        """Fill this runtime's unset fields from *other*.

        Upstream treats every field except ``previous`` with ``or`` semantics,
        so a *falsy* value in ``other`` (``{}``, ``[]``, ``0``) does **not**
        shadow a set value here. ``previous`` is the exception: only ``None``
        falls back, so a legitimate ``False``/``0`` result survives.
        """
        return Runtime(
            context=other.context or self.context,
            store=other.store or self.store,
            stream_writer=(
                other.stream_writer
                if other.stream_writer is not _no_op_stream_writer
                else self.stream_writer
            ),
            heartbeat=(
                other.heartbeat if other.heartbeat is not _no_op_heartbeat else self.heartbeat
            ),
            previous=self.previous if other.previous is None else other.previous,
            execution_info=other.execution_info or self.execution_info,
            server_info=other.server_info or self.server_info,
            control=other.control or self.control,
        )

    def override(self, **overrides: Any) -> Runtime[ContextT]:
        """Return a copy with *overrides* applied."""
        return replace(self, **overrides)

    @property
    def drain_requested(self) -> bool:
        """Whether the stream was asked to drain and stop."""
        return bool(getattr(self.control, "drain_requested", False))

    @property
    def drain_reason(self) -> str | None:
        """Why the stream is draining (or ``None``)."""
        return getattr(self.control, "drain_reason", None)


_current_config: ContextVar[dict[str, Any] | None] = ContextVar(
    "reactivegraph_child_config", default=None
)
_current_runtime: ContextVar[Runtime[Any] | None] = ContextVar(
    "reactivegraph_runtime", default=None
)

_OUTSIDE_RUN = "Called get_config outside of a runnable context"

# ``configurable`` is the upstream LangGraph key holding the run's Runtime.
_PREGEL_RUNTIME_KEY = "__pregel_runtime"


@lru_cache(maxsize=1)
def _langchain_config_var() -> ContextVar[Any] | None:
    """Return langchain-core's child-runnable config ContextVar, if present.

    ``langgraph.config.get_config``/``get_stream_writer`` and
    ``langgraph.runtime.get_runtime`` are thin readers of this variable
    upstream. Hosts that were written against LangGraph import those helpers
    directly, so publishing the same variable is what makes this engine a
    drop-in replacement instead of forcing every host to rewrite its imports.

    langchain-core is an optional integration here (the core package does not
    depend on it), so a missing module simply disables the bridge.
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config
    except ImportError:
        return None
    return var_child_runnable_config


def _langgraph_bridged_config(
    config: dict[str, Any] | None,
    runtime: Runtime[Any] | None,
) -> dict[str, Any]:
    """Return the RunnableConfig shape upstream helpers expect.

    The caller's ``configurable`` mapping is copied rather than mutated, and
    the run's :class:`Runtime` is published under the upstream
    ``__pregel_runtime`` key so ``langgraph.runtime.get_runtime()`` and
    ``get_stream_writer()`` resolve to the same object the engine bound.
    """
    bridged = dict(config) if config is not None else {}
    raw_configurable = bridged.get("configurable")
    configurable = dict(raw_configurable) if isinstance(raw_configurable, dict) else {}
    if runtime is not None:
        configurable[_PREGEL_RUNTIME_KEY] = runtime
    bridged["configurable"] = configurable
    return bridged


@contextlib.contextmanager
def runtime_context(
    *,
    config: dict[str, Any] | None = None,
    runtime: Runtime[Any] | None = None,
) -> Iterator[None]:
    """Publish *config* / *runtime* for the duration of the block.

    The previous values are restored on exit, and because both live in
    context variables the binding is visible only to the current task and its
    children.
    """
    tokens: list[tuple[ContextVar, Any]] = []
    if config is not None or runtime is not None:
        # One bridged view is published everywhere, matching upstream: inside
        # a run ``get_config()`` always succeeds and its ``configurable``
        # mapping carries the active Runtime under ``__pregel_runtime``.
        bridged = _langgraph_bridged_config(config, runtime)
        tokens.append((_current_config, _current_config.set(bridged)))
        langchain_var = _langchain_config_var()
        if langchain_var is not None:
            tokens.append((langchain_var, langchain_var.set(bridged)))
    if runtime is not None:
        tokens.append((_current_runtime, _current_runtime.set(runtime)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _host_helper(name: str) -> Any | None:
    """Return ``langgraph.config.<name>`` when the host module is importable.

    DeerFlow middleware was written against ``langgraph.config``. Modules that
    were switched to the engine import must keep working when they are *hosted*
    by a real LangGraph run (a graph still using ``langgraph.graph``), so the
    engine helpers delegate to the host helper instead of raising. Delegating
    to the module attribute — rather than re-deriving the writer from the host
    config — also preserves the seam host code (and its tests) install there.
    """
    try:
        import langgraph.config as host_config
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return getattr(host_config, name, None)


def _host_runtime_helper() -> Any | None:
    """Return ``langgraph.runtime.get_runtime`` when the host is importable.

    ``langgraph.config`` has never exposed ``get_runtime`` — it only carries
    ``get_config``/``get_store``/``get_stream_writer``. The accessor lives in
    ``langgraph.runtime``, so probing ``langgraph.config`` alone made every
    engine ``get_runtime()`` call from inside a *host* graph node raise
    ``RuntimeError`` (measured against LangGraph 1.2.9: DeerFlow's MCP
    interceptor then reported request-scoped secrets missing and denied the
    tool call).
    """
    try:
        import langgraph.runtime as host_runtime
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return getattr(host_runtime, "get_runtime", None)


def get_config() -> dict[str, Any]:
    """Return the active run config, or raise outside a run.

    Raising (rather than returning ``{}``) is what lets callers distinguish
    "no thread id because we are not in a run" from a real empty config.
    """
    config = _current_config.get()
    if config is not None:
        return config
    host_get_config = _host_helper("get_config")
    if host_get_config is None:
        raise RuntimeError(_OUTSIDE_RUN)
    return host_get_config()


def get_runtime(context_schema: type | None = None) -> Runtime[Any]:
    """Return the active :class:`Runtime`, or raise outside a run.

    *context_schema* is accepted for call-site type inference only; the
    returned runtime is the one bound by :func:`runtime_context`.
    """
    runtime = _current_runtime.get()
    if runtime is not None:
        return runtime
    host_get_runtime = _host_runtime_helper()
    if host_get_runtime is None:
        raise RuntimeError(_OUTSIDE_RUN)
    # ``context_schema`` is forwarded: the host accessor accepts it for call-site
    # type inference, exactly as this one does.
    return host_get_runtime(context_schema)


def get_stream_writer() -> StreamWriter:
    """Return the writer that publishes custom frames for the active run.

    The engine's own binding always wins; outside an engine run this is the
    host helper, so a module written against either import keeps emitting.
    """
    runtime = _current_runtime.get()
    if runtime is not None:
        return runtime.stream_writer
    host_get_stream_writer = _host_helper("get_stream_writer")
    if host_get_stream_writer is None:
        raise RuntimeError(_OUTSIDE_RUN)
    return host_get_stream_writer()

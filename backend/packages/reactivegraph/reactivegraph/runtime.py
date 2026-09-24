"""Run-scoped runtime context for tools, middleware and node functions.

A node rarely receives the run's config as an argument: it is called deep
inside a tool or middleware that only knows about its own parameters. The
engine therefore publishes the active config and runtime on a
:class:`~contextvars.ContextVar`, and these helpers read it back.

The value is scoped to the *task* that set it, so concurrent runs in one
process never observe each other's context.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = (
    "Runtime",
    "get_config",
    "get_runtime",
    "get_stream_writer",
    "runtime_context",
)

StreamWriter = Callable[[Any], None]


def _no_op_stream_writer(_chunk: Any) -> None:
    """Default writer for a runtime that is not attached to a stream."""


def _no_op_heartbeat() -> None:
    """Default progress signal for a runtime with no idle timeout to refresh."""


@dataclass
class Runtime:
    """Run-scoped context and utilities handed to nodes and middleware.

    ``config`` is deliberately *not* a field here: read it with
    :func:`get_config`, which returns the whole active config including
    ``configurable`` and tracing metadata.
    """

    context: Any = None
    store: Any = None
    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    previous: Any = None
    execution_info: Any = None
    server_info: Any = None
    control: Any = None

    def merge(self, other: Runtime) -> Runtime:
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

    def override(self, **overrides: Any) -> Runtime:
        """Return a copy with *overrides* applied."""
        return replace(self, **overrides)

    @property
    def drain_requested(self) -> bool:
        return bool(getattr(self.control, "drain_requested", False))

    @property
    def drain_reason(self) -> str | None:
        return getattr(self.control, "drain_reason", None)


_current_config: ContextVar[dict[str, Any] | None] = ContextVar(
    "reactivegraph_child_config", default=None
)
_current_runtime: ContextVar[Runtime | None] = ContextVar(
    "reactivegraph_runtime", default=None
)

_OUTSIDE_RUN = "Called get_config outside of a runnable context"


@contextlib.contextmanager
def runtime_context(
    *,
    config: dict[str, Any] | None = None,
    runtime: Runtime | None = None,
) -> Iterator[None]:
    """Publish *config* / *runtime* for the duration of the block.

    The previous values are restored on exit, and because both live in
    context variables the binding is visible only to the current task and its
    children.
    """
    tokens: list[tuple[ContextVar, Any]] = []
    if config is not None:
        tokens.append((_current_config, _current_config.set(config)))
    if runtime is not None:
        tokens.append((_current_runtime, _current_runtime.set(runtime)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def get_config() -> dict[str, Any]:
    """Return the active run config, or raise outside a run.

    Raising (rather than returning ``{}``) is what lets callers distinguish
    "no thread id because we are not in a run" from a real empty config.
    """
    config = _current_config.get()
    if config is None:
        raise RuntimeError(_OUTSIDE_RUN)
    return config


def get_runtime(context_schema: type | None = None) -> Runtime:
    """Return the active :class:`Runtime`, or raise outside a run.

    *context_schema* is accepted for call-site type inference only; the
    returned runtime is the one bound by :func:`runtime_context`.
    """
    del context_schema
    runtime = _current_runtime.get()
    if runtime is None:
        raise RuntimeError(_OUTSIDE_RUN)
    return runtime


def get_stream_writer() -> StreamWriter:
    """Return the writer that publishes custom frames for the active run."""
    return get_runtime().stream_writer

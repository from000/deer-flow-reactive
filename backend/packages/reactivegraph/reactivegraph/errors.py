"""Control-flow signals and error codes for graph execution.

Two families live here and they must stay distinct:

* **Bubble-up signals** (:class:`GraphBubbleUp` and its subclasses) are engine
  control flow, not failures. A middleware that catches ``Exception`` to wrap a
  tool failure must re-raise these untouched, or it will swallow an interrupt
  or a parent-graph command.
* **Ordinary errors** (:class:`InvalidUpdateError`,
  :class:`GraphRecursionError`) are failures the caller may handle or report.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from reactivegraph.types import Command

__all__ = (
    "EmptyChannelError",
    "ErrorCode",
    "GraphBubbleUp",
    "GraphInterrupt",
    "GraphRecursionError",
    "InvalidUpdateError",
    "ParentCommand",
    "create_error_message",
)


class ErrorCode(Enum):
    """Stable identifiers used in troubleshooting links."""

    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"


def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    """Append the standard troubleshooting link for *error_code*."""
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/"
        f"errors/{error_code.value}"
    )


def _host_error_class(name: str) -> type[Exception] | None:
    """Return LangGraph's ``langgraph.errors.<name>`` when it is installed."""
    try:
        import langgraph.errors as host_errors
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    host = getattr(host_errors, name, None)
    return host if isinstance(host, type) else None


HostGraphBubbleUp = _host_error_class("GraphBubbleUp")
HostGraphInterrupt = _host_error_class("GraphInterrupt")
HostParentCommand = _host_error_class("ParentCommand")
HostGraphRecursionError = _host_error_class("GraphRecursionError")


def _fallback_bubble_up() -> type[Exception]:
    """Engine-native control-flow root, used when LangGraph is absent."""

    class _GraphBubbleUp(Exception):
        """Base class for engine control flow that must not be swallowed."""

    return _GraphBubbleUp


def _fallback_interrupt() -> type[Exception]:
    """Engine-native ``GraphInterrupt``, used when LangGraph is absent."""

    class GraphInterrupt(GraphBubbleUp):  # type: ignore[misc, valid-type]
        """Raised internally when a subgraph interrupts; the root graph converts
        it into the interrupt payload instead of surfacing this exception."""

        def __init__(self, interrupts: Sequence[Any] = ()) -> None:
            super().__init__(interrupts)

    return GraphInterrupt


def _fallback_parent_command() -> type[Exception]:
    """Engine-native ``ParentCommand``, used when LangGraph is absent."""

    class _ParentCommand(GraphBubbleUp):  # type: ignore[misc, valid-type]
        """Carries a ``Command(graph=Command.PARENT)`` up to the parent graph."""

        args: tuple[Command]

        def __init__(self, command: Command) -> None:
            super().__init__(command)

    return _ParentCommand


def _fallback_recursion_error() -> type[Exception]:
    """Engine-native ``GraphRecursionError``, used when LangGraph is absent."""

    class _GraphRecursionError(RecursionError):
        """The run exhausted its step budget without reaching a stop condition."""

    return _GraphRecursionError


# ``except SomeClass`` matches the raised object's class *by identity*: a
# virtual subclass or a widened metaclass ``__instancecheck__`` is not enough.
# The consumer here is the *host's* code — DeerFlow providers and third-party
# middleware raise ``langgraph.errors.GraphBubbleUp`` from inside engine hooks,
# and engine middleware catches ``reactivegraph.errors.GraphBubbleUp`` around
# them. The two spellings therefore have to name the same object, or the
# control-flow signal is swallowed by the surrounding ``except Exception`` and
# the run silently continues (measured: ~20 middleware failures across the
# fork). Aliasing is exact: these classes carry no engine-specific behaviour,
# and their constructors already match upstream's.
#
# The ordinary errors below instead *subclass* the host classes, because there
# the host is the one doing the ``except`` (``Pregel`` catching its own
# ``EmptyChannelError``); a subclass satisfies that direction.
GraphBubbleUp = HostGraphBubbleUp or _fallback_bubble_up()
GraphInterrupt = HostGraphInterrupt or _fallback_interrupt()
ParentCommand = HostParentCommand or _fallback_parent_command()
GraphRecursionError = HostGraphRecursionError or _fallback_recursion_error()


def _host_invalid_update_error() -> type[Exception] | None:
    """Return LangGraph's ``InvalidUpdateError`` when it is installed.

    Same identity contract as :class:`EmptyChannelError`: hosts guard reducer
    failures with ``pytest.raises(langgraph.errors.InvalidUpdateError)`` or
    ``except InvalidUpdateError``, and an unrelated engine class escapes those
    guards. Subclassing the host class keeps both spellings working; the
    engine also catches its own class, and a subclass is still caught.
    """
    try:
        from langgraph.errors import InvalidUpdateError as HostInvalidUpdateError
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HostInvalidUpdateError


HostInvalidUpdateError = _host_invalid_update_error()


class InvalidUpdateError(
    *(  # type: ignore[misc]
        (HostInvalidUpdateError,)
        if HostInvalidUpdateError is not None
        else (Exception,)
    )
):  # type: ignore[misc, unused-ignore]
    """A channel received an update it cannot apply."""


def _host_empty_channel_error() -> type[Exception] | None:
    """Return LangGraph's ``EmptyChannelError`` when it is installed.

    Upstream's Pregel catches *its own* class by identity::

        try:
            return channels[chan].get()
        except EmptyChannelError:
            ...

    An engine channel raising an unrelated class therefore aborts a whole
    state read instead of being treated as "this field is empty". Subclassing
    the host class keeps both ``except`` clauses working; the engine also
    catches ``EmptyChannelError`` internally, and a subclass is still caught.
    """
    try:
        from langgraph.errors import EmptyChannelError as HostEmptyChannelError
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HostEmptyChannelError


HostEmptyChannelError = _host_empty_channel_error()


class EmptyChannelError(
    *((HostEmptyChannelError,) if HostEmptyChannelError is not None else (Exception,))  # type: ignore[misc]
):
    """Raised when reading a channel that has never been written.

    Upstream defines this as ``Exception`` (not ``ValueError``); hosts catch
    the bare class, so the base must match. When LangGraph is installed this
    also subclasses the host's class, because upstream's Pregel catches that
    exact class around ``channel.get()``.
    """

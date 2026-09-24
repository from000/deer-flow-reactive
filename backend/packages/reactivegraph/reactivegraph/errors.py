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


class GraphBubbleUp(Exception):
    """Base class for engine control flow that must not be swallowed."""


class GraphInterrupt(GraphBubbleUp):
    """Raised internally when a subgraph interrupts; the root graph converts it
    into the interrupt payload instead of surfacing this exception."""

    def __init__(self, interrupts: Sequence[Any] = ()) -> None:
        super().__init__(interrupts)


class ParentCommand(GraphBubbleUp):
    """Carries a ``Command(graph=Command.PARENT)`` up to the parent graph."""

    args: tuple[Command]

    def __init__(self, command: Command) -> None:
        super().__init__(command)


class GraphRecursionError(RecursionError):
    """The run exhausted its step budget without reaching a stop condition."""


class InvalidUpdateError(Exception):
    """A channel received an update it cannot apply."""

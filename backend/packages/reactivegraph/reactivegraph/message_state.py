"""Message reducers and state markers used by agent middleware.

These are engine primitives rather than host conveniences: ``AgentState``
declares its ``messages`` field with :func:`add_messages`, and middleware that
sets ``jump_to`` needs a marker that the graph compiler can recognise as
ephemeral.  Keeping them here means the middleware contract is executable
without importing LangGraph.

Portions of this module are adapted from upstream `langgraph`
(https://github.com/langchain-ai/langgraph) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import partial
from typing import Any, Generic, Literal, TypeVar, cast

from reactivegraph.constants import REMOVE_ALL_MESSAGES
from reactivegraph.messages import (
    BaseMessage,
    BaseMessageChunk,
    MessageLikeRepresentation,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)

__all__ = (
    "Messages",
    "add_messages",
    "EphemeralValue",
)

Messages = list[MessageLikeRepresentation] | MessageLikeRepresentation
Value = TypeVar("Value")


def _host_ephemeral_value() -> type | None:
    """Return LangGraph's channel class when it is importable.

    A state annotation may name a channel *class*. Host compilers resolve it by
    checking ``issubclass(item, BaseChannel)``; a plain marker class is instead
    silently compiled into a persistent ``LastValue``, so middleware directives
    such as ``jump_to`` never clear. Inheriting the real host channel is the
    only way for the annotation to mean the same thing on both sides of the
    bridge. LangGraph stays optional: without it the native marker below keeps
    the engine self-contained.
    """
    try:
        from langgraph.channels.ephemeral_value import EphemeralValue as Host
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return Host


_HostEphemeralValue = _host_ephemeral_value()


if _HostEphemeralValue is None:

    class _EphemeralValueBase(Generic[Value]):
        __slots__ = ("guard", "typ")

        def __init__(self, typ: Any, guard: bool = True) -> None:
            self.typ = typ
            self.guard = guard

        def __eq__(self, value: object) -> bool:
            return isinstance(value, EphemeralValue) and value.guard == self.guard

else:

    class _EphemeralValueBase(  # type: ignore[misc, no-redef, valid-type, unused-ignore]
        _HostEphemeralValue, Generic[Value]  # type: ignore[misc, valid-type]
    ):
        __slots__ = ()

        def __init__(self, typ: Any, guard: bool = True) -> None:
            super().__init__(typ, guard)


class EphemeralValue(_EphemeralValueBase):
    """Channel whose value lasts for one super-step.

    The engine records this marker in state annotations; runtimes that
    understand it clear the value after the step. When LangGraph is installed
    the class is also the host's own ``EphemeralValue`` channel, so a host
    compiler gives the annotation identical semantics.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"EphemeralValue({self.typ!r}, guard={self.guard!r})"


def _add_messages_wrapper(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    def _add_messages(
        left: Messages | None = None,
        right: Messages | None = None,
        **kwargs: Any,
    ) -> Messages | Callable[[Messages, Messages], Messages]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        if left is not None or right is not None:
            received = "left" if left else "right"
            msg = (
                "Must specify non-null arguments for both 'left' and 'right'. "
                f"Only received: '{received}'."
            )
            raise ValueError(msg)
        return partial(func, **kwargs)

    _add_messages.__doc__ = func.__doc__
    return cast(Callable[..., Any], _add_messages)


def _assign_missing_ids(messages: list[BaseMessage]) -> int | None:
    """Give every message a stable id; return the ``RemoveMessage`` all-index."""
    remove_all_idx: int | None = None
    for idx, message in enumerate(messages):
        if message.id is None:
            message.id = str(uuid.uuid4())
        if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx
    return remove_all_idx


def _merge_messages(
    left: list[BaseMessage], right: list[BaseMessage]
) -> list[BaseMessage]:
    """Merge *right* into *left* by id, honouring ``RemoveMessage`` tombstones."""
    merged = list(left)
    merged_by_id = {message.id: i for i, message in enumerate(merged)}
    ids_to_remove: set[str] = set()
    for message in right:
        existing_idx = merged_by_id.get(message.id)
        if existing_idx is not None:
            if isinstance(message, RemoveMessage):
                assert message.id is not None
                ids_to_remove.add(message.id)
            else:
                ids_to_remove.discard(message.id)
                merged[existing_idx] = message
        elif isinstance(message, RemoveMessage):
            raise ValueError(
                "Attempting to delete a message with an ID that doesn't exist "
                f"('{message.id}')"
            )
        else:
            merged_by_id[message.id] = len(merged)
            merged.append(message)
    return [message for message in merged if message.id not in ids_to_remove]


def _format_langchain_openai(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Round-trip messages through the langchain-core OpenAI converter.

    Upstream's ``add_messages(format="langchain-openai")`` is implemented as
    ``convert_to_messages(convert_to_openai_messages(messages))``. That
    converter lives in langchain-core (>=0.3.11); reproducing its full
    multi-provider block translation natively is out of scope, so it is used
    when importable and reported clearly when it is not — never silently
    substituting a different message shape.
    """
    try:
        from langchain_core.messages import convert_to_openai_messages
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise NotImplementedError(
            "add_messages(format='langchain-openai') requires "
            "langchain-core>=0.3.11 for convert_to_openai_messages; install "
            "the 'compat' extra or drop the format argument."
        ) from exc
    return [
        message_chunk_to_message(cast(BaseMessageChunk, message))
        for message in convert_to_messages(convert_to_openai_messages(messages))
    ]


@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """Merge two message lists by id.

    Appends messages from *right* onto *left*, replacing any message whose id
    already exists and removing messages tombstoned by ``RemoveMessage``.
    ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` drops everything before it in
    *right*. With ``format="langchain-openai"`` the result is round-tripped
    through langchain-core's OpenAI converter.
    """
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    left_messages = [
        message_chunk_to_message(cast(BaseMessageChunk, message))
        for message in convert_to_messages(
            list(cast("list[MessageLikeRepresentation]", left))
        )
    ]
    right_messages = [
        message_chunk_to_message(cast(BaseMessageChunk, message))
        for message in convert_to_messages(
            list(cast("list[MessageLikeRepresentation]", right))
        )
    ]
    _assign_missing_ids(left_messages)
    remove_all_idx = _assign_missing_ids(right_messages)

    if remove_all_idx is not None:
        return cast(Messages, right_messages[remove_all_idx + 1 :])

    merged = _merge_messages(left_messages, right_messages)

    if format == "langchain-openai":
        merged = _format_langchain_openai(merged)
    elif format:
        msg = f"Unrecognized format={format!r}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    return cast(Messages, merged)

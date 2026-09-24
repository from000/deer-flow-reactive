"""Message reducers and state markers used by agent middleware.

These are engine primitives rather than host conveniences: ``AgentState``
declares its ``messages`` field with :func:`add_messages`, and middleware that
sets ``jump_to`` needs a marker that the graph compiler can recognise as
ephemeral.  Keeping them here means the middleware contract is executable
without importing LangGraph.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import partial
from typing import Any, Generic, Literal, TypeVar, cast

from reactivegraph.constants import REMOVE_ALL_MESSAGES
from reactivegraph.messages import (
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


class EphemeralValue(Generic[Value]):
    """Marker channel whose value lasts for one super-step.

    The native engine records the marker in state annotations; graph runtimes
    that understand it clear the value after the step.  The class deliberately
    does not inherit from a host channel implementation, so middleware state
    schemas remain engine-neutral.
    """

    __slots__ = ("guard", "typ")

    def __init__(self, typ: Any, guard: bool = True) -> None:
        self.typ = typ
        self.guard = guard

    def __repr__(self) -> str:
        return f"EphemeralValue({self.typ!r}, guard={self.guard!r})"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, EphemeralValue) and value.guard == self.guard


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


@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """Merge messages by id, replacing same-id entries and deleting on RemoveMessage."""
    remove_all_idx = None
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]

    left_messages = [
        message_chunk_to_message(cast(BaseMessageChunk, message))
        for message in convert_to_messages(left)
    ]
    right_messages = [
        message_chunk_to_message(cast(BaseMessageChunk, message))
        for message in convert_to_messages(right)
    ]

    for message in left_messages:
        if message.id is None:
            message.id = str(uuid.uuid4())
    for idx, message in enumerate(right_messages):
        if message.id is None:
            message.id = str(uuid.uuid4())
        if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    if remove_all_idx is not None:
        return right_messages[remove_all_idx + 1 :]

    merged = list(left_messages)
    merged_by_id = {message.id: i for i, message in enumerate(merged)}
    ids_to_remove: set[str] = set()
    for message in right_messages:
        existing_idx = merged_by_id.get(message.id)
        if existing_idx is not None:
            if isinstance(message, RemoveMessage):
                ids_to_remove.add(message.id)
            else:
                ids_to_remove.discard(message.id)
                merged[existing_idx] = message
        elif isinstance(message, RemoveMessage):
            msg = (
                "Attempting to delete a message with an ID that doesn't exist "
                f"('{message.id}')"
            )
            raise ValueError(msg)
        else:
            merged_by_id[message.id] = len(merged)
            merged.append(message)

    merged = [message for message in merged if message.id not in ids_to_remove]

    if format == "langchain-openai":
        # The native message layer has no OpenAI-specific formatter yet.  Fail
        # closed instead of returning a silently different message shape.
        raise NotImplementedError(
            "add_messages(format='langchain-openai') requires the native "
            "convert_to_openai_messages port, which is not implemented yet."
        )
    if format:
        msg = f"Unrecognized format={format!r}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    return merged

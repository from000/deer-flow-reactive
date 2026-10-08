"""Message rendering, trimming and approximate token accounting.

This module owns the small, framework-independent portion of the message
utility contract required by :mod:`reactivegraph.summarization`.  It is a
behavioural port of the corresponding ``langchain_core.messages.utils``
functions (MIT), not a re-export: ReactiveGraph must remain importable in an
environment where neither LangChain nor LangGraph is installed.

The functions deliberately operate structurally as well as on native
ReactiveGraph messages.  That keeps the boundary usable when a host passes a
foreign message object whose identity checks cannot be bridged by subclassing.

Portions of this module are adapted from upstream `langchain-core`
(https://github.com/langchain-ai/langchain) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast
from xml.sax.saxutils import escape, quoteattr

from reactivegraph.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)

__all__ = (
    "count_tokens_approximately",
    "get_buffer_string",
    "trim_messages",
)

MessageType = str | type[BaseMessage] | Sequence[str | type[BaseMessage]]
TokenCounter = (
    Callable[[list[BaseMessage]], int] | Callable[[BaseMessage], int] | Literal["approximate"] | Any
)

_XML_CONTENT_BLOCK_MAX_LEN = 500


def _is_message(value: Any) -> bool:
    """Return whether *value* exposes the message surface used here."""
    return isinstance(value, BaseMessage) or (
        isinstance(getattr(value, "type", None), str)
        and hasattr(value, "content")
        and callable(getattr(value, "model_copy", None))
    )


def _coerce_messages(messages: Iterable[MessageLikeRepresentation]) -> list[Any]:
    """Coerce message-like inputs while preserving foreign message objects.

    ``convert_to_messages`` intentionally rebuilds foreign classes as native
    messages.  Legacy foreign classes (for example ``FunctionMessage``) have no
    native equivalent, so they are retained structurally instead.
    """
    if hasattr(messages, "to_messages"):
        messages = cast("Any", messages).to_messages()
    return [
        message if _is_message(message) else convert_to_messages([message])[0]
        for message in messages
    ]


def _message_type(message: Any) -> str:
    value = getattr(message, "type", "")
    return value if isinstance(value, str) else ""


def _is_ai_message(message: Any) -> bool:
    return isinstance(message, AIMessage) or _message_type(message) in {
        "ai",
        "AIMessageChunk",
    }


def _is_human_message(message: Any) -> bool:
    return isinstance(message, HumanMessage) or _message_type(message) in {
        "human",
        "HumanMessageChunk",
    }


def _is_system_message(message: Any) -> bool:
    return isinstance(message, SystemMessage) or _message_type(message) in {
        "system",
        "SystemMessageChunk",
    }


def _is_tool_message(message: Any) -> bool:
    return isinstance(message, ToolMessage) or _message_type(message) in {
        "tool",
        "ToolMessageChunk",
    }


def _is_chat_message(message: Any) -> bool:
    return isinstance(message, ChatMessage) or _message_type(message) in {
        "chat",
        "ChatMessageChunk",
    }


def _is_function_message(message: Any) -> bool:
    return _message_type(message) in {"function", "FunctionMessageChunk"} or type(
        message
    ).__name__.startswith("FunctionMessage")


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if text is not None:
        return str(text)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _has_base64_data(block: dict[str, Any]) -> bool:
    if block.get("base64"):
        return True
    url = block.get("url", "")
    if isinstance(url, str) and url.startswith("data:"):
        return True
    image_url = block.get("image_url", {})
    return bool(
        isinstance(image_url, dict)
        and isinstance(image_url.get("url"), str)
        and image_url["url"].startswith("data:")
    )


def _truncate(text: str, max_len: int = _XML_CONTENT_BLOCK_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


_XML_MEDIA_TAGS = {"image", "audio", "video"}


def _xml_text_or_none(value: Any, *, truncate: bool = False) -> str | None:
    """Escaped text payload, or ``None`` when the value is empty."""
    if not value:
        return None
    text = str(value)
    return escape(_truncate(text) if truncate else text)


def _xml_media_block(block_type: str, block: dict[str, Any]) -> str | None:
    """``<image|audio|video url=...|file_id=... />``."""
    url = block.get("url")
    if url:
        return f"<{block_type} url={quoteattr(str(url))} />"
    file_id = block.get("file_id")
    if file_id:
        return f"<{block_type} file_id={quoteattr(str(file_id))} />"
    return None


def _xml_image_url_block(block: dict[str, Any]) -> str | None:
    """``<image url=... />`` for the OpenAI ``image_url`` block shape."""
    image_url = block.get("image_url", {})
    if not isinstance(image_url, dict):
        return None
    url = image_url.get("url", "")
    if url and not str(url).startswith("data:"):
        return f"<image url={quoteattr(str(url))} />"
    return None


def _xml_server_tool_call(block: dict[str, Any]) -> str:
    """``<server_tool_call>`` with escaped id/name/args attributes."""
    tc_id = quoteattr(str(block.get("id") or ""))
    tc_name = quoteattr(str(block.get("name") or ""))
    tc_args_json = json.dumps(block.get("args", {}), ensure_ascii=False)
    tc_args = escape(_truncate(tc_args_json))
    return f"<server_tool_call id={tc_id} name={tc_name}>{tc_args}</server_tool_call>"


def _xml_server_tool_result(block: dict[str, Any]) -> str:
    """``<server_tool_result>`` with escaped ids/status and truncated output."""
    tool_call_id = quoteattr(str(block.get("tool_call_id") or ""))
    status = quoteattr(str(block.get("status") or ""))
    output = block.get("output")
    output_str = escape(_truncate(json.dumps(output, ensure_ascii=False))) if output else ""
    return (
        f"<server_tool_result tool_call_id={tool_call_id} status={status}>"
        f"{output_str}</server_tool_result>"
    )


def _format_content_block_xml(block: dict[str, Any]) -> str | None:
    """Format one standard content block as XML, or ``None`` to skip it."""
    if _has_base64_data(block):
        return None
    block_type = block.get("type", "")
    if block_type == "text":
        return _xml_text_or_none(block.get("text", ""))
    if block_type == "reasoning":
        reasoning = _xml_text_or_none(block.get("reasoning", ""))
        return f"<reasoning>{reasoning}</reasoning>" if reasoning else None
    if block_type in _XML_MEDIA_TAGS:
        return _xml_media_block(block_type, block)
    if block_type == "image_url":
        return _xml_image_url_block(block)
    if block_type == "text-plain":
        return _xml_text_or_none(block.get("text", ""), truncate=True)
    if block_type == "server_tool_call":
        return _xml_server_tool_call(block)
    if block_type == "server_tool_result":
        return _xml_server_tool_result(block)
    return None


def _role_for_message(
    message: Any,
    human_prefix: str,
    ai_prefix: str,
    system_prefix: str,
    function_prefix: str,
    tool_prefix: str,
) -> str:
    if _is_human_message(message):
        return human_prefix
    if _is_ai_message(message):
        return ai_prefix
    if _is_system_message(message):
        return system_prefix
    if _is_function_message(message):
        return function_prefix
    if _is_tool_message(message):
        return tool_prefix
    if _is_chat_message(message):
        role = getattr(message, "role", None)
        if isinstance(role, str):
            return role
    msg = f"Got unsupported message type: {message}"
    raise ValueError(msg)


def _prefix_line(
    message: Any,
    role: str,
) -> str:
    """Render one message in the ``prefix`` transcript format."""
    line = f"{role}: {_message_text(message)}"
    if not _is_ai_message(message):
        return line
    tool_calls = getattr(message, "tool_calls", None)
    additional_kwargs = getattr(message, "additional_kwargs", {})
    if tool_calls:
        return line + str(tool_calls)
    if isinstance(additional_kwargs, dict) and "function_call" in additional_kwargs:
        return line + str(additional_kwargs["function_call"])
    return line


def _xml_content_parts(message: Any) -> list[str]:
    """Escaped/formatted content fragments of one message."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return [escape(content)] if content else []
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(escape(block))
        elif isinstance(block, dict):
            formatted = _format_content_block_xml(block)
            if formatted:
                parts.append(formatted)
    return parts


def _xml_tool_call_lines(tool_calls: Any) -> list[str]:
    lines: list[str] = []
    for tool_call in tool_calls:
        tc_id = quoteattr(str(tool_call.get("id") or ""))
        tc_name = quoteattr(str(tool_call.get("name") or ""))
        tc_args = escape(json.dumps(tool_call.get("args", {}), ensure_ascii=False))
        lines.append(f"  <tool_call id={tc_id} name={tc_name}>{tc_args}</tool_call>")
    return lines


def _xml_message(message: Any, role: str) -> str:
    """Render one message in the ``xml`` transcript format."""
    msg_type = str(message.role) if _is_chat_message(message) else role.lower()
    content_parts = _xml_content_parts(message)

    tool_calls = getattr(message, "tool_calls", None) if _is_ai_message(message) else None
    additional_kwargs = getattr(message, "additional_kwargs", {})
    has_function_call = bool(
        _is_ai_message(message)
        and not tool_calls
        and isinstance(additional_kwargs, dict)
        and "function_call" in additional_kwargs
    )

    if not tool_calls and not has_function_call:
        return f"<message type={quoteattr(msg_type)}>{' '.join(content_parts)}</message>"

    parts = [f"<message type={quoteattr(msg_type)}>"]
    if content_parts:
        parts.append(f"  <content>{' '.join(content_parts)}</content>")
    if tool_calls:
        parts.extend(_xml_tool_call_lines(tool_calls))
    else:
        function_call = additional_kwargs["function_call"]
        fc_name = quoteattr(str(function_call.get("name") or ""))
        fc_args = escape(str(function_call.get("arguments") or "{}"))
        parts.append(f"  <function_call name={fc_name}>{fc_args}</function_call>")
    parts.append("</message>")
    return "\n".join(parts)


def get_buffer_string(
    messages: Sequence[Any],
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
    *,
    system_prefix: str = "System",
    function_prefix: str = "Function",
    tool_prefix: str = "Tool",
    message_separator: str = "\n",
    format: Literal["prefix", "xml"] = "prefix",  # noqa: A002
) -> str:
    """Render messages using the upstream prefix or XML transcript contract.

    ``prefix`` produces ``Role: text`` lines, appending serialized tool calls
    for AI messages. ``xml`` produces the escaped ``<message>`` transcript
    upstream uses for summarization prompts.
    """
    if format not in {"prefix", "xml"}:
        msg = f"Unrecognized format={format!r}. Supported formats are 'prefix' and 'xml'."
        raise ValueError(msg)

    rendered: list[str] = []
    for message in messages:
        role = _role_for_message(
            message,
            human_prefix,
            ai_prefix,
            system_prefix,
            function_prefix,
            tool_prefix,
        )
        rendered.append(
            _prefix_line(message, role) if format == "prefix" else _xml_message(message, role)
        )
    return message_separator.join(rendered)


def _get_message_openai_role(message: Any) -> str:
    if _is_ai_message(message):
        return "assistant"
    if _is_human_message(message):
        return "user"
    if _is_tool_message(message):
        return "tool"
    if _is_system_message(message):
        additional_kwargs = getattr(message, "additional_kwargs", {})
        role = (
            additional_kwargs.get("__openai_role__", "system")
            if isinstance(additional_kwargs, dict)
            else "system"
        )
        if not isinstance(role, str):
            msg = f"Expected '__openai_role__' to be a str, got {type(role).__name__}"
            raise TypeError(msg)
        return role
    if _is_function_message(message):
        return "function"
    if _is_chat_message(message):
        return str(message.role)
    msg = f"Unknown BaseMessage type {message.__class__}."
    raise ValueError(msg)


def _tool_schema_chars(tools: list[Any]) -> int:
    """Serialized character count of CLI tool schemas (dicts pass through)."""
    from reactivegraph.tools import convert_to_openai_tool

    return sum(
        len(json.dumps(tool if isinstance(tool, dict) else convert_to_openai_tool(tool)))
        for tool in tools
    )


def _content_block_chars(content: list[Any], tokens_per_image: int) -> tuple[int, int]:
    """Split list-content into ``(characters, image_tokens)``.

    Image blocks are billed at a flat rate rather than by payload size, so the
    two contributions are returned separately.
    """
    chars = 0
    image_tokens = 0
    for block in content:
        if isinstance(block, str):
            chars += len(block)
        elif isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type in {"image", "image_url"}:
                image_tokens += tokens_per_image
            elif block_type == "text":
                chars += len(block.get("text", ""))
            else:
                chars += len(repr(block))
        else:
            chars += len(repr(block))
    return chars, image_tokens


def _message_chars(
    message: Any,
    *,
    content: Any,
    count_name: bool,
) -> int:
    """Character budget of one message, mirroring langchain-core's accounting."""
    if isinstance(content, str):
        chars = len(content)
    elif isinstance(content, list):
        chars, _ = _content_block_chars(content, 0)
    else:
        chars = len(repr(content))

    if _is_ai_message(message) and not isinstance(content, list):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            chars += len(repr(tool_calls))

    if _is_tool_message(message):
        chars += len(str(getattr(message, "tool_call_id", "")))

    chars += len(_get_message_openai_role(message))
    name = getattr(message, "name", None)
    if name and count_name:
        chars += len(str(name))
    return chars


def _model_provider(message: Any) -> str | None:
    """Provider recorded in ``response_metadata``, when present and valid."""
    response_metadata = getattr(message, "response_metadata", {})
    if not isinstance(response_metadata, dict):
        return None
    value = response_metadata.get("model_provider")
    return value if isinstance(value, str) else None


def _usage_total_tokens(message: Any) -> int | None:
    """``usage_metadata.total_tokens`` when the message reports one."""
    usage_metadata = getattr(message, "usage_metadata", None)
    if not isinstance(usage_metadata, dict):
        return None
    total = usage_metadata.get("total_tokens")
    return total if isinstance(total, int) else None


def count_tokens_approximately(
    messages: Iterable[MessageLikeRepresentation],
    *,
    chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
    count_name: bool = True,
    tokens_per_image: int = 85,
    use_usage_metadata_scaling: bool = False,
    tools: list[Any] | None = None,
) -> int:
    """Approximate message and tool-schema token usage like langchain-core.

    ``use_usage_metadata_scaling`` mirrors upstream: when every AI message
    reports the same ``model_provider`` and the last one carries a
    ``usage_metadata.total_tokens``, the estimate is scaled by the observed
    ratio, clamped to ``[1.0, 1.25]``.
    """
    converted_messages = _coerce_messages(messages)
    token_count = 0.0

    if tools:
        token_count += math.ceil(_tool_schema_chars(tools) / chars_per_token)

    provider: str | None = None
    provider_conflict = False
    last_ai_total: int | None = None
    approx_at_last_ai: float | None = None

    for message in converted_messages:
        content = getattr(message, "content", "")
        image_tokens = 0
        if isinstance(content, list):
            _, image_tokens = _content_block_chars(content, tokens_per_image)
        token_count += image_tokens
        chars = _message_chars(message, content=content, count_name=count_name)
        token_count += math.ceil(chars / chars_per_token)
        token_count += extra_tokens_per_message

        if not (use_usage_metadata_scaling and _is_ai_message(message)):
            continue
        current_provider = _model_provider(message)
        if provider is None:
            provider = current_provider
        elif current_provider != provider:
            provider_conflict = True
        total = _usage_total_tokens(message)
        if total is not None:
            last_ai_total = total
            approx_at_last_ai = token_count

    if (
        use_usage_metadata_scaling
        and len(converted_messages) > 1
        and not provider_conflict
        and provider is not None
        and last_ai_total is not None
        and approx_at_last_ai
        and approx_at_last_ai > 0
    ):
        scale_factor = last_ai_total / approx_at_last_ai
        token_count *= min(1.25, max(1.0, scale_factor))

    return math.ceil(token_count)


def _is_message_type(message: Any, type_: MessageType) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [item for item in types if isinstance(item, str)]
    types_types = tuple(item for item in types if isinstance(item, type))
    return _message_type(message) in types_str or isinstance(message, types_types)


def _default_text_splitter(text: str) -> list[str]:
    splits = text.split("\n")
    return [split + "\n" for split in splits[:-1]] + splits[-1:]


def _largest_fitting_prefix(
    count: int,
    fits: Callable[[int], bool],
) -> int:
    """Return the largest ``i`` in ``[0, count]`` for which ``fits(i)`` holds.

    ``fits`` must be monotone: once it returns ``False`` it must stay ``False``
    for larger inputs.  Both call sites (whole-message prefixes and
    text-split prefixes) satisfy that because token counts grow with input
    size.
    """
    left, right = 0, count
    for _ in range(count.bit_length()):
        if left >= right:
            break
        mid = (left + right + 1) // 2
        if fits(mid):
            left = mid
        else:
            right = mid - 1
    return left


def _trailing_messages_matching(
    messages: list[Any], end_on: MessageType
) -> list[Any]:
    """Drop trailing messages until the last one matches ``end_on``."""
    for _ in range(len(messages)):
        if not _is_message_type(messages[-1], end_on):
            messages.pop()
        else:
            break
    return messages


def _partial_block_message(
    message: Any,
    *,
    max_tokens: int,
    token_counter: Callable[[list[Any]], int],
    prefix: list[Any],
    strategy: Literal["first", "last"],
) -> tuple[Any | None, Any]:
    """Try to keep a prefix/suffix of a multimodal message's content blocks.

    Returns ``(trimmed, working)``.  ``trimmed`` is the message with the
    fitting block subset when one was found (``None`` otherwise); ``working``
    is the deep-copied message the algorithm used, whose content may still be
    in reverse order for ``strategy="last"`` — the caller reuses it for the
    text-splitting fallback exactly like the original implementation did.
    """
    working = message.model_copy(deep=True)
    num_block = len(working.content)
    if strategy == "last":
        working.content = list(reversed(working.content))
    for _ in range(1, num_block):
        working.content = working.content[:-1]
        if token_counter([*prefix, working]) <= max_tokens:
            if strategy == "last":
                working.content = list(reversed(working.content))
            return working, working
    return None, working


def _first_text_block(message: Any) -> str | None:
    """Return the first text payload of a message whose content is a list."""
    content = getattr(message, "content", None)
    if not isinstance(content, list) or not content:
        return None
    for block in content:
        if isinstance(block, str):
            return block
        if isinstance(block, dict) and block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str):
                return value
    return None


def _split_text_to_fit(
    message: Any,
    text: str,
    *,
    max_tokens: int,
    token_counter: Callable[[list[Any]], int],
    text_splitter: Callable[[str], list[str]],
    prefix: list[Any],
    strategy: Literal["first", "last"],
) -> Any | None:
    """Trim ``text`` to the largest split sequence that fits the budget."""
    split_texts = text_splitter(text)
    base = token_counter(prefix)
    if strategy == "last":
        split_texts = list(reversed(split_texts))
    left = _largest_fitting_prefix(
        len(split_texts),
        lambda mid: (
            base + token_counter([_with_content(message, "".join(split_texts[:mid]))])
        )
        <= max_tokens,
    )
    if left <= 0:
        return None
    content_splits = split_texts[:left]
    if strategy == "last":
        content_splits = list(reversed(content_splits))
    return _with_content(message, "".join(content_splits))


def _with_content(message: Any, content: Any) -> Any:
    """Set ``content`` on *message* in place and return it (trim helpers only)."""
    message.content = content
    return message


def _first_max_tokens(
    messages: Sequence[Any],
    *,
    max_tokens: int,
    token_counter: Callable[[list[Any]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Literal["first", "last"] | None = None,
    end_on: MessageType | None = None,
) -> list[Any]:
    """Keep the longest leading run of *messages* that fits *max_tokens*."""
    remaining: list[Any] = list(messages)
    if not remaining:
        return remaining

    if token_counter(remaining) <= max_tokens:
        if end_on is None:
            return remaining
        return _trailing_messages_matching(list(remaining), end_on)

    idx = _largest_fitting_prefix(
        len(remaining),
        lambda mid: token_counter(remaining[:mid]) <= max_tokens,
    )

    if partial_strategy and idx < len(remaining):
        candidate = remaining[idx]
        content = getattr(candidate, "content", None)
        working = candidate.model_copy(deep=True) if not isinstance(content, list) else None
        trimmed: Any | None = None
        if isinstance(content, list):
            trimmed, working = _partial_block_message(
                candidate,
                max_tokens=max_tokens,
                token_counter=token_counter,
                prefix=remaining[:idx],
                strategy=partial_strategy,
            )
        if trimmed is None:
            assert working is not None
            text = content if isinstance(content, str) else _first_text_block(working)
            if text:
                trimmed = _split_text_to_fit(
                    working,
                    text,
                    max_tokens=max_tokens,
                    token_counter=token_counter,
                    text_splitter=text_splitter,
                    prefix=remaining[:idx],
                    strategy=partial_strategy,
                )
        if trimmed is not None:
            remaining = [*remaining[:idx], trimmed]
            idx += 1

    if end_on:
        for _ in range(idx):
            if idx > 0 and not _is_message_type(remaining[idx - 1], end_on):
                idx -= 1
            else:
                break
    return remaining[:idx]


def _last_max_tokens(
    messages: Sequence[Any],
    *,
    max_tokens: int,
    token_counter: Callable[[list[Any]], int],
    text_splitter: Callable[[str], list[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: MessageType | None = None,
    end_on: MessageType | None = None,
) -> list[Any]:
    messages = list(messages)
    if not messages:
        return []

    if end_on:
        for _ in range(len(messages)):
            if not _is_message_type(messages[-1], end_on):
                messages.pop()
            else:
                break

    system_message = None
    if include_system and messages and _is_system_message(messages[0]):
        system_message = messages[0]
        messages = messages[1:]

    reversed_result = _first_max_tokens(
        messages[::-1],
        max_tokens=max_tokens - (token_counter([system_message]) if system_message else 0),
        token_counter=token_counter,
        text_splitter=text_splitter,
        partial_strategy="last" if allow_partial else None,
        end_on=start_on,
    )
    result = reversed_result[::-1]
    return [system_message, *result] if system_message else result


def trim_messages(
    messages: Iterable[MessageLikeRepresentation],
    *,
    max_tokens: int,
    token_counter: TokenCounter,
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: MessageType | None = None,
    start_on: MessageType | None = None,
    include_system: bool = False,
    text_splitter: Callable[[str], list[str]] | Any | None = None,
) -> list[Any]:
    """Trim messages to a token budget using the upstream algorithm."""
    if start_on and strategy == "first":
        msg = "start_on parameter is only valid with strategy='last'"
        raise ValueError(msg)
    if include_system and strategy == "first":
        msg = "include_system parameter is only valid with strategy='last'"
        raise ValueError(msg)

    converted = _coerce_messages(messages)
    if isinstance(token_counter, str):
        if token_counter != "approximate":
            msg = (
                f"Invalid token_counter shortcut '{token_counter}'. "
                "Available shortcuts: 'approximate'."
            )
            raise ValueError(msg)
        list_token_counter: Callable[[list[Any]], int] = count_tokens_approximately
    elif hasattr(token_counter, "get_num_tokens_from_messages"):
        list_token_counter = token_counter.get_num_tokens_from_messages
    elif callable(token_counter):
        first_parameter = next(iter(inspect.signature(token_counter).parameters.values()), None)
        first_annotation = getattr(first_parameter, "annotation", inspect.Parameter.empty)
        if first_annotation is BaseMessage or (
            isinstance(first_annotation, type)
            and first_annotation.__name__ == "BaseMessage"
            and first_annotation.__module__.startswith("langchain_core")
        ):
            single_counter = cast("Callable[[Any], int]", token_counter)

            def list_token_counter(values: list[Any]) -> int:
                """Token counter treating each list item as one token."""
                return sum(single_counter(message) for message in values)
        else:
            list_token_counter = cast("Callable[[list[Any]], int]", token_counter)
    else:
        msg = (
            "'token_counter' expected to be a model that implements "
            "'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(token_counter)}."
        )
        raise ValueError(msg)

    if hasattr(text_splitter, "split_text"):
        text_splitter_fn = cast("Any", text_splitter).split_text
    elif callable(text_splitter):
        text_splitter_fn = cast("Callable[[str], list[str]]", text_splitter)
    else:
        text_splitter_fn = _default_text_splitter

    if strategy == "first":
        return _first_max_tokens(
            converted,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    if strategy == "last":
        return _last_max_tokens(
            converted,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    msg = f"Unrecognized strategy={strategy!r}. Supported strategies are 'last' and 'first'."
    raise ValueError(msg)

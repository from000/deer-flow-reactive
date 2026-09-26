"""Message rendering, trimming and approximate token accounting.

This module owns the small, framework-independent portion of the message
utility contract required by :mod:`reactivegraph.summarization`.  It is a
behavioural port of the corresponding ``langchain_core.messages.utils``
functions (MIT), not a re-export: ReactiveGraph must remain importable in an
environment where neither LangChain nor LangGraph is installed.

The functions deliberately operate structurally as well as on native
ReactiveGraph messages.  That keeps the boundary usable when a host passes a
foreign message object whose identity checks cannot be bridged by subclassing.
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


def _format_content_block_xml(block: dict[str, Any]) -> str | None:
    """Format one standard content block as XML, or ``None`` to skip it."""
    block_type = block.get("type", "")
    if _has_base64_data(block):
        return None

    if block_type == "text":
        text = block.get("text", "")
        return escape(text) if text else None

    if block_type == "reasoning":
        reasoning = block.get("reasoning", "")
        return f"<reasoning>{escape(reasoning)}</reasoning>" if reasoning else None

    if block_type in {"image", "audio", "video"}:
        url = block.get("url")
        file_id = block.get("file_id")
        if url:
            return f"<{block_type} url={quoteattr(str(url))} />"
        if file_id:
            return f"<{block_type} file_id={quoteattr(str(file_id))} />"
        return None

    if block_type == "image_url":
        image_url = block.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
            if url and not str(url).startswith("data:"):
                return f"<image url={quoteattr(str(url))} />"
        return None

    if block_type == "text-plain":
        text = block.get("text", "")
        return escape(_truncate(text)) if text else None

    if block_type == "server_tool_call":
        tc_id = quoteattr(str(block.get("id") or ""))
        tc_name = quoteattr(str(block.get("name") or ""))
        tc_args_json = json.dumps(block.get("args", {}), ensure_ascii=False)
        tc_args = escape(_truncate(tc_args_json))
        return f"<server_tool_call id={tc_id} name={tc_name}>{tc_args}</server_tool_call>"

    if block_type == "server_tool_result":
        tool_call_id = quoteattr(str(block.get("tool_call_id") or ""))
        status = quoteattr(str(block.get("status") or ""))
        output = block.get("output")
        output_str = escape(_truncate(json.dumps(output, ensure_ascii=False))) if output else ""
        return (
            f"<server_tool_result tool_call_id={tool_call_id} status={status}>"
            f"{output_str}</server_tool_result>"
        )

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
    """Render messages using the upstream prefix or XML transcript contract."""
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

        if format == "prefix":
            line = f"{role}: {_message_text(message)}"
            tool_info = ""
            if _is_ai_message(message):
                tool_calls = getattr(message, "tool_calls", None)
                additional_kwargs = getattr(message, "additional_kwargs", {})
                if tool_calls:
                    tool_info = str(tool_calls)
                elif isinstance(additional_kwargs, dict) and "function_call" in additional_kwargs:
                    tool_info = str(additional_kwargs["function_call"])
            rendered.append(line + tool_info)
            continue

        msg_type = role.lower()
        if _is_chat_message(message):
            msg_type = str(message.role)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            content_parts = [escape(content)] if content else []
        else:
            content_parts = []
            for block in content:
                if isinstance(block, str):
                    if block:
                        content_parts.append(escape(block))
                elif isinstance(block, dict):
                    formatted = _format_content_block_xml(block)
                    if formatted:
                        content_parts.append(formatted)

        tool_calls = getattr(message, "tool_calls", None) if _is_ai_message(message) else None
        additional_kwargs = getattr(message, "additional_kwargs", {})
        has_function_call = bool(
            _is_ai_message(message)
            and not tool_calls
            and isinstance(additional_kwargs, dict)
            and "function_call" in additional_kwargs
        )
        if tool_calls or has_function_call:
            parts = [f"<message type={quoteattr(msg_type)}>"]
            if content_parts:
                parts.append(f"  <content>{' '.join(content_parts)}</content>")
            if tool_calls:
                for tool_call in tool_calls:
                    tc_id = quoteattr(str(tool_call.get("id") or ""))
                    tc_name = quoteattr(str(tool_call.get("name") or ""))
                    tc_args = escape(json.dumps(tool_call.get("args", {}), ensure_ascii=False))
                    parts.append(f"  <tool_call id={tc_id} name={tc_name}>{tc_args}</tool_call>")
            else:
                function_call = additional_kwargs["function_call"]
                fc_name = quoteattr(str(function_call.get("name") or ""))
                fc_args = escape(str(function_call.get("arguments") or "{}"))
                parts.append(f"  <function_call name={fc_name}>{fc_args}</function_call>")
            parts.append("</message>")
            rendered.append("\n".join(parts))
        else:
            rendered.append(
                f"<message type={quoteattr(msg_type)}>{' '.join(content_parts)}</message>"
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
    """Approximate message and tool-schema token usage like langchain-core."""
    converted_messages = _coerce_messages(messages)
    token_count = 0.0
    ai_model_provider: str | None = None
    invalid_model_provider = False
    last_ai_total_tokens: int | None = None
    approx_at_last_ai: float | None = None

    if tools:
        from reactivegraph.tools import convert_to_openai_tool

        tools_chars = sum(
            len(json.dumps(tool if isinstance(tool, dict) else convert_to_openai_tool(tool)))
            for tool in tools
        )
        token_count += math.ceil(tools_chars / chars_per_token)

    for message in converted_messages:
        message_chars = 0
        content = getattr(message, "content", "")
        if isinstance(content, str):
            message_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    message_chars += len(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in {"image", "image_url"}:
                        token_count += tokens_per_image
                    elif block_type == "text":
                        text = block.get("text", "")
                        message_chars += len(text)
                    else:
                        message_chars += len(repr(block))
                else:
                    message_chars += len(repr(block))
        else:
            message_chars += len(repr(content))

        if _is_ai_message(message) and not isinstance(content, list):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                message_chars += len(repr(tool_calls))

        if _is_tool_message(message):
            message_chars += len(str(getattr(message, "tool_call_id", "")))

        message_chars += len(_get_message_openai_role(message))
        name = getattr(message, "name", None)
        if name and count_name:
            message_chars += len(str(name))

        token_count += math.ceil(message_chars / chars_per_token)
        token_count += extra_tokens_per_message

        if use_usage_metadata_scaling and _is_ai_message(message):
            response_metadata = getattr(message, "response_metadata", {})
            model_provider = (
                response_metadata.get("model_provider")
                if isinstance(response_metadata, dict)
                else None
            )
            if ai_model_provider is None:
                ai_model_provider = model_provider
            elif model_provider != ai_model_provider:
                invalid_model_provider = True

            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                total_tokens = usage_metadata.get("total_tokens")
                if isinstance(total_tokens, int):
                    last_ai_total_tokens = total_tokens
                    approx_at_last_ai = token_count

    if (
        use_usage_metadata_scaling
        and len(converted_messages) > 1
        and not invalid_model_provider
        and ai_model_provider is not None
        and last_ai_total_tokens is not None
        and approx_at_last_ai
        and approx_at_last_ai > 0
    ):
        scale_factor = last_ai_total_tokens / approx_at_last_ai
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


def _first_max_tokens(
    messages: Sequence[Any],
    *,
    max_tokens: int,
    token_counter: Callable[[list[Any]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Literal["first", "last"] | None = None,
    end_on: MessageType | None = None,
) -> list[Any]:
    messages = list(messages)
    if not messages:
        return messages

    if token_counter(messages) <= max_tokens:
        if end_on:
            for _ in range(len(messages)):
                if not _is_message_type(messages[-1], end_on):
                    messages.pop()
                else:
                    break
        return messages

    left, right = 0, len(messages)
    for _ in range(len(messages).bit_length()):
        if left >= right:
            break
        mid = (left + right + 1) // 2
        if token_counter(messages[:mid]) <= max_tokens:
            left = mid
        else:
            right = mid - 1
    idx = left

    if partial_strategy and idx < len(messages):
        included_partial = False
        copied = False
        content = getattr(messages[idx], "content", None)
        if isinstance(content, list):
            excluded = messages[idx].model_copy(deep=True)
            copied = True
            num_block = len(excluded.content)
            if partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
            for _ in range(1, num_block):
                excluded.content = excluded.content[:-1]
                if token_counter([*messages[:idx], excluded]) <= max_tokens:
                    messages = [*messages[:idx], excluded]
                    idx += 1
                    included_partial = True
                    break
            if included_partial and partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))

        if not included_partial:
            if not copied:
                excluded = messages[idx].model_copy(deep=True)
            text: str | None = None
            excluded_content = getattr(excluded, "content", None)
            if isinstance(excluded_content, str):
                text = excluded_content
            elif isinstance(excluded_content, list) and excluded_content:
                for block in excluded_content:
                    if isinstance(block, str):
                        text = block
                        break
                    if isinstance(block, dict) and block.get("type") == "text":
                        value = block.get("text")
                        if isinstance(value, str):
                            text = value
                            break

            if text:
                split_texts = text_splitter(text)
                base_message_count = token_counter(messages[:idx])
                if partial_strategy == "last":
                    split_texts = list(reversed(split_texts))
                left, right = 0, len(split_texts)
                for _ in range(len(split_texts).bit_length()):
                    if left >= right:
                        break
                    mid = (left + right + 1) // 2
                    excluded.content = "".join(split_texts[:mid])
                    if base_message_count + token_counter([excluded]) <= max_tokens:
                        left = mid
                    else:
                        right = mid - 1
                if left > 0:
                    content_splits = split_texts[:left]
                    if partial_strategy == "last":
                        content_splits = list(reversed(content_splits))
                    excluded.content = "".join(content_splits)
                    messages = [*messages[:idx], excluded]
                    idx += 1

    if end_on:
        for _ in range(idx):
            if idx > 0 and not _is_message_type(messages[idx - 1], end_on):
                idx -= 1
            else:
                break
    return messages[:idx]


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

"""Chat message models, interoperable with ``langchain_core.messages``.

DeerFlow imports message types in more places than any other LangChain
module, so this is the layer an engine swap must reproduce exactly. The
models below are pydantic v2 models with the same field names, field order,
``type`` discriminators, ``model_dump()`` shape and constructor coercions as
their upstream counterparts.

The intent is behavioural parity for the surface a host actually consumes:
construction, ``model_dump()`` round-trips, ``text`` extraction, chunk
concatenation, and the conversion helpers. It is deliberately *not* a
re-export of LangChain: the engine owns its message contract, and hosts
convert at the model boundary instead of importing a second copy of the
class hierarchy.

Portions of this module are adapted from upstream `langchain-core`
(https://github.com/langchain-ai/langchain) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import json
import uuid
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, TypedDict

try:  # pragma: no cover - exercised by the optional-integration tests
    from langchain_core import messages as _lc_messages
except ImportError:  # pragma: no cover - langchain-core is an optional host dep
    _lc_messages = None


def _upstream(name: str) -> tuple[type, ...]:
    """Return langchain-core's message class as an extra base, if installed.

    DeerFlow performs ``isinstance(message, langchain_core...Message)`` checks
    in 161 places across the harness. Those checks are identity checks, so a
    structurally identical pydantic model silently fails them and the host
    takes the wrong branch (observed: tool results dropped, summaries never
    persisted). Inheriting the real upstream class is the only way the engine
    can be a drop-in base without rewriting every consumer.

    ``BaseMessage`` permits real subclassing; LangChain only disables the
    *virtual* ``register()`` path. The upstream class contributes behaviour we
    must not reimplement (serialization envelope, ``pretty_repr``), while every
    field and validator below stays ours so ``model_dump()`` keeps the exact
    shape this module pins.

    langchain-core remains optional: without it this returns no extra bases and
    the classes are plain pydantic models with the identical surface.
    """
    if _lc_messages is None:
        return ()
    upstream = getattr(_lc_messages, name, None)
    return (upstream,) if isinstance(upstream, type) else ()

__all__ = (
    "AIMessage",
    "AIMessageChunk",
    "AnyMessage",
    "BaseMessage",
    "BaseMessageChunk",
    "ChatMessage",
    "HumanMessage",
    "HumanMessageChunk",
    "InputTokenDetails",
    "InvalidToolCall",
    "MessageLikeRepresentation",
    "parse_partial_json",
    "OutputTokenDetails",
    "RemoveMessage",
    "SystemMessage",
    "SystemMessageChunk",
    "TextAccessor",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "UsageMetadata",
    "add_ai_message_chunks",
    "add_usage",
    "convert_to_messages",
    "default_tool_chunk_parser",
    "default_tool_parser",
    "get_buffer_string",
    "invalid_tool_call",
    "merge_obj",
    "message_chunk_to_message",
    "message_to_dict",
    "message_to_foreign_dict",
    "messages_from_dict",
    "messages_to_dict",
    "subtract_usage",
    "tool_call",
    "tool_call_chunk",
)

InputTokenDetails = dict[str, int]
OutputTokenDetails = dict[str, int]
UsageMetadata = dict[str, Any]


class ToolCall(TypedDict):
    """A model's request to call a tool (declared order matches upstream)."""

    name: str
    args: dict[str, Any]
    id: str | None
    type: NotRequired[Literal["tool_call"]]


class ToolCallChunk(TypedDict):
    """A streamed fragment of a tool call.

    ``index`` mirrors ``langchain_core.messages.tool.ToolCallChunk``: it is a
    plain (not ``NotRequired``) key typed ``int | None``, so pydantic always
    materialises ``index: None`` when the caller omits it.
    """

    name: str | None
    args: str | None
    id: str | None
    index: int | None
    type: NotRequired[Literal["tool_call_chunk"]]


class InvalidToolCall(TypedDict):
    """A tool call the model produced but that could not be parsed."""

    type: Literal["invalid_tool_call"]
    id: str | None
    name: str | None
    args: str | None
    error: str | None
    index: NotRequired[int | str]
    extras: NotRequired[dict[str, Any]]


def tool_call(*, name: str, args: dict[str, Any], id: str | None) -> ToolCall:
    """Strict keyword-only factory, mirroring ``langchain_core``'s."""
    return ToolCall(name=name, args=args, id=id, type="tool_call")


def tool_call_chunk(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    index: int | None = None,
) -> ToolCallChunk:
    """Strict keyword-only factory, mirroring ``langchain_core``'s."""
    return ToolCallChunk(name=name, args=args, id=id, index=index, type="tool_call_chunk")


def invalid_tool_call(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    error: str | None = None,
) -> InvalidToolCall:
    """Strict keyword-only factory, mirroring ``langchain_core``'s."""
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


class TextAccessor(str):
    """String subclass that supports both ``.text`` and legacy ``.text()``.

    LangChain renamed the text accessor from a method to a property. Keeping
    the string callable preserves hosts written against the old spelling
    while still comparing equal to a plain ``str``.
    """

    __slots__ = ()

    def __call__(self) -> str:
        warnings.warn(
            "Calling .text() as a method is deprecated. "
            "Use .text as a property instead (e.g., message.text).",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(self)


def _try_neq_default(value: Any, field: Any) -> bool:
    """Upstream's default-comparison, defensive against exotic values."""
    try:
        return bool(field.get_default() != value)
    except Exception:  # noqa: BLE001 - mirror upstream's defensive compare
        try:
            return all(field.get_default() != value)
        except Exception:  # noqa: BLE001
            try:
                return value is not field.default
            except Exception:  # noqa: BLE001
                return False


def _coerce_id(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


_MESSAGE_BASE = _upstream("BaseMessage") or (BaseModel,)


class BaseMessage(*_MESSAGE_BASE):  # type: ignore[misc, unused-ignore]
    """Base class for chat messages.

    Field order matches upstream so ``model_dump()`` produces the same key
    order, which matters for journals that persist the dump verbatim.
    """

    content: str | list[str | dict[Any, Any]]
    additional_kwargs: dict[Any, Any] = Field(default_factory=dict)
    response_metadata: dict[Any, Any] = Field(default_factory=dict)
    type: str = ""
    name: str | None = None
    id: str | None = None

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if content_blocks is not None:
            super().__init__(content=content_blocks, **kwargs)
        else:
            super().__init__(content=content, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _coerce_numeric_id(cls, values: Any) -> Any:
        if isinstance(values, dict) and "id" in values:
            values = {**values, "id": _coerce_id(values["id"])}
        return values

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        """Return standard v1 content blocks for this message.

        Mirrors ``langchain_core.messages.base.BaseMessage.content_blocks``:
        known v1 block types pass through, strings become text blocks, and
        anything else is wrapped as ``non_standard`` before the v0 /
        OpenAI-Chat-Completions translators get a chance to unpack it.
        """
        blocks: list[dict[str, Any]] = []
        content = [self.content] if isinstance(self.content, str) and self.content else self.content
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item.get("type") not in _KNOWN_BLOCK_TYPES or "source_type" in item:
                    blocks.append({"type": "non_standard", "value": item})
                else:
                    blocks.append(dict(item))

        for translate in (_translate_v0_multimodal, _translate_openai_chat_completions):
            blocks = translate(blocks)
        return blocks

    @property
    def text(self) -> TextAccessor:
        """Text content: plain strings, or concatenated ``type: text`` blocks."""
        if isinstance(self.content, str):
            return TextAccessor(self.content)
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if block.get("type") == "text" and isinstance(text, str):
                    parts.append(text)
        return TextAccessor("".join(parts))

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump declared fields plus ``extra="allow"`` extras, in field order."""
        dumped = super().model_dump(**kwargs)
        extra = self.__pydantic_extra__
        if extra:
            for key, value in extra.items():
                dumped.setdefault(key, value)
        return dumped

    def __repr_args__(self) -> Any:
        """Yield repr pairs, dropping fields equal to their upstream default.

        Mirrors ``langchain_core.load.serializable.Serializable.__repr_args__``
        including its use of ``FieldInfo.get_default()``: fields whose default
        is ``PydanticUndefined`` (required fields, or ``default_factory``
        fields) always compare unequal to their value and are therefore kept.
        """
        return [
            (key, value)
            for key, value in super().__repr_args__()
            if key not in type(self).model_fields
            or _try_neq_default(value, type(self).model_fields[key])
        ]

    def __str__(self) -> str:
        return " ".join(f"{k}={v!r}" for k, v in self.__repr_args__() if k is not None)


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse JSON that may be truncated mid-stream, mirroring upstream."""
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    new_chars: list[str] = []
    stack: list[str] = []
    is_inside_string = False
    escaped = False
    for char in s:
        new_char = char
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                new_char = "\\n"
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return None
        new_chars.append(new_char)

    if is_inside_string:
        if escaped:
            new_chars.pop()
        new_chars.append('"')

    stack.reverse()
    while new_chars:
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            new_chars.pop()
    return json.loads(s, strict=strict)


def default_tool_parser(
    raw_tool_calls: list[dict[str, Any]],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """Best-effort parse of OpenAI-format ``additional_kwargs["tool_calls"]``."""
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
        function_name = raw_tool_call["function"]["name"]
        try:
            function_args = json.loads(raw_tool_call["function"]["arguments"])
            tool_calls.append(
                tool_call(
                    name=function_name or "",
                    args=function_args or {},
                    id=raw_tool_call.get("id"),
                )
            )
        except json.JSONDecodeError:
            invalid_tool_calls.append(
                invalid_tool_call(
                    name=function_name,
                    args=raw_tool_call["function"]["arguments"],
                    id=raw_tool_call.get("id"),
                    error=None,
                )
            )
    return tool_calls, invalid_tool_calls


def default_tool_chunk_parser(
    raw_tool_calls: list[dict[str, Any]],
) -> list[ToolCallChunk]:
    """Best-effort parse of OpenAI-format tool-call chunks."""
    parsed: list[ToolCallChunk] = []
    for raw in raw_tool_calls:
        if "function" not in raw:
            function_args = None
            function_name = None
        else:
            function_args = raw["function"]["arguments"]
            function_name = raw["function"]["name"]
        parsed.append(
            tool_call_chunk(
                name=function_name,
                args=function_args,
                id=raw.get("id"),
                index=raw.get("index"),
            )
        )
    return parsed


def merge_content(
    first_content: str | list[str | dict[Any, Any]],
    *contents: str | list[str | dict[Any, Any]],
) -> str | list[str | dict[Any, Any]]:
    """Merge message contents, mirroring upstream string/list rules."""
    merged: str | list[str | dict[Any, Any]]
    merged = "" if first_content is None else first_content

    for content in contents:
        if isinstance(merged, str):
            if isinstance(content, str):
                merged += content
            else:
                merged = [merged, *content]
        elif isinstance(content, list):
            merged = _merge_lists(merged, content)
        elif merged and isinstance(merged[-1], str):
            merged[-1] += content
        elif content == "":
            pass
        elif merged:
            merged.append(content)
    return merged


def _merge_lists(left: list[Any], right: list[Any]) -> list[Any]:
    """Merge two content lists; adjacent strings concatenate, dicts merge by index."""
    merged = list(left)
    for item in right:
        if isinstance(item, str) and merged and isinstance(merged[-1], str):
            merged[-1] = merged[-1] + item
        elif isinstance(item, dict) and merged and isinstance(merged[-1], dict):
            merged[-1] = {**merged[-1], **item}
        else:
            merged.append(item)
    return merged


_KNOWN_BLOCK_TYPES = frozenset(
    {
        "audio",
        "file",
        "image",
        "invalid_tool_call",
        "non_standard",
        "reasoning",
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_result",
        "text",
        "text-plain",
        "tool_call",
        "tool_call_chunk",
        "video",
    }
)


def _ensure_id(value: str | None) -> str:
    """Return ``value`` or a fresh ``lc_``-prefixed id, like upstream."""
    return value or f"lc_{uuid.uuid4()}"


def _extras(block: dict[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if key not in known}


def _attach_extras(block: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Attach non-None extras, dropping the key entirely when empty."""
    if extras:
        block["extras"] = extras
    return block


def _translate_v0_multimodal(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unpack legacy v0 ``source_type`` multimodal blocks into v1 blocks."""
    out: list[dict[str, Any]] = []
    for block in (b["value"] if b.get("type") == "non_standard" else b for b in blocks):
        block_type = block.get("type")
        if block_type in {"image", "audio", "file"} and "source_type" in block:
            out.append(_convert_legacy_v0_block(block))
        elif block_type in _KNOWN_BLOCK_TYPES:
            out.append(block)
        else:
            out.append({"type": "non_standard", "value": block})
    return out


def _convert_legacy_v0_block(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block["type"]
    source_type = block.get("source_type")
    if source_type == "url":
        known = {"mime_type", "type", "source_type", "url"}
        converted: dict[str, Any] = {
            "type": block_type,
            "id": block.get("id") or _ensure_id(None),
            "url": block["url"],
        }
        if block.get("mime_type"):
            converted["mime_type"] = block["mime_type"]
        return _attach_extras(converted, _extras(block, known))
    if source_type == "base64":
        known = {"mime_type", "type", "source_type", "data"}
        converted = {
            "type": block_type,
            "id": block.get("id") or _ensure_id(None),
            "base64": block["data"],
        }
        if block.get("mime_type"):
            converted["mime_type"] = block["mime_type"]
        return _attach_extras(converted, _extras(block, known))
    if source_type == "id" and block_type == "image":
        known = {"type", "source_type", "id"}
        converted = {"type": "image", "file_id": block["id"]}
        return _attach_extras(converted, _extras(block, known))
    return block


def _translate_openai_chat_completions(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Unpack OpenAI Chat Completions blocks (``image_url`` etc.) into v1."""
    out: list[dict[str, Any]] = []
    for block in (b["value"] if b.get("type") == "non_standard" else b for b in blocks):
        block_type = block.get("type")
        if block_type in {"image_url", "input_audio", "file"}:
            converted = _convert_openai_block(block)
            if converted.get("type") in _KNOWN_BLOCK_TYPES:
                out.append(converted)
                continue
        if block_type in _KNOWN_BLOCK_TYPES:
            out.append(block)
        else:
            out.append({"type": "non_standard", "value": block})
    return out


def _convert_openai_block(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block["type"]
    if block_type == "image_url":
        image_url = block.get("image_url")
        if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
            return block
        extras = _extras(block, {"type", "image_url"})
        for key, value in _extras(image_url, {"url"}).items():
            extras["detail" if key == "detail" else f"image_url_{key}"] = value
        converted = _attach_extras(
            {"type": "image", "id": _ensure_id(None), "url": image_url["url"]},
            extras,
        )
        return converted
    if block_type == "input_audio":
        audio = block.get("input_audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            return block
        extras = _extras(block, {"type", "input_audio"})
        for key, value in _extras(audio, {"data", "format"}).items():
            extras[f"audio_{key}"] = value
        return _attach_extras(
            {
                "type": "audio",
                "id": _ensure_id(None),
                "base64": audio["data"],
                "mime_type": f"audio/{audio.get('format')}",
            },
            extras,
        )
    if block_type == "file":
        file_block = block.get("file")
        if not isinstance(file_block, dict):
            return block
        if isinstance(file_block.get("file_id"), str):
            extras = _extras(block, {"type", "file"})
            for key, value in _extras(file_block, {"file_id"}).items():
                extras[f"file_{key}"] = value
            return _attach_extras(
                {"type": "file", "id": _ensure_id(None), "file_id": file_block["file_id"]},
                extras,
            )
        data_uri = _parse_data_uri(file_block.get("file_data"))
        if data_uri is None:
            return block
        extras = _extras(block, {"type", "file"})
        for key, value in _extras(file_block, {"file_data", "filename"}).items():
            extras[f"file_{key}"] = value
        converted = {
            "type": "file",
            "id": _ensure_id(None),
            "base64": data_uri["data"],
            "mime_type": "application/pdf",
        }
        if file_block.get("filename"):
            converted["filename"] = file_block["filename"]
        return _attach_extras(converted, extras)
    return block


def _parse_data_uri(value: Any) -> dict[str, str] | None:
    """Parse ``data:<mime>;base64,<payload>`` into its two halves."""
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header, _, payload = value[len("data:") :].partition(",")
    mime_type = header.split(";", 1)[0]
    if not mime_type or ";base64" not in header:
        return None
    return {"mime_type": mime_type, "data": payload}


def merge_obj(left: Any, right: Any) -> Any:
    """Merge two tool artifacts, mirroring ``langchain_core.utils._merge``."""
    if left is None or right is None:
        return left if left is not None else right
    if type(left) is not type(right):
        msg = (
            f"left and right are of different types. Left type:  {type(left)}. Right "
            f"type: {type(right)}."
        )
        raise TypeError(msg)
    if isinstance(left, str):
        return left + right
    if isinstance(left, dict):
        return _merge_dicts(left, right)
    if isinstance(left, list):
        return _merge_lists(left, right)
    if left == right:
        return left
    msg = (
        f"Unable to merge {left=} and {right=}. Both must be of type str, dict, or "
        f"list, or else be two equal objects."
    )
    raise ValueError(msg)


def _merge_dicts(*dicts: dict[Any, Any]) -> dict[Any, Any]:
    merged: dict[Any, Any] = {}
    for item in dicts:
        for key, value in item.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged


def add_usage(left: UsageMetadata | None, right: UsageMetadata | None) -> UsageMetadata:
    """Recursively add two usage-metadata dicts."""
    if not (left or right):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not (left and right):
        return dict(cast("UsageMetadata", left or right))
    return _dict_int_op(left, right, lambda a, b: a + b)


def subtract_usage(left: UsageMetadata | None, right: UsageMetadata | None) -> UsageMetadata:
    """Recursively subtract ``right`` token counts from ``left``."""
    if not (left and right):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return _dict_int_op(left, right, lambda a, b: max(a - b, 0))


def _dict_int_op(
    left: dict[str, Any],
    right: dict[str, Any],
    op: Any,
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict[str, Any]:
    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict[str, Any] = {}
    for key in set(left).union(right):
        left_value = left.get(key, default)
        right_value = right.get(key, default)
        if isinstance(left_value, int) and isinstance(right_value, int):
            combined[key] = op(left_value, right_value)
        elif isinstance(left.get(key, {}), dict) and isinstance(right.get(key, {}), dict):
            combined[key] = _dict_int_op(
                left.get(key, {}),
                right.get(key, {}),
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(item[key]) for item in (left, right) if key in item]
            msg = f"Unknown value types: {types}. Only dict and int values are supported."
            raise ValueError(msg)
    return combined


def _best_chunk_id(left: AIMessageChunk, others: Sequence[AIMessageChunk]) -> str | None:
    """Pick the most trustworthy stream id, preferring provider-assigned ids."""
    best_rank = -1
    chosen: str | None = None
    for candidate in (left.id, *(other.id for other in others)):
        if not candidate:
            continue
        if not candidate.startswith("lc_") and not candidate.startswith("lc_run-"):
            return candidate
        rank = 1 if candidate.startswith("lc_run-") else 0
        if rank > best_rank:
            best_rank = rank
            chosen = candidate
    return chosen


def add_ai_message_chunks(
    left: AIMessageChunk, *others: AIMessageChunk
) -> AIMessageChunk:
    """Concatenate AI chunks, merging content, tool-call chunks and usage."""
    content = merge_content(left.content, *(other.content for other in others))
    additional_kwargs = _merge_dicts(
        left.additional_kwargs, *(other.additional_kwargs for other in others)
    )
    response_metadata = _merge_dicts(
        left.response_metadata, *(other.response_metadata for other in others)
    )

    raw_chunks: list[ToolCallChunk] = []
    for chunk in (left, *others):
        raw_chunks.extend(chunk.tool_call_chunks)
    tool_call_chunks = _merge_tool_call_chunks(raw_chunks)

    usage_metadata: UsageMetadata | None = None
    if left.usage_metadata or any(other.usage_metadata is not None for other in others):
        usage_metadata = left.usage_metadata
        for other in others:
            usage_metadata = add_usage(usage_metadata, other.usage_metadata)

    chunk_position: Literal["last"] | None = (
        "last"
        if any(item.chunk_position == "last" for item in (left, *others))
        else None
    )
    return left.__class__(
        content=content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        id=_best_chunk_id(left, others),
        tool_call_chunks=tool_call_chunks,
        chunk_position=chunk_position,
    )


def _merge_tool_call_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        index = chunk.get("index")
        target = None
        if index is not None:
            target = next((item for item in merged if item.get("index") == index), None)
        if target is None:
            merged.append(dict(chunk))
            continue
        for key in ("name", "args", "id"):
            value = chunk.get(key)
            if value is None:
                continue
            existing = target.get(key)
            target[key] = value if existing is None else existing + value
    return merged


class BaseMessageChunk(BaseMessage, *_upstream("BaseMessageChunk")):  # type: ignore[misc, unused-ignore]
    """Message chunk that can be concatenated with other chunks."""

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, BaseMessageChunk):
            return self.__class__(
                id=self.id,
                content=merge_content(self.content, other.content),
                additional_kwargs=_merge_dicts(self.additional_kwargs, other.additional_kwargs),
                response_metadata=_merge_dicts(self.response_metadata, other.response_metadata),
            )
        if isinstance(other, list) and all(isinstance(o, BaseMessageChunk) for o in other):
            content = merge_content(self.content, *(o.content for o in other))
            additional_kwargs = _merge_dicts(
                self.additional_kwargs, *(o.additional_kwargs for o in other)
            )
            response_metadata = _merge_dicts(
                self.response_metadata, *(o.response_metadata for o in other)
            )
            return self.__class__(
                id=self.id,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        msg = (
            'unsupported operand type(s) for +: "'
            f"{self.__class__.__name__}"
            f'" and "{other.__class__.__name__}"'
        )
        raise TypeError(msg)


class HumanMessage(BaseMessage, *_upstream("HumanMessage")):  # type: ignore[misc, unused-ignore]
    """Message from the user."""

    type: Literal["human"] = "human"


class HumanMessageChunk(  # type: ignore[misc, unused-ignore]
    HumanMessage, BaseMessageChunk, *_upstream("HumanMessageChunk")  # type: ignore[misc]
):
    """Streaming chunk of a human message."""

    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]


class SystemMessage(BaseMessage, *_upstream("SystemMessage")):  # type: ignore[misc, unused-ignore]
    """System instruction message."""

    type: Literal["system"] = "system"


class SystemMessageChunk(  # type: ignore[misc, unused-ignore]
    SystemMessage, BaseMessageChunk, *_upstream("SystemMessageChunk")  # type: ignore[misc]
):
    """Streaming chunk of a system message."""

    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]


class ChatMessage(BaseMessage, *_upstream("ChatMessage")):  # type: ignore[misc, unused-ignore]
    """Message with an arbitrary speaker role."""

    role: str
    type: Literal["chat"] = "chat"


class AIMessage(BaseMessage, *_upstream("AIMessage")):  # type: ignore[misc, unused-ignore]
    """Message from the model, including tool calls and usage metadata."""

    type: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    usage_metadata: UsageMetadata | None = None

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        """Base blocks plus one ``tool_call`` block per parsed tool call."""
        blocks = super().content_blocks
        if self.tool_calls:
            present = {
                block.get("id")
                for block in self.content
                if isinstance(block, dict) and block.get("type") == "tool_call"
            }
            for call in self.tool_calls:
                call_id = call.get("id")
                if not call_id or call_id in present:
                    continue
                block: dict[str, Any] = {
                    "type": "tool_call",
                    "id": call_id,
                    "name": call["name"],
                    "args": call["args"],
                }
                call_mapping = cast("Mapping[str, Any]", call)
                if "index" in call_mapping:
                    block["index"] = call_mapping["index"]
                if "extras" in call_mapping:
                    block["extras"] = call_mapping["extras"]
                blocks.append(block)
        return blocks

    @model_validator(mode="before")
    @classmethod
    def _backwards_compat_tool_calls(cls, values: Any) -> Any:
        """Normalise tool calls exactly like ``langchain_core.messages.ai``.

        Only the legacy ``additional_kwargs["tool_calls"]`` path is parsed
        leniently (OpenAI wire format, best effort). Calls passed directly as
        ``tool_calls=[...]`` are validated strictly through the keyword-only
        factories, so malformed input raises instead of being silently
        downgraded to an "invalid" call.
        """
        if not isinstance(values, dict):
            return values
        values = dict(values)

        check_additional_kwargs = not any(
            values.get(key)
            for key in ("tool_calls", "invalid_tool_calls", "tool_call_chunks")
        )
        raw_tool_calls = (
            values.get("additional_kwargs", {}).get("tool_calls")
            if check_additional_kwargs
            else None
        )
        if raw_tool_calls:
            try:
                if issubclass(cls, AIMessageChunk):
                    values["tool_call_chunks"] = default_tool_chunk_parser(raw_tool_calls)
                else:
                    parsed, invalid = default_tool_parser(raw_tool_calls)
                    values["tool_calls"] = parsed
                    values["invalid_tool_calls"] = invalid
            except Exception:  # noqa: BLE001 - upstream logs and moves on
                pass

        if tool_calls := values.get("tool_calls"):
            values["tool_calls"] = [
                tool_call(**{k: v for k, v in call.items() if k not in {"type", "extras"}})
                for call in tool_calls
            ]
        if invalid := values.get("invalid_tool_calls"):
            values["invalid_tool_calls"] = [
                invalid_tool_call(**{k: v for k, v in call.items() if k != "type"})
                for call in invalid
            ]
        if chunks := values.get("tool_call_chunks"):
            values["tool_call_chunks"] = [
                tool_call_chunk(**{k: v for k, v in chunk.items() if k != "type"})
                for chunk in chunks
            ]
        return values


class AIMessageChunk(  # type: ignore[misc, unused-ignore]
    AIMessage, BaseMessageChunk, *_upstream("AIMessageChunk")  # type: ignore[misc]
):
    """Streaming chunk of an AI message."""

    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment]
    tool_call_chunks: list[ToolCallChunk] = Field(default_factory=list)
    chunk_position: Literal["last"] | None = None

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, AIMessageChunk):
            return add_ai_message_chunks(self, other)
        if isinstance(other, (list, tuple)) and all(
            isinstance(item, AIMessageChunk) for item in other
        ):
            return add_ai_message_chunks(self, *other)
        return super().__add__(other)

    @model_validator(mode="after")
    def _init_tool_calls(self) -> AIMessageChunk:
        """Derive ``tool_call_chunks``/``tool_calls`` exactly like upstream."""
        if not self.tool_call_chunks:
            if self.tool_calls:
                self.tool_call_chunks = [
                    tool_call_chunk(
                        name=call["name"],
                        args=json.dumps(call["args"]),
                        id=call["id"],
                        index=None,
                    )
                    for call in self.tool_calls
                ]
            if self.invalid_tool_calls:
                chunks = self.tool_call_chunks
                chunks.extend(
                    tool_call_chunk(
                        name=call["name"],
                        args=call["args"],
                        id=call["id"],
                        index=None,
                    )
                    for call in self.invalid_tool_calls
                )
                self.tool_call_chunks = chunks
            return self

        parsed_calls: list[ToolCall] = []
        parsed_invalid: list[InvalidToolCall] = []
        for chunk in self.tool_call_chunks:
            try:
                args = parse_partial_json(chunk["args"]) if chunk["args"] else {}
                if isinstance(args, dict):
                    parsed_calls.append(
                        tool_call(name=chunk["name"] or "", args=args, id=chunk["id"])
                    )
                else:
                    parsed_invalid.append(
                        invalid_tool_call(
                            name=chunk["name"],
                            args=chunk["args"],
                            id=chunk["id"],
                            error=None,
                        )
                    )
            except Exception:  # noqa: BLE001 - upstream treats any failure as invalid
                parsed_invalid.append(
                    invalid_tool_call(
                        name=chunk["name"],
                        args=chunk["args"],
                        id=chunk["id"],
                        error=None,
                    )
                )
        self.tool_calls = parsed_calls
        self.invalid_tool_calls = parsed_invalid
        return self


def _host_tool_output_mixin() -> type | None:
    """Return langchain-core's ``ToolOutputMixin`` when it is installed.

    ``langgraph.types.Command`` and langchain-core's own ``ToolMessage`` are
    members of that marker, and DeerFlow returns both from tools. The engine
    must therefore recognise the host marker's members as pass-through
    outputs, or a control-flow command is silently stringified into a
    ``ToolMessage`` and the graph never jumps.
    """
    try:
        from langchain_core.messages.tool import ToolOutputMixin as HostMixin
    except ImportError:  # pragma: no cover - langchain-core is optional
        return None
    return HostMixin


HostToolOutputMixin = _host_tool_output_mixin()


class ToolOutputMixin(
    *((HostToolOutputMixin,) if HostToolOutputMixin is not None else ())  # type: ignore[misc]
):
    """Marker for objects a tool may return directly.

    When a tool is invoked with a model tool call and returns an object that is
    not a tool output, the runtime wraps the normalized result in a
    ``ToolMessage``. Returning a ``ToolMessage`` directly bypasses that wrap.

    When langchain-core is installed this class also inherits its
    ``ToolOutputMixin``, so engine-built outputs satisfy the host's identity
    checks. Host-built objects (``langgraph.types.Command``, langchain-core
    messages) are covered by :func:`is_tool_output`, because they cannot be
    added to this class's MRO after the fact.
    """


def is_tool_output(value: object) -> bool:
    """Return whether *value* is a tool output from either side of the bridge.

    LangGraph's ``Command`` subclasses the host ``ToolOutputMixin`` directly
    and is not an instance of our subclass, so checking only
    :class:`ToolOutputMixin` silently re-wraps host-constructed commands. Both
    markers must be consulted; this is the consumer half of the identity
    bridge in this module.
    """
    if isinstance(value, ToolOutputMixin):
        return True
    return HostToolOutputMixin is not None and isinstance(value, HostToolOutputMixin)


class ToolMessage(  # type: ignore[misc, unused-ignore]
    BaseMessage, ToolOutputMixin, *_upstream("ToolMessage")  # type: ignore[misc]
):
    """Result of a tool call."""

    type: Literal["tool"] = "tool"
    tool_call_id: str
    artifact: Any = None
    status: Literal["success", "error"] = "success"

    # Upstream hides these two inherited fields from repr.
    additional_kwargs: dict[Any, Any] = Field(default_factory=dict, repr=False)
    response_metadata: dict[Any, Any] = Field(default_factory=dict, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _coerce_args(cls, values: Any) -> Any:
        """Coerce content and tool-call id exactly like upstream.

        A missing ``tool_call_id`` raises ``KeyError`` (not a pydantic
        ``ValidationError``) because upstream indexes the input dict directly.
        """
        if not isinstance(values, dict):
            return values
        values = dict(values)

        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)
        if not isinstance(content, (str, list)):
            values["content"] = str(content)
        elif isinstance(content, list):
            coerced: list[Any] = []
            for item in content:
                coerced.append(item if isinstance(item, (str, dict)) else str(item))
            values["content"] = coerced

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (uuid.UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

class ToolMessageChunk(  # type: ignore[misc, unused-ignore]
    ToolMessage, BaseMessageChunk, *_upstream("ToolMessageChunk")  # type: ignore[misc]
):
    """Streaming chunk of a tool message."""

    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                msg = "Cannot concatenate ToolMessageChunks with different tool_call_ids."
                raise ValueError(msg)
            status: Literal["success", "error"] = (
                "error" if "error" in {self.status, other.status} else "success"
            )
            return self.__class__(
                id=self.id,
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                artifact=merge_obj(self.artifact, other.artifact),
                status=status,
                additional_kwargs=_merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=_merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )
        return super().__add__(other)


class RemoveMessage(BaseMessage, *_upstream("RemoveMessage")):  # type: ignore[misc, unused-ignore]
    """Instruction to delete another message from history."""

    type: Literal["remove"] = "remove"

    def __init__(self, id: str, **kwargs: Any) -> None:
        if kwargs.pop("content", None):
            msg = "RemoveMessage does not support 'content' field."
            raise ValueError(msg)
        super().__init__(content="", id=id, **kwargs)


AnyMessage = (
    AIMessage
    | AIMessageChunk
    | ChatMessage
    | HumanMessage
    | HumanMessageChunk
    | SystemMessage
    | SystemMessageChunk
    | ToolMessage
    | ToolMessageChunk
)

MessageLikeRepresentation = (
    BaseMessage
    | list[str]
    | tuple[str, str | list[str | dict[str, Any]]]
    | str
    | dict[str, Any]
)

_MESSAGE_TYPES: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "user": HumanMessage,
    "ai": AIMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "developer": SystemMessage,
    "tool": ToolMessage,
    "remove": RemoveMessage,
    "chat": ChatMessage,
    "AIMessageChunk": AIMessageChunk,
    "HumanMessageChunk": HumanMessageChunk,
    "SystemMessageChunk": SystemMessageChunk,
    "ToolMessageChunk": ToolMessageChunk,
}

# Chunk class names collapse onto their non-chunk role, matching upstream's
# `_LC_CONSTRUCTOR_NAME_TO_TYPE` in `langchain_core.messages.utils`.
_LC_CONSTRUCTOR_NAME_TO_TYPE = {
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "RemoveMessage": "remove",
    "ChatMessage": "chat",
}


_MESSAGE_TYPE_HINTS = (
    "'human', 'user', 'ai', 'assistant', 'system', 'developer', "
    "'tool', 'remove', or 'chat'"
)


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style ``function`` entries into native tool calls."""
    converted: list[dict[str, Any]] = []
    for call in tool_calls:
        if "function" not in call:
            converted.append(call)
            continue
        arguments = call["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments, strict=False)
        converted.append(
            {
                "name": call["function"]["name"],
                "args": arguments,
                "id": call["id"],
                "type": "tool_call",
            }
        )
    return converted


def _message_kwargs(
    message_type: str,
    *,
    name: str | None,
    tool_call_id: str | None,
    id: str | None,
    additional_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Assemble constructor kwargs from the raw ``**additional_kwargs`` funnel.

    ``response_metadata``/``usage_metadata`` are promoted out of the catch-all
    because token accounting reads them directly: ``model_dump()`` puts
    ``usage_metadata`` at the top level, and leaving it in
    ``additional_kwargs`` silently disables every budget check. Everything else
    (including a nested ``additional_kwargs`` mapping) stays in
    ``additional_kwargs``, matching LangChain's object path.
    """
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        response_metadata = additional_kwargs.pop("response_metadata", None)
        if response_metadata:
            kwargs["response_metadata"] = response_metadata
        usage_metadata = additional_kwargs.pop("usage_metadata", None)
        extra = dict(additional_kwargs)
        nested = extra.pop("additional_kwargs", None)
        if nested:
            extra.update(nested)
        kwargs["additional_kwargs"] = extra
        if usage_metadata is not None and message_type in {"ai", "assistant"}:
            kwargs["usage_metadata"] = usage_metadata
    if id is not None:
        kwargs["id"] = id
    return kwargs


def _create_message_from_message_type(
    message_type: str,
    content: str | list[str | dict[str, Any]],
    name: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    id: str | None = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """Build a message from a role string and content, upstream style."""
    kwargs = _message_kwargs(
        message_type,
        name=name,
        tool_call_id=tool_call_id,
        id=id,
        additional_kwargs=additional_kwargs,
    )
    if tool_calls is not None:
        kwargs["tool_calls"] = _normalize_tool_calls(tool_calls)

    if message_type in {"human", "user"}:
        return HumanMessage(content=content, **kwargs)
    if message_type in {"ai", "assistant"}:
        return AIMessage(content=content, **kwargs)
    if message_type in {"system", "developer"}:
        if message_type == "developer":
            kwargs.setdefault("additional_kwargs", {})
            kwargs["additional_kwargs"]["__openai_role__"] = "developer"
        return SystemMessage(content=content, **kwargs)
    if message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        status = kwargs.get("additional_kwargs", {}).pop("status", None)
        if status is not None:
            kwargs["status"] = status
        return ToolMessage(content=content, artifact=artifact, **kwargs)
    if message_type == "remove":
        return RemoveMessage(**kwargs)
    if message_type == "chat":
        return ChatMessage(content=content, **kwargs)

    msg = f"Unexpected message type: '{message_type}'. Use one of {_MESSAGE_TYPE_HINTS}."
    raise ValueError(msg)


def convert_to_messages(
    messages: Iterable[MessageLikeRepresentation],
) -> list[BaseMessage]:
    """Convert strings, ``(role, content)`` tuples and dicts to messages."""
    return [_convert_to_message(message) for message in messages]


def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    if isinstance(message, BaseMessage):
        return message
    if isinstance(message, str):
        return HumanMessage(content=message)
    if isinstance(message, Sequence):
        try:
            message_type, template = cast(Sequence[Any], message)[:2]
        except ValueError as error:
            msg = "Message as a sequence must be (role string, template)"
            raise NotImplementedError(msg) from error
        return _create_message_from_message_type(message_type, template)
    if isinstance(message, dict):
        if (
            message.get("lc") == 1
            and message.get("type") == "constructor"
            and isinstance(message.get("id"), list)
            and message["id"]
            and isinstance(message.get("kwargs"), dict)
        ):
            mapped = _LC_CONSTRUCTOR_NAME_TO_TYPE.get(message["id"][-1])
            if mapped is not None:
                return _convert_to_message({"type": mapped, **message["kwargs"]})

        kwargs = dict(message)
        try:
            try:
                msg_type = kwargs.pop("role")
            except KeyError:
                msg_type = kwargs.pop("type")
            msg_content = kwargs.pop("content") or ""
        except KeyError as error:
            msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
            raise ValueError(msg) from error
        return _create_message_from_message_type(msg_type, msg_content, **kwargs)

    # A foreign message object: a host's own class hierarchy (e.g.
    # ``langchain_core.messages``) is disjoint from ours and LangChain blocks
    # virtual-subclassing of ``BaseMessage``, so identity checks can never
    # bridge the two. Convert structurally instead: any object carrying the
    # upstream ``type`` + ``model_dump()`` surface is rebuilt as *our* class,
    # which is what the engine's ``isinstance``-based reducers require.
    dumped = _dump_foreign_message(message)
    if dumped is not None:
        return _convert_to_message(dumped)

    msg = f"Unsupported message type: {type(message)}"
    raise NotImplementedError(msg)


def _dump_foreign_message(message: Any) -> dict[str, Any] | None:
    """Return the upstream flat shape for a foreign message, else ``None``.

    Fail closed: an object without a usable ``type`` *and* a dict-returning
    ``model_dump()`` is not treated as a message. Returning ``None`` keeps the
    original ``NotImplementedError`` (and its message) intact for real
    non-message values.
    """
    if isinstance(message, BaseMessage):
        return None
    message_type = getattr(message, "type", None)
    if not isinstance(message_type, str) or not message_type:
        return None
    dump = getattr(message, "model_dump", None)
    if not callable(dump):
        return None
    try:
        dumped = dump()
    except Exception:  # noqa: BLE001 - a failing dump is not a message
        return None
    if not isinstance(dumped, dict):
        return None
    if dumped.get("type", message_type) != message_type:
        return None
    return {"type": message_type, **dumped}


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Wrap a message dump in the upstream ``{"type", "data"}`` envelope."""
    return {"type": message.type, "data": message.model_dump()}


def message_to_foreign_dict(message: BaseMessage) -> dict[str, Any]:
    """Flatten one of our messages into the shape a foreign host accepts.

    ``langchain_core.messages.convert_to_messages`` reads a dict carrying
    ``type`` plus the message's own fields (not the ``{"type", "data"}``
    envelope). Chat models coerce that flat dict back into *their* class, so
    this is the outbound half of the boundary bridge.
    """
    if isinstance(message, BaseMessage):
        return {"type": message.type, **message.model_dump()}
    # Host middleware splices *its own* message objects into a request (e.g.
    # DeerFlow's DanglingToolCallMiddleware). Those are already in the target
    # shape, so re-encode them rather than rejecting the request.
    dumped = _dump_foreign_message(message)
    if dumped is not None:
        return dumped
    msg = f"message_to_foreign_dict expects a message, got {type(message)}"
    raise TypeError(msg)


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a sequence of messages to the ``{"type", "data"}`` envelopes."""
    return [message_to_dict(message) for message in messages]


def messages_from_dict(messages: Sequence[dict[str, Any]]) -> list[BaseMessage]:
    """Restore messages produced by :func:`messages_to_dict`."""
    restored: list[BaseMessage] = []
    for message in messages:
        message_type = message["type"]
        cls = _MESSAGE_TYPES.get(message_type)
        if cls is None:
            msg = f"Got unexpected message type: {message_type}"
            raise ValueError(msg)
        restored.append(cls(**message["data"]))
    return restored


def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
    """Promote a streaming chunk to its full message class."""
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    ignore = {"type"}
    if isinstance(chunk, AIMessageChunk):
        ignore |= {"tool_call_chunks", "chunk_position"}
    data = {key: value for key, value in chunk.__dict__.items() if key not in ignore}
    # Chunk classes always list their non-chunk counterpart as first parent.
    return cast("BaseMessage", type(chunk).__mro__[1](**data))


def get_buffer_string(
    messages: Sequence[BaseMessage],
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
    *,
    system_prefix: str = "System",
    function_prefix: str = "Function",
    tool_prefix: str = "Tool",
    message_separator: str = "\n",
    format: Literal["prefix", "xml"] = "prefix",  # noqa: A002
) -> str:
    """Render messages as a prefix or escaped XML transcript.

    The implementation lives in :mod:`reactivegraph.message_utils`; the local
    import avoids a module cycle while keeping this public helper available
    from :mod:`reactivegraph.messages`.
    """
    from reactivegraph.message_utils import get_buffer_string as _get_buffer_string

    return _get_buffer_string(
        messages,
        human_prefix,
        ai_prefix,
        system_prefix=system_prefix,
        function_prefix=function_prefix,
        tool_prefix=tool_prefix,
        message_separator=message_separator,
        format=format,
    )

"""Safe typed serialization for durable ReactiveGraph state.

This module deliberately does not use pickle. Unknown Python objects fail
closed instead of importing arbitrary callables from checkpoint data. The
msgpack extension layout follows LangGraph's JsonPlusSerializer where practical
so rows can be inspected by the host ecosystem, but deserialization is stricter:
a reconstructed class must already be loaded in this process and must pass an
explicit type check.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import decimal
import json
import pathlib
import re
import sys
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import msgpack
from pydantic import BaseModel

# LangGraph-compatible extension codes.
EXT_CONSTRUCTOR_SINGLE_ARG = 0
EXT_CONSTRUCTOR_POS_ARGS = 1
EXT_CONSTRUCTOR_KW_ARGS = 2
EXT_METHOD_SINGLE_ARG = 3
EXT_PYDANTIC_V1 = 4
EXT_PYDANTIC_V2 = 5
EXT_NUMPY_ARRAY = 6
EXT_DELTA_SNAPSHOT = 7

_JSON_TAG = "__reactivegraph_type__"
_JSON_TAG_VERSION = 1
_MAX_DECODE_DEPTH = 100


class SerializationError(TypeError):
    """Raised when a value cannot be represented without unsafe fallback."""


@dataclasses.dataclass(frozen=True)
class _TupleMarker:
    items: tuple[Any, ...]


class ReactiveSerializer:
    """Typed msgpack serializer with a no-pickle security boundary."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None:
            return "null", b""
        if isinstance(obj, bytes):
            return "bytes", obj
        if isinstance(obj, bytearray):
            return "bytearray", bytes(obj)
        return "msgpack", self._pack(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, payload = data
        if type_ == "null":
            return None
        if type_ == "bytes":
            return bytes(payload)
        if type_ == "bytearray":
            return bytearray(payload)
        if type_ == "json":
            return json.loads(payload)
        if type_ == "msgpack":
            return self._unpack(payload)
        if type_ == "pickle":
            raise SerializationError("pickle deserialization is disabled")
        raise SerializationError(f"unknown serialization type: {type_!r}")

    def dumps_json(self, value: Any) -> str:
        return json.dumps(
            _json_encode(value, depth=0),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def loads_json(self, value: str | bytes) -> Any:
        decoded = json.loads(value)
        return _json_decode(decoded, depth=0)

    # -- msgpack ---------------------------------------------------------

    def _pack(self, value: Any) -> bytes:
        prepared = _prepare_msgpack(value, depth=0)
        return self._pack_prepared(prepared)

    def _pack_prepared(self, value: Any) -> bytes:
        try:
            return msgpack.packb(
                value,
                default=self._default,
                use_bin_type=True,
                strict_types=False,
            )
        except (TypeError, ValueError, OverflowError, SerializationError) as exc:
            if isinstance(exc, SerializationError):
                raise
            raise SerializationError(
                f"cannot serialize {type(value).__module__}.{type(value).__qualname__} "
                "without an unsafe fallback"
            ) from exc

    def _unpack(self, payload: bytes) -> Any:
        try:
            return msgpack.unpackb(
                payload,
                raw=False,
                strict_map_key=False,
                ext_hook=self._ext_hook,
            )
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError("invalid msgpack checkpoint payload") from exc

    def _default(self, obj: Any) -> msgpack.ExtType:
        if isinstance(obj, _DeltaSnapshot):
            return self._ext(EXT_DELTA_SNAPSHOT, obj.value)
        if isinstance(obj, _TupleMarker):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                ("builtins", "tuple", list(obj.items)),
            )
        if isinstance(obj, BaseModel):
            return self._ext(
                EXT_PYDANTIC_V2,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.model_dump(),
                    "model_validate_json",
                ),
            )
        if isinstance(obj, uuid.UUID):
            return self._ext(
                EXT_CONSTRUCTOR_SINGLE_ARG,
                (obj.__class__.__module__, obj.__class__.__name__, obj.hex),
            )
        if isinstance(obj, decimal.Decimal):
            return self._ext(
                EXT_CONSTRUCTOR_SINGLE_ARG,
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            )
        if isinstance(obj, dt.datetime):
            return self._ext(
                EXT_METHOD_SINGLE_ARG,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.isoformat(),
                    "fromisoformat",
                ),
            )
        if isinstance(obj, dt.timedelta):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    (obj.days, obj.seconds, obj.microseconds),
                ),
            )
        if isinstance(obj, dt.date):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    (obj.year, obj.month, obj.day),
                ),
            )
        if isinstance(obj, dt.time):
            return self._ext(
                EXT_CONSTRUCTOR_KW_ARGS,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    {
                        "hour": obj.hour,
                        "minute": obj.minute,
                        "second": obj.second,
                        "microsecond": obj.microsecond,
                        "tzinfo": obj.tzinfo,
                        "fold": obj.fold,
                    },
                ),
            )
        if isinstance(obj, dt.timezone):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.__getinitargs__(),  # type: ignore[attr-defined]
                ),
            )
        if isinstance(obj, (set, frozenset, deque)):
            return self._ext(
                EXT_CONSTRUCTOR_SINGLE_ARG,
                (obj.__class__.__module__, obj.__class__.__name__, tuple(obj)),
            )
        if isinstance(obj, pathlib.Path):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                (obj.__class__.__module__, obj.__class__.__name__, tuple(obj.parts)),
            )
        if isinstance(obj, re.Pattern):
            return self._ext(
                EXT_CONSTRUCTOR_POS_ARGS,
                ("re", "compile", (obj.pattern, obj.flags)),
            )
        if isinstance(obj, Enum):
            return self._ext(
                EXT_CONSTRUCTOR_SINGLE_ARG,
                (obj.__class__.__module__, obj.__class__.__name__, obj.value),
            )
        raise SerializationError(
            f"cannot serialize {type(obj).__module__}.{type(obj).__qualname__}; "
            "pickle fallback is disabled"
        )

    def _ext(self, code: int, value: Any) -> msgpack.ExtType:
        # ``value`` is already normalised by ``_prepare_msgpack`` (its children
        # may contain marker objects), so encode it without re-running the
        # top-level preparation pass.
        return msgpack.ExtType(code, self._pack_prepared(value))

    def _ext_hook(self, code: int, data: bytes) -> Any:
        value = self._unpack(data)
        if code == EXT_CONSTRUCTOR_POS_ARGS:
            module, name, args = value
            if (module, name) == ("builtins", "tuple"):
                return tuple(args)
            if module == "datetime":
                cls = _safe_builtin_class(
                    module,
                    name,
                    (dt.date, dt.datetime, dt.time, dt.timedelta, dt.tzinfo),
                )
                if name in {"date", "datetime", "timedelta", "timezone"}:
                    return cls(*args)
            if module == "pathlib":
                cls = _safe_builtin_class(module, name, (pathlib.PurePath,))
                return cls(*args)
            if module == "re" and name == "compile":
                return re.compile(*args)
            raise SerializationError(f"blocked constructor: {module}.{name}")
        if code == EXT_CONSTRUCTOR_SINGLE_ARG:
            module, name, arg = value
            if module == "builtins":
                cls = _safe_builtin_class(module, name, (set, frozenset))
                return cls(arg)
            if module == "collections" and name == "deque":
                return deque(arg)
            if module == "uuid":
                return uuid.UUID(arg)
            if module == "decimal":
                return decimal.Decimal(arg)
            # Enums carry their real module/name (e.g. ``myapp.Color``), so
            # resolve them only from an already-loaded module and require the
            # target to actually be an Enum subclass.
            try:
                enum_cls = _loaded_enum_class(module, name)
            except SerializationError:
                raise SerializationError(f"blocked constructor: {module}.{name}") from None
            return enum_cls(arg)
        if code == EXT_CONSTRUCTOR_KW_ARGS:
            module, name, kwargs = value
            if module == "datetime" and name == "time":
                return dt.time(**kwargs)
            raise SerializationError(f"blocked keyword constructor: {module}.{name}")
        if code == EXT_METHOD_SINGLE_ARG:
            module, name, arg, method = value
            if (module, name, method) == ("datetime", "datetime", "fromisoformat"):
                return dt.datetime.fromisoformat(arg)
            raise SerializationError(f"blocked method call: {module}.{name}.{method}")
        if code in (EXT_PYDANTIC_V1, EXT_PYDANTIC_V2):
            module, name, dumped = value[:3]
            cls = _loaded_pydantic_class(module, name)
            if code == EXT_PYDANTIC_V1:
                try:
                    return cls(**dumped)
                except Exception:
                    return cls.construct(**dumped)
            return cls.model_validate(dumped)
        if code == EXT_DELTA_SNAPSHOT:
            return _restore_delta_snapshot(value)
        raise SerializationError(f"blocked msgpack extension code: {code}")


@dataclasses.dataclass(frozen=True)
class _DeltaSnapshot:
    value: Any


def _is_delta_snapshot(value: Any) -> bool:
    """Recognise only the engine or host delta snapshot marker."""
    if isinstance(value, _DeltaSnapshot):
        return True
    cls = type(value)
    return (
        cls.__name__ == "_DeltaSnapshot"
        and cls.__module__ == "langgraph.checkpoint.serde.types"
        and getattr(cls, "_fields", None) == ("value",)
    )


def _restore_delta_snapshot(value: Any) -> Any:
    """Restore the host marker when available, otherwise the native marker.

    The host module is already loaded whenever a host-created marker is being
    decoded; looking it up in ``sys.modules`` preserves the identity without
    adding LangGraph as an import-time dependency of the engine.
    """
    host_module = sys.modules.get("langgraph.checkpoint.serde.types")
    host_type = getattr(host_module, "_DeltaSnapshot", None) if host_module else None
    if isinstance(host_type, type) and getattr(host_type, "_fields", None) == ("value",):
        return host_type(value)
    return _DeltaSnapshot(value)


def _prepare_msgpack(value: Any, *, depth: int) -> Any:
    if depth > _MAX_DECODE_DEPTH:
        raise SerializationError("serialization nesting exceeds the safety limit")
    if value is None or isinstance(value, (str, bytes, int, float, bool, bytearray)):
        return value
    if _is_delta_snapshot(value):
        return _DeltaSnapshot(_prepare_msgpack(value.value, depth=depth + 1))
    if isinstance(value, tuple):
        # ``default`` only sees values msgpack cannot encode. A plain list would
        # erase the tuple identity, so keep a marker object that the encoder
        # receives after recursion has already converted its children.
        return _TupleMarker(
            tuple(_prepare_msgpack(item, depth=depth + 1) for item in value)
        )
    if isinstance(value, list):
        return [_prepare_msgpack(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            key: _prepare_msgpack(item, depth=depth + 1) for key, item in value.items()
        }
    return value


def _safe_builtin_class(module: str, name: str, expected: tuple[type, ...]) -> type:
    loaded = sys.modules.get(module)
    if loaded is None:
        raise SerializationError(f"module is not loaded: {module}")
    cls = getattr(loaded, name, None)
    if not isinstance(cls, type) or not issubclass(cls, expected):
        raise SerializationError(f"blocked class: {module}.{name}")
    return cls


def _loaded_enum_class(module: str, name: str) -> type[Enum]:
    """Resolve an enum class only from an already-loaded module."""
    loaded = sys.modules.get(module)
    if loaded is None:
        raise SerializationError(f"enum module is not loaded: {module}")
    cls = getattr(loaded, name, None)
    if not isinstance(cls, type) or not issubclass(cls, Enum):
        raise SerializationError(f"blocked enum class: {module}.{name}")
    return cls


def _loaded_pydantic_class(module: str, name: str) -> type[BaseModel]:
    loaded = sys.modules.get(module)
    if loaded is None:
        raise SerializationError(
            f"refusing to import module while deserializing: {module}"
        )
    cls = getattr(loaded, name, None)
    if cls is None and module == "__main__":
        # A script may define its model in the top-level namespace; this is
        # still an already-loaded module, never an import triggered by data.
        cls = globals().get(name)
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise SerializationError(f"blocked pydantic class: {module}.{name}")
    return cls


# -- JSON codec used by the long-term SQLite store -----------------------


def _json_encode(value: Any, *, depth: int) -> Any:
    if depth > _MAX_DECODE_DEPTH:
        raise SerializationError("JSON nesting exceeds the safety limit")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return _tag("bytes", base64.b64encode(value).decode("ascii"))
    if isinstance(value, bytearray):
        return _tag("bytearray", base64.b64encode(bytes(value)).decode("ascii"))
    if isinstance(value, dt.datetime):
        return _tag("datetime", value.isoformat())
    if isinstance(value, dt.date):
        return _tag("date", value.isoformat())
    if isinstance(value, dt.time):
        return _tag("time", value.isoformat())
    if isinstance(value, dt.timedelta):
        return _tag("timedelta", value.total_seconds())
    if isinstance(value, uuid.UUID):
        return _tag("uuid", str(value))
    if isinstance(value, decimal.Decimal):
        return _tag("decimal", str(value))
    if isinstance(value, tuple):
        return _tag("tuple", [_json_encode(v, depth=depth + 1) for v in value])
    if isinstance(value, (set, frozenset)):
        return _tag(
            type(value).__name__,
            [_json_encode(v, depth=depth + 1) for v in value],
        )
    if isinstance(value, deque):
        return _tag("deque", [_json_encode(v, depth=depth + 1) for v in value])
    if isinstance(value, Enum):
        return _tag(
            "enum",
            {
                "module": value.__class__.__module__,
                "name": value.__class__.__name__,
                "value": _json_encode(value.value, depth=depth + 1),
            },
        )
    if isinstance(value, BaseModel):
        return _tag(
            "pydantic",
            {
                "module": value.__class__.__module__,
                "name": value.__class__.__name__,
                "value": _json_encode(value.model_dump(), depth=depth + 1),
            },
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SerializationError("JSON store values require string dictionary keys")
        return {
            key: _json_encode(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_encode(item, depth=depth + 1) for item in value]
    raise SerializationError(
        f"cannot serialize {type(value).__module__}.{type(value).__qualname__} to JSON"
    )


def _json_decode(value: Any, *, depth: int) -> Any:
    if depth > _MAX_DECODE_DEPTH:
        raise SerializationError("JSON nesting exceeds the safety limit")
    if isinstance(value, list):
        return [_json_decode(item, depth=depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_JSON_TAG)
    if tag is not None and value.get("v") == _JSON_TAG_VERSION and set(value) == {
        _JSON_TAG,
        "v",
        "value",
    }:
        payload = value["value"]
        if tag == "bytes":
            return base64.b64decode(payload)
        if tag == "bytearray":
            return bytearray(base64.b64decode(payload))
        if tag == "datetime":
            return dt.datetime.fromisoformat(payload)
        if tag == "date":
            return dt.date.fromisoformat(payload)
        if tag == "time":
            return dt.time.fromisoformat(payload)
        if tag == "timedelta":
            return dt.timedelta(seconds=payload)
        if tag == "uuid":
            return uuid.UUID(payload)
        if tag == "decimal":
            return decimal.Decimal(payload)
        if tag == "tuple":
            return tuple(_json_decode(item, depth=depth + 1) for item in payload)
        if tag == "set":
            return set(_json_decode(item, depth=depth + 1) for item in payload)
        if tag == "frozenset":
            return frozenset(_json_decode(item, depth=depth + 1) for item in payload)
        if tag == "deque":
            return deque(_json_decode(item, depth=depth + 1) for item in payload)
        if tag == "enum":
            cls = _loaded_enum_class(payload["module"], payload["name"])
            return cls(_json_decode(payload["value"], depth=depth + 1))
        if tag == "pydantic":
            cls = _loaded_pydantic_class(payload["module"], payload["name"])
            return cls.model_validate(_json_decode(payload["value"], depth=depth + 1))
        raise SerializationError(f"unknown JSON type tag: {tag!r}")
    return {key: _json_decode(item, depth=depth + 1) for key, item in value.items()}


def _tag(name: str, value: Any) -> dict[str, Any]:
    return {_JSON_TAG: name, "v": _JSON_TAG_VERSION, "value": value}


__all__ = (
    "ReactiveSerializer",
    "SerializationError",
    "EXT_CONSTRUCTOR_SINGLE_ARG",
    "EXT_CONSTRUCTOR_POS_ARGS",
    "EXT_CONSTRUCTOR_KW_ARGS",
    "EXT_METHOD_SINGLE_ARG",
    "EXT_PYDANTIC_V1",
    "EXT_PYDANTIC_V2",
    "EXT_NUMPY_ARRAY",
    "EXT_DELTA_SNAPSHOT",
)

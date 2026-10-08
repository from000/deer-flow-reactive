"""RGP/1 wire protocol, Python side (docs/spec/rgp-1.md).

Mirrors packages/protocol/src/{messages,framing,codec}.ts. Byte-identical with the
TypeScript implementation for the shared fixture corpus.

Canonical values: None/bool/int (arbitrary precision, int64 range)/float/str/bytes/
list/dict (str keys)/datetime (msgpack timestamp ext -1)/decimal (custom ext type 0,
UTF-8 decimal string).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import msgpack

DECIMAL_EXT_TYPE = 0
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 32 * 1024 * 1024
HEADER_BYTES = 4

METHODS = frozenset(
    {
        "DRIVER_HELLO",
        "COMPILE_GRAPH",
        "RELEASE_GRAPH",
        "RUN",
        "RESUME",
        "GET_STATE",
        "EXPORT_DOT",
        "TASK_INVOKE",
        "TASK_RESULT",
        "STREAM_EVENT",
        "CHECKPOINT_OP",
        "STORE_OP",
        "VECTOR_UPSERT",
        "VECTOR_SEARCH",
        "TRACE_QUERY",
        "CANCEL",
        "SHUTDOWN",
    }
)

KINDS = frozenset({"request", "response", "event", "cancel"})


class ProtocolError(Exception):
    """Wire-level protocol violation (framing, unknown version, duplicate id...)."""


class CodecError(Exception):
    """A value is not representable in the RGP/1 canonical value set."""


@dataclass(frozen=True)
class Envelope:
    """Immutable RGP/1 envelope (docs/spec/rgp-1.md §2)."""

    id: str
    kind: str
    method: str
    payload: Any = None
    version: int = PROTOCOL_VERSION
    trace: dict | None = None

    def with_trace(self, **ctx: str) -> Envelope:
        """Return a copy carrying trace/telemetry metadata."""
        merged = dict(self.trace or {})
        merged.update(ctx)
        return Envelope(
            id=self.id,
            kind=self.kind,
            method=self.method,
            payload=self.payload,
            version=self.version,
            trace=merged,
        )


def is_method(value: Any) -> bool:
    """Whether *value* names a known protocol method."""
    return isinstance(value, str) and value in METHODS


def make_id() -> str:
    """Globally unique id within a session (uuid4 fallback, test-stable optional)."""
    import uuid

    return str(uuid.uuid4())


def is_envelope(value: Any) -> bool:
    """Whether *value* has the RGP/1 envelope shape."""
    if not isinstance(value, dict):
        return False
    if value.get("version") != PROTOCOL_VERSION:
        return False
    if not isinstance(value.get("id"), str) or not value["id"]:
        return False
    if not is_method(value.get("method")):
        return False
    if value.get("kind") not in KINDS:
        return False
    trace = value.get("trace")
    if trace is not None and not isinstance(trace, dict):
        return False
    return True


# --------------------------------------------------------------------------
# Canonical value codec
# --------------------------------------------------------------------------

_DECIMAL_TAG: dict[str, Any] = {"__type__": "decimal"}


def _is_decimal_marker(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("__type__") == "decimal"
        and isinstance(value.get("value"), str)
    )


def _assert_canonical(
    value: Any,
    path: str = "$",
    seen: set[int] | None = None,
    depth: int = 0,
) -> None:
    if depth > 256:
        raise CodecError(f"max depth exceeded at {path}")
    if value is None:
        return
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and value != value:  # NaN
            raise CodecError(f"non-finite number at {path}")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        return
    if isinstance(value, datetime):
        return
    if _is_decimal_marker(value):
        return
    if isinstance(value, (list, tuple)):
        if seen is None:
            seen = set()
        if id(value) in seen:
            raise CodecError(f"cyclic reference at {path}")
        seen.add(id(value))
        for i, item in enumerate(value):
            _assert_canonical(item, f"{path}[{i}]", seen, depth + 1)
        seen.remove(id(value))
        return
    if isinstance(value, dict):
        if seen is None:
            seen = set()
        if id(value) in seen:
            raise CodecError(f"cyclic reference at {path}")
        seen.add(id(value))
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodecError(f"non-string map key {key!r} at {path}")
            _assert_canonical(item, f"{path}.{key}", seen, depth + 1)
        seen.remove(id(value))
        return
    raise CodecError(f"non-canonical value of type {type(value).__name__} at {path}")


def _encode_decimal(value: str) -> msgpack.ExtType:
    return msgpack.ExtType(DECIMAL_EXT_TYPE, value.encode("utf-8"))


def _decode_decimal(data: bytes) -> dict[str, Any]:
    return {"__type__": "decimal", "value": data.decode("utf-8")}


def _ext_hook(code: int, data: bytes) -> Any:
    if code == DECIMAL_EXT_TYPE:
        return _decode_decimal(data)
    return msgpack.ExtType(code, data)


def encode_value(value: Any) -> bytes:
    """Encode a Python value into the wire representation."""
    _assert_canonical(value)
    return msgpack.packb(
        value,
        use_bin_type=True,
        datetime=True,
        default=_encode_decimal,
    )


def decode_value(data: bytes) -> Any:
    """Decode a wire value into the Python representation."""
    return msgpack.unpackb(
        data,
        raw=False,
        strict_map_key=True,
        timestamp=3,  # 3 = datetime
        ext_hook=_ext_hook,
        # Bound frame payload sizes: a malicious peer must not be able to
        # allocate unbounded structures (maps/arrays/strings) via one frame.
        max_map_len=1_000_000,
        max_array_len=1_000_000,
        max_str_len=32 * 1024 * 1024,
        max_bin_len=32 * 1024 * 1024,
    )


def canonicalize_envelope_payload(payload: Any) -> bytes:
    """Serialize the payload to canonical bytes (for byte-identity checks)."""
    return encode_value(payload)


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def encode_frame(payload: bytes) -> bytes:
    """[4-byte big-endian length][payload]."""
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"payload length {len(payload)} exceeds maxFrameBytes {MAX_FRAME_BYTES}"
        )
    return len(payload).to_bytes(HEADER_BYTES, "big") + payload


class FrameDecoder:
    """Incremental frame decoder over a byte stream (mirror of TS FrameDecoder)."""

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES) -> None:
        self.max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        """Accept one more response frame for matching."""
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            if len(self._buffer) < HEADER_BYTES:
                break
            length = int.from_bytes(self._buffer[:HEADER_BYTES], "big")
            if length > self.max_frame_bytes:
                self.reset()
                raise ProtocolError(
                    f"frame length {length} exceeds maxFrameBytes {self.max_frame_bytes}"
                )
            if len(self._buffer) < HEADER_BYTES + length:
                break
            frame = bytes(self._buffer[HEADER_BYTES : HEADER_BYTES + length])
            del self._buffer[: HEADER_BYTES + length]
            frames.append(frame)
        return frames

    def reset(self) -> None:
        """Discard buffered frames and start a fresh stream."""
        self._buffer.clear()
"""Host-boundary value transform: RGP/1 canonical form <-> Python objects.

The RGP/1 canonical value set (docs/spec/rgp-1.md §3) contains no class
instances, but real agent state is ``{"messages": [BaseMessage, ...]}``. The
Driver owns the state store, so messages must be representable on the wire.

Messages are encoded as a *tagged plain dict* rather than a new msgpack ext
type. Two reasons:

* the TypeScript codec needs no change, and the tagged dict is a plain map, so
  ``canonicalHash`` bytes are identical on both sides (this is what makes
  patch pre-image checks pass for message paths);
* an unknown ext type would be preserved opaquely by the Driver, which cannot
  hash or diff it consistently with Python.

The tag ``__type__: reactivegraph_message`` is reserved: hosts must not store
user data under that exact marker.
"""

from __future__ import annotations

from typing import Any

from reactivegraph.messages import BaseMessage, message_to_dict, messages_from_dict

__all__ = ("MESSAGE_WIRE_TAG", "from_wire_value", "to_wire_value")

MESSAGE_WIRE_TAG = "reactivegraph_message"
_TAGGED = "__type__"


def _encode_message(message: BaseMessage) -> dict[str, Any]:
    return {_TAGGED: MESSAGE_WIRE_TAG, "value": message_to_dict(message)}


def _decode_message(payload: Any) -> BaseMessage:
    if not isinstance(payload, dict):
        raise ValueError("reactivegraph_message payload must be a dict")
    (restored,) = messages_from_dict([payload])
    return restored


def to_wire_value(value: Any) -> Any:
    """Recursively convert Python values into RGP/1 canonical values.

    Only messages are transformed; every other value is passed through (or
    rebuilt for containers) so the canonical codec still validates it.
    """
    if isinstance(value, BaseMessage):
        return _encode_message(value)
    if isinstance(value, dict):
        return {key: to_wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire_value(item) for item in value]
    return value


def from_wire_value(value: Any) -> Any:
    """Inverse of :func:`to_wire_value` for decoded wire values."""
    if isinstance(value, dict):
        if value.get(_TAGGED) == MESSAGE_WIRE_TAG and "value" in value:
            return _decode_message(value["value"])
        return {key: from_wire_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_wire_value(item) for item in value]
    return value

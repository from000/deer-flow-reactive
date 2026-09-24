"""State-channel resolution for the ``create_agent`` compatibility surface.

LangGraph compiles a state schema into a table of channels: plain fields become
``LastValue``, two-argument callables in ``Annotated`` metadata become
``BinaryOperatorAggregate``, and channel *instances* in the metadata (DeerFlow
annotates ``messages`` with ``DeltaChannel`` in delta mode) are used as-is.

Hosts read that table back off the compiled graph. DeerFlow's checkpoint
mutation path calls ``graph_reducer_channels`` and wraps replacement writes in
``Overwrite`` for every entry it finds — with no table, reducer channels are
silently replaced instead of merged. The same table drives this engine's own
write folding, so a middleware write to a reducer field accumulates rather than
clobbering what an earlier turn recorded.

``langgraph`` is an optional integration dependency here, exactly as
``langchain_core`` is for :mod:`reactivegraph.messages`: when it is installed
the engine hands back the host's real channel classes, so ``isinstance`` checks
in the storage layer keep working. Without it, engine-native descriptors with
the same reducer semantics are used.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Required, get_type_hints

from reactivegraph.message_state import EphemeralValue as _EphemeralValueMarker

try:  # pragma: no cover - exercised by the optional-integration tests
    from langgraph import channels as _lg_channels
except ImportError:  # pragma: no cover - langgraph is an optional host dep
    _lg_channels = None


__all__ = ("resolve_channels", "channel_reducer")


@dataclass(frozen=True)
class NativeLastValue:
    """Engine-native stand-in for ``langgraph.channels.LastValue``."""

    typ: Any
    key: str = ""


@dataclass(frozen=True)
class NativeBinaryOperatorAggregate:
    """Engine-native stand-in for ``langgraph.channels.BinaryOperatorAggregate``."""

    typ: Any
    operator: Callable[[Any, Any], Any]
    key: str = ""


def _unwrap(annotation: Any) -> Any:
    """Strip ``Required``/``NotRequired`` wrappers (LangGraph does the same)."""
    while getattr(annotation, "__origin__", None) in (Required,):
        annotation = annotation.__args__[0]
    from typing_extensions import NotRequired

    while getattr(annotation, "__origin__", None) in (NotRequired,):
        annotation = annotation.__args__[0]
    return annotation


def _is_two_arg_callable(value: Any) -> bool:
    if not callable(value) or isinstance(value, type):
        return False
    import inspect

    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) == 2


def _channel_for(name: str, annotation: Any) -> Any:
    annotation = _unwrap(annotation)
    metadata = getattr(annotation, "__metadata__", ())
    typ = getattr(annotation, "__origin__", annotation)

    if _lg_channels is not None:
        for item in metadata:
            if isinstance(item, _lg_channels.BaseChannel):
                channel = copy.copy(item)
                if hasattr(channel, "key"):
                    channel.key = name
                return channel
            if isinstance(item, type) and issubclass(item, _lg_channels.BaseChannel):
                return item(typ)
            if item is _EphemeralValueMarker or isinstance(item, _EphemeralValueMarker):
                return _lg_channels.EphemeralValue(typ)
        if metadata and _is_two_arg_callable(metadata[-1]):
            return _lg_channels.BinaryOperatorAggregate(typ, metadata[-1])
        return _lg_channels.LastValue(typ)

    for item in metadata:
        if item is _EphemeralValueMarker or isinstance(item, _EphemeralValueMarker):
            return NativeLastValue(typ=typ, key=name)
    if metadata and _is_two_arg_callable(metadata[-1]):
        return NativeBinaryOperatorAggregate(typ=typ, operator=metadata[-1], key=name)
    return NativeLastValue(typ=typ, key=name)


def resolve_channels(schemas: list[type]) -> dict[str, Any]:
    """Merge *schemas* in order and compile their channel table.

    Later schemas win on field conflicts, matching the upstream
    ``middleware schemas first, base schema last`` resolution order.
    """
    resolved: dict[str, Any] = {}
    for schema in schemas:
        # ``get_type_hints`` is deliberately not wrapped: a schema whose
        # annotations cannot be resolved must raise here, exactly as upstream's
        # ``_get_channels`` does. Falling back to raw (string) annotations
        # would silently compile every reducer to last-value and drop writes.
        hints = get_type_hints(schema, include_extras=True)
        for field_name, annotation in hints.items():
            if field_name == "__slots__":
                continue
            resolved[field_name] = _channel_for(field_name, annotation)
    return resolved


def channel_reducer(channel: Any) -> Callable[[Any, Any], Any] | None:
    """Return the two-argument fold for *channel*, or ``None`` for last-value.

    ``BinaryOperatorAggregate`` folds a single write against the current value.
    ``DeltaChannel``'s reducer takes the whole *sequence* of writes for a step,
    so it is adapted to the same one-write signature here.
    """
    if _lg_channels is not None:
        if isinstance(channel, _lg_channels.DeltaChannel):
            reducer = channel.reducer

            def fold_delta(current: Any, write: Any) -> Any:
                return reducer(current, [write])

            return fold_delta
        if isinstance(channel, _lg_channels.BinaryOperatorAggregate):
            return channel.operator
        return None
    if isinstance(channel, NativeBinaryOperatorAggregate):
        return channel.operator
    return None

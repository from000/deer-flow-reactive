"""Store batch operations and their duck-typed accessors.

The backend implementations share one operation vocabulary (``GetOp``,
``SearchOp``, ``PutOp``, ``ListNamespacesOp``) and accept host-provided
namedtuples that look the same. The ``_op_*`` helpers read either shape, which
is why they live together with the dataclasses rather than inside a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GetOp:
    """Retrieve one item by ``(namespace, key)``."""

    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True


@dataclass(frozen=True)
class SearchOp:
    """Search within a namespace prefix, with optional filter and ranking."""

    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool = True


@dataclass(frozen=True)
class PutOp:
    """Write ``value`` (or delete when ``value is None``) at a key."""

    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None = None
    index: Any = None
    ttl: Any = None


@dataclass(frozen=True)
class MatchCondition:
    """A prefix or suffix namespace pattern; ``*`` matches any one label."""

    match_type: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class ListNamespacesOp:
    """List namespaces matching all conditions, truncated to ``max_depth``."""

    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0


_OWN_OPS = (GetOp, PutOp, SearchOp, ListNamespacesOp)

# Tuple form: ``(kind, namespace, key, value)`` — used by callers that build
# operations inline. Namespace may be a tuple or a list.
_TUPLE_KINDS = {"get", "put", "search", "delete", "list_namespaces"}


def _op_kind(op: Any) -> str:
    """Classify an operation, accepting our own objects, a host's, or tuples."""
    if isinstance(op, GetOp):
        return "get"
    if isinstance(op, PutOp):
        return "put"
    if isinstance(op, SearchOp):
        return "search"
    if isinstance(op, ListNamespacesOp):
        return "list_namespaces"
    # Foreign (host) operation objects: classify by class name so a host can
    # pass its own GetOp/PutOp/SearchOp/ListNamespacesOp without conversion.
    # Checked BEFORE the tuple branch because a host's op may be a NamedTuple.
    name = type(op).__name__
    if name == "GetOp":
        return "get"
    if name == "PutOp":
        return "put"
    if name == "SearchOp":
        return "search"
    if name == "ListNamespacesOp":
        return "list_namespaces"
    # Inline tuple form: ``("put", namespace, key, value)``.
    if type(op) is tuple:
        if op and isinstance(op[0], str) and op[0] in _TUPLE_KINDS:
            return op[0]
        raise ValueError(f"Unknown operation type: {type(op)}")
    raise ValueError(f"Unknown operation type: {type(op)}")


def _op_attr(op: Any, name: str, default: Any) -> Any:
    if type(op) is tuple:
        return default
    return getattr(op, name, default)


def _op_namespace(op: Any) -> tuple[str, ...]:
    if type(op) is tuple:
        return tuple(op[1])
    raw = getattr(op, "namespace", None)
    if raw is None:
        raw = getattr(op, "namespace_prefix", ())
    return tuple(raw or ())


def _op_key(op: Any) -> str:
    if type(op) is tuple:
        return str(op[2]) if len(op) > 2 else ""
    return str(getattr(op, "key", ""))


def _op_value(op: Any) -> Any:
    if type(op) is tuple:
        return op[3] if len(op) > 3 else None
    return getattr(op, "value", None)


def _op_filter(op: Any) -> dict[str, Any] | None:
    return _op_attr(op, "filter", None)


def _op_ttl(op: Any) -> Any:
    """Normalise a host's NOT_PROVIDED sentinel to None (no TTL)."""
    ttl = _op_attr(op, "ttl", None)
    if ttl is None or type(ttl).__name__ == "NotProvided":
        return None
    return ttl





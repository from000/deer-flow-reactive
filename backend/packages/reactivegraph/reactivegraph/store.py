"""Long-term key-value store for ReactiveGraph.

This module owns the *storage* contract for the engine's long-term memory:
namespaced items with searchable filters, plus a namespace listing API. It is
deliberately engine-agnostic and imports nothing from LangChain/LangGraph.

A host engine that demands a specific store interface adapts
:class:`MemoryStore` in its own package. Keeping the semantics here means every
host shares one tested implementation instead of re-deriving it.

Semantics are pinned against the real ``langgraph.store.memory.InMemoryStore``
by differential probing, including its quirks:

* ``get`` on an unknown namespace *materialises* that namespace for
  ``list_namespaces`` (upstream uses a ``defaultdict``);
* a repeated ``put`` resets both ``created_at`` and ``updated_at``;
* ``batch`` dedupes ``PutOp``s by ``(namespace, key)``, last write wins;
* a ``search`` with no query returns items in insertion order and slices by
  ``offset``/``limit`` *after* filtering.

Scope today is the in-process ``memory`` backend. Durable backends fail closed
in :func:`get_store` rather than silently degrading to process-local state,
because that would be data loss presented as durability.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = (
    "BaseStore",
    "InvalidNamespaceError",
    "Item",
    "MemoryStore",
    "SearchItem",
    "get_store",
    "reset_store",
)


class InvalidNamespaceError(ValueError):
    """Provided namespace is invalid."""


def _validate_namespace(namespace: Sequence[str]) -> None:
    """Reject namespaces upstream rejects, with the same messages."""
    if not namespace:
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in namespace:
        if not isinstance(label, str):
            raise InvalidNamespaceError(
                f"Invalid namespace label '{label}' found in {namespace}. Namespace labels"
                f" must be strings, but got {type(label).__name__}."
            )
        if "." in label:
            raise InvalidNamespaceError(
                f"Invalid namespace label '{label}' found in {namespace}. "
                f"Namespace labels cannot contain periods ('.')."
            )
        if not label:
            raise InvalidNamespaceError(
                f"Namespace labels cannot be empty strings. Got {label} in {namespace}"
            )
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError(
            f'Root label for namespace cannot be "langgraph". Got: {namespace}'
        )


class _NativeBaseStore:
    """LangGraph's ``BaseStore`` surface, implemented natively.

    Used only when ``langgraph`` cannot be imported. As with the checkpoint
    saver, unimplemented operations raise instead of silently reporting an
    empty namespace: a host that forgets to configure a backend must see a
    failure, not data loss. Concrete stores (including :class:`MemoryStore`)
    are duck-typed and need not inherit from this class.
    """

    supports_ttl: bool = False
    ttl_config: Any = None

    def get(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        del namespace, key, refresh_ttl
        raise NotImplementedError

    def put(
        self,
        namespace: Sequence[str],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        del namespace, key, value, index, ttl
        raise NotImplementedError

    def delete(self, namespace: Sequence[str], key: str) -> None:
        del namespace, key
        raise NotImplementedError

    def search(
        self,
        namespace_prefix: Sequence[str],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        del namespace_prefix, query, filter, limit, offset, refresh_ttl
        raise NotImplementedError

    def list_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        del prefix, suffix, max_depth, limit, offset
        raise NotImplementedError

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        del ops
        raise NotImplementedError

    async def aget(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        del namespace, key, refresh_ttl
        raise NotImplementedError

    async def aput(
        self,
        namespace: Sequence[str],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        del namespace, key, value, index, ttl
        raise NotImplementedError

    async def adelete(self, namespace: Sequence[str], key: str) -> None:
        del namespace, key
        raise NotImplementedError

    async def asearch(
        self,
        namespace_prefix: Sequence[str],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        del namespace_prefix, query, filter, limit, offset, refresh_ttl
        raise NotImplementedError

    async def alist_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        del prefix, suffix, max_depth, limit, offset
        raise NotImplementedError

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        del ops
        raise NotImplementedError


try:  # pragma: no cover - exercised by the optional-integration tests
    from langgraph.store.base import BaseStore as BaseStore
except ImportError:  # pragma: no cover - langgraph is an optional host dep
    BaseStore = _NativeBaseStore  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class Item:
    """A stored item with its identity and timestamps.

    ``namespace`` is normalised to a tuple so deserialised payloads behave like
    freshly written ones.
    """

    value: dict[str, Any]
    key: str
    namespace: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", tuple(self.namespace))

    def __hash__(self) -> int:
        # Upstream hashes identity, not payload: a mutable ``value`` dict must
        # not make the item unhashable.
        return hash((self.namespace, self.key))

    def dict(self) -> dict[str, Any]:
        return {
            "namespace": list(self.namespace),
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class SearchItem(Item):
    """An :class:`Item` returned from ``search``, optionally ranked."""

    score: float | None = None

    def dict(self) -> dict[str, Any]:
        result = super().dict()
        result["score"] = self.score
        return result


def _does_match(match_type: str, path: Sequence[str], namespace: Sequence[str]) -> bool:
    """Prefix/suffix namespace match with ``*`` wildcards."""
    if len(namespace) < len(path):
        return False
    if match_type == "prefix":
        pairs = zip(namespace, path, strict=False)
    elif match_type == "suffix":
        pairs = zip(reversed(namespace), reversed(path), strict=False)
    else:
        raise ValueError(f"Unsupported match type: {match_type}")
    for ns_elem, path_elem in pairs:
        if path_elem == "*":
            continue
        if ns_elem != path_elem:
            return False
    return True


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compare values JSONB-style, including ``$`` operators and nested dicts."""
    if isinstance(filter_value, dict):
        if any(key.startswith("$") for key in filter_value):
            return all(
                _apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(
            _compare_values(item_value.get(key), value)
            for key, value in filter_value.items()
        )
    if isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                _compare_values(item_elem, filter_elem)
                for item_elem, filter_elem in zip(item_value, filter_value, strict=False)
            )
        )
    return item_value == filter_value


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Apply a comparison operator, matching PostgreSQL's JSONB behaviour."""
    if operator == "$eq":
        return value == op_value
    if operator == "$gt":
        return float(value) > float(op_value)
    if operator == "$gte":
        return float(value) >= float(op_value)
    if operator == "$lt":
        return float(value) < float(op_value)
    if operator == "$lte":
        return float(value) <= float(op_value)
    if operator == "$ne":
        return value != op_value
    raise ValueError(f"Unsupported operator: {operator}")


class MemoryStore:
    """In-process namespaced key-value store with structured search.

    Not vector-search enabled: an ``index`` argument is accepted and ignored,
    exactly like upstream when no index configuration is supplied. TTL is
    unsupported and fails closed rather than pretending to expire items.
    """

    supports_ttl: bool = False
    ttl_config: dict[str, Any] | None = None

    def __init__(self, *, index: Any | None = None) -> None:
        # ``index`` is accepted for signature compatibility with hosts that
        # pass an index config; without an embedding function it is inert.
        self.index_config = index
        self._data: dict[tuple[str, ...], dict[str, Item]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Batch core
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute operations in order; ``PutOp``s dedupe last-write-wins."""
        return self._run(ops)

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return self._run(ops)

    def _run(self, ops: Iterable[Any]) -> list[Any]:
        materialised = list(ops)
        results: list[Any] = [None] * len(materialised)
        puts: dict[tuple[tuple[str, ...], str], tuple[Any, ...]] = {}
        searches: list[tuple[int, tuple[Any, ...]]] = []

        for index, op in enumerate(materialised):
            kind = _op_kind(op)
            if kind == "get":
                namespace, key = _op_namespace(op), _op_key(op)
                with self._lock:
                    # Upstream touches the defaultdict, so a read of an unknown
                    # namespace registers it for list_namespaces().
                    results[index] = self._data.setdefault(namespace, {}).get(key)
            elif kind == "put":
                namespace, key = _op_namespace(op), _op_key(op)
                _validate_namespace(namespace)
                self._check_ttl(_op_ttl(op))
                puts[(namespace, key)] = (namespace, key, _op_value(op), index)
            elif kind == "search":
                searches.append((index, op))
            elif kind == "list_namespaces":
                results[index] = self._list_namespaces(op)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        # Writes are applied after reads (upstream order), deduped last-wins.
        with self._lock:
            now = datetime.now(timezone.utc)
            for namespace, key, value, _index in puts.values():
                if value is None:
                    self._data.setdefault(namespace, {}).pop(key, None)
                else:
                    self._data.setdefault(namespace, {})[key] = Item(
                        value=copy.deepcopy(value),
                        key=key,
                        namespace=namespace,
                        created_at=now,
                        updated_at=now,
                    )

        for index, op in searches:
            results[index] = self._search(op)
        return results

    def _check_ttl(self, ttl: Any) -> None:
        if ttl is None:
            return
        raise NotImplementedError(
            f"TTL is not supported by {type(self).__name__}. "
            f"Use a store implementation that supports TTL or set ttl=None."
        )

    # ------------------------------------------------------------------
    # Search / listing
    # ------------------------------------------------------------------

    def _search(self, op: Any) -> list[SearchItem]:
        prefix = _op_namespace(op)
        filter_dict = _op_filter(op)
        limit = _op_attr(op, "limit", 10)
        offset = _op_attr(op, "offset", 0)
        query = _op_attr(op, "query", None)

        with self._lock:
            candidates: list[Item] = []
            for namespace, bucket in self._data.items():
                if not _does_match("prefix", prefix, namespace):
                    continue
                for item in bucket.values():
                    if filter_dict and not all(
                        _compare_values(item.value.get(key), value)
                        for key, value in filter_dict.items()
                    ):
                        continue
                    candidates.append(item)

        if query:
            # No embedding configuration: upstream falls back to the unranked
            # branch, returning score=None in insertion order.
            del query
        window = candidates[offset : offset + limit]
        return [
            SearchItem(
                value=item.value,
                key=item.key,
                namespace=item.namespace,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=None,
            )
            for item in window
        ]

    def _list_namespaces(self, op: Any) -> list[tuple[str, ...]]:
        conditions = _op_attr(op, "match_conditions", None) or ()
        max_depth = _op_attr(op, "max_depth", None)
        limit = _op_attr(op, "limit", 100)
        offset = _op_attr(op, "offset", 0)

        with self._lock:
            namespaces = list(self._data.keys())
        if conditions:
            namespaces = [
                namespace
                for namespace in namespaces
                if all(
                    _does_match(condition.match_type, condition.path, namespace)
                    for condition in conditions
                )
            ]
        if max_depth is not None:
            namespaces = sorted({namespace[:max_depth] for namespace in namespaces})
        else:
            namespaces = sorted(namespaces)
        return namespaces[offset : offset + limit]

    # ------------------------------------------------------------------
    # Convenience API (mirrors the host store surface)
    # ------------------------------------------------------------------

    def get(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        del refresh_ttl
        return self.batch([GetOp(tuple(namespace), str(key))])[0]

    async def aget(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        return self.get(namespace, key, refresh_ttl=refresh_ttl)

    def search(
        self,
        namespace_prefix: Sequence[str],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        del refresh_ttl
        return self.batch([
            SearchOp(tuple(namespace_prefix), filter, limit, offset, query)
        ])[0]

    async def asearch(
        self,
        namespace_prefix: Sequence[str],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return self.search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    def put(
        self,
        namespace: Sequence[str],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        del index
        self.batch([PutOp(tuple(namespace), str(key), value, ttl=ttl)])

    async def aput(
        self,
        namespace: Sequence[str],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        self.put(namespace, key, value, index, ttl=ttl)

    def delete(self, namespace: Sequence[str], key: str) -> None:
        self.batch([PutOp(tuple(namespace), str(key), None, ttl=None)])

    async def adelete(self, namespace: Sequence[str], key: str) -> None:
        self.delete(namespace, key)

    def list_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        conditions = []
        if prefix:
            conditions.append(MatchCondition("prefix", tuple(prefix)))
        if suffix:
            conditions.append(MatchCondition("suffix", tuple(suffix)))
        return self.batch([
            ListNamespacesOp(tuple(conditions), max_depth, limit, offset)
        ])[0]

    async def alist_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return self.list_namespaces(
            prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset
        )


# ---------------------------------------------------------------------------
# Operations
#
# These mirror the host engines' operation tuples so ``batch`` accepts both our
# own objects and a host's (which are matched structurally, see ``_op_kind``).
# ---------------------------------------------------------------------------


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
    return tuple(raw)


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


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_store() -> MemoryStore:
    """Return the process-wide store, creating it on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryStore()
        return _store


def reset_store() -> None:
    """Drop the process-wide store (tests, config reload)."""
    global _store
    with _store_lock:
        _store = None

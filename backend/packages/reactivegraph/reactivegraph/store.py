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

Native ``memory``, ``sqlite``, and ``postgres`` backends are provided.
:func:`get_store` rejects unknown backends rather than silently degrading to
process-local state, because that would be data loss presented as durability.
"""

from __future__ import annotations

import asyncio
import copy
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

__all__ = (
    "BaseStore",
    "InvalidNamespaceError",
    "Item",
    "MemoryStore",
    "SearchItem",
    "SqliteStore",
    "PostgresStore",
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
    BaseStore = _NativeBaseStore  # type: ignore[misc, assignment, unused-ignore]


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
                results[index] = self._search(op)
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


class SqliteStore:
    """Durable namespaced key-value store backed by SQLite.

    The table layout is compatible with LangGraph's SQLite store, while values
    use ReactiveGraph's typed, no-pickle JSON codec. TTL is implemented natively:
    expired rows are hidden from reads and searches, and may be refreshed when
    ``refresh_ttl`` is requested.
    """

    supports_ttl: bool = True
    ttl_config: dict[str, Any] = {"refresh_on_read": True}

    def __init__(
        self,
        conn_string: str | Path = ":memory:",
        *,
        index: Any | None = None,
        serde: Any | None = None,
    ) -> None:
        from reactivegraph.serde import ReactiveSerializer

        self.conn_string = str(conn_string)
        if self.conn_string != ":memory:":
            Path(self.conn_string).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self.index_config = index
        self.serde = serde if serde is not None else ReactiveSerializer()
        self.conn = sqlite3.connect(
            self.conn_string,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self._db_lock = threading.RLock()
        self._closed = False
        self.setup()

    def setup(self) -> None:
        """Create or migrate the LangGraph-compatible store schema."""
        with self._db_lock:
            self._ensure_open()
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS store (
                    prefix TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    ttl_minutes REAL,
                    PRIMARY KEY (prefix, key)
                );
                """
            )
            # A database created by an older LangGraph version may not have the
            # TTL columns yet. Migrate before creating an index over the column;
            # SQLite validates index expressions against the current schema.
            columns = {
                row["name"] for row in self.conn.execute("PRAGMA table_info(store)")
            }
            if "expires_at" not in columns:
                self.conn.execute("ALTER TABLE store ADD COLUMN expires_at TIMESTAMP")
            if "ttl_minutes" not in columns:
                self.conn.execute("ALTER TABLE store ADD COLUMN ttl_minutes REAL")
            self.conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS store_prefix_idx ON store (prefix);
                CREATE INDEX IF NOT EXISTS idx_store_expires_at
                    ON store (expires_at) WHERE expires_at IS NOT NULL;
                """
            )
            # LangGraph records schema versions here. Recording the migrations
            # we already materialised prevents its setup from replaying ALTERs
            # against an existing database.
            self.conn.execute(
                "INSERT OR IGNORE INTO store_migrations (v) VALUES (0), (1), (2), (3), (4)"
            )

    def close(self) -> None:
        with self._db_lock:
            if not self._closed:
                self.conn.close()
                self._closed = True

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    # ------------------------------------------------------------------
    # Batch core
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute store operations, preserving upstream ordering semantics."""
        return self._run(ops)

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return await asyncio.to_thread(self._run, list(ops))

    def _run(self, ops: Iterable[Any]) -> list[Any]:
        materialised = list(ops)
        results: list[Any] = [None] * len(materialised)
        puts: dict[tuple[tuple[str, ...], str], tuple[Any, ...]] = {}

        # Reads are resolved before writes, matching MemoryStore and upstream's
        # grouped batch implementation.
        for index, op in enumerate(materialised):
            kind = _op_kind(op)
            if kind == "get":
                results[index] = self._get_item(
                    _op_namespace(op),
                    _op_key(op),
                    refresh_ttl=bool(_op_attr(op, "refresh_ttl", True)),
                )
            elif kind == "put":
                namespace, key = _op_namespace(op), _op_key(op)
                _validate_namespace(namespace)
                self._normalise_ttl(_op_ttl(op))
                puts[(namespace, key)] = (namespace, key, _op_value(op), index)
            elif kind == "search":
                results[index] = self._search(op)
            elif kind == "list_namespaces":
                results[index] = self._list_namespaces(op)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        with self._db_lock:
            self._ensure_open()
            now = datetime.now(timezone.utc)
            with self.conn:
                for namespace, key, value, _index in puts.values():
                    prefix = self._namespace_to_text(namespace)
                    if value is None:
                        self.conn.execute(
                            "DELETE FROM store WHERE prefix = ? AND key = ?",
                            (prefix, key),
                        )
                        continue
                    ttl = self._normalise_ttl(_op_ttl(materialised[_index]))
                    expires_at = self._expiry_text(now, ttl)
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO store
                            (prefix, key, value, created_at, updated_at,
                             expires_at, ttl_minutes)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prefix,
                            key,
                            self.serde.dumps_json(value),
                            now.isoformat(),
                            now.isoformat(),
                            expires_at,
                            ttl,
                        ),
                    )

        return results

    @staticmethod
    def _normalise_ttl(ttl: Any) -> float | None:
        if ttl is None:
            return None
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
            raise TypeError("ttl must be a number of minutes or None")
        return float(ttl)

    @staticmethod
    def _namespace_to_text(namespace: Sequence[str]) -> str:
        return ".".join(namespace)

    @staticmethod
    def _expiry_text(now: datetime, ttl: Any) -> str | None:
        if ttl is None:
            return None
        # Keep SQLite's textual CURRENT_TIMESTAMP ordering intact. isoformat()
        # uses a ``T`` separator, which sorts after the space SQLite emits.
        return str(now + timedelta(minutes=float(ttl)))

    @staticmethod
    def _decode_namespace(prefix: str) -> tuple[str, ...]:
        return tuple(prefix.split("."))

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _expires_at(self, row: sqlite3.Row) -> float | None:
        value = row["expires_at"]
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return self._parse_datetime(value).timestamp()

    def _row_to_item(
        self, row: sqlite3.Row, *, search: bool = False
    ) -> Item | SearchItem:
        item_type = SearchItem if search else Item
        kwargs: dict[str, Any] = {
            "value": self.serde.loads_json(row["value"]),
            "key": row["key"],
            "namespace": self._decode_namespace(row["prefix"]),
            "created_at": self._parse_datetime(row["created_at"]),
            "updated_at": self._parse_datetime(row["updated_at"]),
        }
        if item_type is SearchItem:
            kwargs["score"] = None
        return item_type(**kwargs)

    def _get_item(
        self,
        namespace: Sequence[str],
        key: str,
        *,
        refresh_ttl: bool = True,
    ) -> Item | None:
        _validate_namespace(namespace)
        prefix = self._namespace_to_text(namespace)
        with self._db_lock:
            self._ensure_open()
            row = self.conn.execute(
                """
                SELECT prefix, key, value, created_at, updated_at,
                       expires_at, ttl_minutes
                FROM store WHERE prefix = ? AND key = ?
                """,
                (prefix, str(key)),
            ).fetchone()
            if row is None:
                return None
            expires_at = self._expires_at(row)
            now = datetime.now(timezone.utc).timestamp()
            if expires_at is not None and expires_at <= now:
                return None
            if (
                refresh_ttl
                and expires_at is not None
                and row["ttl_minutes"] is not None
                and self.ttl_config.get("refresh_on_read", False)
            ):
                refreshed = self._expiry_text(
                    datetime.fromtimestamp(now, tz=timezone.utc),
                    row["ttl_minutes"],
                )
                assert refreshed is not None
                self.conn.execute(
                    "UPDATE store SET expires_at = ? WHERE prefix = ? AND key = ?",
                    (refreshed, prefix, str(key)),
                )
            return self._row_to_item(row)

    # ------------------------------------------------------------------
    # Search / listing
    # ------------------------------------------------------------------

    def _search(self, op: Any) -> list[SearchItem]:
        prefix = _op_namespace(op)
        filter_dict = _op_filter(op)
        limit = int(_op_attr(op, "limit", 10))
        offset = int(_op_attr(op, "offset", 0))
        query = _op_attr(op, "query", None)
        refresh_ttl = bool(_op_attr(op, "refresh_ttl", True))

        candidates: list[SearchItem] = []
        with self._db_lock:
            self._ensure_open()
            rows = self.conn.execute(
                """
                SELECT prefix, key, value, created_at, updated_at,
                       expires_at, ttl_minutes
                FROM store ORDER BY updated_at DESC, rowid DESC
                """
            ).fetchall()
            now = datetime.now(timezone.utc).timestamp()
            for row in rows:
                namespace = self._decode_namespace(row["prefix"])
                if not _does_match("prefix", prefix, namespace):
                    continue
                expires_at = self._expires_at(row)
                if expires_at is not None and expires_at <= now:
                    continue
                value = self.serde.loads_json(row["value"])
                if filter_dict and not all(
                    _compare_values(value.get(key), expected)
                    for key, expected in filter_dict.items()
                ):
                    continue
                if (
                    refresh_ttl
                    and expires_at is not None
                    and row["ttl_minutes"] is not None
                    and self.ttl_config.get("refresh_on_read", False)
                ):
                    self.conn.execute(
                        "UPDATE store SET expires_at = ? WHERE prefix = ? AND key = ?",
                        (
                            self._expiry_text(
                                datetime.fromtimestamp(now, tz=timezone.utc),
                                row["ttl_minutes"],
                            ),
                            row["prefix"],
                            row["key"],
                        ),
                    )
                candidates.append(
                    cast("SearchItem", self._row_to_item(row, search=True))
                )

        if query:
            # No embedding configuration: upstream returns the unranked branch.
            del query
        return candidates[offset : offset + limit]

    def _list_namespaces(self, op: Any) -> list[tuple[str, ...]]:
        conditions = _op_attr(op, "match_conditions", None) or ()
        max_depth = _op_attr(op, "max_depth", None)
        limit = int(_op_attr(op, "limit", 100))
        offset = int(_op_attr(op, "offset", 0))
        now = datetime.now(timezone.utc).timestamp()

        with self._db_lock:
            self._ensure_open()
            rows = self.conn.execute(
                "SELECT DISTINCT prefix, expires_at FROM store"
            ).fetchall()
        namespaces: set[tuple[str, ...]] = set()
        for row in rows:
            expires_at = self._expires_at(row)
            if expires_at is None or expires_at > now:
                namespaces.add(self._decode_namespace(row["prefix"]))
        if conditions:
            namespaces = {
                namespace
                for namespace in namespaces
                if all(
                    _does_match(condition.match_type, condition.path, namespace)
                    for condition in conditions
                )
            }
        if max_depth is not None:
            values = sorted({namespace[:max_depth] for namespace in namespaces})
        else:
            values = sorted(namespaces)
        return values[offset : offset + limit]

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def get(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        return self._get_item(
            namespace,
            str(key),
            refresh_ttl=True if refresh_ttl is None else refresh_ttl,
        )

    async def aget(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        return await asyncio.to_thread(self.get, namespace, key, refresh_ttl=refresh_ttl)

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
        await asyncio.to_thread(self.put, namespace, key, value, index, ttl=ttl)

    def delete(self, namespace: Sequence[str], key: str) -> None:
        self.batch([PutOp(tuple(namespace), str(key), None, ttl=None)])

    async def adelete(self, namespace: Sequence[str], key: str) -> None:
        await asyncio.to_thread(self.delete, namespace, key)

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
        return self.batch([
            SearchOp(
                tuple(namespace_prefix),
                filter,
                limit,
                offset,
                query,
                True if refresh_ttl is None else refresh_ttl,
            )
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
        return await asyncio.to_thread(
            self.search,
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

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
        return await asyncio.to_thread(
            self.list_namespaces,
            prefix=prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLite store is closed")



class PostgresStore:
    """Durable PostgreSQL namespaced store using ReactiveGraph serialization."""

    supports_ttl: bool = True
    ttl_config: dict[str, Any] = {"refresh_on_read": True}

    def __init__(
        self,
        conn_string: str,
        *,
        schema: str = "public",
        index: Any | None = None,
        serde: Any | None = None,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional production extra
            raise ImportError(
                "psycopg is required for the PostgreSQL store. "
                "Install it with: pip install 'reactivegraph[postgres]'"
            ) from exc

        from reactivegraph.serde import ReactiveSerializer

        self.conn_string = str(conn_string)
        self.schema = str(schema or "public")
        self.index_config = index
        self.serde = serde if serde is not None else ReactiveSerializer()
        self.conn = psycopg.connect(self.conn_string, autocommit=False)
        self._db_lock = threading.RLock()
        self._closed = False
        self.setup()

    def setup(self) -> None:
        with self._db_lock:
            self._ensure_open()
            with self.conn.transaction():
                # CREATE SCHEMA IF NOT EXISTS still races in PostgreSQL:
                # concurrent transactions can both observe a missing
                # namespace, then one fails on the unique catalog index.
                # Serialize bootstrap per target schema with a lock that
                # is released automatically when this transaction ends.
                self.conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                    ("reactivegraph_schema_bootstrap", self.schema),
                )
                self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                self.conn.execute(f'SET search_path TO "{self.schema}"')
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS store (
                        prefix TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ,
                        ttl_minutes DOUBLE PRECISION,
                        PRIMARY KEY (prefix, key)
                    )
                    """
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS store_prefix_idx ON store (prefix)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_store_expires_at "
                    "ON store (expires_at) WHERE expires_at IS NOT NULL"
                )

    def close(self) -> None:
        with self._db_lock:
            if not self._closed:
                self.conn.close()
                self._closed = True

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()):
        if self._closed:
            raise RuntimeError("PostgreSQL store is closed")
        self.conn.execute(f'SET search_path TO "{self.schema}"')
        return self.conn.execute(sql, tuple(params))

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        return self._run(ops)

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return await asyncio.to_thread(self._run, list(ops))

    def _run(self, ops: Iterable[Any]) -> list[Any]:
        materialised = list(ops)
        results: list[Any] = [None] * len(materialised)
        puts: dict[tuple[tuple[str, ...], str], tuple[Any, ...]] = {}
        for index, op in enumerate(materialised):
            kind = _op_kind(op)
            if kind == "get":
                results[index] = self._get_item(
                    _op_namespace(op),
                    _op_key(op),
                    refresh_ttl=bool(_op_attr(op, "refresh_ttl", True)),
                )
            elif kind == "put":
                namespace, key = _op_namespace(op), _op_key(op)
                _validate_namespace(namespace)
                self._normalise_ttl(_op_ttl(op))
                puts[(namespace, key)] = (namespace, key, _op_value(op), index)
            elif kind == "search":
                results[index] = self._search(op)
            elif kind == "list_namespaces":
                results[index] = self._list_namespaces(op)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        now = datetime.now(timezone.utc)
        with self._db_lock:
            self._ensure_open()
            with self.conn.transaction():
                for namespace, key, value, index in puts.values():
                    prefix = self._namespace_to_text(namespace)
                    if value is None:
                        self._execute(
                            "DELETE FROM store WHERE prefix = %s AND key = %s",
                            (prefix, key),
                        )
                        continue
                    ttl = self._normalise_ttl(_op_ttl(materialised[index]))
                    self._execute(
                        """
                        INSERT INTO store (
                            prefix, key, value, created_at, updated_at,
                            expires_at, ttl_minutes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (prefix, key) DO UPDATE SET
                            value = EXCLUDED.value,
                            created_at = EXCLUDED.created_at,
                            updated_at = EXCLUDED.updated_at,
                            expires_at = EXCLUDED.expires_at,
                            ttl_minutes = EXCLUDED.ttl_minutes
                        """,
                        (
                            prefix,
                            key,
                            self.serde.dumps_json(value),
                            now,
                            now,
                            now + timedelta(minutes=ttl) if ttl is not None else None,
                            ttl,
                        ),
                    )
        return results

    @staticmethod
    def _normalise_ttl(ttl: Any) -> float | None:
        if ttl is None:
            return None
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
            raise TypeError("ttl must be a number of minutes or None")
        return float(ttl)

    @staticmethod
    def _namespace_to_text(namespace: Sequence[str]) -> str:
        return ".".join(namespace)

    @staticmethod
    def _decode_namespace(prefix: str) -> tuple[str, ...]:
        return tuple(prefix.split("."))

    def _row_to_item(self, row: Any, *, search: bool = False) -> Item | SearchItem:
        item_type = SearchItem if search else Item
        kwargs: dict[str, Any] = {
            "value": self.serde.loads_json(row[2]),
            "key": row[1],
            "namespace": self._decode_namespace(row[0]),
            "created_at": row[3],
            "updated_at": row[4],
        }
        return item_type(**kwargs)

    def _get_item(
        self,
        namespace: Sequence[str],
        key: str,
        *,
        refresh_ttl: bool = True,
    ) -> Item | None:
        _validate_namespace(namespace)
        prefix = self._namespace_to_text(namespace)
        with self._db_lock:
            self._ensure_open()
            row = self._execute(
                """
                SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM store WHERE prefix = %s AND key = %s
                """,
                (prefix, str(key)),
            ).fetchone()
            if row is None:
                return None
            if row[5] is not None and row[5] <= datetime.now(timezone.utc):
                return None
            if (
                refresh_ttl
                and row[5] is not None
                and row[6] is not None
                and self.ttl_config.get("refresh_on_read", False)
            ):
                self._execute(
                    """
                    UPDATE store SET expires_at = %s WHERE prefix = %s AND key = %s
                    """,
                    (
                        datetime.now(timezone.utc) + timedelta(minutes=row[6]),
                        prefix,
                        str(key),
                    ),
                )
            return self._row_to_item(row)

    def _search(self, op: Any) -> list[SearchItem]:
        prefix = _op_namespace(op)
        filter_dict = _op_filter(op)
        limit = int(_op_attr(op, "limit", 10))
        offset = int(_op_attr(op, "offset", 0))
        query = _op_attr(op, "query", None)
        refresh_ttl = bool(_op_attr(op, "refresh_ttl", True))
        with self._db_lock:
            self._ensure_open()
            rows = self._execute(
                """
                SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM store ORDER BY updated_at DESC, key DESC
                """
            ).fetchall()
            candidates: list[SearchItem] = []
            now = datetime.now(timezone.utc)
            for row in rows:
                namespace = self._decode_namespace(row[0])
                if not _does_match("prefix", prefix, namespace):
                    continue
                if row[5] is not None and row[5] <= now:
                    continue
                value = self.serde.loads_json(row[2])
                if filter_dict and not all(
                    _compare_values(value.get(key), expected)
                    for key, expected in filter_dict.items()
                ):
                    continue
                if (
                    refresh_ttl
                    and row[5] is not None
                    and row[6] is not None
                    and self.ttl_config.get("refresh_on_read", False)
                ):
                    self._execute(
                        "UPDATE store SET expires_at = %s WHERE prefix = %s AND key = %s",
                        (now + timedelta(minutes=row[6]), row[0], row[1]),
                    )
                candidates.append(
                    cast("SearchItem", self._row_to_item(row, search=True))
                )
        if query:
            del query
        return candidates[offset : offset + limit]

    def _list_namespaces(self, op: Any) -> list[tuple[str, ...]]:
        conditions = _op_attr(op, "match_conditions", None) or ()
        max_depth = _op_attr(op, "max_depth", None)
        limit = int(_op_attr(op, "limit", 100))
        offset = int(_op_attr(op, "offset", 0))
        with self._db_lock:
            self._ensure_open()
            rows = self._execute(
                """
                SELECT DISTINCT prefix FROM store
                WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                """
            ).fetchall()
        namespaces = {self._decode_namespace(row[0]) for row in rows}
        if conditions:
            namespaces = {
                namespace
                for namespace in namespaces
                if all(_does_match(c.match_type, c.path, namespace) for c in conditions)
            }
        if max_depth is not None:
            values = sorted({namespace[:max_depth] for namespace in namespaces})
        else:
            values = sorted(namespaces)
        return values[offset : offset + limit]

    def get(
        self,
        namespace: Sequence[str],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return self._get_item(
            namespace,
            key,
            refresh_ttl=True if refresh_ttl is None else refresh_ttl,
        )

    async def aget(
        self,
        namespace: Sequence[str],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return await asyncio.to_thread(
            self.get,
            namespace,
            key,
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
        await asyncio.to_thread(
            self.put,
            namespace,
            key,
            value,
            index,
            ttl=ttl,
        )

    def delete(self, namespace: Sequence[str], key: str) -> None:
        self.batch([PutOp(tuple(namespace), str(key), None, ttl=None)])

    async def adelete(self, namespace: Sequence[str], key: str) -> None:
        await asyncio.to_thread(self.delete, namespace, key)

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
        return self.batch(
            [
                SearchOp(
                    tuple(namespace_prefix),
                    filter,
                    limit,
                    offset,
                    query,
                    True if refresh_ttl is None else refresh_ttl,
                )
            ]
        )[0]

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
        return await asyncio.to_thread(
            self.search,
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

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
        return self.batch([ListNamespacesOp(tuple(conditions), max_depth, limit, offset)])[0]

    async def alist_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return await asyncio.to_thread(
            self.list_namespaces,
            prefix=prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PostgreSQL store is closed")


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


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: MemoryStore | SqliteStore | PostgresStore | None = None
_store_lock = threading.Lock()


def get_store(
    config: dict[str, Any] | Any | None = None,
) -> MemoryStore | SqliteStore | PostgresStore:
    """Return the configured process-wide store.

    ``memory``, ``sqlite``, and ``postgres`` are implemented natively. Other
    durable backends fail closed rather than silently degrading to
    process-local state.
    """
    global _store
    if config is None or not isinstance(config, dict):
        config = {"type": "memory"}
    backend = str(config.get("type", "memory"))
    if backend not in {"memory", "sqlite", "postgres"}:
        raise NotImplementedError(
            f"store backend {backend!r} is not ported to ReactiveGraph yet. "
            "Refusing to silently fall back to process-local state."
        )
    with _store_lock:
        if _store is None:
            if backend == "sqlite":
                connection_string = config.get("connection_string") or config.get("path")
                if not connection_string:
                    raise ValueError(
                        "store.connection_string is required for the sqlite backend"
                    )
                _store = SqliteStore(connection_string)
            elif backend == "postgres":
                _store = PostgresStore(
                    config["connection_string"], schema=config.get("schema", "public")
                )
            else:
                _store = MemoryStore()
        return _store


def reset_store() -> None:
    """Drop the process-wide store (tests, config reload)."""
    global _store
    with _store_lock:
        if isinstance(_store, (SqliteStore, PostgresStore)):
            _store.close()
        _store = None

"""Store data types and the engine-native base ``BaseStore`` surface.

``Item``/``SearchItem`` and the namespace validation rules are shared by every
backend (memory, SQLite, Postgres) and by callers that only construct or
inspect store records. Splitting them out keeps ``store.py`` focused on the
backend implementations while ``reactivegraph.store`` re-exports these names,
so existing imports keep working.

``BaseStore`` resolves to LangGraph's class when that host package is
installed and to the native fallback below otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
        """Return the checkpoint dict for *config*, or ``None`` when absent.

        Derived from :meth:`get_tuple` so subclasses only implement storage."""
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
        """Persist *checkpoint* and return the updated config mapping."""
        del namespace, key, value, index, ttl
        raise NotImplementedError

    def delete(self, namespace: Sequence[str], key: str) -> None:
        """Delete the item stored at *namespace*/*key* (missing keys are ignored)."""
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
        """Search one namespace, returning items ordered by score when supported."""
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
        """List namespaces, optionally preferring items matching *prefix*."""
        del prefix, suffix, max_depth, limit, offset
        raise NotImplementedError

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Apply a sequence of get/put/search/list/delete operations."""
        del ops
        raise NotImplementedError

    async def aget(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        """Async twin of :meth:`get`."""
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
        """Async twin of :meth:`put`."""
        del namespace, key, value, index, ttl
        raise NotImplementedError

    async def adelete(self, namespace: Sequence[str], key: str) -> None:
        """Async twin of :meth:`delete`."""
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
        """Async twin of :meth:`search`."""
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
        """Async twin of :meth:`list_namespaces`."""
        del prefix, suffix, max_depth, limit, offset
        raise NotImplementedError

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Async twin of :meth:`batch`."""
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
        """Return the item as a plain dict (upstream-compatible shape)."""
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
        """Return the item as a plain dict (upstream-compatible shape)."""
        result = super().dict()
        result["score"] = self.score
        return result



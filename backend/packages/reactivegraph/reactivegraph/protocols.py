"""Public extension contracts.

These ``Protocol`` classes are the supported way to plug a custom backend into
the engine. They are structural: any object implementing the methods below
satisfies the contract, and no inheritance from this module is required.

Three extension points are defined:

``StoreProtocol``
    Long-term key/value storage (``graph.store_get``/``store_put``/...). The
    engine ships ``MemoryStore``, ``SqliteStore`` and ``PostgresStore``; a
    host plugs in its own when it already has a data platform.

``CheckpointSaverProtocol``
    Thread checkpoint persistence. The engine ships an in-memory store plus
    SQLite and PostgreSQL savers; a host plugs in its own to reuse existing
    infrastructure or to apply its own retention rules.

``RetrieverProtocol``
    Document retrieval for RAG pipelines, consumed by ``reactivechain``.

A custom implementation only has to provide the methods it will actually be
called with. ``supports`` documents which are required for each integration
level, and the engine fails loudly (never silently) when a required method is
missing — see ``check_protocol``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from reactivegraph.checkpoint_types import CheckpointTuple
from reactivegraph.store_types import Item, SearchItem

__all__ = (
    "CheckpointSaverProtocol",
    "RetrieverProtocol",
    "StoreProtocol",
    "check_protocol",
)


@runtime_checkable
class StoreProtocol(Protocol):
    """Long-term storage contract.

    ``namespace`` is a tuple of string segments, ``key`` a string unique within
    that namespace. Implementations may raise :class:`NotImplementedError` for
    operations they do not support; the engine surfaces that to the caller
    rather than degrading to an empty result, so a misconfigured backend is
    visible instead of looking like "no data".
    """

    def get(
        self, namespace: Sequence[str], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        """Return the stored item, or ``None`` when absent."""
        ...

    def put(
        self,
        namespace: Sequence[str],
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
    ) -> None:
        """Write *value* under ``namespace``/``key``, optionally with a TTL."""
        ...

    def delete(self, namespace: Sequence[str], key: str) -> None:
        """Remove one item; missing keys are ignored."""
        ...

    def search(
        self,
        namespace_prefix: Sequence[str],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Return items under *namespace_prefix*, newest/highest-score first."""
        ...

    def list_namespaces(
        self,
        *,
        prefix: Sequence[str] | None = None,
        suffix: Sequence[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Enumerate namespaces, optionally filtered by prefix/suffix."""
        ...


@runtime_checkable
class CheckpointSaverProtocol(Protocol):
    """Thread checkpoint persistence contract.

    ``config`` is the LangGraph-style mapping
    ``{"configurable": {"thread_id": ..., "checkpoint_ns": ...,
    "checkpoint_id": ...}}``. Implementations that follow the LangGraph
    ``BaseCheckpointSaver`` shape can be passed directly — the engine detects
    and uses the host methods when present.
    """

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist *checkpoint* and return the config identifying it."""
        ...

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Record pending writes for *task_id*."""
        ...

    def get_tuple(self, config: dict[str, Any] | None) -> CheckpointTuple | None:
        """Return the checkpoint identified by *config*, or ``None``."""
        ...

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate checkpoints newest-first."""
        ...

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint belonging to *thread_id*."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Document retrieval contract used by RAG pipelines.

    Return objects exposing ``page_content`` and ``metadata`` (any
    LangChain-compatible ``Document`` satisfies this).
    """

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        """Return the documents most relevant to *query*."""
        ...


def check_protocol(obj: Any, protocol: type, *, context: str = "") -> None:
    """Raise a helpful error when *obj* does not satisfy *protocol*.

    ``runtime_checkable`` only verifies attribute presence, so this reports
    exactly which methods are missing — a bare ``TypeError`` from deep inside a
    run would be much harder to act on.

    Raises:
        TypeError: listing the missing methods.
    """
    if isinstance(obj, protocol):
        return
    # ``runtime_checkable`` only checks attribute *presence*; enumerate the
    # protocol's own callables so the error can name exactly what is missing.
    # ``Protocol`` classes expose their members through ``__protocol_attrs__``
    # on some versions only, so fall back to the class namespace.
    declared = getattr(protocol, "__protocol_attrs__", None)
    if declared is None:
        declared = {
            name
            for name, value in vars(protocol).items()
            if not name.startswith("_") and callable(value)
        }
    missing = sorted(n for n in declared if not callable(getattr(obj, n, None)))
    where = f" for {context}" if context else ""
    raise TypeError(
        f"{type(obj).__name__} does not satisfy {protocol.__name__}{where}; "
        f"missing: {', '.join(missing) if missing else 'unknown'}"
    )

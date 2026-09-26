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

import abc
import collections.abc
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, get_type_hints

from typing_extensions import Required

from reactivegraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)
from reactivegraph.message_state import EphemeralValue as _EphemeralValueMarker
from reactivegraph.types import Overwrite

try:  # pragma: no cover - exercised by the optional-integration tests
    from langgraph import channels as _lg_channels
except ImportError:  # pragma: no cover - langgraph is an optional host dep
    _lg_channels = None


__all__ = (
    "BaseChannel",
    "BinaryOperatorAggregate",
    "DeltaChannel",
    "channel_reducer",
    "resolve_channels",
)

Value = TypeVar("Value")
_Update = TypeVar("_Update")
_Checkpoint = TypeVar("_Checkpoint")

def _host_missing_sentinel() -> object | None:
    """Return LangGraph's private ``MISSING`` sentinel when installed.

    Upstream's Pregel passes *its* sentinel across the channel boundary:
    ``channels_from_checkpoint`` calls ``spec.from_checkpoint(...get(k, MISSING))``
    and ``create_checkpoint`` skips a value only when ``v is MISSING``. A
    module-private sentinel is therefore not interchangeable — a host-driven
    graph would persist the placeholder as real state and fold it into the
    first reducer call. Sharing the object keeps both directions aligned.
    """
    try:
        from langgraph._internal._typing import MISSING as HOST_MISSING
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HOST_MISSING


HOST_MISSING = _host_missing_sentinel()

_MISSING = HOST_MISSING if HOST_MISSING is not None else object()
"""Sentinel for "this channel has never been written".

A module-private object rather than ``None`` so a legitimate ``None`` value can
still be stored, matching upstream's ``MISSING``. When LangGraph is installed
this *is* upstream's sentinel, because upstream compares against it by
identity on both the inbound (``from_checkpoint``) and outbound
(``create_checkpoint``) paths.
"""

_OVERWRITE = "__overwrite__"


class BaseChannel(Generic[Value, _Update, _Checkpoint], abc.ABC):
    """Base class for state channels.

    A channel owns one state field: how writes fold into a value, and how that
    value is (de)serialised into a checkpoint. Hosts subclass this directly
    (DeerFlow's ``TaskNotesChannel``) and the storage layer does ``isinstance``
    checks against it, so the class must be importable from the engine rather
    than only from the host that previously supplied it.
    """

    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key

    @property
    @abc.abstractmethod
    def ValueType(self) -> Any:  # noqa: N802 - LangGraph API name
        """The type of the value stored in the channel."""

    @property
    @abc.abstractmethod
    def UpdateType(self) -> Any:  # noqa: N802 - LangGraph API name
        """The type of the update received by the channel."""

    def copy(self) -> BaseChannel[Value, _Update, _Checkpoint]:
        """Return an identical channel, by default via a checkpoint round-trip."""
        return self.from_checkpoint(self.checkpoint())

    def checkpoint(self) -> Any:
        """Return a serialisable representation of the channel's state.

        An empty channel yields :data:`_MISSING` rather than raising, matching
        upstream: the caller stores the sentinel and replays writes on restore.
        """
        try:
            return self.get()
        except EmptyChannelError:
            return _MISSING

    @abc.abstractmethod
    def from_checkpoint(self, checkpoint: Any) -> BaseChannel[Value, _Update, _Checkpoint]:
        """Return a new identical channel, optionally seeded from *checkpoint*."""

    @abc.abstractmethod
    def get(self) -> Value:
        """Return the current value; raise :class:`EmptyChannelError` if empty."""

    def is_available(self) -> bool:
        """Whether the channel holds a value (cheaper than catching ``get``)."""
        try:
            self.get()
            return True
        except EmptyChannelError:
            return False

    @abc.abstractmethod
    def update(self, values: Sequence[_Update]) -> bool:
        """Apply one super-step's writes; return whether the value changed."""

    def consume(self) -> bool:
        """Notify the channel that a subscribed task ran.

        Part of the upstream ``BaseChannel`` write contract. Pregel calls this
        on every subscribed channel; channels that consume their value override
        it to prevent it being seen again.
        """
        return False

    def finish(self) -> bool:
        """Notify the channel that the Pregel run is finishing.

        Part of the upstream ``BaseChannel`` write contract. Pregel calls
        ``finish()`` on every channel once the run stops triggering tasks;
        without it a host-compiled graph raises ``AttributeError`` on the
        engine's channels.
        """
        return False


def _strip_extras(t: Any) -> Any:
    """Strip ``Annotated``/``Required``/``NotRequired`` wrappers from a type."""
    from typing_extensions import NotRequired

    while getattr(t, "__origin__", None) in (Required, NotRequired):
        t = t.__args__[0]
    if hasattr(t, "__origin__"):
        return _strip_extras(t.__origin__)
    return t


def _concrete_type(typ: Any) -> Any:
    """Map abstract ``collections.abc`` types to their instantiable concrete form."""
    typ = _strip_extras(typ)
    if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
        return list
    if typ in (collections.abc.Set, collections.abc.MutableSet):
        return set
    if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
        return dict
    return typ


def _host_overwrite_type() -> type | None:
    """The host's ``langgraph.types.Overwrite`` class, when importable.

    Cached only on success so a partially-initialised host module is retried.
    """
    global _HOST_OVERWRITE
    if _HOST_OVERWRITE is not None:
        return _HOST_OVERWRITE
    try:
        from langgraph.types import Overwrite as HostOverwrite
    except Exception:
        return None
    _HOST_OVERWRITE = HostOverwrite
    return HostOverwrite


def _is_foreign_overwrite(value: Any) -> bool:
    """Whether *value* is a host ``Overwrite`` (a different class object).

    The host's dataclass is duck-type identical but not ours; comparing on the
    ``type`` marker keeps the two families interoperable without importing the
    host eagerly.
    """
    host = _host_overwrite_type()
    if host is None or type(value) is Overwrite:
        return False
    return isinstance(value, host)


_HOST_OVERWRITE: type | None = None


def _get_overwrite(value: Any) -> tuple[bool, Any]:
    """Return ``(is_overwrite, payload)`` for every Overwrite form.

    Four forms must be recognised: our dataclass, the host's ``Overwrite``
    (a distinct class object that DeerFlow's write paths construct), the
    sentinel-keyed dict, and the dataclass-erased
    ``{"value": ..., "type": "__overwrite__"}`` form produced when a state
    update round-trips through JSON.
    """
    if isinstance(value, Overwrite):
        return True, value.value
    if _is_foreign_overwrite(value):
        return True, value.value
    if isinstance(value, dict):
        if len(value) == 1 and _OVERWRITE in value:
            return True, value[_OVERWRITE]
        if value.get("type") == _OVERWRITE and "value" in value:
            return True, value["value"]
    return False, None


def _operators_equal(a: Callable, b: Callable) -> bool:
    """Whether two reducers should compare equal.

    Every lambda shares the name ``<lambda>``, so identity comparison is
    unreliable for them; treat any pairing involving a lambda as equal.
    """
    if a.__name__ == "<lambda>" or b.__name__ == "<lambda>":
        return True
    return a is b


def _is_host_binop(item: Any) -> bool:
    """Whether *item* is LangGraph's ``BinaryOperatorAggregate`` instance/class."""
    if _lg_channels is None:
        return False
    host = getattr(_lg_channels, "BinaryOperatorAggregate", None)
    if host is None:  # pragma: no cover - a host without the class is unusable
        return False
    if isinstance(item, host):
        return True
    return item is host


class _BinopMeta(abc.ABCMeta):
    """Recognise the host's ``BinaryOperatorAggregate`` as a virtual sibling.

    ``_channel_for`` deliberately hands back the *host* class for a
    two-argument reducer annotation, because the host storage layer
    isinstance-checks against it. The reverse direction cannot use ``register``
    (the host ABC already virtually inherits from our class through
    ``_bridge_host_channel_identity``, which would create a cycle), so
    ``isinstance`` against our class is widened explicitly and only for the
    host's own class.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        if cls.__name__ == "BinaryOperatorAggregate" and _is_host_binop(instance):
            return True
        return super().__instancecheck__(instance)


class BinaryOperatorAggregate(
    Generic[Value], BaseChannel[Value, Value, Value], metaclass=_BinopMeta
):
    """Folds each write into the running value with a two-argument operator.

    ``Annotated[dict, merge]`` compiles to one of these, and the checkpoint
    layer patches :meth:`update` for the ``Overwrite``-on-empty-channel bug
    (DeerFlow #4380), so the method must stay a normal overridable attribute.
    """

    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]) -> None:
        super().__init__(typ)
        self.operator = operator
        concrete = _concrete_type(typ)
        try:
            self.value = concrete()
        except Exception:
            # Union types and other non-instantiable forms start empty; the
            # first write seeds the channel instead.
            self.value = _MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, BinaryOperatorAggregate) and _operators_equal(
            self.operator, value.operator
        )

    @property
    def ValueType(self) -> type[Value]:  # noqa: N802 - LangGraph API name
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:  # noqa: N802 - LangGraph API name
        return self.typ

    def copy(self) -> BinaryOperatorAggregate[Value]:
        empty = self.__class__(self.typ, self.operator)
        empty.key = self.key
        empty.value = self.value
        return empty

    def from_checkpoint(self, checkpoint: Value) -> BinaryOperatorAggregate[Value]:
        empty = self.__class__(self.typ, self.operator)
        empty.key = self.key
        if checkpoint is not _MISSING:
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        seen_overwrite = False
        if self.value is _MISSING:
            # The first write normally seeds the channel verbatim, but an
            # Overwrite must still mean "replace": a Union-typed channel has
            # no constructible default, so it starts empty and a replace
            # write would otherwise persist the wrapper itself (DeerFlow
            # #4380). Upstream LangGraph still has this bug and DeerFlow
            # monkeypatches it at import time; the engine implements the
            # fixed behaviour directly.
            is_overwrite, overwrite_value = _get_overwrite(values[0])
            self.value = overwrite_value if is_overwrite else values[0]
            seen_overwrite = is_overwrite
            values = values[1:]
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                self.value = overwrite_value
                seen_overwrite = True
                continue
            if not seen_overwrite:
                self.value = self.operator(self.value, value)
        return True

    def get(self) -> Value:
        if self.value is _MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not _MISSING

    def checkpoint(self) -> Value:
        return self.value


def _is_host_delta(item: Any) -> bool:
    """Whether *item* is LangGraph's ``DeltaChannel`` instance or class."""
    if _lg_channels is None:
        return False
    if isinstance(item, _lg_channels.DeltaChannel):
        return True
    return item is _lg_channels.DeltaChannel


@dataclass
class _DeltaSnapshot:
    """A full-value blob written on the snapshot cadence.

    Only the engine's own storage layer produces and consumes this, so it is
    deliberately private; a host reads the reconstructed value, never the blob.
    """

    value: Any


def _host_delta_snapshot(value: Any) -> Any:
    """Build the host's private snapshot blob when LangGraph is installed.

    Host savers serialise blobs through their own serde, and host
    ``DeltaChannel.from_checkpoint`` recognises only its own namedtuple.
    Falling back to the local dataclass keeps the engine usable without the
    optional host dependency.
    """
    try:
        from langgraph.checkpoint.serde.types import _DeltaSnapshot as HostDeltaSnapshot
    except ImportError:
        return _DeltaSnapshot(value)
    return HostDeltaSnapshot(value)


def _is_host_delta_snapshot(value: Any) -> bool:
    """Whether *value* is LangGraph's private ``_DeltaSnapshot`` blob.

    Upstream's ``StateGraph`` writes ``langgraph.checkpoint.serde.types._DeltaSnapshot``
    into ``channel_values``. Duck-typing the two-field namedtuple shape keeps
    the engine free of a module-scope LangGraph import while still restoring
    checkpoints written by the host.
    """
    return (
        type(value).__name__ == "_DeltaSnapshot"
        and hasattr(value, "value")
        and hasattr(value, "_fields")
    )


class _DeltaChannelMeta(abc.ABCMeta):
    """Recognise the host's ``DeltaChannel`` as this class' virtual sibling.

    The host ABC is the one that must accept *our* instances (done by
    ``register`` below). Registering the reverse direction would create an
    inheritance cycle, so ``isinstance`` against our ``DeltaChannel`` is
    widened explicitly, and only for the host base.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        if cls.__name__ == "DeltaChannel" and _is_host_delta(instance):
            return True
        return super().__instancecheck__(instance)


class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any], metaclass=_DeltaChannelMeta):
    """Reducer channel storing a sentinel plus per-step writes.

    The reducer receives the accumulated value and the *whole* batch of writes
    for one super-step (``reducer(state, [w1, w2, ...])``), so it must be
    deterministic and batching-invariant: folding ``xs`` then ``ys`` must equal
    folding ``xs + ys``. That property is what lets a restore replay ancestor
    writes in larger batches than they were produced without changing state.
    """

    __slots__ = ("value", "reducer", "snapshot_frequency")

    def __init__(
        self,
        reducer: Callable[[Any, Sequence[Any]], Any],
        typ: type[Value] | None = None,
        *,
        snapshot_frequency: int = 1000,
    ) -> None:
        if snapshot_frequency <= 0:
            raise ValueError(
                f"snapshot_frequency must be a positive int, got {snapshot_frequency}"
            )
        super().__init__(typ if typ is not None else list)
        self.reducer = reducer
        self.snapshot_frequency = snapshot_frequency
        self.typ = _concrete_type(typ if typ is not None else list)
        self.value: Any = _MISSING

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeltaChannel):
            return False
        if self.snapshot_frequency != other.snapshot_frequency:
            return False
        return _operators_equal(self.reducer, other.reducer)

    @property
    def ValueType(self) -> Any:  # noqa: N802 - LangGraph API name
        return self.typ

    @property
    def UpdateType(self) -> Any:  # noqa: N802 - LangGraph API name
        return self.typ

    def copy(self) -> DeltaChannel[Value]:
        new = self.__class__(
            self.reducer, self.typ, snapshot_frequency=self.snapshot_frequency
        )
        new.key = self.key
        new.value = self.value if self.value is _MISSING else copy.copy(self.value)
        return new

    def from_checkpoint(self, checkpoint: Any) -> DeltaChannel[Value]:
        """Restore from a blob.

        ``_MISSING`` starts empty for the caller to replay writes into,
        :class:`_DeltaSnapshot` restores a full value, and a plain value is the
        legacy ``BinaryOperatorAggregate`` blob shape.
        """
        new = self.__class__(
            self.reducer, self.typ, snapshot_frequency=self.snapshot_frequency
        )
        new.key = self.key
        if checkpoint is _MISSING:
            new.value = self.typ()
        elif isinstance(checkpoint, _DeltaSnapshot) or _is_host_delta_snapshot(
            checkpoint
        ):
            new.value = checkpoint.value
        else:
            new.value = checkpoint
        return new

    def replay_writes(self, writes: Sequence[tuple[Any, Any, Any]]) -> None:
        """Apply ancestor writes oldest-to-newest in a single reducer call.

        An ``Overwrite`` resets the base to its payload; only writes after the
        last one are folded.
        """
        values = [v for _, _, v in writes]
        if not values:
            return
        base = self.value
        start = 0
        for i, v in enumerate(values):
            is_ow, ow_value = _get_overwrite(v)
            if is_ow:
                base = copy.copy(ow_value) if ow_value is not None else self.typ()
                start = i + 1
        remaining = values[start:]
        self.value = self.reducer(base, remaining) if remaining else base

    def update(self, values: Sequence[Any]) -> bool:
        if not values:
            return False
        overwrite_idx: int | None = None
        for i, v in enumerate(values):
            is_ow, _ = _get_overwrite(v)
            if is_ow:
                if overwrite_idx is not None:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                overwrite_idx = i
        if overwrite_idx is not None:
            _, overwrite_value = _get_overwrite(values[overwrite_idx])
            self.value = (
                copy.copy(overwrite_value) if overwrite_value is not None else self.typ()
            )
            return True
        base = self.typ() if self.value is _MISSING else self.value
        self.value = self.reducer(base, list(values))
        return True

    def get(self) -> Any:
        if self.value is _MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not _MISSING

    def checkpoint(self) -> Any:
        """Always ``_MISSING``.

        Snapshot decisions belong to ``create_checkpoint``, which writes a
        :class:`_DeltaSnapshot` on the cadence; a non-snapshot step simply
        omits the channel and is rebuilt by replaying ancestor writes.
        """
        return _MISSING


def _bridge_host_channel_identity() -> None:
    """Register our channels as virtual subclasses of the host's ABCs.

    LangGraph's compiler accepts only instances of *its* ``BaseChannel`` and
    special-cases its ``DeltaChannel``. Registering our concrete classes is the
    supported ABC mechanism: upstream then uses the instance verbatim, keeping
    the delta reducer and snapshot cadence instead of silently compiling a
    ``LastValue``. The reverse direction cannot use ``register`` because the
    host ABC already virtually inherits from our registered class (that would
    form an inheritance cycle); the engine recognises host instances with the
    local ``_is_host_channel`` / ``_is_host_delta`` predicates instead.
    """
    if _lg_channels is None:
        return
    try:
        _lg_channels.BaseChannel.register(BaseChannel)
        _lg_channels.BaseChannel.register(BinaryOperatorAggregate)
        _lg_channels.BaseChannel.register(DeltaChannel)
        _lg_channels.DeltaChannel.register(DeltaChannel)
    except (AttributeError, TypeError):
        # A host that changes its ABCs must not make importing the engine fail;
        # channel resolution still works through the native implementations.
        return


_bridge_host_channel_identity()


@dataclass(frozen=True)
class LastValue:
    """Engine-native stand-in for ``langgraph.channels.LastValue``."""

    typ: Any
    key: str = ""


@dataclass(frozen=True)
class BinaryOperatorAggregateNative:
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


def _is_host_channel(item: Any) -> bool:
    """Whether *item* is a LangGraph channel instance or subclass.

    Hosts that installed LangGraph before us may annotate a schema with one of
    *its* channel objects. Those are honoured verbatim (copied, so the schema's
    shared instance is never mutated) because the storage layer does
    ``isinstance`` checks against the host classes it was built with.
    """
    if _lg_channels is None:
        return False
    if isinstance(item, _lg_channels.BaseChannel):
        return True
    return isinstance(item, type) and issubclass(item, _lg_channels.BaseChannel)


def _channel_for(name: str, annotation: Any) -> Any:
    annotation = _unwrap(annotation)
    metadata = getattr(annotation, "__metadata__", ())
    typ = getattr(annotation, "__origin__", annotation)

    for item in metadata:
        # A caller-supplied channel *instance* wins outright: DeerFlow's delta
        # mode annotates ``messages`` with a configured DeltaChannel, and the
        # snapshot cadence on that instance is part of the schema's meaning.
        if isinstance(item, BaseChannel):
            channel = copy.copy(item)
            if hasattr(channel, "key"):
                channel.key = name
            return channel
        if isinstance(item, type) and issubclass(item, BaseChannel):
            return item(typ)
        if _is_host_channel(item):
            if isinstance(item, type):
                # A *class* annotation must be instantiated, never copied:
                # ``copy.copy`` on a class returns the class itself, so writing
                # ``key`` would shadow ``BaseChannel``'s slot descriptor on the
                # host class process-wide. Every later construction anywhere in
                # the interpreter then died with "attribute 'key' is read-only".
                channel = item(typ)
                if hasattr(channel, "key"):
                    channel.key = name
                return channel
            channel = copy.copy(item)
            if hasattr(channel, "key"):
                channel.key = name
            return channel
        if item is _EphemeralValueMarker or isinstance(item, _EphemeralValueMarker):
            if _lg_channels is not None:
                return _lg_channels.EphemeralValue(typ)
            return LastValue(typ=typ, key=name)
    if metadata and _is_two_arg_callable(metadata[-1]):
        # Prefer the host's class when it is importable: the storage layer does
        # ``isinstance(channel, BinaryOperatorAggregate)`` against the classes
        # it was built with, so handing back our look-alike would silently stop
        # a reducer field from being recognised as one.
        if _lg_channels is not None:
            return _lg_channels.BinaryOperatorAggregate(typ, metadata[-1])
        return BinaryOperatorAggregate(typ, metadata[-1])
    if _lg_channels is not None:
        return _lg_channels.LastValue(typ)
    return LastValue(typ=typ, key=name)


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


def _delta_fold(channel: Any) -> Callable[[Any, Any], Any]:
    """Adapt a batch reducer to the single-write fold this engine applies."""
    reducer = channel.reducer

    def fold_delta(current: Any, write: Any) -> Any:
        return reducer(current, [write])

    return fold_delta


def channel_reducer(channel: Any) -> Callable[[Any, Any], Any] | None:
    """Return the two-argument fold for *channel*, or ``None`` for last-value.

    ``BinaryOperatorAggregate`` folds a single write against the current value.
    ``DeltaChannel``'s reducer takes the whole *sequence* of writes for a step,
    so it is adapted to the same one-write signature here. Host channels are
    recognised too, so a schema still annotated with LangGraph's classes folds
    identically.
    """
    if isinstance(channel, DeltaChannel):
        return _delta_fold(channel)
    if isinstance(channel, BinaryOperatorAggregate):
        return channel.operator
    if isinstance(channel, BinaryOperatorAggregateNative):
        return channel.operator
    if _is_host_delta(channel):
        return _delta_fold(channel)
    if _is_host_binop(channel):
        return channel.operator
    return None

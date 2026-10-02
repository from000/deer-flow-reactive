"""Engine-level primitives shared by every host binding.

``Overwrite`` is part of the state-update contract, not of any one host: a
node that returns ``Overwrite(value)`` asks the channel to replace its
accumulated value instead of merging through the reducer. The class also
recognises the two serialized forms the value can take after a JSON round
trip, because a wrapper must survive transport between processes. When
LangGraph is importable the engine class *subclasses* the host class, because
the host's own ``BinaryOperatorAggregate.update`` checks
``isinstance(value, langgraph.types.Overwrite)`` before it bypasses the
reducer; a look-alike class would be fed to the reducer and raise.

``Interrupt`` is the *payload* a paused node surfaces to its client. It is
deliberately not an exception: middleware that catches ``Exception`` to wrap
ordinary failures must never swallow a human-in-the-loop pause. The control
flow signal is :class:`reactivegraph.errors.GraphInterrupt`.

Portions of this module are adapted from upstream `langgraph`
(https://github.com/langchain-ai/langgraph) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast

from typing_extensions import final

from reactivegraph.checkpoint import CheckpointStore
from reactivegraph.messages import ToolOutputMixin
from reactivegraph.warnings import ReactiveGraphDeprecatedSinceV10

OVERWRITE: Literal["__overwrite__"] = "__overwrite__"
"""Sentinel key/discriminator for the serialized ``Overwrite`` forms."""

DEFAULT_INTERRUPT_ID = "placeholder-id"
"""Id carried by an ``Interrupt`` created without a namespace."""

Checkpointer: TypeAlias = None | bool | CheckpointStore
"""Checkpointer selector, mirroring the upstream ``Checkpointer`` alias.

``None`` inherits the parent graph's checkpointer, ``True`` enables the
inherited one, ``False`` disables checkpointing, and a
:class:`~reactivegraph.checkpoint.CheckpointStore` instance supplies its own.
"""

def _hash_namespace(ns: str) -> str:
    """Return the 128-bit xxh3 digest LangGraph derives interrupt ids from.

    The digest is an interoperability contract: a client that resumed an
    interrupt before the engine swap must keep resolving the same id after it.
    ``xxhash`` is therefore a hard dependency rather than an optional one.
    """
    from xxhash import xxh3_128_hexdigest

    return xxh3_128_hexdigest(ns.encode())


__all__ = (
    "DEFAULT_INTERRUPT_ID",
    "OVERWRITE",
    "Checkpointer",
    "Command",
    "Interrupt",
    "Overwrite",
    "Send",
)


class _OverwriteBehavior:
    """Semantics shared by the native and host-bridged ``Overwrite``."""

    __slots__ = ()

    @staticmethod
    def is_overwrite(value: Any) -> tuple[bool, Any]:
        """Return ``(is_overwrite, overwrite_value)`` for any accepted form.

        Recognises the typed instance (engine or host), plus both serialized
        shapes: ``{"__overwrite__": value}`` and the dataclass-erased
        ``{"value": ..., "type": "__overwrite__"}`` that JSON produces.
        """
        if isinstance(value, Overwrite):
            return True, value.value
        if isinstance(value, dict):
            if len(value) == 1 and OVERWRITE in value:
                return True, value[OVERWRITE]
            if value.get("type") == OVERWRITE and "value" in value:
                return True, value["value"]
        return False, None


def _host_overwrite() -> type | None:
    """Return LangGraph's ``Overwrite`` class when it is importable."""
    try:
        from langgraph.types import Overwrite as HostOverwrite
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HostOverwrite


HostOverwrite = _host_overwrite()


class _OverwriteMeta(type):
    """Recognise host-created ``Overwrite`` objects as engine overwrites.

    Engine-produced overwrites are host subclasses. External callers may still
    construct the host class directly (DeerFlow's rollback paths did), and
    engine ``isinstance`` checks must accept those too. Only the framework
    base is widened, never unrelated subclasses.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        if HostOverwrite is not None and isinstance(instance, HostOverwrite):
            return True
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: type) -> bool:
        if HostOverwrite is not None and issubclass(subclass, HostOverwrite):
            return True
        return super().__subclasscheck__(subclass)


if HostOverwrite is None:

    @dataclass(slots=True)
    class Overwrite(_OverwriteBehavior):
        """Bypass a reducer and write the wrapped value directly to a channel.

        Receiving multiple ``Overwrite`` values for the same channel in one
        super-step is an error; the channel raises rather than guessing which
        one wins.
        """

        value: Any
        type: Literal["__overwrite__"] = OVERWRITE

else:

    class Overwrite(  # type: ignore[misc, no-redef, valid-type, unused-ignore]
        _OverwriteBehavior, HostOverwrite, metaclass=_OverwriteMeta  # type: ignore[misc, valid-type]
    ):
        """Engine ``Overwrite`` that is also a ``langgraph.types.Overwrite``.

        The host class owns the dataclass field declarations, ``__init__``,
        equality and unhashability; declaring the slots again keeps the
        engine's ``__slots__`` surface pinned and stops an instance dict from
        appearing.
        """

        __slots__ = ("value", "type")


@final
@dataclass(init=False, slots=True)
class _NativeInterrupt:
    """Engine-native interrupt payload, used when LangGraph is absent.

    ``id`` is stable for a given node namespace, which is what lets a client
    resume a specific interrupt instead of only the most recent one.
    """

    value: Any
    """The value associated with the interrupt."""

    id: str
    """The ID of the interrupt; can be used to resume it directly."""

    def __init__(
        self, value: Any, id: str = DEFAULT_INTERRUPT_ID, **deprecated_kwargs: Any
    ) -> None:
        self.value = value

        # ``ns`` was removed in favour of ``id`` but old callers still pass it.
        # A sequence of namespace segments hashes to the same id ``from_ns``
        # would produce; anything else is ignored.
        ns = deprecated_kwargs.get("ns")
        if id == DEFAULT_INTERRUPT_ID and isinstance(ns, Sequence):
            self.id = _hash_namespace("|".join(ns))
        else:
            self.id = id

    @classmethod
    def from_ns(cls, value: Any, ns: str) -> _NativeInterrupt:
        """Build an interrupt whose id is derived from *ns*."""
        return cls(value=value, id=_hash_namespace(ns))

    @property
    def interrupt_id(self) -> str:
        """Deprecated alias for :attr:`id`."""
        warnings.warn(
            "`interrupt_id` is deprecated. Use `id` instead.",
            ReactiveGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return self.id


def _host_interrupt() -> type | None:
    """Return LangGraph's ``Interrupt`` class when it is importable.

    The payload crosses package boundaries: DeerFlow's serializer checks
    ``isinstance(obj, reactivegraph.Interrupt)`` on objects raised by
    ``langgraph.types.interrupt`` inside graphs still running on the host
    runtime. Only the *same object* satisfies both directions, so the engine
    re-exports the host class when present. The import stays lazy so the
    engine remains usable without LangGraph.
    """
    try:
        from langgraph.types import Interrupt as HostInterrupt
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HostInterrupt


HostInterrupt = _host_interrupt()

# ``Interrupt`` has no engine-specific behaviour to add: field names, the
# xxh3 id derivation and the slot-only storage are all pinned against the
# host. Aliasing (rather than subclassing) is required because host code also
# checks engine-produced payloads against ``langgraph.types.Interrupt``.
Interrupt = HostInterrupt or _NativeInterrupt


class _CommandBehavior:
    """Engine semantics shared by the native and host-bridged ``Command``.

    LangGraph's ``Command`` is a frozen slotted dataclass; subclassing it is
    the only way host ``isinstance(..., langgraph.types.Command)`` checks can
    accept engine-produced commands. The host class also owns the field
    declarations and equality/hash behaviour, so this mixin carries only the
    parts the engine must keep control of: mutation (the engine's public
    contract), repr, and update normalisation.
    """

    __slots__ = ()

    PARENT: ClassVar[str] = "__parent__"
    update: Any

    def __repr__(self) -> str:
        # Upstream omits every falsy field, so ``Command(resume=0)`` prints as
        # ``Command()``; keeping that quirk means log output stays comparable.
        contents = ", ".join(
            f"{key}={value!r}"
            for key, value in asdict(cast("Any", self)).items()
            if value
        )
        return f"Command({contents})"

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        """Normalise ``update`` into ``(channel, value)`` pairs.

        The host implementation also consults annotated dataclass keys for
        non-dict updates; the engine's tool contract instead treats a plain
        value as a root update. Keeping this method here prevents a bridge
        choice from silently changing ``ToolNode`` behaviour.
        """
        if isinstance(self.update, dict):
            return list(self.update.items())
        if isinstance(self.update, (list, tuple)) and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in self.update
        ):
            return list(self.update)
        if self.update is not None:
            return [("__root__", self.update)]
        return []


N = TypeVar("N", bound=Hashable)
"""Type parameter of the host ``Command``; mirrored so annotations match."""


@dataclass(slots=True, repr=False)
class _NativeCommand(_CommandBehavior, ToolOutputMixin, Generic[N]):
    """The engine's own ``Command`` used when LangGraph is not installed."""

    graph: str | None = None
    update: Any | None = None
    resume: Any | None = None
    goto: Any = ()


def _host_command() -> type | None:
    """Return LangGraph's ``Command`` class when it is importable.

    ``Command`` is a plain class, so it cannot act as a virtual ABC base.
    Subclassing it keeps the host's own ``isinstance`` checks true without
    requiring the host to know about the engine; the import stays lazy so the
    engine remains usable without LangGraph.
    """
    try:
        from langgraph.types import Command as HostCommand
    except ImportError:
        return None
    return HostCommand


HostCommand = _host_command()


class _CommandMeta(type):
    """Recognise host-created ``Command`` objects as engine commands.

    Engine-produced commands are host subclasses. External tools may still
    construct the host class directly, and the engine's ``isinstance`` checks
    must accept those too. Only the framework base is widened, never unrelated
    subclasses.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        if HostCommand is not None and isinstance(instance, HostCommand):
            return True
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: type) -> bool:
        if HostCommand is not None and issubclass(subclass, HostCommand):
            return True
        return super().__subclasscheck__(subclass)


if HostCommand is None:
    Command = _NativeCommand
else:
    class Command(  # type: ignore[no-redef, misc, valid-type, unused-ignore]
        _CommandBehavior, HostCommand, Generic[N], metaclass=_CommandMeta  # type: ignore[misc, valid-type]
    ):
        """Engine ``Command`` that is also a ``langgraph.types.Command``."""

        __slots__ = ("graph", "update", "resume", "goto")

        def __init__(
            self,
            graph: str | None = None,
            update: Any | None = None,
            resume: Any | None = None,
            goto: Any = (),
        ) -> None:
            object.__setattr__(self, "graph", graph)
            object.__setattr__(self, "update", update)
            object.__setattr__(self, "resume", resume)
            object.__setattr__(self, "goto", goto)

        def __setattr__(self, name: str, value: Any) -> None:
            # The host base is frozen; the engine's contract is mutable and
            # callers do assign to fields after construction.
            object.__setattr__(self, name, value)

        def __delattr__(self, name: str) -> None:
            object.__delattr__(self, name)


class Send:
    """A task pushed to one node with an input private to that node.

    Used from a routing function for map-reduce fan-out: the sent state can
    differ from the graph's own state.
    """

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Send) and self.node == other.node and self.arg == other.arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

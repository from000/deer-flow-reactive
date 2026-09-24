"""Engine-level primitives shared by every host binding.

``Overwrite`` is part of the state-update contract, not of any one host: a
node that returns ``Overwrite(value)`` asks the channel to replace its
accumulated value instead of merging through the reducer. The class also
recognises the two serialized forms the value can take after a JSON round
trip, because a wrapper must survive transport between processes.

``Interrupt`` is the *payload* a paused node surfaces to its client. It is
deliberately not an exception: middleware that catches ``Exception`` to wrap
ordinary failures must never swallow a human-in-the-loop pause. The control
flow signal is :class:`reactivegraph.errors.GraphInterrupt`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, TypeAlias, final

from reactivegraph.checkpoint import CheckpointStore
from reactivegraph.warnings import ReactiveGraphDeprecatedSinceV10

OVERWRITE = "__overwrite__"
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


@dataclass(slots=True)
class Overwrite:
    """Bypass a reducer and write the wrapped value directly to a channel.

    Receiving multiple ``Overwrite`` values for the same channel in one
    super-step is an error; the channel raises rather than guessing which one
    wins.
    """

    value: Any
    type: Literal["__overwrite__"] = OVERWRITE

    @staticmethod
    def is_overwrite(value: Any) -> tuple[bool, Any]:
        """Return ``(is_overwrite, overwrite_value)`` for any accepted form.

        Recognises the typed instance plus both serialized shapes:
        ``{"__overwrite__": value}`` and the dataclass-erased
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


@final
@dataclass(init=False, slots=True)
class Interrupt:
    """Information about an interrupt that occurred in a node.

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
    def from_ns(cls, value: Any, ns: str) -> Interrupt:
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


@dataclass(slots=True)
class Command:
    """Update the graph's state and/or route execution to other nodes.

    A node (or a tool) returns ``Command`` when it needs to do more than merge
    a value into state: jump to another node, hand a task to a specific node
    with a private input (``Send``), or resume an interrupt.

    ``goto`` accepts one node name, a sequence of node names, a ``Send``, or a
    sequence of ``Send`` objects. ``Command.PARENT`` targets the closest parent
    graph.
    """

    graph: str | None = None
    update: Any | None = None
    resume: Any | None = None
    goto: Any = ()

    PARENT: ClassVar[str] = "__parent__"

    def __repr__(self) -> str:
        # Upstream omits every falsy field, so ``Command(resume=0)`` prints as
        # ``Command()``; keeping that quirk means log output stays comparable.
        contents = ", ".join(
            f"{key}={value!r}" for key, value in asdict(self).items() if value
        )
        return f"Command({contents})"

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        """Normalise ``update`` into ``(channel, value)`` pairs."""
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

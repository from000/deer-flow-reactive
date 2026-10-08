"""Call-shape resolution for `ReactiveGraph`'s dual call surface.

One graph object serves two audiences:

* the **native** surface — ``invoke(event, payload, config=...)`` — where the
  caller names the entry event explicitly, and
* the **Runnable** surface — ``invoke(input, config=None)`` — which LangChain's
  composition helpers (``|``, ``.batch``, ``.with_retry``, ``.with_fallbacks``)
  call positionally.

Before this module existed the Runnable surface was identity-only: the graph
inherited ``langchain_core.Runnable`` (so ``coerce_to_runnable`` accepted it)
but every composed call failed, because ``RunnableSequence`` invokes steps as
``step.invoke(input, config)`` while the graph expected ``(event, payload)``.
Measured before the fix: ``pipe.invoke`` / ``pipe.batch`` / ``pipe.stream`` and
both retry/fallback wrappers failed — 7 of 7 composed forms.

Both surfaces can receive two arguments, so the rules are explicit and
conservative; a mistyped event must keep failing loudly instead of silently
re-running the graph with the event name as payload:

1. one positional argument is the Runnable form;
2. two arguments whose first names a *declared* event are the native form;
3. an undeclared string plus a config that looks exactly like a LangChain
   ``RunnableConfig`` (non-empty, keys all drawn from the known set, at least
   one of the load-bearing keys) is the Runnable form with a string payload;
4. anything else falls back to the native reading, so the engine's own
   ``no task routes for event ...`` hint (with its fix suggestion) is raised
   rather than a bare ``TypeError``.

Everything here is pure — no graph state is touched — so the rules stay unit
testable and the graph module stays a thin binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reactivegraph.errors import GraphBuildError

__all__ = (
    "MISSING",
    "configs_for",
    "declared_events",
    "default_event",
    "resolve_batch_call",
    "resolve_call",
)

_ENTRY_HINT = (
    " — Hint: 用 b.entry(<event>) 声明入口事件,或改用 "
    "invoke(event, payload) 显式指定事件"
)

# Keys LangChain puts on a RunnableConfig. The whole set is used as an
# allow-list so a plain payload dict is never mistaken for a config.
_CONFIG_KEYS = frozenset(
    {
        "callbacks",
        "configurable",
        "metadata",
        "tags",
        "recursion_limit",
        "run_name",
        "run_id",
    }
)
# At least one of these must be present: they are what the framework itself
# always supplies, so a random payload dict with a "metadata" key stays native.
_LOAD_BEARING_CONFIG_KEYS = frozenset(
    {"callbacks", "configurable", "run_id", "recursion_limit"}
)


class _Missing:
    """Sentinel distinguishing "argument omitted" from an explicit ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


def declared_events(definition: Any) -> tuple[str, ...]:
    """Every event routed by at least one task, in declaration order."""
    seen: list[str] = []
    for task in getattr(definition, "tasks", ()):
        for event in getattr(task, "on", ()):
            if event not in seen:
                seen.append(str(event))
    return tuple(seen)


def default_event(definition: Any) -> str:
    """The entry event used by the Runnable call form.

    ``GraphBuilder.entry`` wins when declared; a graph routing exactly one
    event infers it; anything else fails closed rather than guessing which
    branch the caller meant.
    """
    events = declared_events(definition)
    declared = getattr(definition, "entry_event", None)
    if declared is not None:
        if declared in events:
            return str(declared)
        raise GraphBuildError(
            f"entry event {declared!r} is not routed by any task; declared "
            f"events: {', '.join(events) or '(none)'}{_ENTRY_HINT}"
        )
    if len(events) == 1:
        return events[0]
    if not events:
        raise GraphBuildError(
            "no task routes: this graph declares no events, so the Runnable "
            "call form has no entry event" + _ENTRY_HINT
        )
    raise GraphBuildError(
        f"cannot infer the entry event from {len(events)} declared events "
        f"({', '.join(events)}){_ENTRY_HINT}"
    )


def _as_runnable_config(value: Any) -> dict[str, Any] | None:
    """Return *value* as a config when it can only be a ``RunnableConfig``."""
    if not isinstance(value, Mapping) or not value:
        return None
    keys = set(value)
    if not keys <= _CONFIG_KEYS or not (keys & _LOAD_BEARING_CONFIG_KEYS):
        return None
    return dict(value)


def resolve_call(
    definition: Any,
    first: Any,
    second: Any = MISSING,
    config: dict | None = None,
) -> tuple[str, Any, dict | None]:
    """Resolve ``(event, payload, config)`` for a single-run call."""
    if second is MISSING:
        if isinstance(first, str) and first in declared_events(definition):
            # ``invoke("run")`` on a graph routing ``run`` is a missing-payload
            # mistake, not a string payload: fail loudly instead of running the
            # graph with the event name as its input.
            raise TypeError(
                f"invoke(event, payload) requires a payload; got only the "
                f"declared event {first!r}. Pass invoke({first!r}, <payload>)."
            )
        return default_event(definition), first, config
    if isinstance(first, str) and first in declared_events(definition):
        return first, second, config
    if isinstance(first, str) and config is None:
        runnable_config = _as_runnable_config(second)
        if runnable_config is not None:
            return default_event(definition), first, runnable_config
    if not isinstance(first, str) and config is None and isinstance(second, Mapping):
        # ``step.invoke(input, config)`` from RunnableSequence: a non-string
        # entry can only be the payload.
        return default_event(definition), first, dict(second)
    return first, second, config


def resolve_batch_call(
    definition: Any,
    first: Any,
    second: Any = MISSING,
    config: dict | Sequence[dict] | None = None,
) -> tuple[str, Sequence[Any], dict | Sequence[dict] | None]:
    """Resolve ``(event, inputs, configs)`` for a batch call.

    The native form always names its event first, and the Runnable form always
    passes a sequence of inputs first, so a string in the first position is
    unambiguous.
    """
    if isinstance(first, str) and second is not MISSING:
        inputs = second
        if isinstance(inputs, (str, bytes, bytearray)) or not isinstance(inputs, Sequence):
            raise TypeError(
                "batch(event, payloads) expects a sequence of payloads, "
                f"got {type(inputs).__name__}"
            )
        return first, inputs, config
    if isinstance(first, str):
        # ``batch("run")`` / ``batch("a string")``: a bare string is not a
        # sequence of inputs, and iterating it would batch its characters.
        raise TypeError(
            "batch expects a sequence of inputs (or the native "
            "batch(event, payloads) form); got a bare string"
        )
    if second is not MISSING and config is None:
        return default_event(definition), first, second
    return default_event(definition), first, config


def configs_for(
    configs: dict | Sequence[dict] | None, length: int
) -> list[dict | None]:
    """Expand a batch's config argument to one config per input.

    LangChain passes a *list* of configs when a sequence runs through a
    pipeline step; a single mapping means "same config for every input".
    """
    if configs is None:
        return [None] * length
    if isinstance(configs, Mapping):
        return [dict(configs) for _ in range(length)]
    if isinstance(configs, (str, bytes, bytearray)) or not isinstance(configs, Sequence):
        raise TypeError(
            "config must be a mapping or a list of mappings, "
            f"got {type(configs).__name__}"
        )
    if len(configs) != length:
        raise ValueError(
            "config must be a list of the same length as inputs, "
            f"but got {len(configs)} configs for {length} inputs"
        )
    return [dict(item) if isinstance(item, Mapping) else None for item in configs]

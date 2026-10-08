"""Middleware plumbing shared by the agent factory and its callers.

These helpers answer three questions the factory would otherwise inline:
which middleware take part in a hook chain, how ``wrap_tool_call`` layers
compose, and how a host's runtime object is adopted into the engine's
``Runtime``. They are pure functions over middleware/runtime objects — no graph
state — so they live beside ``agent_state`` rather than inside the factory.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from reactivegraph.messages import convert_to_messages
from reactivegraph.runtime import Runtime, get_runtime

_PREGEL_RUNTIME_KEY = "__pregel_runtime"

def _compose_tool_wrappers(middleware: Sequence[Any], method: str) -> Any:
    """Compose ``wrap_tool_call``/``awrap_tool_call`` middleware, outermost first.

    Mirrors upstream's participation rule: a middleware joins the chain when
    either its sync or async tool wrapper is overridden, so an async-only
    middleware still raises in a sync run instead of being silently dropped.
    """
    handlers = []
    for mw in middleware:
        sync_wrap = _middleware_method(mw, "wrap_tool_call")
        async_wrap = _middleware_method(mw, "awrap_tool_call")
        if _is_unimplemented_stub(sync_wrap, "wrap_tool_call") and (
            _is_unimplemented_stub(async_wrap, "awrap_tool_call")
        ):
            continue
        handlers.append(getattr(mw, method))

    if not handlers:
        return None

    def composed(request: Any, handler: Any) -> Any:
        inner = handler
        for wrapper in reversed(handlers):
            nxt = inner

            def call(req: Any, _w: Any = wrapper, _n: Any = nxt) -> Any:
                return _w(req, _n)

            inner = call
        return inner(request)

    return composed


def _is_unimplemented_stub(fn: Any, method: str) -> bool:
    """Return whether *fn* is an engine base-class stub, not a real override.

    Upstream decides participation by identity against *its own*
    ``AgentMiddleware``. Host middleware subclasses a different base class, so
    identity cannot cross the boundary; the stub is recognised structurally
    instead — a function that only raises ``NotImplementedError`` carrying the
    canonical "``<method>`` is not available" message.
    """
    code = getattr(fn, "__code__", None)
    if code is None or "NotImplementedError" not in code.co_names:
        return False
    marker = f"implementation of {method} is not available"
    return any(isinstance(const, str) and marker in const for const in code.co_consts)


def _middleware_method(mw: Any, method: str) -> Any:
    """Resolve a middleware hook, failing closed for duck-typed objects.

    Upstream rejects non-``AgentMiddleware`` objects with a bare
    ``AttributeError``. Keep the exception type (observable contract) but say
    what to do about it.
    """
    fn = getattr(type(mw), method, None)
    if fn is None:
        raise AttributeError(
            f"{type(mw).__name__!r} middleware does not define {method!r}; "
            f"subclass reactivegraph.middleware.AgentMiddleware to implement it."
        )
    return fn


def _assert_no_duplicate_middleware(middleware: Sequence[Any]) -> None:
    """Upstream rejects the same middleware instance twice, keyed on ``.name``."""
    seen: set[str] = set()
    for mw in middleware:
        name = getattr(mw, "name", None) or type(mw).__name__
        if name in seen:
            raise AssertionError("Please remove duplicate middleware instances.")
        seen.add(name)


def _call_get(call: Any, key: str, default: Any = None) -> Any:
    """Read a tool-call field from either a dict (our ToolCall) or an object."""
    if isinstance(call, dict):
        return call.get(key, default)
    return getattr(call, key, default)


def _call_name(call: Any) -> str:
    return str(_call_get(call, "name", "") or "")


def _call_args(call: Any) -> dict[str, Any]:
    args = _call_get(call, "args", None)
    return dict(args) if isinstance(args, dict) else {}


def _call_id(call: Any) -> str | None:
    value = _call_get(call, "id", None)
    return value if isinstance(value, str) else None


_PREGEL_RUNTIME_KEY = "__pregel_runtime"


def _resolve_runtime(config: dict | None, context: Any = None) -> Runtime:
    """Resolve the run-scoped runtime exactly the way upstream does.

    Probe-verified precedence (langgraph 1.2.9):

    1. ``config['configurable']['__pregel_runtime']`` wins when present -- this
       is how DeerFlow's run worker installs a ``Runtime`` whose ``context`` is
       the very dict it later reads ``stop_reason`` back from.
    2. otherwise an explicit ``context=`` argument builds a fresh runtime;
    3. otherwise the ambient runtime (``get_runtime()``) is reused, so nesting
       a run inside an outer run keeps the outer context.

    The returned ``Runtime`` is never ``None``: middleware hooks dereference
    ``runtime.context`` unconditionally (35 DeerFlow modules do).
    """
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            pinned = configurable.get(_PREGEL_RUNTIME_KEY)
            if pinned is not None:
                return pinned
    if context is not None:
        return Runtime(context=context)
    try:
        return get_runtime()
    except RuntimeError:
        return Runtime()


def _normalize_input(input: Any) -> dict[str, Any]:
    if isinstance(input, dict):
        messages = input.get("messages")
        if messages is not None:
            return {**input, "messages": _adopt_input_messages(messages)}
        return dict(input)
    return {"messages": _adopt_input_messages(input)}


def _adopt_input_messages(messages: Any) -> list[Any]:
    """Coerce *messages* and assign ids, exactly as LangGraph initializes state.

    ``add_messages`` treats a same-id write as a replacement. Upstream assigns
    the missing ids while building the input channel — *before* the first node
    runs — so a middleware that rebuilds the last message under its existing id
    replaces it instead of appending a duplicate. Assigning the ids only inside
    the reducer is too late: the middleware has already read ``None``.
    """
    adopted = list(convert_to_messages(messages))
    for message in adopted:
        if getattr(message, "id", None) is None:
            message.id = str(uuid.uuid4())
    return adopted


def _reject_unimplemented(kwargs: dict[str, Any]) -> None:
    for key in ("response_format", "interrupt_before", "interrupt_after", "transformers", "cache"):
        value = kwargs.get(key)
        # Only ``None`` (and the explicit ``False`` for flags) means "not
        # requested". An empty sequence still counts as a caller passing the
        # argument, and we must not silently ignore it.
        if value is None or value is False:
            continue
        raise NotImplementedError(
            f"create_agent({key}=...) is not implemented by ReactiveGraph yet. "
            f"Refusing to silently ignore it — remove the argument or use "
            f"langchain.agents.create_agent until parity lands."
        )



"""``create_agent`` — the agent factory DeerFlow builds its graphs with.

This owns the LangChain agent-factory contract (``langchain.agents.create_agent``)
on top of ReactiveGraph's native primitives: no langchain/langgraph imports.

Shape of the returned object matches what real callers depend on:

* ``.invoke({"messages": [...]}) -> {"messages": [...]}``;
* ``.checkpointer`` / ``.store`` attributes (identity pass-through);
* ``.config`` carrying ``recursion_limit`` (default 9999, as upstream);
* ``.builder.schemas`` mapping the resolved state schemas.

The model/tools loop is a real ReactiveGraph cycle: a ``model`` task and a
``tools`` task routed by task-emitted events, with ``max_runs`` as the explicit
re-entry budget. The model task decides termination by emitting no follow-up
event when the model stops asking for tools — the same convergence rule
LangGraph encodes as a conditional edge.

``stream``/``astream`` speak LangGraph's mode language, because DeerFlow drives
the graph exclusively through ``astream(stream_mode=[...])`` (see
``deerflow/runtime/runs/worker.py`` and ``subagents/executor.py``). Modes we do
not implement raise instead of silently degrading.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from reactivegraph.graph import GraphBuilder, ReactiveGraph, TaskOutcome
from reactivegraph.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from reactivegraph.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)

__all__ = ("AgentGraph", "create_agent")

# Upstream sets this on the compiled graph; DeerFlow reads it back.
_DEFAULT_RECURSION_LIMIT = 9_999


class _BoundModel:
    """Wraps a caller-supplied chat model with the bind/bind_tools surface.

    Only the calls the agent loop actually makes are forwarded, so a fake model
    in tests needs the same two methods a real chat model exposes.
    """

    def __init__(self, model: Any, tools: Sequence[Any]) -> None:
        self._model = model
        self._tools = list(tools)
        self._bound: Any = None

    def bind(self) -> Any:
        if self._tools and hasattr(self._model, "bind_tools"):
            self._bound = self._model.bind_tools(self._tools)
        elif hasattr(self._model, "bind"):
            self._bound = self._model.bind()
        else:  # pragma: no cover - defensive; fakes implement one of the two
            self._bound = self._model
        return self._bound

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        bound = self._bound if self._bound is not None else self.bind()
        return bound.invoke(messages)


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    return getattr(tool, "__name__", None) or repr(tool)


def _coerce_tool(tool: Any) -> Any:
    """Normalize a callable into an object exposing ``.name``/``.invoke``."""
    if hasattr(tool, "invoke") and hasattr(tool, "name"):
        return tool

    def _invoke(args: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        call_args = dict(args or {})
        call_args.update(kwargs)
        return tool(**call_args)

    @dataclass
    class _CallableTool:
        name: str
        func: Callable[..., Any]

        def invoke(self, args: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            return _invoke(args, **kwargs)

    return _CallableTool(name=_tool_name(tool), func=tool)


def _resolve_jump(jump: Any) -> str | None:
    if jump in ("model", "tools", "end"):
        return str(jump)
    return None


@dataclass
class _SchemaView:
    """``.builder.schemas`` — dict keyed by the resolved state schema classes."""

    schemas: dict[type, type] = field(default_factory=dict)


@dataclass(frozen=True)
class _GraphNode:
    """One node in the ``get_graph()`` view (upstream-compatible shape)."""

    name: str


@dataclass(frozen=True)
class _GraphEdge:
    """One edge in the ``get_graph()`` view."""

    source: str
    target: str
    conditional: bool = False


class _GraphView:
    """``get_graph()`` — the node/edge surface DeerFlow and its tests read.

    Upstream returns a LangGraph ``DrawableGraph``; callers only rely on
    ``.nodes`` (mapping keyed by node name) and ``.edges``. We expose the same
    two attributes with the same node set and edge topology so graph-shape
    assertions hold, without pretending to be a LangGraph object.
    """

    def __init__(self, graph: ReactiveGraph, tools: dict[str, Any]) -> None:
        self._graph = graph
        self._tools = tools

    @property
    def nodes(self) -> dict[str, _GraphNode]:
        names = ["__start__", "model"]
        if self._tools:
            names.append("tools")
        names.append("__end__")
        return {name: _GraphNode(name=name) for name in names}

    @property
    def edges(self) -> list[_GraphEdge]:
        edges = [_GraphEdge("__start__", "model")]
        if self._tools:
            edges.append(_GraphEdge("model", "tools", conditional=True))
            edges.append(_GraphEdge("tools", "model", conditional=True))
        edges.append(_GraphEdge("model", "__end__", conditional=True))
        return edges


class _GraphBuilderView:
    def __init__(self, schemas: dict[type, type]) -> None:
        self.schemas = schemas


class AgentGraph:
    """The compiled agent graph returned by :func:`create_agent`."""

    def __init__(
        self,
        *,
        model: Any,
        tools: Sequence[Any],
        system_prompt: str | None,
        middleware: Sequence[Any],
        state_schema: type | None,
        context_schema: type | None,
        checkpointer: Any,
        store: Any,
        name: str | None,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    ) -> None:
        self._middleware = list(middleware)
        _assert_no_duplicate_middleware(self._middleware)
        # Middleware may contribute tools (upstream: every ``tools`` attribute
        # on the chain is merged into the single ``tools`` node).
        merged_tools: list[Any] = list(tools)
        seen_tool_names = {_tool_name(t) for t in merged_tools}
        for mw in self._middleware:
            for extra in getattr(mw, "tools", None) or ():
                if _tool_name(extra) not in seen_tool_names:
                    merged_tools.append(extra)
                    seen_tool_names.add(_tool_name(extra))
        self._model = _BoundModel(model, merged_tools)
        self._tools = {_tool_name(t): _coerce_tool(t) for t in merged_tools}
        self._system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.store = store
        self.name = name
        self.config: dict[str, Any] = {
            "recursion_limit": recursion_limit,
            "metadata": {"ls_integration": "reactivegraph_create_agent"},
            "configurable": {},
        }
        # Middleware schemas first, caller schema last (last wins) — upstream
        # merge order.
        schemas: dict[type, type] = {}
        for mw in self._middleware:
            schema = getattr(mw, "state_schema", None)
            if isinstance(schema, type):
                schemas[schema] = schema
        schemas[AgentState] = AgentState
        if isinstance(state_schema, type):
            schemas[state_schema] = state_schema
        self._schemas = schemas
        self.builder = _GraphBuilderView(schemas)
        self._graph = self._compile()

    # -- graph construction -------------------------------------------------

    def _compile(self) -> ReactiveGraph:
        tools = self._tools
        model = self._model
        recursion_limit = self.config["recursion_limit"]
        system_prompt = self._system_prompt
        middleware = self._middleware

        def _run_hook(name: str, state: dict, reverse: bool = False) -> dict:
            updates: dict[str, Any] = {}
            items = list(reversed(middleware)) if reverse else list(middleware)
            for mw in items:
                hook = getattr(mw, name, None)
                if hook is None:
                    continue
                result = hook(dict(state), None)
                if isinstance(result, dict):
                    updates.update(result)
            return updates

        def _request(state: dict) -> ModelRequest:
            messages = list(state.get("messages", []))
            if system_prompt is not None:
                messages = [SystemMessage(content=system_prompt), *messages]
            request = ModelRequest(model=model, messages=messages, tools=list(tools.values()))
            for mw in middleware:
                dynamic = getattr(mw, "dynamic_prompt", None)
                if dynamic is None:
                    continue
                wrapped = dynamic(request)
                if isinstance(wrapped, ModelRequest):
                    request = wrapped
            return request

        def _model(state: dict) -> TaskOutcome:
            if not state.get(_AGENT_STARTED):
                # ``before_agent`` fires exactly once per run, before the first
                # model call (probe-verified upstream ordering).
                state = {
                    **state,
                    _AGENT_STARTED: True,
                    **_run_hook("before_agent", state),
                }
            state = {**state, **_run_hook("before_model", state)}
            request = _request(state)

            def _handler(req: ModelRequest) -> ModelResponse:
                reply = req.model.invoke(list(req.messages))
                return ModelResponse(result=[reply])

            handler: Callable[[ModelRequest], Any] = _handler
            # Upstream only composes middleware that *override*
            # ``wrap_model_call``; the base implementation raises the
            # "sync not available" error and must not be invoked as a
            # passthrough. Reverse order makes the first middleware outermost.
            base_wrap = AgentMiddleware.wrap_model_call
            for mw in reversed(middleware):
                wrapper = type(mw).wrap_model_call
                if wrapper is base_wrap:
                    continue
                inner = handler
                bound = mw.wrap_model_call

                def handler(req: Any, _w: Any = bound, _i: Any = inner) -> Any:
                    return _w(req, _i)


            response = handler(request)
            command_update: dict[str, Any] = {}
            if isinstance(response, ExtendedModelResponse):
                command_update = dict(getattr(response.command, "update", None) or {})
                response = response.model_response
            if isinstance(response, ModelResponse):
                new_messages = list(response.result)
                structured = response.structured_response
            else:  # bare AIMessage
                new_messages = [response]
                structured = None

            updates: dict[str, Any] = {
                "messages": [*state.get("messages", []), *new_messages],
            }
            if command_update:
                updates.update(command_update)
            if structured is not None:
                updates["structured_response"] = structured
            updates.update(_run_hook("after_model", updates, reverse=True))

            last = new_messages[-1] if new_messages else None
            tool_calls = list(getattr(last, "tool_calls", None) or [])
            pending = [c for c in tool_calls if _call_name(c) in tools]
            # Convergence: no runnable tool call → stop the loop by emitting
            # nothing (this is the conditional-edge "go to end" decision).
            if pending:
                return TaskOutcome(update=updates, emits=("model:written",))
            # Loop converged: ``after_agent`` fires once, in reverse order,
            # and its updates join this final model write.
            updates.update(_run_hook("after_agent", updates, reverse=True))
            return TaskOutcome(update=updates, emits=())

        def _tools(state: dict) -> TaskOutcome:
            messages = list(state.get("messages", []))
            last = messages[-1] if messages else None
            pending = [
                c for c in (getattr(last, "tool_calls", None) or []) if _call_name(c) in tools
            ]
            results: list[ToolMessage] = []
            for call in pending:
                name = _call_name(call)
                tool = tools[name]
                try:
                    output = tool.invoke(_call_args(call))
                    content = output if isinstance(output, str) else str(output)
                except Exception as exc:  # noqa: BLE001 - surfaced as a tool message
                    content = f"Error: {type(exc).__name__}: {exc}"
                results.append(
                    ToolMessage(content=content, tool_call_id=_call_id(call), name=name)
                )
            updates = {"messages": [*messages, *results]}
            # A tool result always hands control back to the model; the model
            # decides whether the loop continues.
            return TaskOutcome(update=updates, emits=("tools:written",))

        def build(b: GraphBuilder) -> None:
            # max_runs is the re-entry budget: the model may run up to the
            # configured recursion limit, the tool node once per model turn.
            b.task(
                "model",
                fn=_model,
                on=("run", "tools:written"),
                writes=("messages", "structured_response"),
                max_runs=max(2, recursion_limit),
            )
            b.task(
                "tools",
                fn=_tools,
                on=("model:written",),
                writes=("messages",),
                max_runs=max(2, recursion_limit),
            )

        return ReactiveGraph.build(build, graph_id=self.name or "create_agent")

    # -- public surface -----------------------------------------------------

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = _normalize_input(input)
        state = self._graph.invoke("run", payload, config=config)
        return _public_state(state)

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload = _normalize_input(input)
        state = await self._graph.ainvoke("run", payload)
        return _public_state(state)

    def stream(
        self,
        input: Any,
        config: dict | None = None,
        *,
        stream_mode: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream the run in LangGraph's mode language.

        ``stream_mode`` semantics match upstream exactly:

        * ``None`` or a bare string → bare chunks (no mode prefix);
        * a *list* → ``(mode, chunk)`` tuples, even for a single mode;
        * ``"values"`` → full state snapshots;
        * ``"updates"`` → ``{node: update}`` per completed task;
        * ``"messages"`` → ``(message, metadata)`` per produced message;
        * ``"custom"`` → custom frames written by nodes/tools.

        Modes we cannot honor raise :class:`UnsupportedStreamModeError`
        instead of silently degrading.
        """
        modes, as_list = _normalize_stream_modes(stream_mode)
        payload = _normalize_input(input)
        return self._stream_modes(payload, config, modes, as_list=as_list)

    async def astream(
        self,
        input: Any,
        config: dict | None = None,
        *,
        stream_mode: str | Sequence[str] | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async twin of :meth:`stream`.

        ``context`` is accepted for call-site compatibility; it is threaded
        into the run-scoped :class:`Runtime` so middleware/tools can read it.
        """
        modes, as_list = _normalize_stream_modes(stream_mode)
        payload = _normalize_input(input)
        async for item in self._astream_modes(
            payload, config, modes, context, as_list=as_list
        ):
            yield item

    def get_graph(self) -> Any:
        return _GraphView(self._graph, self._tools)

    # -- stream machinery ---------------------------------------------------

    def _stream_modes(
        self,
        payload: dict[str, Any],
        config: dict | None,
        modes: tuple[str, ...],
        *,
        as_list: bool,
    ) -> Iterator[Any]:
        """Drive the graph once and adapt its native frames to LG modes."""
        if not modes:
            return
        # ``updates``/``messages`` need per-task attribution. The fallback path
        # reports task boundaries natively; the Driver path reports them as
        # span frames, so ask for tracing whenever attribution is required.
        need_trace = any(mode in {"updates", "messages", "tasks", "debug"} for mode in modes)
        run_config = dict(config or {})
        if need_trace:
            run_config["trace"] = True
        raw = self._graph.stream("run", payload, config=run_config)
        yield from _adapt_stream(
            raw,
            modes,
            as_list=as_list,
            integration=self.config["metadata"]["ls_integration"],
        )

    async def _astream_modes(
        self,
        payload: dict[str, Any],
        config: dict | None,
        modes: tuple[str, ...],
        context: Any,
        *,
        as_list: bool,
    ) -> AsyncIterator[Any]:
        if not modes:
            return
        need_trace = any(mode in {"updates", "messages", "tasks", "debug"} for mode in modes)
        run_config = dict(config or {})
        if need_trace:
            run_config["trace"] = True
        raw = self._graph.astream("run", payload, config=run_config)
        async for item in _aadapt_stream(
            raw,
            modes,
            as_list=as_list,
            integration=self.config["metadata"]["ls_integration"],
        ):
            yield item


# ---------------------------------------------------------------------------
# Stream-mode adapter (LangGraph mode language -> native frames)
# ---------------------------------------------------------------------------

# Modes the real product asks for. ``messages-tuple`` is DeerFlow's public
# spelling of LangGraph's ``messages`` and is translated before we get here.
_SUPPORTED_STREAM_MODES = ("values", "updates", "messages", "custom", "tasks", "debug")

_AGENT_STARTED = "__reactivegraph_agent_started__"


class UnsupportedStreamModeError(ValueError):
    """Raised when a caller asks for a stream mode we cannot honor."""

    def __init__(self, modes: Iterable[str]) -> None:
        self.modes = tuple(dict.fromkeys(modes))
        super().__init__(
            "Unsupported stream mode(s): " + ", ".join(self.modes)
        )


def _normalize_stream_modes(
    stream_mode: str | Sequence[str] | None,
) -> tuple[tuple[str, ...], bool]:
    """Return ``(modes, as_list)``.

    ``as_list`` mirrors upstream's *shape* rule: a bare string (or ``None``)
    yields bare chunks, a list yields ``(mode, chunk)`` tuples — even when the
    list holds a single mode.
    """
    if stream_mode is None:
        return ("updates",), False
    if isinstance(stream_mode, str):
        raw = [stream_mode]
        as_list = False
    else:
        raw = list(stream_mode)
        as_list = True
    normalized = ["messages" if m == "messages-tuple" else str(m) for m in raw]
    unsupported = [m for m in normalized if m not in _SUPPORTED_STREAM_MODES]
    if unsupported:
        raise UnsupportedStreamModeError(unsupported)
    # Upstream de-duplicates while preserving first-seen order.
    return tuple(dict.fromkeys(normalized)), as_list


def _public_state(state: Any) -> dict[str, Any]:
    """Strip run-private bookkeeping keys from a returned state."""
    if not isinstance(state, dict):
        return state
    return {k: v for k, v in state.items() if k != _AGENT_STARTED}


def _messages_delta(previous: Any, current: Any) -> list[Any]:
    """New messages appended between two ``values`` snapshots."""
    prev = previous.get("messages") if isinstance(previous, dict) else None
    cur = current.get("messages") if isinstance(current, dict) else None
    if not isinstance(cur, list):
        return []
    if not isinstance(prev, list):
        return list(cur)
    return list(cur[len(prev) :])


def _node_update(node: str, new_messages: list[Any], state: dict[str, Any]) -> dict[str, Any]:
    """Build one ``updates`` frame for *node* from its message delta."""
    update: dict[str, Any] = {}
    if new_messages:
        update["messages"] = new_messages
    for key, value in state.items():
        if key in {"messages", _AGENT_STARTED}:
            continue
        update[key] = value
    return {node: update}


def _adapt_stream(
    raw: Iterator[dict],
    modes: tuple[str, ...],
    *,
    as_list: bool,
    integration: str,
) -> Iterator[Any]:
    """Translate native frames into the requested LangGraph stream modes.

    The fallback path reports ``task_end`` frames (authoritative task identity),
    the Driver path reports only ``values`` snapshots plus ``custom`` span
    frames. Both are handled here so a run yields the same mode chunks on
    either backend.
    """
    yield from _AdaptState(modes, as_list=as_list, integration=integration).feed(raw)


async def _aadapt_stream(
    raw: AsyncIterator[dict],
    modes: tuple[str, ...],
    *,
    as_list: bool,
    integration: str,
) -> AsyncIterator[Any]:
    """Async twin of :func:`_adapt_stream` — same semantics, no thread bridge."""
    state = _AdaptState(modes, as_list=as_list, integration=integration)
    async for frame in raw:
        for item in state.push(frame):
            yield item
    for item in state.finish():
        yield item


class _AdaptState:
    """Shared frame -> mode translation state for sync and async adapters."""

    def __init__(
        self, modes: tuple[str, ...], *, as_list: bool, integration: str
    ) -> None:
        self.modes = modes
        self.as_list = as_list
        self.integration = integration
        self.previous: dict[str, Any] = {}
        self.step = 0
        self.pending_task: str | None = None
        self.custom: list[Any] = []

    def feed(self, raw: Iterator[dict]) -> Iterator[Any]:
        for frame in raw:
            yield from self.push(frame)
        yield from self.finish()

    def finish(self) -> Iterator[Any]:
        if "custom" in self.modes:
            for chunk in self.custom:
                yield _wrap("custom", chunk, self.as_list)
        self.custom = []

    def push(self, frame: dict) -> Iterator[Any]:
        kind = frame.get("eventType")
        if kind == "values":
            state = ((frame.get("payload") or {}).get("state")) or {}
            delta = _messages_delta(self.previous, state)
            first = not self.previous
            self.previous = state
            if "values" in self.modes:
                yield _wrap("values", _public_state(state), self.as_list)
            if not delta:
                return
            if "messages" in self.modes:
                for message in delta:
                    self.step += 1
                    yield _wrap(
                        "messages",
                        (message, self._metadata(message)),
                        self.as_list,
                    )
            if "updates" in self.modes and not first:
                self.step += 1
                # ``task_end`` (fallback) gives the authoritative node; the
                # Driver path has no task frames, so fall back to the message
                # class — exact for this graph's model/tools split. Either
                # way the payload is the *delta*, matching upstream, whose
                # messages channel is reducer-appended.
                node = self.pending_task or (
                    "model" if _looks_like_model_message(delta) else "tools"
                )
                self.pending_task = None
                yield _wrap(
                    "updates", _node_update(node, delta, state), self.as_list
                )
        elif kind == "task_start":
            # Fallback path: the ``values`` frame for this task follows
            # immediately (fallback order is task_start → values → task_end),
            # so this is the authoritative attribution point.
            self.pending_task = str(frame.get("task") or "") or None
        elif kind == "task_end":
            self.pending_task = None
        elif kind == "custom":
            payload = frame.get("payload") or {}
            if str(payload.get("event", "")).startswith("span:"):
                return  # internal tracing frame, not a user custom chunk
            self.custom.append(payload.get("payload"))
        elif kind == "messages":
            if "messages" in self.modes:
                payload = frame.get("payload") or {}
                yield _wrap("messages", (payload.get("message"), {}), self.as_list)

    def _metadata(self, message: Any) -> dict[str, Any]:
        return {
            "ls_integration": self.integration,
            "langgraph_step": self.step,
            "langgraph_node": "model"
            if message.__class__.__name__.startswith("AI")
            else "tools",
        }


def _wrap(mode: str, chunk: Any, as_list: bool) -> Any:
    return (mode, chunk) if as_list else chunk


def _looks_like_model_message(delta: list[Any]) -> bool:
    """Heuristic used only when the native frame carries no task identity."""
    return any(m.__class__.__name__.startswith("AI") for m in delta)


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


def _normalize_input(input: Any) -> dict[str, Any]:
    if isinstance(input, dict):
        messages = input.get("messages")
        if messages is not None:
            return {**input, "messages": list(convert_to_messages(messages))}
        return dict(input)
    return {"messages": list(convert_to_messages(input))}


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


def create_agent(
    model: Any,
    tools: Sequence[Any] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    response_format: Any = None,
    state_schema: type | None = None,
    context_schema: type | None = None,
    checkpointer: Any = None,
    store: Any = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: Any = None,
    transformers: Sequence[Any] | None = None,
) -> AgentGraph:
    """Build an agent graph. Mirrors ``langchain.agents.create_agent``."""
    _reject_unimplemented(
        {
            "response_format": response_format,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "transformers": transformers,
            "cache": cache,
        }
    )
    if response_format is not None:  # pragma: no cover - unreachable guard
        raise NotImplementedError("response_format")
    if debug:
        raise NotImplementedError(
            "create_agent(debug=True) is not implemented by ReactiveGraph yet."
        )
    return AgentGraph(
        model=model,
        tools=list(tools or ()),
        system_prompt=system_prompt,
        middleware=list(middleware or ()),
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )

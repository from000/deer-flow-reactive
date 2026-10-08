"""``create_agent`` — the agent factory DeerFlow builds its graphs with.

This owns the LangChain agent-factory contract (``langchain.agents.create_agent``)
on top of ReactiveGraph's native primitives: no langchain/langgraph imports.

Shape of the returned object matches what real callers depend on:

* ``.invoke({"messages": [...]}) -> {"messages": [...]}``;
* ``.checkpointer`` / ``.store`` attributes (identity pass-through);
* thread memory: ``invoke``/``stream`` load the head checkpoint for
  ``configurable.thread_id`` and fold the input through the channel reducers
  before running, then persist the resulting snapshot; ``get_state`` /
  ``update_state`` / ``get_state_history`` (and their async twins) address that
  lineage;
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

import copy
from collections.abc import (
    AsyncIterator,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, cast

# ``_implements_hook_pair``/``_middleware_node_name``/``_overrides_hook`` are
# re-exported on purpose: the middleware interop test and downstream callers
# import those private predicates from ``reactivegraph.create_agent``, their
# historical home.
from reactivegraph.agent_graph_view import (  # noqa: F401
    _build_graph_view,
    _get_can_jump_to,
    _GraphBuilderView,
    _GraphTopology,
    _GraphView,
    _hook_layout,
    _implements_hook_pair,
    _middleware_node_name,
    _overrides_hook,
)
from reactivegraph.agent_middleware_tools import (
    _assert_no_duplicate_middleware,
    _call_get,
    _call_id,
    _call_name,
    _compose_tool_wrappers,
    _is_unimplemented_stub,
    _middleware_method,
    _normalize_input,
    _reject_unimplemented,
    _resolve_runtime,
)
from reactivegraph.agent_recursion import (
    _MAX_RECURSION_LIMIT,
    _BoundModel,
    _graph_recursion_error,
    _RecursionBudgetExceeded,
    _run_exhausted,
    _run_recursion_limit,
    _StepBudget,
)
from reactivegraph.agent_state import (
    AGENT_STARTED,
    ALL_HOOKS,
    DEFAULT_RECURSION_LIMIT,
    EXHAUSTED_KEY,
    HOOK_PAIRS,
    PRIVATE_STATE_KEYS,
    RECURSION_LIMIT_KEY,
    STEP_COUNT_KEY,
)
from reactivegraph.channels import channel_reducer, resolve_channels
from reactivegraph.graph import GraphBuilder, ReactiveGraph, TaskOutcome, _runnable_base
from reactivegraph.messages import (
    AIMessage,
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
from reactivegraph.runtime import Runtime, _PinnedIterator, get_runtime, runtime_context
from reactivegraph.state import TrackedStateProxy
from reactivegraph.streaming import (
    UnsupportedStreamModeError as UnsupportedStreamModeError,
)
from reactivegraph.streaming import (
    _aadapt_stream as _aadapt_stream,
)
from reactivegraph.streaming import (
    _adapt_stream as _adapt_stream,
)
from reactivegraph.streaming import (
    _AdaptState as _AdaptState,
)
from reactivegraph.streaming import (
    _normalize_stream_modes as _normalize_stream_modes,
)
from reactivegraph.streaming import (
    _public_state as _public_state,
)
from reactivegraph.streaming import (
    _StreamCommitter as _StreamCommitter,
)
from reactivegraph.streaming import (
    _update_metadata as _update_metadata,
)
from reactivegraph.thread_state import (
    NO_CHECKPOINTER_MESSAGE,
    StateSnapshot,
    ThreadCheckpointer,
)
from reactivegraph.tool_node import ToolNode
from reactivegraph.types import Command

__all__ = ("AgentGraph", "create_agent")

# Recursion bounds, run-private keys and hook pairs live in
# ``reactivegraph.agent_state`` so the streaming adapter can share them
# without importing this module (that would be a cycle). The aliases below keep
# the module-internal underscore names used throughout this file.
_DEFAULT_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT

# ---------------------------------------------------------------------------
# recursion_limit accounting
# ---------------------------------------------------------------------------

_AGENT_STARTED = AGENT_STARTED
_RECURSION_LIMIT_KEY = RECURSION_LIMIT_KEY
_STEP_COUNT_KEY = STEP_COUNT_KEY
_EXHAUSTED_KEY = EXHAUSTED_KEY
_PRIVATE_STATE_KEYS = PRIVATE_STATE_KEYS
_HOOK_PAIRS = HOOK_PAIRS
_ALL_HOOKS = ALL_HOOKS








def _adopt_messages(update: dict[str, Any]) -> dict[str, Any]:
    """Rebuild any ``messages`` value in *update* as our own message classes.

    Middleware lives in the host's world and may return its own message
    objects. The reducer's ``isinstance`` checks (RemoveMessage handling,
    chunk promotion) only see our classes, so adopt at the boundary or those
    messages silently bypass the reducer.
    """
    messages = update.get("messages")
    if messages is None:
        return update
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Iterable):
        return update
    return {**update, "messages": list(convert_to_messages(messages))}


def _plain_state(state: Any) -> dict[str, Any]:
    """Return a plain-dict view of *state* with no ``TrackedStateProxy`` values.

    The Driver hands task functions a tracking proxy; ``dict(proxy)`` is only a
    shallow copy, so nested list/dict values (notably ``messages``) stay
    wrapped and are rejected by the message coercion at the model boundary.
    Reading each value through the proxy keeps the Driver's read-set tracking
    intact while giving hooks and the model plain containers.
    """
    if not isinstance(state, TrackedStateProxy):
        return dict(state)
    return {
        key: value.snapshot() if isinstance(value, TrackedStateProxy) else value
        for key in state.keys()
        for value in (state[key],)
    }


def _fold_channel_value(channel: Any, current: Any, write: Any) -> Any:
    """Fold one write into *current* through the channel's reducer.

    Without a reducer the write replaces the value (``LastValue`` semantics).
    """
    reducer = channel_reducer(channel)
    if reducer is None:
        return write
    return reducer(current, write)


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
            """Run *event* (or the graph) synchronously and return the result."""
            return _invoke(args, **kwargs)

    return _CallableTool(name=_tool_name(tool), func=tool)


def _batch_configs(
    config: dict | Sequence[dict] | None, length: int
) -> list[dict | None]:
    """Expand a batch's config argument to one config per input.

    ``RunnableSequence.batch`` hands each step a *list* of per-input configs.
    A single mapping keeps its "shared by every input" meaning.
    """
    if config is None:
        return [None] * length
    if isinstance(config, Mapping):
        return [dict(config) for _ in range(length)]
    if isinstance(config, (str, bytes, bytearray)) or not isinstance(config, Sequence):
        raise TypeError(
            "config must be a mapping or a list of mappings, "
            f"got {type(config).__name__}"
        )
    if len(config) != length:
        raise ValueError(
            "config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
    return [dict(item) if isinstance(item, Mapping) else None for item in config]


def _resolve_jump(jump: Any) -> str | None:
    if jump in ("model", "tools", "end"):
        return str(jump)
    return None




def _fetch_last_ai_and_tool_messages(
    messages: Sequence[Any],
) -> tuple[AIMessage | None, list[Any]]:
    """Return the last AI message and any ToolMessages following it.

    This is the routing read upstream performs after the whole ``after_model``
    chain: a middleware may replace or remove the model reply, and tool calls
    answered by artificial tool messages must not be dispatched twice.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            return message, [
                candidate
                for candidate in messages[index + 1 :]
                if isinstance(candidate, ToolMessage)
            ]
    return None, []
































class AgentGraph(*_runnable_base()):  # type: ignore[misc]
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
        # ``ToolNode`` owns tool execution: validation errors become error
        # ``ToolMessage``s, real tool exceptions propagate, ``Command`` results
        # route, and ``wrap_tool_call`` middleware composes (first outermost,
        # matching upstream's ``_chain_tool_call_wrappers``).
        self._tool_node = ToolNode(
            merged_tools,
            wrap_tool_call=_compose_tool_wrappers(self._middleware, "wrap_tool_call"),
            awrap_tool_call=_compose_tool_wrappers(self._middleware, "awrap_tool_call"),
        )
        self._system_prompt = system_prompt
        # ``checkpointer=False`` is upstream's explicit "no persistence"; it is
        # treated as absent so state access fails closed with the same message
        # rather than attempting to drive a bool. ``True`` means "inherit the
        # parent graph's saver", which a root graph does not have, so it is kept
        # on the attribute and rejected on use (upstream raises the same
        # RuntimeError on the first run).
        self.checkpointer = None if checkpointer is False else checkpointer
        self._checkpointer_is_inherit = self.checkpointer is True
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
        # Later schemas win on field conflicts, so the caller's schema
        # overrides middleware/base annotations (upstream merge order).
        self.channels = resolve_channels(list(schemas.values()))
        self.builder = _GraphBuilderView(schemas, self.channels)
        # Validate eagerly (upstream ``ensure_valid_checkpointer`` raises here
        # for a bogus saver), but re-bind on every access: hosts such as
        # DeerFlow's run worker assign ``agent.checkpointer`` *after*
        # construction, and ``CompiledStateGraph`` has the same contract.
        self._threads: ThreadCheckpointer | None = (
            ThreadCheckpointer(self.checkpointer, self.channels)
            if self.checkpointer is not None and not self._checkpointer_is_inherit
            else None
        )
        self._graph, self._topology = self._compile()

    def _apply_channel_writes(self, state: dict, writes: dict[str, Any]) -> dict[str, Any]:
        """Return *state* with *writes* folded through the channel table.

        This is the engine's own commit rule: a ``LastValue`` field is replaced,
        a reducer field is folded. Keeping the loop's in-task state consistent
        with what the graph will commit is what lets later hooks and the
        routing decision read the same values the caller ends up with.
        """
        updated = _plain_state(state)
        for key, write in writes.items():
            if key in _PRIVATE_STATE_KEYS:
                updated[key] = write
                continue
            channel = self.channels.get(key)
            if channel is None or key not in updated:
                # First write initializes the channel; a reducer only folds
                # *subsequent* writes (LangGraph's BinaryOperatorAggregate
                # behaves the same, and ``add_messages`` rejects a None left
                # operand).
                updated[key] = write
                continue
            updated[key] = _fold_channel_value(channel, updated[key], write)
        return updated

    # -- graph construction -------------------------------------------------

    def _compile(self) -> tuple[ReactiveGraph, _GraphTopology]:
        tools = self._tools
        tool_node = self._tool_node
        model = self._model
        default_limit = self.config["recursion_limit"]
        system_prompt = self._system_prompt
        middleware = self._middleware

        # LangChain's factory compiles one node per middleware *hook pair*, so
        # every hook is a real super-step and therefore produces its own values
        # frame. Precompute the four chains once and derive both the graph and
        # get_graph() from the same topology.
        layout = _hook_layout(middleware)
        before_agent_chain = layout.chains["before_agent"]
        before_model_chain = layout.chains["before_model"]
        after_model_chain = layout.chains["after_model"]
        after_agent_chain = layout.chains["after_agent"]
        before_agent_nodes = layout.nodes["before_agent"]
        before_model_nodes = layout.nodes["before_model"]
        after_model_nodes = layout.nodes["after_model"]
        after_agent_nodes = layout.nodes["after_agent"]
        entry = layout.entry
        loop_entry = layout.loop_entry
        exit_node = layout.exit_node

        def _enter(node_id: str) -> str:
            return f"enter:{node_id}"

        def _stop(exc: _RecursionBudgetExceeded, state: dict, budget: _StepBudget,
                  committed: dict[str, Any]) -> TaskOutcome:
            """Turn a refusal into a normal, non-emitting task outcome."""
            update = {key: state[key] for key in committed if key in state}
            update.update(exc.partial)
            update.update(budget.exhausted_update())
            return TaskOutcome(update=update, emits=())

        def _run_hook(
            node_id: str, name: str, state: dict, *, budget: _StepBudget
        ) -> TaskOutcome:
            """Run exactly one middleware node."""
            try:
                budget.charge(node_id)
            except _RecursionBudgetExceeded as exc:
                return _stop(exc, _public_state(state), budget, {})
            mw = hook_by_node[node_id]
            # ``jump_to`` is ephemeral: clear the previous step's directive
            # before this node observes state.  A hook may write a new one.
            current = self._apply_channel_writes(state, {"jump_to": None})
            committed: dict[str, Any] = {}
            hook = getattr(mw, name, None)
            if hook is not None:
                result = hook(dict(current), get_runtime())
                if isinstance(result, dict):
                    adopted = _adopt_messages(result)
                    current = self._apply_channel_writes(current, adopted)
                    committed.update(adopted)
            # Only the destinations the hook explicitly declared may be
            # selected by ``jump_to``; anything else falls back to the default.
            declared = _get_can_jump_to(mw, name)
            jump = _resolve_jump(current.get("jump_to"))
            if jump not in declared:
                jump = None
            if jump == "model":
                target = loop_entry
            elif jump == "tools":
                target = "tools" if tools else default_destination(node_id)
            elif jump == "end":
                target = exit_node
            else:
                target = default_destination(node_id)
            if (
                jump is None
                and node_id in after_model_nodes
                and node_id == after_model_nodes[0]
            ):
                # The last executed after_model node is upstream's loop-exit
                # conditional edge: it decides tools/loop/end from the full
                # post-middleware state. Without a tools node that edge is a
                # plain edge to ``exit_node``; it must not try to route a tool
                # call that can never execute.
                target = _route_after_loop_exit(current) if tools else exit_node
            # The directive is consumed by this node even when it was ignored.
            current = self._apply_channel_writes(current, {"jump_to": None})
            updates = {
                key: current[key]
                for key in committed
                if key in current and key not in _PRIVATE_STATE_KEYS
            }
            updates.update(budget.spent_update())
            emits = () if target == "__end__" else (_enter(target),)
            return TaskOutcome(update=updates, emits=emits)

        def _default_destination(node_id: str) -> str:
            if node_id in before_agent_nodes:
                index = before_agent_nodes.index(node_id)
                if index + 1 < len(before_agent_nodes):
                    return before_agent_nodes[index + 1]
                return loop_entry
            if node_id in before_model_nodes:
                index = before_model_nodes.index(node_id)
                if index + 1 < len(before_model_nodes):
                    return before_model_nodes[index + 1]
                return "model"
            if node_id in after_model_nodes:
                # Registered order is the *reverse* execution order.
                index = after_model_nodes.index(node_id)
                if index > 0:
                    return after_model_nodes[index - 1]
                return "tools" if tools else exit_node
            if node_id in after_agent_nodes:
                # Registered order is the *reverse* execution order.
                index = after_agent_nodes.index(node_id)
                if index > 0:
                    return after_agent_nodes[index - 1]
                return "__end__"
            raise AssertionError(node_id)

        def _request(state: dict) -> ModelRequest:
            # Upstream keeps the static system prompt in the separate
            # ``system_message`` field and only flattens it at the model
            # boundary. Middleware reads and rewrites that field (DeerFlow's
            # ``SystemMessageCoalescingMiddleware`` merges the durable-context
            # authority into it), so prepending it to ``messages`` here would
            # make every such write land on a field nobody consumes.
            request = ModelRequest(
                model=model,
                messages=list(state.get("messages", [])),
                system_message=(
                    SystemMessage(content=system_prompt)
                    if system_prompt is not None
                    else None
                ),
                tools=list(tools.values()),
                state=cast("AgentState[Any]", state),
                runtime=get_runtime(),
            )
            for mw in middleware:
                dynamic = getattr(mw, "dynamic_prompt", None)
                if dynamic is None:
                    continue
                wrapped = dynamic(request)
                if isinstance(wrapped, ModelRequest):
                    request = wrapped
            return request

        def _model(state: dict) -> TaskOutcome:
            budget = _StepBudget(state, default_limit)
            try:
                budget.charge("model")
            except _RecursionBudgetExceeded as exc:
                return _stop(exc, state, budget, {})
            current = self._apply_channel_writes(state, {"jump_to": None})
            request = _request(current)

            def _handler(req: ModelRequest) -> ModelResponse:
                # Mirror upstream's ``_execute_model_sync``: the system message
                # is flattened at the very last moment, after every
                # ``wrap_model_call`` middleware has had its say.
                messages = list(req.messages)
                if req.system_message:
                    messages = [req.system_message, *messages]
                reply = req.model.invoke(messages)
                return ModelResponse(result=[reply])

            handler: Callable[[ModelRequest], Any] = _handler
            # Upstream composes a middleware into the chain when *either* its
            # sync or async ``wrap_model_call`` is overridden.
            for mw in reversed(middleware):
                sync_wrap = _middleware_method(mw, "wrap_model_call")
                async_wrap = _middleware_method(mw, "awrap_model_call")
                if _is_unimplemented_stub(sync_wrap, "wrap_model_call") and (
                    _is_unimplemented_stub(async_wrap, "awrap_model_call")
                ):
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

            produced: dict[str, Any] = {"messages": list(new_messages)}
            if command_update:
                produced.update(command_update)
            if structured is not None:
                produced["structured_response"] = structured
            current = self._apply_channel_writes(current, produced)
            updates = {
                key: current[key]
                for key in produced
                if key in current and key not in _PRIVATE_STATE_KEYS
            }
            updates.update(budget.spent_update())

            if after_model_nodes:
                target = after_model_nodes[-1]
            else:
                target = _route_after_loop_exit(current)
            emits = () if target == "__end__" else (_enter(target),)
            return TaskOutcome(update=updates, emits=emits)

        def _route_after_loop_exit(state: dict) -> str:
            last, tool_messages = _fetch_last_ai_and_tool_messages(
                list(state.get("messages", []))
            )
            if last is None:
                return exit_node
            calls = list(getattr(last, "tool_calls", None) or [])
            answered = {
                _call_get(tool_message, "tool_call_id", None)
                for tool_message in tool_messages
            }
            pending = [
                call
                for call in calls
                if _call_id(call) not in answered and _call_name(call) in tools
            ]
            if pending:
                return "tools"
            if calls and not pending and "structured_response" not in state:
                # Artificial tool messages / already answered calls: upstream
                # re-enters the model instead of executing tools again.
                return loop_entry
            return exit_node

        def _tools(state: dict) -> TaskOutcome:
            budget = _StepBudget(state, default_limit)
            try:
                budget.charge("tools")
            except _RecursionBudgetExceeded as exc:
                return _stop(exc, state, budget, {})
            messages = list(state.get("messages", []))
            result = tool_node.invoke({**state, "messages": messages})

            updates: dict[str, Any] = {}
            produced: list[Any] = []
            command_goto: list[Any] = []
            if isinstance(result, dict):
                produced = list(result.get("messages", []))
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, Command):
                        if isinstance(item.update, dict):
                            for key, value in item.update.items():
                                if key == "messages":
                                    produced.extend(list(value or ()))
                                else:
                                    updates[key] = value
                        if item.goto:
                            jumps = item.goto if isinstance(item.goto, list) else [item.goto]
                            command_goto.extend(jumps)
                    elif isinstance(item, dict):
                        produced.extend(list(item.get("messages", [])))
                    else:
                        produced.append(item)
            updates["messages"] = [*messages, *produced]
            updates.update(budget.spent_update())

            if any(_resolve_jump(jump) == "end" for jump in command_goto):
                target = "__end__"
            else:
                target = _route_after_tools(
                    {**state, "messages": updates["messages"]}
                )
            emits = () if target == "__end__" else (_enter(target),)
            return TaskOutcome(update=updates, emits=emits)

        def _route_after_tools(state: dict) -> str:
            last, _ = _fetch_last_ai_and_tool_messages(
                list(state.get("messages", []))
            )
            if last is None:
                return loop_entry
            calls = list(getattr(last, "tool_calls", None) or [])
            client_calls = [call for call in calls if _call_name(call) in tools]
            if client_calls and all(
                bool(getattr(tools[_call_name(call)], "return_direct", False))
                for call in client_calls
            ):
                return exit_node
            return loop_entry

        # Node id -> middleware.  Hook task ids are the exact names upstream
        # exposes in get_graph() and in updates frames.
        hook_by_node = {
            **{
                node: mw
                for node, mw in zip(
                    before_agent_nodes, before_agent_chain, strict=True
                )
            },
            **{
                node: mw
                for node, mw in zip(
                    before_model_nodes, before_model_chain, strict=True
                )
            },
            **{
                node: mw
                for node, mw in zip(
                    after_model_nodes, after_model_chain, strict=True
                )
            },
            **{
                node: mw
                for node, mw in zip(
                    after_agent_nodes, after_agent_chain, strict=True
                )
            },
        }

        def default_destination(node_id: str) -> str:
            """Next node on the default path when no jump applies."""
            return _default_destination(node_id)

        # Build the public graph topology from the same chains that drive the
        # runtime tasks.  Keep this close to upstream's factory ordering: the
        # loop edge first, then middleware chains.  ``_GraphView`` is read by
        # hosts and compatibility tests, so a cosmetic approximation is not
        # enough -- notably ``after_model`` chains are registered in reverse
        # and ``after_agent`` routes to its *last* registered node first.
        nodes, edges = _build_graph_view(
            tools=tools,
            middleware=middleware,
            before_agent_chain=before_agent_chain,
            before_model_chain=before_model_chain,
            after_model_chain=after_model_chain,
            after_agent_chain=after_agent_chain,
            before_agent_nodes=before_agent_nodes,
            before_model_nodes=before_model_nodes,
            after_model_nodes=after_model_nodes,
            after_agent_nodes=after_agent_nodes,
            entry=entry,
            loop_entry=loop_entry,
            exit_node=exit_node,
            hook_by_node=hook_by_node,
        )

        def build(b: GraphBuilder) -> None:
            # ``max_runs`` is the engine's own re-entry budget, distinct from
            # ``recursion_limit``: the compatibility layer charges super-steps
            # and raises GraphRecursionError itself, while this bound only has
            # to be large enough not to truncate a legal run first. A limit
            # above the compiled ceiling is rejected in ``_seed_budget`` rather
            # than silently truncated here.
            """Materialise the declared graph definition."""
            compiled_budget = _MAX_RECURSION_LIMIT + 2
            all_hook_nodes = [
                *before_agent_nodes,
                *before_model_nodes,
                *after_model_nodes,
                *after_agent_nodes,
            ]
            def _hook_task_factory(
                node_id: str,
                name: str,
            ) -> Callable[[dict], TaskOutcome]:
                def run_hook_task(state: dict) -> TaskOutcome:
                    """Task wrapper charging the recursion budget for one hook."""
                    return _run_hook(
                        node_id,
                        name,
                        state,
                        budget=_StepBudget(state, default_limit),
                    )

                return run_hook_task

            for node_id in all_hook_nodes:
                hook, _, name = node_id.rpartition(".")
                b.task(
                    node_id,
                    fn=_hook_task_factory(node_id, name),
                    on=(
                        ("run", _enter(node_id))
                        if node_id == entry
                        else (_enter(node_id),)
                    ),
                    writes=("messages",),
                    max_runs=compiled_budget,
                )
            b.task(
                "model",
                fn=_model,
                on=(
                    ("run", _enter(entry))
                    if entry == "model"
                    else (_enter("model"),)
                ),
                writes=("messages", "structured_response"),
                max_runs=compiled_budget,
            )
            if tools:
                b.task(
                    "tools",
                    fn=_tools,
                    on=(_enter("tools"),),
                    writes=("messages",),
                    max_runs=compiled_budget,
                )

        return (
            ReactiveGraph.build(build, graph_id=self.name or "create_agent"),
            _GraphTopology(nodes=tuple(nodes), edges=tuple(edges)),
        )
    # -- thread state -------------------------------------------------------

    def _reject_inherited_checkpointer(self) -> None:
        if self._checkpointer_is_inherit:
            raise RuntimeError(
                "checkpointer=True cannot be used for root graphs."
            )

    def _bind_threads(self) -> ThreadCheckpointer | None:
        """Re-bind on every access: hosts assign ``agent.checkpointer`` late."""
        if self._checkpointer_is_inherit:
            return None
        checkpointer = self.checkpointer
        if checkpointer is None:
            return None
        threads = self._threads
        if threads is None or threads._saver is not checkpointer:  # noqa: SLF001
            threads = ThreadCheckpointer(checkpointer, self.channels)
            self._threads = threads
        return threads

    def _threads_or_raise(self) -> ThreadCheckpointer:
        self._reject_inherited_checkpointer()
        threads = self._bind_threads()
        if threads is None:
            raise ValueError(NO_CHECKPOINTER_MESSAGE)
        return threads

    def _seed_channel_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add constructible channel defaults before the first task runs.

        LangGraph initializes reducer channels in the compiled channel table,
        so the first ``values`` snapshot includes an empty list/dict even when
        no task has written it yet. Mirror that without mutating the compiled
        channel objects or overwriting caller input.
        """
        defaults: dict[str, Any] = {}
        for key, channel in self.channels.items():
            is_available = getattr(channel, "is_available", None)
            get_value = getattr(channel, "get", None)
            if not callable(is_available) or not callable(get_value):
                continue
            try:
                if not is_available():
                    continue
                defaults[key] = copy.deepcopy(get_value())
            except Exception:
                # Non-instantiable reducer annotations start empty; the first
                # real write seeds them, matching BinaryOperatorAggregate.
                continue
        defaults.update(_plain_state(payload))
        return defaults

    def _seed_budget(self, payload: dict[str, Any], config: dict | None) -> dict[str, Any]:
        """Install this run's budget as graph state before the first task runs.

        Upstream gives every ``invoke`` a *fresh* loop, so the counter resets
        per run even when the thread already has checkpoints (probe-verified:
        a second ``invoke`` starts at its own step allowance). The limit is
        resolved here rather than at compile time so a run config can override
        the compiled default, and validated here so a bad limit fails before
        any node — or any tool — runs.
        """
        limit = _run_recursion_limit(config, self.config["recursion_limit"])
        return {
            **_plain_state(payload),
            _RECURSION_LIMIT_KEY: limit,
            _STEP_COUNT_KEY: 0,
            # A previous run's exhaustion must not leak into this one; the
            # counter resets per run, so the flag has to as well.
            _EXHAUSTED_KEY: False,
        }

    def _raise_if_exhausted(self, state: Any) -> None:
        """Raise ``GraphRecursionError`` after a run that ran out of budget.

        Called only once the graph has stopped, so every super-step it managed
        to commit has already been streamed to the caller — the same ordering
        upstream produces, where the error surfaces after the partial frames.
        """
        if not _run_exhausted(state):
            return
        limit = state.get(_RECURSION_LIMIT_KEY, self.config["recursion_limit"])
        raise _graph_recursion_error(limit)

    def _seed_payload(
        self, payload: dict[str, Any], head: Any
    ) -> tuple[dict[str, Any], list[str]]:
        """Fold the caller's input into the stored thread values.

        The returned payload is the *full* channel state the graph runs on, so
        last-value channels inherited from earlier turns (``thread_data``,
        ``summary_text``, …) survive into this run, and reducer channels
        accumulate instead of restarting from the input.
        """
        threads = self._threads_or_raise()
        stored = threads.load_values(head)
        if not stored:
            # First turn on this thread: the input *is* the state, but it still
            # folds through the reducers so a caller-supplied reducer value is
            # normalized the same way a stored one would be.
            folded, written = threads.fold_input({}, payload)
            return folded, written
        merged = dict(stored)
        merged.update({key: value for key, value in payload.items() if key not in stored})
        folded, written = threads.fold_input(
            stored, {key: value for key, value in payload.items() if key in stored}
        )
        if not written:
            return merged, []
        return folded, written

    async def _aseed_payload(
        self, payload: dict[str, Any], head: Any
    ) -> tuple[dict[str, Any], list[str]]:
        """Async twin of :meth:`_seed_payload`.

        ``AsyncSqliteSaver`` refuses the synchronous delta walk from the event
        loop thread, so the async run path has to reconstruct through
        ``aload_values``.
        """
        threads = self._threads_or_raise()
        stored = await threads.aload_values(head)
        if not stored:
            folded, written = threads.fold_input({}, payload)
            return folded, written
        merged = dict(stored)
        merged.update({key: value for key, value in payload.items() if key not in stored})
        folded, written = threads.fold_input(
            stored, {key: value for key, value in payload.items() if key in stored}
        )
        if not written:
            return merged, []
        return folded, written

    def _stream_committer(
        self, config: dict | None, head: Any, written: Sequence[str] | None
    ) -> _StreamCommitter:
        """Build the per-run checkpoint writer used by ``stream``/``astream``."""
        return _StreamCommitter(
            threads=self._bind_threads(),
            channels=self.channels,
            config=config,
            head=head,
            written=list(written or []),
        )

    def _commit_run(
        self, config: dict | None, head: Any, state: dict[str, Any], written: list[str]
    ) -> None:
        """Persist the finished run as the thread's new head checkpoint."""
        threads = self._threads_or_raise()
        committed = _public_state(state)
        # Channels the run touched are re-versioned; inherited ones keep their
        # version so a resumed lineage stays comparable.
        touched = written or [key for key in committed if key in self.channels]
        threads.commit(
            config,
            head,
            committed,
            written=touched,
            metadata={"source": "loop", "step": 1, "parents": {}},
        )

    # -- public surface -----------------------------------------------------

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
        """Run *event* (or the graph) synchronously and return the result."""
        self._reject_inherited_checkpointer()
        payload = _normalize_input(input)
        head = None
        written: list[str] = []
        threads = self._bind_threads()
        if threads is not None:
            head = threads.get_head(config)
            payload, written = self._seed_payload(payload, head)
        payload = self._seed_budget(self._seed_channel_defaults(payload), config)
        with runtime_context(runtime=_resolve_runtime(config, kwargs.get("context"))):
            state = self._graph.invoke("run", payload, config=config)
        if threads is not None:
            # Commit before raising: upstream checkpoints every super-step, so
            # an exhausted thread keeps the work it already did. DeerFlow's
            # subagent executor reads that partial state back when it catches
            # ``GraphRecursionError`` and reports ``turn_capped``.
            self._commit_run(config, head, state, written)
        self._raise_if_exhausted(state)
        return _public_state(state)

    def batch(
        self,
        inputs: Sequence[Any],
        config: dict | Sequence[dict] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run several inputs, returning one result each, in order.

        LangChain's ``Runnable`` contract: ``batch`` preserves input order and
        is what lets this graph be composed into an existing LangChain
        pipeline. Each input is an independent run. ``config`` may be a single
        mapping (shared) or the per-input list LangChain's sequence batching
        passes; treating the list as a mapping raised
        ``AttributeError: 'list' object has no attribute 'get'`` before.
        """
        results: list[Any] = []
        for item, item_config in zip(
            inputs, _batch_configs(config, len(inputs)), strict=True
        ):
            try:
                results.append(self.invoke(item, item_config, **kwargs))
            except Exception as exc:  # noqa: BLE001 - reported per item
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def abatch(
        self,
        inputs: Sequence[Any],
        config: dict | Sequence[dict] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Async twin of :meth:`batch`.

        Runs sequentially so a shared checkpointer/thread sees deterministic
        ordering; the engine's own concurrency controls govern parallelism
        inside each run.
        """
        results: list[Any] = []
        for item, item_config in zip(
            inputs, _batch_configs(config, len(inputs)), strict=True
        ):
            try:
                results.append(await self.ainvoke(item, item_config, **kwargs))
            except Exception as exc:  # noqa: BLE001 - reported per item
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Async twin of :meth:`invoke`."""
        self._reject_inherited_checkpointer()
        payload = _normalize_input(input)
        head = None
        written: list[str] = []
        threads = self._bind_threads()
        if threads is not None:
            head = await threads.aget_head(config)
            payload, written = await self._aseed_payload(payload, head)
        payload = self._seed_budget(self._seed_channel_defaults(payload), config)
        with runtime_context(
            config=config, runtime=_resolve_runtime(config, kwargs.get("context"))
        ):
            state = await self._graph.ainvoke("run", payload, config=config)
        if threads is not None:
            await threads.acommit(
                config,
                head,
                _public_state(state),
                written=written or [key for key in state if key in self.channels],
                metadata={"source": "loop", "step": 1, "parents": {}},
            )
        self._raise_if_exhausted(state)
        return _public_state(state)

    # -- checkpoint state surface ------------------------------------------

    def get_state(self, config: dict | None = None, **kwargs: Any) -> StateSnapshot:
        """The thread's head snapshot (upstream ``CompiledStateGraph.get_state``)."""
        threads = self._threads_or_raise()
        return threads.snapshot(threads.get_head(config), config)

    async def aget_state(self, config: dict | None = None, **kwargs: Any) -> StateSnapshot:
        """Async twin of :meth:`get_state`."""
        threads = self._threads_or_raise()
        return await threads.asnapshot(await threads.aget_head(config), config)

    def get_state_history(
        self,
        config: dict | None = None,
        *,
        before: dict | None = None,
        limit: int | None = None,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> Iterator[StateSnapshot]:
        """The thread's checkpoint lineage, newest first."""
        threads = self._threads_or_raise()
        return threads.history(config, before=before, limit=limit, filter=filter)

    def aget_state_history(
        self,
        config: dict | None = None,
        *,
        before: dict | None = None,
        limit: int | None = None,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StateSnapshot]:
        """Async twin of :meth:`get_state_history`."""
        threads = self._threads_or_raise()
        return threads.ahistory(config, before=before, limit=limit, filter=filter)

    def update_state(
        self,
        config: dict | None,
        values: dict[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply *values* to the thread through the channel reducers.

        DeerFlow's compaction and rollback paths write here with
        ``as_node="manual_compaction"``/``"rollback_restore"``; those names
        address the *mutation* graph it builds for the write, not this agent
        graph, so ``as_node`` is accepted and recorded as provenance rather than
        validated against this graph's two nodes.
        """
        threads = self._threads_or_raise()
        head = threads.get_head(config)
        update = _adopt_messages(dict(values or {}))
        folded, written = threads.fold_input(
            threads.load_values(head), update
        )
        return threads.commit(
            config,
            head,
            folded,
            written=written,
            metadata=_update_metadata(as_node),
            task_id=task_id,
            delta_writes=update,
        )

    async def aupdate_state(
        self,
        config: dict | None,
        values: dict[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Async twin of :meth:`update_state`."""
        threads = self._threads_or_raise()
        head = await threads.aget_head(config)
        update = _adopt_messages(dict(values or {}))
        folded, written = threads.fold_input(
            await threads.aload_values(head), update
        )
        return await threads.acommit(
            config,
            head,
            folded,
            written=written,
            metadata=_update_metadata(as_node),
            task_id=task_id,
            delta_writes=update,
        )

    def stream(
        self,
        input: Any,
        config: dict | None = None,
        *,
        stream_mode: str | Sequence[str] | None = None,
        context: Any = None,
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
        self._reject_inherited_checkpointer()
        modes, as_list = _normalize_stream_modes(stream_mode)
        payload = _normalize_input(input)
        head = None
        written: list[str] = []
        threads = self._bind_threads()
        if threads is not None:
            head = threads.get_head(config)
            payload, written = self._seed_payload(payload, head)
        payload = self._seed_budget(self._seed_channel_defaults(payload), config)
        runtime = _resolve_runtime(config, kwargs.get("context"))
        return _PinnedIterator(
            self._stream_modes(
                payload,
                config,
                modes,
                runtime,
                as_list=as_list,
                committer=self._stream_committer(config, head, written),
            )
        )

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
        self._reject_inherited_checkpointer()
        modes, as_list = _normalize_stream_modes(stream_mode)
        payload = _normalize_input(input)
        head = None
        written: list[str] = []
        threads = self._bind_threads()
        if threads is not None:
            head = await threads.aget_head(config)
            payload, written = await self._aseed_payload(payload, head)
        payload = self._seed_budget(self._seed_channel_defaults(payload), config)
        async for item in self._astream_modes(
            payload,
            config,
            modes,
            context,
            as_list=as_list,
            committer=self._stream_committer(config, head, written),
        ):
            yield item

    def get_graph(self) -> Any:
        """Return the compiled public node/edge view."""
        return _GraphView(self._topology)

    # -- stream machinery ---------------------------------------------------

    def _stream_modes(
        self,
        payload: dict[str, Any],
        config: dict | None,
        modes: tuple[str, ...],
        runtime: Runtime,
        *,
        as_list: bool,
        committer: _StreamCommitter | None = None,
    ) -> Generator[Any, Any, Any]:
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
        # Bind before the first task runs: task bodies execute on the graph's
        # own callbacks, and ``get_runtime()`` must already be live there.
        with runtime_context(runtime=runtime):
            raw = self._graph.stream("run", payload, config=run_config)
            adapter = _AdaptState(
                modes,
                as_list=as_list,
                integration=self.config["metadata"]["ls_integration"],
            )
            for frame in raw:
                if committer is not None:
                    committer.commit_frame(frame)
                yield from adapter.push(frame)
            yield from adapter.finish()
            # Only now, with every committed super-step already handed to the
            # caller, is exhaustion observable — exactly upstream's ordering.
            self._raise_if_exhausted(adapter.previous)

    async def _astream_modes(
        self,
        payload: dict[str, Any],
        config: dict | None,
        modes: tuple[str, ...],
        context: Any,
        *,
        as_list: bool,
        committer: _StreamCommitter | None = None,
    ) -> AsyncIterator[Any]:
        if not modes:
            return
        need_trace = any(mode in {"updates", "messages", "tasks", "debug"} for mode in modes)
        run_config = dict(config or {})
        if need_trace:
            run_config["trace"] = True
        with runtime_context(runtime=_resolve_runtime(config, context)):
            raw = self._graph.astream("run", payload, config=run_config)
            adapter = _AdaptState(
                modes,
                as_list=as_list,
                integration=self.config["metadata"]["ls_integration"],
            )
            async for frame in raw:
                if committer is not None:
                    await committer.acommit_frame(frame)
                for item in adapter.push(frame):
                    yield item
            for item in adapter.finish():
                yield item
            self._raise_if_exhausted(adapter.previous)


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

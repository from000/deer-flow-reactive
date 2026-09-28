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
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from reactivegraph.channels import channel_reducer, resolve_channels
from reactivegraph.errors import (
    ErrorCode,
    GraphRecursionError,
    create_error_message,
)
from reactivegraph.graph import GraphBuilder, ReactiveGraph, TaskOutcome
from reactivegraph.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    message_to_foreign_dict,
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
from reactivegraph.thread_state import (
    NO_CHECKPOINTER_MESSAGE,
    StateSnapshot,
    ThreadCheckpointer,
)
from reactivegraph.tool_node import ToolNode
from reactivegraph.types import Command

__all__ = ("AgentGraph", "create_agent")

# Upstream sets this on the compiled graph; DeerFlow reads it back.
_DEFAULT_RECURSION_LIMIT = 9_999

# The compiled task budget is a fixed wire value, so a run whose
# ``recursion_limit`` exceeds it would be silently truncated by the scheduler's
# at-most-once gate instead of raising. Refuse the run instead: the fallback's
# own event-propagation guard sits at the same ceiling, so one bound covers
# both execution paths.
_MAX_RECURSION_LIMIT = 10_000


# ---------------------------------------------------------------------------
# recursion_limit accounting
# ---------------------------------------------------------------------------

# The compiled graph's "the one-shot lifecycle hook already fired" marker.
_AGENT_STARTED = "__reactivegraph_agent_started__"

# Run-private state keys. They ride the graph like any other channel so a task
# re-entry reads the budget its predecessor spent, and they are stripped from
# every public surface (``invoke`` result, ``values``/``updates`` frames,
# checkpoints), exactly as ``_AGENT_STARTED`` already was.
_RECURSION_LIMIT_KEY = "__reactivegraph_recursion_limit__"
_STEP_COUNT_KEY = "__reactivegraph_step_count__"
_EXHAUSTED_KEY = "__reactivegraph_recursion_exhausted__"
_RUN_PRIVATE_KEYS = frozenset(
    {_AGENT_STARTED, _RECURSION_LIMIT_KEY, _STEP_COUNT_KEY, _EXHAUSTED_KEY}
)
# ``jump_to`` is an ``EphemeralValue + PrivateStateAttr`` upstream: a hook can
# use it to choose the next node, but it must not be visible to later nodes or
# to the caller. The engine's compatibility layer owns that consumption.
_PRIVATE_STATE_KEYS = _RUN_PRIVATE_KEYS | {"jump_to"}

# Hook name -> the sync/async pair LangChain compiles into one node. The
# ordering here is the order the nodes run in, which is also the order the
# step budget must be charged in: when the limit lands between two nodes, the
# earlier node's writes are the ones LangGraph committed before giving up.
_HOOK_PAIRS: dict[str, tuple[str, str]] = {
    "before_agent": ("before_agent", "abefore_agent"),
    "before_model": ("before_model", "abefore_model"),
    "after_model": ("after_model", "aafter_model"),
    "after_agent": ("after_agent", "aafter_agent"),
}


_ALL_HOOKS = tuple(hook for pair in _HOOK_PAIRS.values() for hook in pair)


def _middleware_base(middleware_type: type) -> type | None:
    """The middleware base class *middleware_type* ultimately derives from.

    Both frameworks' bases declare the whole hook surface (sync and async of
    every pair) in their own ``__dict__``, so the *most-base* class in the MRO
    that does so is the base the chain was built on — ours
    (``reactivegraph.middleware.AgentMiddleware``) or the host's
    (``langchain.agents.middleware.AgentMiddleware``) — without importing
    either one. A user's intermediate base declares only the hooks it overrides
    and is therefore not mistaken for the framework base.
    """
    for klass in reversed(middleware_type.__mro__):
        if all(hook in klass.__dict__ for hook in _ALL_HOOKS):
            return klass
    return None


def _overrides_hook(middleware_type: type, hook: str) -> bool:
    """Whether *hook* is overridden, using upstream's identity comparison.

    LangChain's factory decides participation with
    ``m.__class__.hook is not AgentMiddleware.hook`` — identity against the
    framework base, not against the most-base definition of *this* hook. That
    distinction matters for a user's intermediate base class: a hook it defines
    is an override even though it is the most-base definition in the MRO.

    A structural "does the body do anything" test cannot substitute here: a
    real override that returns ``None`` compiles to the same bytecode as the
    base's docstring-only stub, and upstream still compiles a node for it.
    """
    resolved = getattr(middleware_type, hook, None)
    if resolved is None:
        return False
    base = _middleware_base(middleware_type)
    if base is not None and resolved is getattr(base, hook, None):
        return False
    return not _is_unimplemented_stub(resolved, hook)


def _implements_hook_pair(middleware: Any, pair: tuple[str, str]) -> bool:
    """Whether *middleware* overrides either side of one sync/async hook pair."""
    middleware_type = type(middleware)
    return any(_overrides_hook(middleware_type, hook) for hook in pair)


def _validate_recursion_limit(limit: Any) -> Any:
    """Validate a run's ``recursion_limit`` exactly the way upstream does.

    ``0``/negative raise ``ValueError``; a non-numeric value raises ``TypeError``
    from the comparison itself, which is what LangGraph's own check does.
    ``None`` means "use the compiled default" and is resolved by the caller.

    One bound is *ours*, not upstream's: the compiled task budgets and the
    fallback event guard are fixed wire values, so a run whose limit exceeds
    them would be truncated silently instead of raising. Fail closed instead.
    """
    if limit is None:
        return None
    if limit < 1:
        raise ValueError("recursion_limit must be at least 1")
    if limit > _MAX_RECURSION_LIMIT:
        raise ValueError(
            f"recursion_limit {limit!r} exceeds ReactiveGraph's compiled ceiling "
            f"of {_MAX_RECURSION_LIMIT}; a larger limit would be silently "
            "truncated by the engine's task budget"
        )
    return limit


def _run_recursion_limit(config: dict | None, default: Any) -> Any:
    """Resolve a run's limit: an explicit config value wins over the default.

    ``config['recursion_limit'] = None`` keeps the compiled default, matching
    upstream (probe-verified against langgraph 1.2.9).
    """
    limit = default
    if config:
        raw = config.get("recursion_limit")
        if raw is not None:
            limit = raw
    return _validate_recursion_limit(limit)


def _run_exhausted(state: Any) -> bool:
    """Whether a finished run spent its whole budget without stopping in time.

    Two independent signals, because the two paths notice exhaustion in
    different places: a task that was *refused* mid-run records
    ``_EXHAUSTED_KEY``; a run that converged normally still owes the
    terminating super-step upstream charges while discovering there is no more
    work (``count + 1``).
    """
    if not isinstance(state, dict):
        return False
    if state.get(_EXHAUSTED_KEY):
        return True
    limit = state.get(_RECURSION_LIMIT_KEY)
    if limit is None:
        return False
    count = int(state.get(_STEP_COUNT_KEY) or 0)
    return count + 1 > limit


def _graph_recursion_error(limit: Any) -> Exception:
    """Upstream's exact exhaustion message, troubleshooting link included."""
    error_type = cast("type[Exception]", GraphRecursionError)
    return error_type(
        create_error_message(
            message=(
                f"Recursion limit of {limit} reached without hitting a stop "
                "condition. You can increase the limit by setting the "
                "`recursion_limit` config key."
            ),
            error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
        )
    )


class _RecursionBudgetExceeded(Exception):
    """Internal control flow: this run spent more super-steps than allowed.

    It is deliberately *not* :class:`GraphRecursionError`: the node that would
    have run must not run, but the nodes that already ran have to keep their
    writes, because LangGraph commits every super-step before it notices the
    limit. The task catches this, returns a stopping outcome, and the
    compatibility layer raises the real error once the graph has stopped.
    """

    def __init__(self, node: str, limit: Any) -> None:
        super().__init__(node)
        self.node = node
        self.limit = limit
        # Filled in by :meth:`AgentGraph._compile._run_hook` when the refusal
        # lands part-way through a middleware chain: upstream compiles one node
        # per middleware instance, so the instances that already ran keep their
        # writes. Empty when the very first node was refused.
        self.partial: dict[str, Any] = {}


class _StepBudget:
    """LangGraph-compatible super-step accounting for one run.

    ``recursion_limit`` counts *super-steps*, not turns. Upstream advances its
    step counter once per loop iteration — including the final iteration that
    finds no runnable task — and refuses to prepare tasks once the counter
    exceeds the limit, so a graph that executes exactly ``limit`` nodes still
    raises. Exhaustion is only observable *after* those nodes committed.

    The engine runs one task per node, so this reproduces that arithmetic:
    every node is charged in upstream's order, and the terminating super-step
    is charged once the graph has stopped (``finish``). The spent count rides
    the graph state, because a task re-entry starts from committed values and
    would otherwise restart the count on every round.
    """

    def __init__(self, state: dict, default_limit: Any) -> None:
        raw = state.get(_RECURSION_LIMIT_KEY)
        self.limit = default_limit if raw is None else raw
        # ``count`` is the *tick index* upstream is currently on. The first
        # tick (index 0) is the one that seeds the input and prepares the
        # entry node, so an untouched run is at 0 and each node advances it.
        raw_count = state.get(_STEP_COUNT_KEY)
        self.count = 0 if raw_count is None else int(raw_count)

    def charge(self, node: str) -> None:
        """Advance one tick to prepare *node*, refusing if the budget is spent.

        ``SyncPregelLoop.tick`` runs while ``self.step <= self.stop`` where
        ``stop = recursion_limit`` for a fresh run, so tick *n* prepares a node
        iff ``n <= limit``. The node prepared at tick ``limit`` therefore still
        executes; only the tick after it is refused.
        """
        self.count += 1
        if self.count > self.limit:
            raise _RecursionBudgetExceeded(node, self.limit)

    def exhausted_update(self) -> dict[str, Any]:
        """The bookkeeping patch a task returns when its budget ran out."""
        return {_STEP_COUNT_KEY: self.count, _EXHAUSTED_KEY: True}

    def spent_update(self) -> dict[str, Any]:
        """The bookkeeping patch every completed task commits."""
        return {_STEP_COUNT_KEY: self.count}


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
        # The model belongs to the host's world (e.g. a real LangChain chat
        # model), and its message coercion rejects our classes outright.
        # ``convert_to_messages`` accepts the flat ``{type, **dump}`` shape,
        # and adopting the reply keeps our reducers' isinstance checks valid.
        payload = [message_to_foreign_dict(message) for message in messages]
        return convert_to_messages([bound.invoke(payload)])[0]


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
            return _invoke(args, **kwargs)

    return _CallableTool(name=_tool_name(tool), func=tool)


def _resolve_jump(jump: Any) -> str | None:
    if jump in ("model", "tools", "end"):
        return str(jump)
    return None


def _get_can_jump_to(middleware: Any, hook_name: str) -> list[str]:
    """Read ``hook_config(can_jump_to=...)`` from either side of the pair.

    Upstream's ``_get_can_jump_to`` checks the sync hook first, then the async
    hook, and only trusts metadata on a method that actually overrides the
    framework base. The identity check matters for host middleware: a bare
    subclass must not inherit a declaration from the base stub.
    """
    middleware_type = type(middleware)
    for hook in (hook_name, f"a{hook_name}"):
        if not _overrides_hook(middleware_type, hook):
            continue
        resolved = getattr(middleware_type, hook, None)
        declared = getattr(resolved, "__can_jump_to__", None)
        if declared:
            return list(declared)
    return []


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


@dataclass(frozen=True)
class _GraphTopology:
    """The compiled node/edge shape shared by ``get_graph()`` and the tasks.

    ``get_graph()`` is not decorative for the hosts we target: DeerFlow and
    LangChain-compatibility tests inspect it to reason about routing and
    recursion.  Building it independently from the task graph had already let
    the two drift once (middleware hooks existed at runtime but not in the
    view).  Keeping one immutable value as the source for both prevents that
    class of bug.
    """

    nodes: tuple[str, ...]
    edges: tuple[_GraphEdge, ...]


class _GraphView:
    """``get_graph()`` — the node/edge surface DeerFlow and its tests read.

    Upstream returns a LangGraph ``DrawableGraph``; callers only rely on
    ``.nodes`` (mapping keyed by node name) and ``.edges``. We expose the same
    two attributes with the same node set and edge topology so graph-shape
    assertions hold, without pretending to be a LangGraph object.
    """

    def __init__(self, topology: _GraphTopology) -> None:
        self._topology = topology

    @property
    def nodes(self) -> dict[str, _GraphNode]:
        return {name: _GraphNode(name=name) for name in self._topology.nodes}

    @property
    def edges(self) -> list[_GraphEdge]:
        return list(self._topology.edges)


class _GraphBuilderView:
    """``.builder`` — the resolved schema table hosts read back.

    Upstream's ``StateGraph.builder`` also exposes ``channels``; DeerFlow's
    checkpoint-mutation path and its factory tests read it.
    """

    def __init__(self, schemas: dict[type, type], channels: dict[str, Any]) -> None:
        self.schemas = schemas
        self.channels = channels


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
        # frame.  Precompute the four chains once and derive both the graph and
        # get_graph() from the same topology.
        before_agent_chain = [
            mw
            for mw in middleware
            if _implements_hook_pair(mw, _HOOK_PAIRS["before_agent"])
        ]
        before_model_chain = [
            mw
            for mw in middleware
            if _implements_hook_pair(mw, _HOOK_PAIRS["before_model"])
        ]
        after_model_chain = [
            mw
            for mw in middleware
            if _implements_hook_pair(mw, _HOOK_PAIRS["after_model"])
        ]
        after_agent_chain = [
            mw
            for mw in middleware
            if _implements_hook_pair(mw, _HOOK_PAIRS["after_agent"])
        ]

        def _node_name(hook: str, mw: Any) -> str:
            return f"{getattr(mw, 'name', None) or type(mw).__name__}.{hook}"

        before_agent_nodes = [
            _node_name("before_agent", mw) for mw in before_agent_chain
        ]
        before_model_nodes = [
            _node_name("before_model", mw) for mw in before_model_chain
        ]
        after_model_nodes = [
            _node_name("after_model", mw) for mw in after_model_chain
        ]
        after_agent_nodes = [
            _node_name("after_agent", mw) for mw in after_agent_chain
        ]

        entry = before_agent_nodes[0] if before_agent_nodes else (
            before_model_nodes[0] if before_model_nodes else "model"
        )
        loop_entry = before_model_nodes[0] if before_model_nodes else "model"
        # Upstream's ``exit_node`` is the *last* registered after_agent node;
        # the chain itself then walks backwards to the first one.
        exit_node = after_agent_nodes[-1] if after_agent_nodes else "__end__"

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
            return _default_destination(node_id)

        # Build the public graph topology from the same chains that drive the
        # runtime tasks.  Keep this close to upstream's factory ordering: the
        # loop edge first, then middleware chains.  ``_GraphView`` is read by
        # hosts and compatibility tests, so a cosmetic approximation is not
        # enough -- notably ``after_model`` chains are registered in reverse
        # and ``after_agent`` routes to its *last* registered node first.
        nodes: list[str] = ["__start__", "model"]
        if tools:
            nodes.append("tools")
        for mw in middleware:
            for hook, chain in (
                ("before_agent", before_agent_chain),
                ("before_model", before_model_chain),
                ("after_model", after_model_chain),
                ("after_agent", after_agent_chain),
            ):
                if any(candidate is mw for candidate in chain):
                    nodes.append(_node_name(hook, mw))
        nodes.append("__end__")

        edges: list[_GraphEdge] = []

        def _append_edge(source: str, target: str, *, conditional: bool = False) -> None:
            edge = _GraphEdge(source=source, target=target, conditional=conditional)
            if edge not in edges:
                edges.append(edge)

        def _add_middleware_edge(
            source: str,
            *,
            default: str,
            can_jump_to: Sequence[str],
        ) -> None:
            """Mirror upstream ``_add_middleware_edge`` in the public view."""
            if not can_jump_to:
                _append_edge(source, default)
                return
            targets = [default]
            if "end" in can_jump_to:
                targets.append(exit_node)
            if "tools" in can_jump_to:
                targets.append("tools")
            if "model" in can_jump_to and source != loop_entry:
                targets.append(loop_entry)
            for target in targets:
                _append_edge(source, target, conditional=True)

        _append_edge("__start__", entry)
        if tools:
            tools_to_model_destinations = [loop_entry]
            if any(
                bool(getattr(tool, "return_direct", False))
                for tool in tools.values()
            ):
                tools_to_model_destinations.append(exit_node)
            for target in tools_to_model_destinations:
                _append_edge("tools", target, conditional=True)

            model_to_tools_destinations = ["tools", exit_node]
            if after_model_nodes:
                model_to_tools_destinations.append(loop_entry)
            loop_exit = after_model_nodes[0] if after_model_nodes else "model"
            for target in model_to_tools_destinations:
                _append_edge(loop_exit, target, conditional=True)
        elif after_model_nodes:
            _add_middleware_edge(
                after_model_nodes[0],
                default=exit_node,
                can_jump_to=_get_can_jump_to(
                    hook_by_node[after_model_nodes[0]], "after_model"
                ),
            )
        else:
            # No tools and no after_model: upstream adds a plain edge here.
            _append_edge("model", exit_node)

        for index, node_id in enumerate(before_agent_nodes):
            default = (
                before_agent_nodes[index + 1]
                if index + 1 < len(before_agent_nodes)
                else loop_entry
            )
            _add_middleware_edge(
                node_id,
                default=default,
                can_jump_to=_get_can_jump_to(hook_by_node[node_id], "before_agent"),
            )

        for index, node_id in enumerate(before_model_nodes):
            default = (
                before_model_nodes[index + 1]
                if index + 1 < len(before_model_nodes)
                else "model"
            )
            _add_middleware_edge(
                node_id,
                default=default,
                can_jump_to=_get_can_jump_to(hook_by_node[node_id], "before_model"),
            )

        if after_model_nodes:
            _append_edge("model", after_model_nodes[-1])
            for index in range(len(after_model_nodes) - 1, 0, -1):
                node_id = after_model_nodes[index]
                _add_middleware_edge(
                    node_id,
                    default=after_model_nodes[index - 1],
                    can_jump_to=_get_can_jump_to(
                        hook_by_node[node_id], "after_model"
                    ),
                )

        for index in range(len(after_agent_nodes) - 1, 0, -1):
            node_id = after_agent_nodes[index]
            _add_middleware_edge(
                node_id,
                default=after_agent_nodes[index - 1],
                can_jump_to=_get_can_jump_to(hook_by_node[node_id], "after_agent"),
            )
        if after_agent_nodes:
            node_id = after_agent_nodes[0]
            _add_middleware_edge(
                node_id,
                default="__end__",
                can_jump_to=_get_can_jump_to(hook_by_node[node_id], "after_agent"),
            )

        def build(b: GraphBuilder) -> None:
            # ``max_runs`` is the engine's own re-entry budget, distinct from
            # ``recursion_limit``: the compatibility layer charges super-steps
            # and raises GraphRecursionError itself, while this bound only has
            # to be large enough not to truncate a legal run first. A limit
            # above the compiled ceiling is rejected in ``_seed_budget`` rather
            # than silently truncated here.
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

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> dict[str, Any]:
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
        payload = self._seed_budget(self._seed_channel_defaults(_normalize_input(input)), config)
        runtime = _resolve_runtime(config, kwargs.get("context"))
        return _PinnedIterator(
            self._stream_modes(payload, config, modes, runtime, as_list=as_list)
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
        payload = self._seed_budget(self._seed_channel_defaults(_normalize_input(input)), config)
        async for item in self._astream_modes(
            payload, config, modes, context, as_list=as_list
        ):
            yield item

    def get_graph(self) -> Any:
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
                for item in adapter.push(frame):
                    yield item
            for item in adapter.finish():
                yield item
            self._raise_if_exhausted(adapter.previous)


# ---------------------------------------------------------------------------
# Stream-mode adapter (LangGraph mode language -> native frames)
# ---------------------------------------------------------------------------

# Modes the real product asks for. ``messages-tuple`` is DeerFlow's public
# spelling of LangGraph's ``messages`` and is translated before we get here.
_SUPPORTED_STREAM_MODES = ("values", "updates", "messages", "custom", "tasks", "debug")


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


def _update_metadata(as_node: str | None) -> dict[str, Any]:
    """Upstream's ``update_state`` provenance metadata, plus the write node."""
    metadata: dict[str, Any] = {"source": "update", "step": 1, "parents": {}}
    if as_node:
        metadata["as_node"] = as_node
    return metadata


def _public_state(state: Any) -> dict[str, Any]:
    """Strip run-private bookkeeping keys from a returned state."""
    if not isinstance(state, dict):
        return state
    return {k: v for k, v in state.items() if k not in _PRIVATE_STATE_KEYS}


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
        if key == "messages" or key in _PRIVATE_STATE_KEYS:
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

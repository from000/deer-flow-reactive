"""Public ``get_graph()`` view construction and middleware-predicate helpers.

``get_graph()`` is not decorative for the hosts this engine targets: DeerFlow
and LangGraph-compatibility tests inspect it to reason about routing and
recursion. Building it independently from the task graph had already let the
two drift once (middleware hooks existed at runtime but not in the view), so
the view is derived from the same middleware chains the runtime uses.

The middleware predicates (``_overrides_hook`` and friends) live here too
because both the view and the factory need them, and neither should import the
other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from reactivegraph.agent_middleware_tools import _is_unimplemented_stub
from reactivegraph.agent_state import ALL_HOOKS, HOOK_PAIRS

# The predicates below were written against the underscored module-internal
# names; keep the aliases so the moved bodies stay identical.
_ALL_HOOKS = ALL_HOOKS
_HOOK_PAIRS = HOOK_PAIRS

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
        """Declared node ids in registration order."""
        return {name: _GraphNode(name=name) for name in self._topology.nodes}

    @property
    def edges(self) -> list[_GraphEdge]:
        """Declared edges in registration order."""
        return list(self._topology.edges)


class _GraphBuilderView:
    """``.builder`` — the resolved schema table hosts read back.

    Upstream's ``StateGraph.builder`` also exposes ``channels``; DeerFlow's
    checkpoint-mutation path and its factory tests read it.
    """

    def __init__(self, schemas: dict[type, type], channels: dict[str, Any]) -> None:
        self.schemas = schemas
        self.channels = channels


@dataclass(frozen=True)
class _HookLayout:
    """Precomputed middleware chains, node ids and entry/exit points."""

    chains: dict[str, list[Any]]
    nodes: dict[str, list[str]]
    entry: str
    loop_entry: str
    exit_node: str


def _hook_layout(middleware: Sequence[Any]) -> _HookLayout:
    """Derive the four middleware chains and their public node ids once.

    LangChain's factory compiles one node per middleware *hook pair*, so every
    hook is a real super-step and therefore produces its own values frame. The
    graph and ``get_graph()`` are both derived from this single layout.
    """
    chains = {
        hook: [mw for mw in middleware if _implements_hook_pair(mw, pair)]
        for hook, pair in _HOOK_PAIRS.items()
    }
    nodes = {
        hook: [_middleware_node_name(hook, mw) for mw in chain]
        for hook, chain in chains.items()
    }
    before_agent_nodes = nodes["before_agent"]
    before_model_nodes = nodes["before_model"]
    after_agent_nodes = nodes["after_agent"]
    entry = (
        before_agent_nodes[0]
        if before_agent_nodes
        else (before_model_nodes[0] if before_model_nodes else "model")
    )
    loop_entry = before_model_nodes[0] if before_model_nodes else "model"
    # Upstream's ``exit_node`` is the *last* registered after_agent node; the
    # chain itself then walks backwards to the first one.
    exit_node = after_agent_nodes[-1] if after_agent_nodes else "__end__"
    return _HookLayout(
        chains=chains,
        nodes=nodes,
        entry=entry,
        loop_entry=loop_entry,
        exit_node=exit_node,
    )


def _middleware_node_name(hook: str, mw: Any) -> str:
    """Public node id upstream exposes for one middleware hook."""
    return f"{getattr(mw, 'name', None) or type(mw).__name__}.{hook}"


def _build_graph_view(
    *,
    tools: dict[str, Any],
    middleware: list[Any],
    before_agent_chain: list[Any],
    before_model_chain: list[Any],
    after_model_chain: list[Any],
    after_agent_chain: list[Any],
    before_agent_nodes: list[str],
    before_model_nodes: list[str],
    after_model_nodes: list[str],
    after_agent_nodes: list[str],
    entry: str,
    loop_entry: str,
    exit_node: str,
    hook_by_node: dict[str, Any],
) -> tuple[list[str], list[_GraphEdge]]:
    """Build the public ``get_graph()`` node/edge view.

    Mirrors upstream's factory ordering: the loop edge first, then the
    middleware chains -- ``after_model`` chains register in reverse and
    ``after_agent`` routes to its last registered node first.
    """
    nodes = _graph_view_nodes(
        tools=tools,
        middleware=middleware,
        chains={
            "before_agent": before_agent_chain,
            "before_model": before_model_chain,
            "after_model": after_model_chain,
            "after_agent": after_agent_chain,
        },
    )
    edges: list[_GraphEdge] = []
    edge = _EdgeCollector(edges)

    edge.plain("__start__", entry)
    _add_loop_edges(
        edge,
        tools=tools,
        entry=entry,
        loop_entry=loop_entry,
        exit_node=exit_node,
        after_model_nodes=after_model_nodes,
        hook_by_node=hook_by_node,
    )
    _add_chain_edges(
        edge,
        nodes=before_agent_nodes,
        tail=loop_entry,
        hook="before_agent",
        hook_by_node=hook_by_node,
    )
    _add_chain_edges(
        edge,
        nodes=before_model_nodes,
        tail="model",
        hook="before_model",
        hook_by_node=hook_by_node,
    )
    _add_chain_edges(
        edge,
        nodes=after_model_nodes,
        tail=None,
        hook="after_model",
        hook_by_node=hook_by_node,
        reverse=True,
    )
    _add_chain_edges(
        edge,
        nodes=after_agent_nodes,
        tail="__end__",
        hook="after_agent",
        hook_by_node=hook_by_node,
        reverse=True,
    )
    return nodes, edges


class _EdgeCollector:
    """Deduplicating edge sink for the public topology view."""

    def __init__(self, edges: list[_GraphEdge]) -> None:
        self._edges = edges

    def plain(self, source: str, target: str) -> None:
        """Record an unconditional edge."""
        self._append(source, target, conditional=False)

    def conditional(self, source: str, target: str) -> None:
        """Record a conditional edge."""
        self._append(source, target, conditional=True)

    def _append(self, source: str, target: str, *, conditional: bool) -> None:
        edge = _GraphEdge(source=source, target=target, conditional=conditional)
        if edge not in self._edges:
            self._edges.append(edge)


def _graph_view_nodes(
    *,
    tools: dict[str, Any],
    middleware: Sequence[Any],
    chains: dict[str, list[Any]],
) -> list[str]:
    """Ordered node ids for the public view (start/model/tools/hooks/end)."""
    ids: list[str] = ["__start__", "model"]
    if tools:
        ids.append("tools")
    for mw in middleware:
        for hook in ("before_agent", "before_model", "after_model", "after_agent"):
            if any(candidate is mw for candidate in chains[hook]):
                ids.append(_middleware_node_name(hook, mw))
    ids.append("__end__")
    return ids


def _add_middleware_edge(
    edge: _EdgeCollector,
    source: str,
    *,
    default: str,
    can_jump_to: Sequence[str],
    exit_node: str,
    loop_entry: str,
) -> None:
    """Mirror upstream ``_add_middleware_edge`` in the public view."""
    if not can_jump_to:
        edge.plain(source, default)
        return
    targets = [default]
    if "end" in can_jump_to:
        targets.append(exit_node)
    if "tools" in can_jump_to:
        targets.append("tools")
    if "model" in can_jump_to and source != loop_entry:
        targets.append(loop_entry)
    for target in targets:
        edge.conditional(source, target)


def _add_loop_edges(
    edge: _EdgeCollector,
    *,
    tools: dict[str, Any],
    entry: str,
    loop_entry: str,
    exit_node: str,
    after_model_nodes: list[str],
    hook_by_node: dict[str, Any],
) -> None:
    """The ``model``/``tools``/loop-exit conditional edges."""
    if tools:
        tools_to_model_destinations = [loop_entry]
        if any(bool(getattr(tool, "return_direct", False)) for tool in tools.values()):
            tools_to_model_destinations.append(exit_node)
        for target in tools_to_model_destinations:
            edge.conditional("tools", target)

        model_to_tools_destinations = ["tools", exit_node]
        if after_model_nodes:
            model_to_tools_destinations.append(loop_entry)
        loop_exit = after_model_nodes[0] if after_model_nodes else "model"
        for target in model_to_tools_destinations:
            edge.conditional(loop_exit, target)
    elif after_model_nodes:
        _add_middleware_edge(
            edge,
            after_model_nodes[0],
            default=exit_node,
            can_jump_to=_get_can_jump_to(hook_by_node[after_model_nodes[0]], "after_model"),
            exit_node=exit_node,
            loop_entry=loop_entry,
        )
    else:
        # No tools and no after_model: upstream adds a plain edge here.
        edge.plain("model", exit_node)


def _add_chain_edges(
    edge: _EdgeCollector,
    *,
    nodes: list[str],
    tail: str | None,
    hook: str,
    hook_by_node: dict[str, Any],
    exit_node: str = "__end__",
    loop_entry: str = "model",
    reverse: bool = False,
) -> None:
    """Edges for one middleware chain, in upstream registration order.

    ``reverse`` chains (``after_model``/``after_agent``) register in the
    opposite order, and their first registered node is the chain head that
    routes back to ``tail`` (or ``__end__`` for after_agent).
    """
    if not nodes:
        return
    if reverse and tail is None:
        # after_model: ``model`` flows into the last registered node, then the
        # chain walks backwards; the head routes to tools/exit in the loop.
        edge.plain("model", nodes[-1])
        for index in range(len(nodes) - 1, 0, -1):
            _add_middleware_edge(
                edge,
                nodes[index],
                default=nodes[index - 1],
                can_jump_to=_get_can_jump_to(hook_by_node[nodes[index]], hook),
                exit_node=exit_node,
                loop_entry=loop_entry,
            )
        return
    if reverse:
        # after_agent: registered reverse, head routes to __end__.
        for index in range(len(nodes) - 1, 0, -1):
            _add_middleware_edge(
                edge,
                nodes[index],
                default=nodes[index - 1],
                can_jump_to=_get_can_jump_to(hook_by_node[nodes[index]], hook),
                exit_node=exit_node,
                loop_entry=loop_entry,
            )
        assert tail is not None  # after_agent always passes "__end__"
        _add_middleware_edge(
            edge,
            nodes[0],
            default=tail,
            can_jump_to=_get_can_jump_to(hook_by_node[nodes[0]], hook),
            exit_node=exit_node,
            loop_entry=loop_entry,
        )
        return
    for index, node_id in enumerate(nodes):
        default = nodes[index + 1] if index + 1 < len(nodes) else (tail or "__end__")
        _add_middleware_edge(
            edge,
            node_id,
            default=default,
            can_jump_to=_get_can_jump_to(hook_by_node[node_id], hook),
            exit_node=exit_node,
            loop_entry=loop_entry,
        )


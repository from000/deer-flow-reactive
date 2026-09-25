"""The *real* lead agent — full middleware stack — must run on ReactiveGraph.

``deerflow.agents.lead_agent.agent`` is the fifth and heaviest
``create_agent`` call site: it is the graph ``langgraph.json`` points at, so
every gateway/CLI/Agent-Server run goes through it. Unlike
``tests/test_lead_agent_model_resolution.py`` (which stubs ``build_middlewares``
down to ``[]`` and stubs ``create_agent`` down to a dict capture), this module
stubs *only the provider model and the tool registry* and then builds the
genuine middleware stack: ThreadDataMiddleware, DanglingToolCallMiddleware,
ToolErrorHandlingMiddleware, LoopDetection, subagent limits, and the rest.

That matters because the middleware stack is the part of DeerFlow the engine
actually has to support: real hooks, real state schemas, real
``langgraph.config.get_config()`` reads from inside a hook, real reducers. A
graph that compiles but cannot drive a turn through that stack is not a swap.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.runnables import Runnable
from reactivegraph.create_agent import AgentGraph


class _ToolBindingFakeChatModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` plus the no-op ``bind_tools`` real models have."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[override]
        return self


def _app_config():
    from deerflow.config.app_config import AppConfig
    from deerflow.config.loop_detection_config import LoopDetectionConfig
    from deerflow.config.model_config import ModelConfig
    from deerflow.config.sandbox_config import SandboxConfig

    model = ModelConfig(
        name="rg-lead-model",
        display_name="rg-lead-model",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="rg-lead-model",
        supports_thinking=False,
        supports_vision=False,
    )
    return AppConfig(
        models=[model],
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        loop_detection=LoopDetectionConfig(),
    )


def _build_lead_agent(monkeypatch, model):
    import deerflow.tools as tools_module
    from deerflow.agents.lead_agent import agent as lead_agent_module

    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "create_chat_model", lambda **kwargs: model)
    # NOTE: build_middlewares is deliberately *not* patched.
    return lead_agent_module._make_lead_agent(
        {"configurable": {"model_name": "rg-lead-model"}},
        app_config=_app_config(),
    )


def test_lead_agent_module_uses_the_reactivegraph_engine() -> None:
    """Identity pin: the lead-agent factory must not silently fall back."""
    from langchain.agents import create_agent as lc_create_agent
    from reactivegraph.create_agent import create_agent as rg_create_agent

    from deerflow.agents.lead_agent import agent as lead_agent_module

    assert lead_agent_module.create_agent is rg_create_agent
    assert lead_agent_module.create_agent is not lc_create_agent


def test_real_lead_agent_assembly_completes_a_turn_on_reactivegraph(monkeypatch) -> None:
    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="lead-ok")]))

    graph = _build_lead_agent(monkeypatch, model)

    assert isinstance(graph, AgentGraph), type(graph).__module__
    assert type(graph).__module__.startswith("reactivegraph"), type(graph).__module__
    out = graph.invoke(
        {"messages": [("user", "hello")]},
        {"configurable": {"thread_id": "rg-lead-thread"}, "recursion_limit": 30},
    )
    assert [m.content for m in out["messages"]][-2:] == ["hello", "lead-ok"]


def test_real_bootstrap_lead_agent_completes_a_turn_on_reactivegraph(monkeypatch) -> None:
    """The bootstrap branch is a *second* ``create_agent`` call in this file.

    Both must be swapped; swapping only the regular branch would leave the
    custom-agent-creation flow on LangChain, where it would keep working and
    silently mask the gap.
    """
    import deerflow.tools as tools_module
    from deerflow.agents.lead_agent import agent as lead_agent_module

    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="boot-ok")]))
    monkeypatch.setattr(tools_module, "get_available_tools", lambda **kwargs: [])
    monkeypatch.setattr(lead_agent_module, "create_chat_model", lambda **kwargs: model)

    graph = lead_agent_module._make_lead_agent(
        {"configurable": {"model_name": "rg-lead-model", "is_bootstrap": True}},
        app_config=_app_config(),
    )

    assert isinstance(graph, AgentGraph), type(graph).__module__
    out = graph.invoke(
        {"messages": [("user", "hello")]},
        {"configurable": {"thread_id": "rg-bootstrap-thread"}, "recursion_limit": 40},
    )
    assert [m.content for m in out["messages"]][-2:] == ["hello", "boot-ok"]


def test_real_lead_agent_stack_exposes_thread_state_on_reactivegraph(monkeypatch) -> None:
    """A real hook reads ``config.configurable.thread_id`` through LangGraph's
    ``get_config()``; if the engine does not publish the caller config to that
    helper, ``ThreadDataMiddleware.before_agent`` raises instead of writing
    ``thread_data`` into state.
    """
    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="ok")]))

    graph = _build_lead_agent(monkeypatch, model)
    out = graph.invoke(
        {"messages": [("user", "hi")]},
        {"configurable": {"thread_id": "rg-lead-thread-2"}, "recursion_limit": 30},
    )

    thread_data = out.get("thread_data")
    assert thread_data is not None, out.keys()
    assert "rg-lead-thread-2" in thread_data["workspace_path"]

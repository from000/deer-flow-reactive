"""The *real* ``SubagentExecutor`` must build its agent on ReactiveGraph.

``deerflow.subagents.executor`` is the delegation path: every ``task`` tool
call in a DeerFlow run lands here. It used to call LangChain's
``create_agent`` and catch LangGraph's ``GraphRecursionError``; after the swap
it must do both against our engine.

Unlike ``tests/test_subagent_executor.py`` — which replaces the executor
package with a ``MagicMock`` to break a production import cycle and therefore
can only assert *mocked* wiring — this module imports the genuine module and
drives the genuine ``_create_agent``: only the model and the middleware
builder are stubbed, because a real provider key is not available in CI. A
silent fall back to LangChain fails on the ``AgentGraph`` identity assertion,
and a broken state/tool boundary fails on the completed turn.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.runnables import Runnable
from reactivegraph.create_agent import AgentGraph


class _ToolBindingFakeChatModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` plus the no-op ``bind_tools`` real models have."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[override]
        return self


def _real_executor_module():
    """Import the real executor, not the cycle-breaking conftest mock."""
    pytest.importorskip("deerflow.agents.thread_state")
    sys.modules.pop("deerflow.subagents.executor", None)
    package = sys.modules.get("deerflow.subagents")
    if package is not None and hasattr(package, "executor"):
        delattr(package, "executor")
    return importlib.import_module("deerflow.subagents.executor")


def _stub_app_config() -> SimpleNamespace:
    return SimpleNamespace(
        tool_search=SimpleNamespace(enabled=False, auto_promote_top_k=5),
        authorization=SimpleNamespace(enabled=False),
        skills=SimpleNamespace(deferred_discovery=False, container_path="/mnt/skills"),
        skill_evolution=SimpleNamespace(enabled=False),
    )


def _executor(module, model, **kwargs):
    from deerflow.subagents.config import SubagentConfig

    module.create_chat_model = lambda **_: model
    middleware_module = ModuleType("deerflow.agents.middlewares.tool_error_handling_middleware")
    middleware_module.build_subagent_runtime_middlewares = lambda **_: []
    sys.modules["deerflow.agents.middlewares.tool_error_handling_middleware"] = middleware_module

    config = SubagentConfig(name="rg-probe", description="probe", model="probe-model")
    return module.SubagentExecutor(
        config=config,
        tools=[],
        app_config=_stub_app_config(),
        parent_model="parent-model",
        **kwargs,
    )


def test_real_subagent_executor_builds_an_engine_graph_and_completes_a_turn() -> None:
    module = _real_executor_module()
    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="done")]))
    executor = _executor(module, model)

    agent = executor._create_agent()

    assert isinstance(agent, AgentGraph), type(agent).__module__
    assert type(agent).__module__.startswith("reactivegraph"), type(agent).__module__
    out = agent.invoke(
        {"messages": [("user", "go")]},
        {"recursion_limit": executor._recursion_limit},
    )
    assert [m.content for m in out["messages"]] == ["go", "done"]


def test_real_subagent_executor_budget_is_two_nodes_per_turn() -> None:
    """``max_turns`` reaches the engine as a super-step budget, not verbatim.

    With no middleware the chain is ``model`` -> ``tools``; one turn costs two
    super-steps. The formula is pinned against installed LangChain in
    ``tests/test_subagent_turn_budget.py``; this checks the real executor
    actually applies it before handing the limit to our engine.
    """
    module = _real_executor_module()
    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="done")]))
    executor = _executor(module, model)

    assert executor._recursion_limit is None  # resolved during assembly
    executor._create_agent()

    assert executor.config.max_turns == 50
    assert executor._recursion_limit == 100


class _AlwaysToolCallingModel(_ToolBindingFakeChatModel):
    """Never stops: every turn asks for the ``ping`` tool again."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[{"name": "ping", "args": {}, "id": "call"}],
                    )
                )
            ]
        )


def test_real_subagent_executor_graph_raises_the_error_its_handler_catches() -> None:
    """End-to-end: our engine raises exactly the class the executor catches.

    A mocked-graph test can only prove the ``except`` clause names a symbol; it
    cannot prove the engine raises that same symbol. Here the *real* executor
    builds the graph and drives it past its own budget.
    """
    from langchain_core.tools import tool
    from reactivegraph.errors import GraphRecursionError as ReactiveGraphRecursionError

    module = _real_executor_module()
    model = _AlwaysToolCallingModel(messages=iter([]))

    @tool
    def ping() -> str:
        """Return a fixed token."""
        return "pong"

    executor = _executor(module, model)
    executor.tools = [ping]
    agent = executor._create_agent()

    with pytest.raises(ReactiveGraphRecursionError) as excinfo:
        agent.invoke({"messages": [("user", "go")]}, {"recursion_limit": 4})

    assert isinstance(excinfo.value, module.GraphRecursionError)
    assert "recursion" in str(excinfo.value).lower()


def test_engine_error_message_matches_langchain_verbatim() -> None:
    """Operators grep logs for this string; the swap must not change it."""
    from langchain.agents import create_agent as lc_create_agent
    from langchain_core.tools import tool
    from langgraph.errors import GraphRecursionError as LangGraphRecursionError
    from reactivegraph.create_agent import create_agent as rg_create_agent
    from reactivegraph.errors import GraphRecursionError as ReactiveGraphRecursionError

    @tool
    def ping() -> str:
        """Return a fixed token."""
        return "pong"

    def message(create_agent, error, limit: int) -> str:
        agent = create_agent(model=_AlwaysToolCallingModel(messages=iter([])), tools=[ping])
        with pytest.raises(error) as excinfo:
            agent.invoke({"messages": [("user", "go")]}, {"recursion_limit": limit})
        return str(excinfo.value)

    for limit in (2, 3, 4, 7):
        assert message(rg_create_agent, ReactiveGraphRecursionError, limit) == message(
            lc_create_agent, LangGraphRecursionError, limit
        ), limit


def test_real_subagent_executor_catches_the_engine_recursion_error() -> None:
    """The executor's recovery branch must catch *our* error, not LangGraph's."""
    module = _real_executor_module()
    from langgraph.errors import GraphRecursionError as LangGraphRecursionError
    from reactivegraph.errors import GraphRecursionError as ReactiveGraphRecursionError

    assert module.GraphRecursionError is ReactiveGraphRecursionError
    assert module.GraphRecursionError is not LangGraphRecursionError
    assert issubclass(ReactiveGraphRecursionError, RecursionError)

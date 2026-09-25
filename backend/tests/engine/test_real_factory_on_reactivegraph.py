"""The *real* DeerFlow agent factory must run on the ReactiveGraph engine.

``create_deerflow_agent`` is the single entry point every DeerFlow execution
path goes through: the CLI, the gateway run worker and the subagent executor
all call it. If that factory builds its middleware stack on our engine and the
resulting graph completes a turn, the swap is real rather than cosmetic.

This test monkeypatches only the import site (``deerflow.agents.factory``),
exactly like the production switch will, and then asserts that the graph
completes a full turn through the genuine DeerFlow middleware chain --
including ``ThreadDataMiddleware``, which reads the ambient run config through
``langgraph.config.get_config()``.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.runnables import Runnable

from reactivegraph.create_agent import create_agent as rg_create_agent
from reactivegraph.messages import HumanMessage


class _ToolBindingFakeChatModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` plus the no-op ``bind_tools`` real models have."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[override]
        return self


def test_real_deerflow_factory_runs_on_reactivegraph(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = pytest.importorskip("deerflow.agents.factory")
    monkeypatch.setattr(factory, "create_agent", rg_create_agent)

    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="ok")]))
    agent = factory.create_deerflow_agent(model, tools=[], name="rg-factory-probe")

    out = agent.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "rg-factory-thread"}},
    )

    assert [m.content for m in out["messages"]] == ["hi", "ok"]

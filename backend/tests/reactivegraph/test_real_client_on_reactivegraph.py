"""``DeerFlowClient`` must build its agent on the ReactiveGraph engine.

``deerflow.client`` is the embedded SDK path (``DeerFlowClient.chat`` /
``.stream``) and one of the five ``create_agent`` call sites in the harness.
This test drives the *real* ``_ensure_agent`` — with only the model, prompt and
tool assembly stubbed — and then completes a turn through the resulting graph,
so a silent fall back to LangChain would fail here rather than pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.runnables import Runnable

from reactivegraph.create_agent import AgentGraph


class _ToolBindingFakeChatModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` plus the no-op ``bind_tools`` real models have."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[override]
        return self


def _client_with_stubbed_assembly(model, monkeypatch):
    from deerflow.client import DeerFlowClient

    config_mock = MagicMock()
    config_mock.models = [MagicMock()]
    config_mock.checkpointer = None
    config_mock.database.checkpoint_channel_mode = "full"
    config_mock.database.checkpoint_delta.snapshot_frequency = 10
    config_mock.skills.deferred_discovery = False
    config_mock.skills.container_path = "/mnt/skills"
    config_mock.tool_search.enabled = False
    config_mock.tool_search.auto_promote_top_k = 5

    monkeypatch.setattr("deerflow.client.get_app_config", lambda: config_mock)
    monkeypatch.setattr("deerflow.client.create_chat_model", lambda **_: model)
    monkeypatch.setattr("deerflow.client.build_middlewares", lambda *a, **k: [])
    monkeypatch.setattr("deerflow.client.apply_prompt_template", lambda **_: "be brief")
    monkeypatch.setattr("deerflow.client.get_enabled_skills_for_config", lambda *a, **k: [])
    monkeypatch.setattr(DeerFlowClient, "_get_tools", staticmethod(lambda **_: []))
    return DeerFlowClient(checkpointer=None)


def test_client_builds_an_engine_graph_and_completes_a_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("deerflow.client")
    model = _ToolBindingFakeChatModel(messages=iter([LCAIMessage(content="ok")]))
    client = _client_with_stubbed_assembly(model, monkeypatch)

    config = client._get_runnable_config("rg-client-thread")
    client._ensure_agent(config)

    assert isinstance(client._agent, AgentGraph), type(client._agent).__module__
    out = client._agent.invoke(
        {"messages": [{"role": "user", "content": "hi"}]},
        config=config,
    )
    assert [m.content for m in out["messages"]] == ["hi", "ok"]

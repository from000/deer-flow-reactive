"""Does ReactiveGraph's ``create_agent`` drive *real* LangChain models/tools?

DeerFlow hands ``create_agent`` a genuine ``BaseChatModel`` and genuine
``BaseTool`` objects. The engine owns its own message classes, so the boundary
between those two worlds is the whole ballgame: if a real chat model cannot
consume the messages we hand it, the swap is cosmetic.

These tests use upstream LangChain objects (no fakes of our own) so a green
result means the model/tool boundary is genuinely honoured.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from reactivegraph.create_agent import create_agent
from reactivegraph.messages import HumanMessage


class ToolCallingFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel + the no-op bind_tools real providers have.

    The base fake raises NotImplementedError there, and upstream create_agent
    raises identically, so this is a property of the fake rather than of the
    boundary under test.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[override]
        return self


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_real_langchain_tool_is_a_basetool() -> None:
    assert isinstance(add, BaseTool)


def test_real_langchain_model_consumes_engine_messages() -> None:
    """A real chat model must be able to read the transcript we hand it."""
    model = ToolCallingFakeChatModel(messages=iter([LCAIMessage(content="hello")]))
    agent = create_agent(model)

    out = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert [m.content for m in out["messages"]] == ["hi", "hello"]


def test_real_langchain_model_and_tool_complete_a_tool_loop() -> None:
    """model -> tool -> model, with a real LangChain tool executing."""
    model = ToolCallingFakeChatModel(
        messages=iter(
            [
                LCAIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "args": {"a": 2, "b": 3}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                LCAIMessage(content="5"),
            ]
        )
    )
    agent = create_agent(model, [add])

    out = agent.invoke({"messages": [HumanMessage(content="2+3?")]})

    tool_messages = [m for m in out["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1, [type(m).__name__ for m in out["messages"]]
    assert tool_messages[0].content == "5"
    assert tool_messages[0].tool_call_id == "c1"
    assert out["messages"][-1].content == "5"


@tool
def route(city: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Route to a city, returning a graph Command instead of a value."""
    return Command(
        update={
            "messages": [
                LCToolMessage(content=f"routed:{city}", tool_call_id=tool_call_id)
            ]
        }
    )


@tool
def describe_runtime(runtime: ToolRuntime) -> str:
    """Report which runtime dataclass was injected."""
    return f"{type(runtime).__name__}:{runtime.tool_call_id}"


def _drive(tool_obj: BaseTool, name: str, args: dict) -> dict:
    model = ToolCallingFakeChatModel(
        messages=iter(
            [
                LCAIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": "c1", "type": "tool_call"}
                    ],
                ),
                LCAIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(model, [tool_obj])
    return agent.invoke({"messages": [HumanMessage(content="go")]})


def test_real_tool_returning_command_routes_instead_of_stringifying() -> None:
    """27 harness tools return ``Command``; a stringified one would corrupt state."""
    out = _drive(route, "route", {"city": "Paris"})

    tool_messages = [m for m in out["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1, [type(m).__name__ for m in out["messages"]]
    assert tool_messages[0].content == "routed:Paris"
    assert tool_messages[0].tool_call_id == "c1"
    assert out["messages"][-1].content == "done"


def test_real_tool_receives_its_own_declared_runtime_type() -> None:
    """Host tools validate the injected runtime against *their* dataclass.

    Passing our own ``ToolRuntime`` raises a pydantic ``ValidationError``, so
    the engine must rebuild the declared type from the active runtime.
    """
    out = _drive(describe_runtime, "describe_runtime", {})

    tool_messages = [m for m in out["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1, [type(m).__name__ for m in out["messages"]]
    assert tool_messages[0].content == "ToolRuntime:c1"

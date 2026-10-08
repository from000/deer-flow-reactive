"""Native prebuilt agents (M3-F7).

High-level, dependency-free building blocks on top of the native API — the
React (ReAct) agent loop plus a tool registry. No external layout service is
required: the model callable is injected, so tests and apps use fakes or any
LLM SDK of their choice.

Design goals (why this is "native", not a langgraph mirror):

* ``ToolNode`` — a deterministic batch tool executor with per-call error
  capture, so a failing tool becomes an error message the agent can recover
  from (never an unhandled crash mid-loop);
* ``ReactiveAgent`` — the model/tools loop expressed as plain message
  threading; it composes with the reaction API later when a graph builds on
  top of the loop's final state.

Messages follow the shared shape ``{"role", "content", ...}`` with assistant
``tool_calls`` and ``{"role": "tool", "tool_call_id", "name", "content"}``
results.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = (
    "AGENT_STATE_SCHEMA",
    "AgentIterationError",
    "AgentState",
    "ReactiveAgent",
    "ToolNode",
    "ToolSpec",
    "create_react_agent",
)

# A message is a plain dict: {"role", "content", ...}. An assistant message may
# carry "tool_calls": [{"id", "name", "arguments": {...}}]; a tool result is
# {"role": "tool", "tool_call_id", "name", "content"}.
Message = dict[str, Any]
ToolCall = dict[str, Any]


class AgentIterationError(RuntimeError):
    """Raised when the ReAct loop exceeds ``max_iterations``."""


AGENT_STATE_SCHEMA = "reactivegraph.agent-state.v1"


@dataclass(frozen=True)
class AgentState:
    """Stable, versioned state shape for native agents.

    The schema is intentionally a plain dictionary at the boundary (safe over
    RGP/1) while this dataclass provides typed construction and read-only
    inspection in Python. It does not model LangGraph channels or reducers.
    """

    messages: tuple[Message, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)
    receipts: dict[str, str] = field(default_factory=dict)
    interrupts: tuple[dict[str, Any], ...] = ()
    human_decisions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this value to a plain dict."""
        return {
            "schema": AGENT_STATE_SCHEMA,
            "messages": [dict(message) for message in self.messages],
            "tool_calls": [dict(call) for call in self.tool_calls],
            "artifacts": dict(self.artifacts),
            "receipts": dict(self.receipts),
            "interrupts": [dict(item) for item in self.interrupts],
            "human_decisions": [dict(item) for item in self.human_decisions],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentState:
        """Reconstruct this value from a plain dict."""
        schema = value.get("schema")
        if schema != AGENT_STATE_SCHEMA:
            raise ValueError(f"unsupported agent state schema: {schema!r}")
        messages = value.get("messages", [])
        tool_calls = value.get("tool_calls", [])
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) and isinstance(message.get("role"), str)
            for message in messages
        ):
            raise ValueError("agent state messages must be dicts with a string role")
        if not isinstance(tool_calls, list) or not all(
            isinstance(call, dict) and isinstance(call.get("id"), str)
            for call in tool_calls
        ):
            raise ValueError("agent state tool_calls must be dicts with a string id")
        return cls(
            messages=tuple(dict(message) for message in messages),
            tool_calls=tuple(dict(call) for call in tool_calls),
            artifacts=dict(value.get("artifacts", {})),
            receipts=dict(value.get("receipts", {})),
            interrupts=tuple(dict(item) for item in value.get("interrupts", [])),
            human_decisions=tuple(dict(item) for item in value.get("human_decisions", [])),
        )


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool plus optional ReactiveGraph execution metadata."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    parameters: dict | None = None  # JSON Schema for the tool call
    kind: str = "effect"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    receipt: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ToolSpec.name must be a non-empty string")


class ToolNode:
    """Executes a batch of tool calls.

    Each call runs independently; an exception inside a tool is captured into
    an ``Error: ...`` tool message so the agent can adapt (standard ReAct
    behavior) instead of aborting the whole loop.
    """

    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        self._by_name: dict[str, ToolSpec] = {}
        for tool in tools:
            if tool.name in self._by_name:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._by_name[tool.name] = tool
        self.tools: tuple[ToolSpec, ...] = tuple(tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __call__(
        self, tool_calls: Iterable[ToolCall], *, concurrency: int = 1
    ) -> list[Message]:
        """Execute tool calls and return one tool message per call.

        ``concurrency=1`` preserves the historical serial behavior. Values above
        one execute independent calls on a bounded thread pool while preserving
        the original tool-call order in the returned messages. Tool exceptions
        are captured per call, so one failure never hides sibling results.
        """
        calls = list(tool_calls)
        if concurrency < 1:
            raise ValueError("ToolNode concurrency must be >= 1")

        def execute(call: ToolCall) -> Message:
            """Execute this tool with the given arguments."""
            name = call.get("name")
            tool = self._by_name.get(name) if isinstance(name, str) else None
            if tool is None:
                return self._error_message(call, f"unknown tool: {name!r}")
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            try:
                result = tool.fn(**arguments)
            except Exception as exc:  # noqa: BLE001 - captured for the agent
                return self._error_message(call, f"{type(exc).__name__}: {exc}")
            return {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": _stringify(result),
            }

        if concurrency == 1:
            return [execute(call) for call in calls]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(execute, calls))

    @staticmethod
    def _error_message(call: ToolCall, detail: str) -> Message:
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": call.get("name", ""),
            "content": f"Error: {detail}",
        }

    def specs(self) -> list[dict]:
        """Tool specifications in OpenAI tool-call schema form (for the model)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
        ]


@dataclass
class ReactiveAgent:
    """A ReAct agent loop: model tool_calls → tools → model … until done.

    ``model`` receives the full message history and returns either a message
    dict, a list of messages, or a plain string (wrapped as assistant content).
    The loop terminates when the most recent assistant message has no
    ``tool_calls`` or ``max_iterations`` is reached.
    """

    model: Callable[[Sequence[Message]], Message | list[Message] | str]
    tools: ToolNode
    max_iterations: int = 10
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()
    _iteration: int = field(default=0, init=False)
    _approved: set[str] = field(default_factory=set, init=False)
    _completed: set[str] = field(default_factory=set, init=False)

    def invoke(self, messages: Sequence[Message]) -> dict:
        """Run the loop over a copy of the messages; return {"messages": [...]}."""
        history: list[Message] = list(messages)
        for _ in range(self.max_iterations):
            self._iteration += 1
            response = self.model(history)
            assistant = _as_assistant_message(response)
            history.append(assistant)
            tool_calls = assistant.get("tool_calls")
            if not tool_calls:
                return {"messages": history}
            history.extend(self.tools(tool_calls))
        raise AgentIterationError(
            f"agent exceeded max_iterations={self.max_iterations} without an answer"
        )

    def invoke_state(self, state: AgentState) -> AgentState:
        """Run from a typed state, returning typed state (including interrupts)."""
        has_tool_result = any(message.get("role") == "tool" for message in state.messages)
        if not has_tool_result:
            pending_before = self._pending_interrupts(state.messages, "before")
            if pending_before:
                return AgentState(messages=state.messages, interrupts=tuple(pending_before))

        if state.tool_calls:
            tool_messages = self.tools(list(state.tool_calls))
            messages = (*state.messages, *tool_messages)
            after_interrupts = tuple(self._pending_interrupts(messages, "after"))
            return AgentState(messages=messages, interrupts=after_interrupts)

        response = self.model(list(state.messages))
        assistant = _as_assistant_message(response)
        tool_calls = assistant.get("tool_calls") or []
        before_calls = [
            call
            for call in tool_calls
            if isinstance(call.get("name"), str) and call["name"] in self.interrupt_before
        ]
        if before_calls:
            interrupts = tuple(
                {"id": f"before:{call['name']}", "value": {"taskId": call["name"]}}
                for call in before_calls
                if f"before:{call['name']}" not in self._approved
            )
            if interrupts:
                messages = (*state.messages, assistant)
                return AgentState(
                    messages=messages,
                    tool_calls=tuple(tool_calls),
                    interrupts=interrupts,
                )

        result = self.invoke(state.messages)
        after_interrupts = tuple(self._pending_interrupts(result["messages"], "after"))
        return AgentState(
            messages=tuple(result["messages"]),
            tool_calls=tuple(
                call
                for message in result["messages"]
                for call in message.get("tool_calls", [])
                if isinstance(message.get("tool_calls"), list)
            ),
            interrupts=after_interrupts,
        )

    def resume_state(self, state: AgentState, response: dict[str, Any]) -> AgentState:
        """Resume one structured human decision without mutating input state."""
        interrupt_id = response.get("id")
        if not isinstance(interrupt_id, str):
            raise ValueError("human response requires a string id")
        target = next(
            (item for item in state.interrupts if item.get("id") == interrupt_id), None
        )
        if target is None:
            raise ValueError(f"unknown interrupt: {interrupt_id}")
        self._approved.add(interrupt_id)
        return self.invoke_state(state)

    def _pending_interrupts(
        self, messages: Sequence[Message], phase: Literal["before", "after"]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        boundaries = self.interrupt_before if phase == "before" else self.interrupt_after
        for index, message in enumerate(messages):
            task_id = message.get("name") or message.get("role")
            if task_id not in boundaries:
                continue
            suffix = f":{index}" if phase == "before" else ""
            interrupt_id = f"{phase}:{task_id}{suffix}"
            if interrupt_id in self._approved:
                continue
            if phase == "after" and interrupt_id in self._completed:
                continue
            if phase == "after":
                self._completed.add(interrupt_id)
            out.append({"id": interrupt_id, "value": {"taskId": task_id}})
        return out

    def stream(self, messages: Sequence[Message]) -> Iterator[dict]:
        """Run the loop emitting per-step events (LLM token-level streaming).

        Events (same shape as ``ReactiveGraph.stream``):
          - ``{"eventType": "messages", "payload": {"message": {...}}}``: if the
            model returns a generator of tokens, one event PER TOKEN (the
            assistant message is re-assembled from the tokens); otherwise one
            event per assistant/tool message.
          - ``{"eventType": "result", "payload": {"messages": [...]}}``: the
            final message history, exactly like ``invoke``.
        """
        history: list[Message] = list(messages)
        for _ in range(self.max_iterations):
            self._iteration += 1
            response = self.model(history)
            if isinstance(response, Iterator):
                tokens: list[str] = []
                for tok in response:
                    tokens.append(str(tok))
                    yield {
                        "eventType": "messages",
                        "payload": {"message": {"role": "assistant", "content": str(tok)}},
                    }
                assistant = _as_assistant_message("".join(tokens))
            else:
                assistant = _as_assistant_message(response)
                yield {"eventType": "messages", "payload": {"message": assistant}}
            history.append(assistant)
            tool_calls = assistant.get("tool_calls")
            if not tool_calls:
                yield {"eventType": "result", "payload": {"messages": history}}
                return
            tool_messages = self.tools(tool_calls)
            for tm in tool_messages:
                yield {"eventType": "messages", "payload": {"message": tm}}
            history.extend(tool_messages)
        raise AgentIterationError(
            f"agent exceeded max_iterations={self.max_iterations} without an answer"
        )

    def __call__(self, messages: Sequence[Message]) -> dict:
        return self.invoke(messages)


def create_react_agent(
    model: Callable[[Sequence[Message]], Message | list[Message] | str],
    tools: Sequence[ToolSpec] | ToolNode,
    *,
    max_iterations: int = 10,
) -> ReactiveAgent:
    """Build a ReAct agent over the native primitives.

    ``tools`` may be a list of :class:`ToolSpec` or an existing :class:`ToolNode`.
    """
    tool_node = tools if isinstance(tools, ToolNode) else ToolNode(tools)
    return ReactiveAgent(model=model, tools=tool_node, max_iterations=max_iterations)


def _as_assistant_message(response: Message | list[Message] | str) -> Message:
    if isinstance(response, str):
        return {"role": "assistant", "content": response}
    if isinstance(response, list):
        if not response:
            return {"role": "assistant", "content": ""}
        return dict(response[-1])
    if not isinstance(response, dict):
        return {"role": "assistant", "content": str(response)}
    msg = dict(response)
    msg.setdefault("role", "assistant")
    return msg


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        import json

        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(result)

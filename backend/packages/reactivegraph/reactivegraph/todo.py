"""Planning / todo-list middleware, owned by the engine.

DeerFlow's ``TodoMiddleware`` subclasses ``TodoListMiddleware`` to add
context-loss detection and premature-exit prevention. That base class, the
``Todo`` item type, the ``PlanningState`` schema, and the ``write_todos`` tool
used to come from LangChain; importing LangChain pulls in LangGraph, so the
surface moves here while keeping upstream behaviour (system-prompt injection and
the one-``write_todos``-call-per-turn rule).

Prompts are copied verbatim from ``langchain.agents.middleware.todo`` so a host
that tuned its agent against upstream keeps the same model-facing text.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, override

from reactivegraph.messages import AIMessage, SystemMessage, ToolMessage
from reactivegraph.middleware import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)
from reactivegraph.tools import StructuredTool, ToolRuntime
from reactivegraph.types import Command

__all__ = (
    "WRITE_TODOS_SYSTEM_PROMPT",
    "WRITE_TODOS_TOOL_DESCRIPTION",
    "PlanningState",
    "Todo",
    "TodoListMiddleware",
    "WriteTodosInput",
    "write_todos",
)


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


class PlanningState(AgentState[ResponseT]):
    """State schema for the todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""


class WriteTodosInput(BaseModel):
    """Input schema for the ``write_todos`` tool."""

    todos: list[Todo]


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress and organize complex tasks.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool

Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool

1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool

It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
    - pending: Task not yet started
    - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
    - completed: Task finished successfully

2. **Task Management**:
    - Update task status in real-time as you work
    - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
    - Complete current tasks before starting new ones
    - Remove tasks that are no longer relevant from the list entirely
    - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
    - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress.

3. **Task Completion Requirements**:
    - ONLY mark a task as completed when you have FULLY accomplished it
    - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
    - When blocked, create a new task describing what needs to be resolved
    - Never mark a task as completed if:
        - There are unresolved issues or errors
        - Work is partial or incomplete
        - You encountered blockers that prevent completion
        - You couldn't find necessary resources or dependencies
        - Quality standards haven't been met

4. **Task Breakdown**:
    - Create specific, actionable items
    - Break complex tasks into smaller, manageable steps
    - Use clear, descriptive task names

Being proactive with task management ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

## When You Finish

`write_todos` tracks your work; it does not deliver the answer. Whatever the user asked for — computations, summaries, comparisons, data — must appear as text content in a message after your final `write_todos` call. Marking the last todo complete is not itself an answer to the user."""  # noqa: E501

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember

- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## Finishing a task

When you finish all work, write your final answer in the message AFTER your last `write_todos` call — not in the same turn as that call. Start the final message with the substantive content the user asked for — the data, computation, summary, or analysis. The user wants the result, not confirmation that the work is done."""  # noqa: E501


def _write_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    f"Updated todo list to {todos}", tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


async def _awrite_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return _write_todos(runtime, todos)


class TodoListMiddleware(AgentMiddleware[PlanningState[ResponseT], ContextT, ResponseT]):
    """Provide ``write_todos`` plus the prompts that teach the model to use it."""

    state_schema = PlanningState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.tools = [
            StructuredTool.from_function(
                name="write_todos",
                description=tool_description,
                func=_write_todos,
                coroutine=_awrite_todos,
                args_schema=WriteTodosInput,
                infer_schema=False,
            )
        ]

    def _with_prompt(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Append the todo system prompt, then run the model call."""
        return handler(self._with_prompt(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Async form of :meth:`wrap_model_call`."""
        return await handler(self._with_prompt(request))

    @override
    def after_model(
        self, state: PlanningState[ResponseT], runtime: Any = None
    ) -> dict[str, Any] | None:
        """Reject parallel ``write_todos`` calls; the tool replaces the list."""
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None
        )
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        write_todos_calls = [
            tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"
        ]
        if len(write_todos_calls) <= 1:
            return None

        error_messages = [
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id=tc["id"],
                status="error",
            )
            for tc in write_todos_calls
        ]
        return {"messages": error_messages}

    @override
    async def aafter_model(
        self, state: PlanningState[ResponseT], runtime: Any = None
    ) -> dict[str, Any] | None:
        """Async form of :meth:`after_model`."""
        return self.after_model(state, runtime)


# A module-level tool mirroring upstream's decorated ``write_todos``. It uses
# the same explicit schema so hosts can import and call it directly.
write_todos = StructuredTool.from_function(
    name="write_todos",
    description=WRITE_TODOS_TOOL_DESCRIPTION,
    func=lambda todos, tool_call_id: Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    ),
    args_schema=WriteTodosInput,
    infer_schema=False,
)

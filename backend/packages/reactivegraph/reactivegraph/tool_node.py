"""Object-model tool execution node.

This module owns the runtime contract a host uses to execute model tool calls:
input parsing, argument injection, parallel dispatch, error policy, wrapper
interception, and ``Command`` normalisation. It intentionally mirrors the
observable behaviour of the upstream engine's prebuilt tool node while using
only this package's message, tool, middleware, and control-flow primitives.

The node supports four input shapes:

* a graph-state mapping containing a messages key;
* a message list whose latest AI message carries tool calls;
* a bare list of tool-call mappings;
* a ``ToolCallWithContext`` mapping dispatched through the ``Send`` API.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import warnings
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ValidationError

from reactivegraph.constants import REMOVE_ALL_MESSAGES
from reactivegraph.errors import GraphBubbleUp
from reactivegraph.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
    _dump_foreign_message,
    convert_to_messages,
)
from reactivegraph.middleware import AgentState, ToolCallRequest, ToolRuntime
from reactivegraph.runtime import Runtime
from reactivegraph.tools import (
    TOOL_MESSAGE_BLOCK_TYPES,
    BaseTool,
    InjectedToolArg,
    ToolException,
    _get_all_basemodel_annotations,
    _get_type_hints_or_empty,
    _is_injected_arg_type,
)
from reactivegraph.tools import (
    tool as create_tool,
)
from reactivegraph.types import Command, Send

__all__ = (
    "AsyncToolCallWrapper",
    "INVALID_TOOL_NAME_ERROR_TEMPLATE",
    "InjectedState",
    "InjectedStore",
    "TOOL_CALL_ERROR_TEMPLATE",
    "TOOL_EXECUTION_ERROR_TEMPLATE",
    "TOOL_INVOCATION_ERROR_TEMPLATE",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolInvocationError",
    "ToolNode",
    "msg_content_output",
    "tools_condition",
)

CONF = "configurable"
CONFIG_KEY_READ = "__pregel_read"

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
TOOL_EXECUTION_ERROR_TEMPLATE = (
    "Error executing tool '{tool_name}' with kwargs {tool_kwargs} with error:\n"
    " {error}\n"
    " Please fix the error and try again."
)
TOOL_INVOCATION_ERROR_TEMPLATE = (
    "Error invoking tool '{tool_name}' with kwargs {tool_kwargs} with error:\n"
    " {error}\n"
    " Please fix the error and try again."
)


class _ToolCallRequestOverrides(TypedDict, total=False):
    """Possible overrides for :meth:`ToolCallRequest.override`."""

    tool_call: ToolCall
    tool: Any
    state: Any


class ToolCallWithContext(TypedDict):
    """A tool call plus the state snapshot carried by a ``Send`` payload."""

    tool_call: ToolCall
    __type: Literal["tool_call_with_context"]
    state: Any


ToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]],
    ToolMessage | Command,
]
AsyncToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
    Awaitable[ToolMessage | Command],
]
_ToolResult = ToolMessage | Command | list[ToolMessage | Command]


def msg_content_output(output: Any) -> str | list[dict]:
    """Convert a tool result into ``ToolMessage`` content form.

    Strings and lists of supported content blocks pass through unchanged;
    everything else is JSON encoded with a ``str()`` fallback.
    """
    if isinstance(output, str) or (
        isinstance(output, list)
        and all(
            isinstance(item, dict) and item.get("type") in TOOL_MESSAGE_BLOCK_TYPES
            for item in output
        )
    ):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - parity with the upstream fallback
        return str(output)


class ToolInvocationError(ToolException):
    """Raised when a tool call fails argument validation."""

    def __init__(
        self,
        tool_name: str,
        source: ValidationError,
        tool_kwargs: dict[str, Any],
        filtered_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        if filtered_errors is not None:
            error_str_parts: list[str] = []
            for error in filtered_errors:
                loc_str = ".".join(str(loc) for loc in error.get("loc", ()))
                msg = error.get("msg", "Unknown error")
                error_str_parts.append(f"{loc_str}: {msg}")
            error_display_str = "\n".join(error_str_parts)
        else:
            error_display_str = str(source)

        self.message = TOOL_INVOCATION_ERROR_TEMPLATE.format(
            tool_name=tool_name,
            tool_kwargs=tool_kwargs,
            error=error_display_str,
        )
        self.tool_name = tool_name
        self.tool_kwargs = tool_kwargs
        self.source = source
        self.filtered_errors = filtered_errors
        super().__init__(self.message)


def _default_handle_tool_errors(e: Exception) -> str:
    """Return the message for invocation errors and re-raise everything else."""
    if isinstance(e, ToolInvocationError):
        return e.message
    raise e


def _handle_tool_error(
    e: Exception,
    *,
    flag: bool
    | str
    | Callable[..., str]
    | type[Exception]
    | tuple[type[Exception], ...],
) -> str:
    """Build error content according to ``handle_tool_errors``."""
    if isinstance(flag, (bool, tuple)) or (
        isinstance(flag, type) and issubclass(flag, Exception)
    ):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)  # type: ignore[assignment, call-arg]
    else:
        msg = (
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)
    return content


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception], ...]:
    """Infer the exception types accepted by a custom error handler."""
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = _get_type_hints_or_empty(handler)
        if first_param.name in type_hints:
            exception_type = type_hints[first_param.name]
            origin = get_origin(exception_type)
            if origin in [Union, UnionType]:
                args = get_args(exception_type)
                if all(
                    isinstance(arg, type) and issubclass(arg, Exception)
                    for arg in args
                ):
                    return tuple(args)
                msg = (
                    "All types in the error handler error annotation must be "
                    "Exception types. For example, "
                    "`def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )
                raise ValueError(msg)

            if isinstance(exception_type, type) and Exception in exception_type.__mro__:
                return (exception_type,)
            msg = (
                f"Arbitrary types are not supported in the error handler "
                f"signature. Please annotate the error with either a "
                f"specific Exception type or a union of Exception types. "
                "For example, `def custom_handler(e: ValueError)` or "
                "`def custom_handler(e: Union[ValueError, TypeError])`. "
                f"Got '{exception_type}' instead."
            )
            raise ValueError(msg)

    return (Exception,)


def _filter_validation_errors(
    validation_error: ValidationError,
    injected_args: _InjectedArgs | None,
) -> list[dict[str, Any]]:
    """Keep only validation errors for arguments controlled by the model."""
    injected_arg_names: set[str] = set()
    if injected_args:
        if injected_args.state:
            injected_arg_names.update(injected_args.state.keys())
        if injected_args.store:
            injected_arg_names.add(injected_args.store)
        if injected_args.runtime:
            injected_arg_names.add(injected_args.runtime)

    filtered_errors: list[dict[str, Any]] = []
    for error in validation_error.errors():
        if error["loc"] and error["loc"][0] not in injected_arg_names:
            error_copy: dict[str, Any] = {**error}
            if isinstance(error_copy.get("input"), dict):
                input_dict = error_copy["input"]
                error_copy["input"] = {
                    key: value
                    for key, value in input_dict.items()
                    if key not in injected_arg_names
                }
            filtered_errors.append(error_copy)
    return filtered_errors


@dataclass
class _InjectedArgs:
    """Injection requirements discovered for one tool."""

    state: dict[str, str | None]
    store: str | None
    runtime: str | None
    all_injected_keys: set[str]
    _optional_state_args: set[str]
    runtime_type: Any = None


class ToolNode:
    """Execute a batch of tool calls and return their normalized results."""

    name: str = "tools"

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool
        | str
        | Callable[..., str]
        | type[Exception]
        | tuple[type[Exception], ...] = _default_handle_tool_errors,
        messages_key: str = "messages",
        wrap_tool_call: ToolCallWrapper | None = None,
        awrap_tool_call: AsyncToolCallWrapper | None = None,
    ) -> None:
        self.name = name
        self.tags = tags
        self._tools_by_name: dict[str, Any] = {}
        self._injected_args: dict[str, _InjectedArgs] = {}
        self._handle_tool_errors = handle_tool_errors
        self._messages_key = messages_key
        self._wrap_tool_call = wrap_tool_call
        self._awrap_tool_call = awrap_tool_call
        for item in tools:
            tool_ = cast("BaseTool", item if _is_tool_like(item) else create_tool(item))
            self._tools_by_name[tool_.name] = tool_
            self._injected_args[tool_.name] = _get_all_injected_args(tool_)

    @property
    def tools_by_name(self) -> dict[str, Any]:
        """Mapping from tool name to tool instance."""
        return self._tools_by_name

    def invoke(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute tool calls synchronously."""
        del kwargs
        return self._func(input, _ensure_config(config), _active_runtime())

    async def ainvoke(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute tool calls asynchronously."""
        del kwargs
        return await self._afunc(input, _ensure_config(config), _active_runtime())

    def _func(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: dict[str, Any],
        runtime: Runtime,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        config_list = _get_config_list(config, len(tool_calls))

        tool_runtimes: list[ToolRuntime] = []
        for call, cfg in zip(tool_calls, config_list, strict=True):
            state = self._extract_state(input, cfg)
            tool_runtimes.append(
                ToolRuntime(
                    state=cast("AgentState[Any]", state),
                    tool_call_id=call.get("id"),
                    config=cfg,
                    context=getattr(runtime, "context", None),
                    store=getattr(runtime, "store", None),
                    stream_writer=getattr(runtime, "stream_writer", None),
                    tools=list(self.tools_by_name.values()),
                    execution_info=getattr(runtime, "execution_info", None),
                    server_info=getattr(runtime, "server_info", None),
                )
            )

        input_types = [input_type] * len(tool_calls)
        outputs = _map_in_context(
            self._run_one,
            tool_calls,
            input_types,
            tool_runtimes,
            max_workers=config.get("max_concurrency"),
        )
        return self._combine_tool_outputs(outputs, input_type)

    async def _afunc(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: dict[str, Any],
        runtime: Runtime,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        config_list = _get_config_list(config, len(tool_calls))

        tool_runtimes: list[ToolRuntime] = []
        for call, cfg in zip(tool_calls, config_list, strict=True):
            state = self._extract_state(input, cfg)
            tool_runtimes.append(
                ToolRuntime(
                    state=cast("AgentState[Any]", state),
                    tool_call_id=call.get("id"),
                    config=cfg,
                    context=getattr(runtime, "context", None),
                    store=getattr(runtime, "store", None),
                    stream_writer=getattr(runtime, "stream_writer", None),
                    tools=list(self.tools_by_name.values()),
                    execution_info=getattr(runtime, "execution_info", None),
                    server_info=getattr(runtime, "server_info", None),
                )
            )

        coros = [
            self._arun_one(call, input_type, tool_runtime)
            for call, tool_runtime in zip(tool_calls, tool_runtimes, strict=True)
        ]
        outputs = await asyncio.gather(*coros)
        return self._combine_tool_outputs(outputs, input_type)

    def _combine_tool_outputs(
        self,
        outputs: list[ToolMessage | Command | list[ToolMessage | Command]],
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> list[ToolMessage | Command | list[ToolMessage] | dict[str, list[ToolMessage]]]:
        flat_outputs: list[ToolMessage | Command]
        if any(isinstance(output, list) for output in outputs):
            flat_outputs = []
            for output in outputs:
                if isinstance(output, list):
                    flat_outputs.extend(output)
                else:
                    flat_outputs.append(output)
        else:
            flat_outputs = cast("list[ToolMessage | Command]", outputs)

        if not any(isinstance(output, Command) for output in flat_outputs):
            # Preserve the LangGraph contract: list input returns a list update,
            # while dict/model/tool_calls input returns a state mapping.
            if input_type == "list":
                return cast(
                    "list[ToolMessage | Command | list[ToolMessage] | "
                    "dict[str, list[ToolMessage]]]",
                    flat_outputs,
                )
            return cast(
                "list[ToolMessage | Command | list[ToolMessage] | "
                "dict[str, list[ToolMessage]]]",
                {self._messages_key: flat_outputs},
            )

        combined_outputs: list[
            ToolMessage | Command | list[ToolMessage] | dict[str, list[ToolMessage]]
        ] = []
        parent_command: Command | None = None
        for output in flat_outputs:
            if isinstance(output, Command):
                if (
                    output.graph is Command.PARENT
                    and isinstance(output.goto, list)
                    and all(isinstance(send, Send) for send in output.goto)
                ):
                    if parent_command:
                        parent_command = replace(
                            parent_command,
                            goto=cast("list[Send]", parent_command.goto) + output.goto,
                        )
                    else:
                        parent_command = Command(graph=Command.PARENT, goto=output.goto)
                else:
                    combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self._messages_key: [output]}
                )

        if parent_command:
            combined_outputs.append(parent_command)
        return combined_outputs

    def _execute_tool_sync(
        self,
        request: ToolCallRequest,
        input_type: Literal["list", "dict", "tool_calls"],
        config: dict[str, Any],
    ) -> _ToolResult:
        call = request.tool_call
        tool = request.tool

        if tool is None:
            if invalid_tool_message := self._validate_tool_call(call):
                return invalid_tool_message
            msg = f"Tool {call['name']} is not registered with ToolNode"
            raise TypeError(msg)

        injected_call = self._inject_tool_args(call, request.runtime, tool)
        call_args = {**injected_call, "type": "tool_call"}

        try:
            try:
                response = tool.invoke(call_args, config)
            except ValidationError as exc:
                injected = self._injected_args.get(call["name"])
                filtered_errors = _filter_validation_errors(exc, injected)
                raise ToolInvocationError(
                    call["name"], exc, call["args"], filtered_errors
                ) from exc

            return self._normalize_tool_response(response, request.tool_call, input_type)
        except GraphBubbleUp:
            raise
        except Exception as e:
            handled_types: tuple[type[Exception], ...]
            if isinstance(self._handle_tool_errors, type) and issubclass(
                self._handle_tool_errors, Exception
            ):
                handled_types = (self._handle_tool_errors,)
            elif isinstance(self._handle_tool_errors, tuple):
                handled_types = self._handle_tool_errors
            elif callable(self._handle_tool_errors) and not isinstance(
                self._handle_tool_errors, type
            ):
                handled_types = _infer_handled_types(self._handle_tool_errors)
            else:
                handled_types = (Exception,)

            if not self._handle_tool_errors or not isinstance(e, handled_types):
                raise

            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

    async def _execute_tool_async(
        self,
        request: ToolCallRequest,
        input_type: Literal["list", "dict", "tool_calls"],
        config: dict[str, Any],
    ) -> _ToolResult:
        call = request.tool_call
        tool = request.tool

        if tool is None:
            if invalid_tool_message := self._validate_tool_call(call):
                return invalid_tool_message
            msg = f"Tool {call['name']} is not registered with ToolNode"
            raise TypeError(msg)

        injected_call = self._inject_tool_args(call, request.runtime, tool)
        call_args = {**injected_call, "type": "tool_call"}

        try:
            try:
                response = await tool.ainvoke(call_args, config)
            except ValidationError as exc:
                injected = self._injected_args.get(call["name"])
                filtered_errors = _filter_validation_errors(exc, injected)
                raise ToolInvocationError(
                    call["name"], exc, call["args"], filtered_errors
                ) from exc

            return self._normalize_tool_response(response, request.tool_call, input_type)
        except GraphBubbleUp:
            raise
        except Exception as e:
            handled_types: tuple[type[Exception], ...]
            if isinstance(self._handle_tool_errors, type) and issubclass(
                self._handle_tool_errors, Exception
            ):
                handled_types = (self._handle_tool_errors,)
            elif isinstance(self._handle_tool_errors, tuple):
                handled_types = self._handle_tool_errors
            elif callable(self._handle_tool_errors) and not isinstance(
                self._handle_tool_errors, type
            ):
                handled_types = _infer_handled_types(self._handle_tool_errors)
            else:
                handled_types = (Exception,)

            if not self._handle_tool_errors or not isinstance(e, handled_types):
                raise

            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime: ToolRuntime,
    ) -> _ToolResult:
        tool = self.tools_by_name.get(call["name"])
        tool_request = ToolCallRequest(
            tool_call=call,
            tool=tool,
            state=tool_runtime.state,
            runtime=tool_runtime,
        )
        config = tool_runtime.config

        if self._wrap_tool_call is None:
            return self._execute_tool_sync(tool_request, input_type, config)

        def execute(req: ToolCallRequest) -> ToolMessage | Command:
            return cast(
                "ToolMessage | Command",
                self._execute_tool_sync(req, input_type, config),
            )

        try:
            return self._wrap_tool_call(tool_request, execute)
        except Exception as e:
            if not self._handle_tool_errors:
                raise
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=tool_request.tool_call["name"],
                tool_call_id=tool_request.tool_call["id"],
                status="error",
            )

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime: ToolRuntime,
    ) -> _ToolResult:
        tool = self.tools_by_name.get(call["name"])
        tool_request = ToolCallRequest(
            tool_call=call,
            tool=tool,
            state=tool_runtime.state,
            runtime=tool_runtime,
        )
        config = tool_runtime.config

        if self._awrap_tool_call is None and self._wrap_tool_call is None:
            return await self._execute_tool_async(tool_request, input_type, config)

        async def execute(req: ToolCallRequest) -> ToolMessage | Command:
            return cast(
                "ToolMessage | Command",
                await self._execute_tool_async(req, input_type, config),
            )

        def _sync_execute(req: ToolCallRequest) -> ToolMessage | Command:
            return cast(
                "ToolMessage | Command",
                self._execute_tool_sync(req, input_type, config),
            )

        try:
            if self._awrap_tool_call is not None:
                return await self._awrap_tool_call(tool_request, execute)
            wrap = cast("ToolCallWrapper", self._wrap_tool_call)
            return wrap(tool_request, _sync_execute)
        except Exception as e:
            if not self._handle_tool_errors:
                raise
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=tool_request.tool_call["name"],
                tool_call_id=tool_request.tool_call["id"],
                status="error",
            )

    def _parse_input(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
    ) -> tuple[list[ToolCall], Literal["list", "dict", "tool_calls"]]:
        input_type: Literal["list", "dict", "tool_calls"]
        if isinstance(input, list):
            if input and isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                return cast("list[ToolCall]", input), "tool_calls"
            input_type = "list"
            messages = input
        elif isinstance(input, dict) and input.get("__type") == "tool_call_with_context":
            input_with_ctx = cast("ToolCallWithContext", input)
            return [input_with_ctx["tool_call"]], "tool_calls"
        elif isinstance(input, dict) and (messages := input.get(self._messages_key, [])):
            input_type = "dict"
        elif messages := getattr(input, self._messages_key, []):
            input_type = "dict"
        else:
            msg = "No message found in input"
            raise ValueError(msg)

        try:
            latest_ai_message = next(
                message for message in reversed(messages) if isinstance(message, AIMessage)
            )
        except StopIteration:
            msg = "No AIMessage found in input"
            raise ValueError(msg) from None

        return list(latest_ai_message.tool_calls), input_type

    def _validate_tool_call(self, call: ToolCall) -> ToolMessage | None:
        requested_tool = call["name"]
        if requested_tool not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name),
            )
            return ToolMessage(
                content, name=requested_tool, tool_call_id=call["id"], status="error"
            )
        return None

    def _extract_state(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: dict[str, Any],
    ) -> list[AnyMessage] | dict[str, Any] | BaseModel:
        if isinstance(input, dict) and input.get("__type") == "tool_call_with_context":
            return input["state"]
        if (
            isinstance(input, list)
            and input
            and isinstance(input[-1], dict)
            and input[-1].get("type") == "tool_call"
        ):
            read = config.get(CONF, {}).get(CONFIG_KEY_READ)
            if read is None:
                return {}
            channels = read.args[1]
            return cast("dict[str, Any]", read(list(channels), True))
        return input

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
        tool: BaseTool | None = None,
    ) -> ToolCall:
        injected = self._injected_args.get(tool_call["name"])
        if not injected and tool is not None:
            injected = _get_all_injected_args(tool)
        if not injected:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        injected_args: dict[str, Any] = {}

        if injected.state:
            state = tool_runtime.state
            if isinstance(state, list):
                required_fields = list(injected.state.values())
                if (
                    len(required_fields) == 1
                    and required_fields[0] == self._messages_key
                ) or required_fields[0] is None:
                    state = {self._messages_key: state}
                else:
                    err_msg = (
                        f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                        f"graph state dict as input."
                    )
                    if any(state_field for state_field in injected.state.values()):
                        required_fields_str = ", ".join(f for f in required_fields if f)
                        err_msg += (
                            f" State should contain fields {required_fields_str}."
                        )
                    raise ValueError(err_msg)

            if isinstance(state, dict):
                for tool_arg, state_field in injected.state.items():
                    if not state_field:
                        injected_args[tool_arg] = state
                    elif state_field in state:
                        injected_args[tool_arg] = cast("dict[str, Any]", state)[state_field]
                    elif tool_arg not in injected._optional_state_args:
                        raise KeyError(state_field)
            else:
                for tool_arg, state_field in injected.state.items():
                    if not state_field:
                        injected_args[tool_arg] = state
                    elif hasattr(state, state_field):
                        injected_args[tool_arg] = getattr(state, state_field)
                    elif tool_arg not in injected._optional_state_args:
                        raise AttributeError(state_field)

        if injected.store:
            if tool_runtime.store is None:
                msg = (
                    "Cannot inject store into tools with InjectedStore annotations - "
                    "please compile your graph with a store."
                )
                raise ValueError(msg)
            injected_args[injected.store] = tool_runtime.store

        if injected.runtime:
            injected_args[injected.runtime] = _coerce_tool_runtime(
                tool_runtime, injected.runtime_type
            )

        stripped_args = {
            key: value
            for key, value in tool_call_copy["args"].items()
            if key not in injected.all_injected_keys
        }
        tool_call_copy["args"] = {**stripped_args, **injected_args}
        return tool_call_copy

    def _normalize_tool_response(
        self,
        response: Any,
        tool_call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> ToolMessage | Command | list[Command | ToolMessage]:
        if isinstance(response, Command):
            return self._validate_tool_command(response, tool_call, input_type)
        if _is_foreign_command(response):
            command = _adopt_foreign_command(response)
            return self._validate_tool_command(command, tool_call, input_type)
        if isinstance(response, ToolMessage):
            response.content = cast("str | list", msg_content_output(response.content))
            return response
        foreign_message = _adopt_foreign_tool_message(response)
        if foreign_message is not None:
            foreign_message.content = cast(
                "str | list", msg_content_output(foreign_message.content)
            )
            return foreign_message
        if isinstance(response, list):
            adopted = [
                _adopt_foreign_command(item)
                if _is_foreign_command(item)
                else _adopt_foreign_tool_message(item) or item
                for item in response
            ]
            if all(isinstance(item, (Command, ToolMessage)) for item in adopted):
                return self._validate_tool_command_list(adopted, tool_call, input_type)
            msg = (
                f"Tool {tool_call['name']} returned a list with invalid element "
                "types: expected all Command or ToolMessage"
            )
            raise TypeError(msg)
        msg = f"Tool {tool_call['name']} returned unexpected type: {type(response)}"
        raise TypeError(msg)

    def _validate_tool_command_list(
        self,
        response: list[Command | ToolMessage],
        tool_call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> list[Command | ToolMessage]:
        expected_id = tool_call["id"]
        terminator_count = 0
        for item in response:
            if isinstance(item, ToolMessage):
                if item.tool_call_id == expected_id:
                    terminator_count += 1
            elif isinstance(item, Command) and isinstance(item.update, dict):
                for message in item.update.get(self._messages_key, []):
                    if (
                        isinstance(message, ToolMessage)
                        and message.tool_call_id == expected_id
                    ):
                        terminator_count += 1

        if terminator_count != 1:
            msg = (
                f"Tool {tool_call['name']} returned a list with "
                f"{terminator_count} messages bound to tool_call_id "
                f"{expected_id!r}; expected exactly one terminating ToolMessage."
            )
            raise ValueError(msg)

        validated: list[Command | ToolMessage] = []
        for item in response:
            if isinstance(item, Command):
                validated.append(
                    self._validate_tool_command(
                        item, tool_call, input_type, require_terminator=False
                    )
                )
            else:
                item.content = cast("str | list", msg_content_output(item.content))
                validated.append(item)
        return validated

    def _validate_tool_command(
        self,
        command: Command,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        *,
        require_terminator: bool = True,
    ) -> Command:
        if isinstance(command.update, dict):
            if input_type not in ("dict", "tool_calls"):
                msg = (
                    "Tools can provide a dict in Command.update only when using dict "
                    f"with '{self._messages_key}' key as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )
                raise ValueError(msg)
            updated_command = deepcopy(command)
            state_update = cast("dict[str, Any]", updated_command.update) or {}
            messages_update = state_update.get(self._messages_key, [])
        elif isinstance(command.update, list):
            if input_type != "list":
                msg = (
                    "Tools can provide a list of messages in Command.update "
                    "only when using list of messages as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )
                raise ValueError(msg)
            updated_command = deepcopy(command)
            messages_update = updated_command.update
        else:
            return command

        messages_update = convert_to_messages(messages_update)

        if messages_update == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]:
            return updated_command

        has_matching_tool_message = False
        for message in messages_update:
            if not isinstance(message, ToolMessage):
                continue
            if message.tool_call_id == call["id"]:
                message.name = call["name"]
                has_matching_tool_message = True

        if (
            require_terminator
            and updated_command.graph is None
            and not has_matching_tool_message
        ):
            example_update = (
                '`Command(update={"messages": '
                '[ToolMessage("Success", tool_call_id=tool_call_id), ...]}, ...)`'
                if input_type == "dict"
                else "`Command(update="
                '[ToolMessage("Success", tool_call_id=tool_call_id), ...], ...)`'
            )
            msg = (
                "Expected to have a matching ToolMessage in Command.update "
                f"for tool '{call['name']}', got: {messages_update}. "
                "Every tool call (LLM requesting to call a tool) "
                "in the message history MUST have a corresponding ToolMessage. "
                f"You can fix it by modifying the tool to return {example_update}."
            )
            raise ValueError(msg)
        return updated_command


def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Route to ``tools`` when the latest message carries tool calls."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif (isinstance(state, dict) and (messages := state.get(messages_key, []))) or (
        messages := getattr(state, messages_key, [])
    ):
        ai_message = messages[-1]
    else:
        msg = f"No messages found in input state to tool_edge: {state}"
        raise ValueError(msg)
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class InjectedState(InjectedToolArg):
    """Annotation for injecting graph state into a tool argument."""

    def __init__(self, field: str | None = None) -> None:
        self.field = field


class InjectedStore(InjectedToolArg):
    """Annotation for injecting the persistent store into a tool argument."""


def _is_injection(
    type_arg: Any,
    injection_type: type[InjectedState | InjectedStore | ToolRuntime],
) -> bool:
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ in (Union, UnionType):
        return any(_is_injection(arg, injection_type) for arg in get_args(type_arg))
    if origin_ is not None and (
        origin_ is injection_type
        or (isinstance(origin_, type) and issubclass(origin_, injection_type))
    ):
        return True
    return False


def _get_injection_from_type(
    type_: Any,
    injection_type: type[InjectedState | InjectedStore | ToolRuntime],
) -> Any | None:
    type_args = get_args(type_)
    matches = [arg for arg in type_args if _is_injection(arg, injection_type)]

    if len(matches) > 1:
        msg = (
            f"A tool argument should not be annotated with {injection_type.__name__} "
            f"more than once. Found: {matches}"
        )
        raise ValueError(msg)

    if len(matches) == 1:
        return matches[0]
    if _is_injection(type_, injection_type):
        return True
    return None


def _is_tool_like(item: Any) -> bool:
    """Whether ``item`` is already a runnable tool.

    Hosts define tools with *their* engine's decorator (DeerFlow has 65
    ``@tool`` definitions), so the two class hierarchies can never meet.
    Structural acceptance keys on the two members ``ToolNode`` actually uses:
    a string ``name`` and a callable ``invoke``. Anything else still goes
    through :func:`create_tool`, preserving the raw-callable contract.
    """
    return isinstance(item, BaseTool) or (
        callable(getattr(item, "invoke", None))
        and isinstance(getattr(item, "name", None), str)
    )


def _coerce_tool_runtime(runtime: ToolRuntime, declared: Any) -> Any:
    """Present the active runtime as the type a host tool declared.

    LangChain validates injected ``runtime`` arguments against the tool's own
    annotation, so handing it our dataclass raises ``ValidationError``. When
    the declared type is ours (or unknown) pass the runtime straight through;
    otherwise rebuild the declared dataclass from the shared field names.
    """
    if declared is None or declared is ToolRuntime:
        return runtime
    # A host may annotate ``ToolRuntime[dict[str, Any], ThreadState]``. The
    # subscripted generic is not a class, so unwrap to its origin before
    # constructing it; otherwise the host tool's validator rejects our type.
    if not isinstance(declared, type):
        origin = get_origin(declared)
        if not isinstance(origin, type) or origin is Annotated:
            return runtime
        declared = origin
    if isinstance(runtime, declared):
        return runtime
    fields = {
        name: getattr(runtime, name, None)
        for name in (
            "state",
            "context",
            "config",
            "stream_writer",
            "tool_call_id",
            "store",
            "tools",
            "execution_info",
            "server_info",
        )
    }
    return declared(**fields)


def _is_foreign_command(value: Any) -> bool:
    """Structural check for a host ``Command`` (never import theirs)."""
    return (
        not isinstance(value, Command)
        and hasattr(value, "update")
        and hasattr(value, "goto")
        and hasattr(value, "graph")
        and hasattr(value, "resume")
    )


def _adopt_foreign_command(value: Any) -> Any:
    """Rebuild a host ``Command`` as ours, converting nested messages."""
    if isinstance(value, Command):
        return value
    update = getattr(value, "update", None)
    if isinstance(update, dict):
        update = {
            key: (
                convert_to_messages(item)
                if key == "messages" and isinstance(item, list)
                else item
            )
            for key, item in update.items()
        }
    elif isinstance(update, list):
        update = convert_to_messages(update)
    return Command(
        graph=getattr(value, "graph", None),
        update=update,
        resume=getattr(value, "resume", None),
        goto=getattr(value, "goto", ()),
    )


def _adopt_foreign_tool_message(value: Any) -> ToolMessage | None:
    """Rebuild a host ``ToolMessage`` as ours, or ``None`` if it is not one."""
    if isinstance(value, ToolMessage):
        return value
    dumped = _dump_foreign_message(value)
    if dumped is None or dumped.get("type") != "tool":
        return None
    kwargs = {
        key: item
        for key, item in dumped.items()
        if key in {"content", "artifact", "status", "name", "id", "tool_call_id"}
    }
    return ToolMessage(**kwargs)


def _get_all_injected_args(tool: BaseTool) -> _InjectedArgs:
    """Discover state, store, and runtime injection requirements for a tool."""
    full_schema = tool.get_input_schema()
    schema_annotations = _get_all_basemodel_annotations(full_schema)

    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    func_annotations = _get_type_hints_or_empty(func) if func else {}

    all_annotations = {**func_annotations, **schema_annotations}

    state_args: dict[str, str | None] = {}
    store_arg: str | None = None
    runtime_arg: str | None = None
    runtime_type: Any = None
    all_injected_keys: set[str] = set()
    _optional_state_args: set[str] = set()

    for name, type_ in all_annotations.items():
        if _is_injected_arg_type(type_):
            all_injected_keys.add(name)

        if name == "runtime":
            runtime_arg = name
            runtime_type = type_

        if state_inj := _get_injection_from_type(type_, InjectedState):
            if isinstance(state_inj, InjectedState) and state_inj.field:
                state_args[name] = state_inj.field
                field_info = full_schema.model_fields.get(name)
                if field_info and not field_info.is_required():
                    _optional_state_args.add(name)
            else:
                state_args[name] = None

        if _get_injection_from_type(type_, InjectedStore):
            store_arg = name

        if _get_injection_from_type(type_, ToolRuntime):
            runtime_arg = name
            runtime_type = type_

    return _InjectedArgs(
        state=state_args,
        store=store_arg,
        runtime=runtime_arg,
        all_injected_keys=all_injected_keys,
        _optional_state_args=_optional_state_args,
        runtime_type=runtime_type,
    )


def _ensure_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _get_config_list(
    config: dict[str, Any] | Sequence[dict[str, Any]] | None,
    length: int,
) -> list[dict[str, Any]]:
    if length < 0:
        msg = f"length must be >= 0, but got {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes, dict)):
        if len(config) != length:
            msg = (
                "config must be a list of the same length as inputs, "
                f"but got {len(config)} configs for {length} inputs"
            )
            raise ValueError(msg)
        return [_ensure_config(item) for item in config]
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = {key: value for key, value in config.items() if key != "run_id"}
        return [
            _ensure_config(subsequent) if index else _ensure_config(config)
            for index in range(length)
        ]
    return [_ensure_config(config) for _ in range(length)]


def _map_in_context(
    fn: Callable[..., Any],
    *iterables: Sequence[Any],
    max_workers: int | None = None,
) -> list[Any]:
    """Run ``fn`` over zipped arguments, preserving order and context."""
    items = list(zip(*iterables, strict=True))
    if not items:
        return []
    workers = max_workers if max_workers and max_workers > 0 else len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(copy_context().run, fn, *args) for args in items
        ]
        return [future.result() for future in futures]


def _active_runtime() -> Runtime:
    """Return the ambient runtime, or an empty one outside a run."""
    from reactivegraph.runtime import get_runtime

    try:
        return get_runtime()
    except RuntimeError:
        return Runtime()

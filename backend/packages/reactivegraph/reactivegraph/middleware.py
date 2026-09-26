"""Agent middleware and model-hook contracts.

DeerFlow's agent layer is built from ``AgentMiddleware`` subclasses rather than
from graphs assembled by hand.  The middleware owns two different kinds of
callback:

* state hooks (``before_agent``/``before_model``/``after_model``/``after_agent``)
  return state updates or a :class:`~reactivegraph.types.Command`;
* wrappers (``wrap_model_call``/``wrap_tool_call``) receive the next handler and
  decide whether to call it, retry it, or short-circuit it.

This module keeps the *shape* of that contract without importing LangChain.
That is what makes it possible to replace the engine underneath a real agent
factory while preserving every middleware's type annotations and runtime
behaviour.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    cast,
)

from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack

from reactivegraph.message_state import EphemeralValue, add_messages
from reactivegraph.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from reactivegraph.runtime import Runtime, StreamWriter
from reactivegraph.types import Command

if TYPE_CHECKING:
    from reactivegraph.runtime import StreamWriter

__all__ = (
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ExtendedModelResponse",
    "InputAgentState",
    "ModelCallResult",
    "ModelRequest",
    "ModelResponse",
    "OmitFromInput",
    "OmitFromOutput",
    "OmitFromSchema",
    "OutputAgentState",
    "PrivateStateAttr",
    "ResponseT",
    "StateT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "ToolRuntime",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "tool_call_writer",
    "wrap_model_call",
    "wrap_tool_call",
)

JumpTo = Literal["tools", "model", "end"]
ResponseT = TypeVar("ResponseT", default=Any)
ContextT = TypeVar("ContextT", default=None)
StateT = TypeVar("StateT", bound="AgentState[Any]", default="AgentState[Any]")
StateT_co = TypeVar(
    "StateT_co", bound="AgentState[Any]", default="AgentState[Any]", covariant=True
)
StateT_contra = TypeVar("StateT_contra", bound="AgentState[Any]", contravariant=True)

# The middleware layer must not require a model implementation at import time;
# DeerFlow supplies real LangChain chat models at the boundary.
BaseChatModel = Any
BaseTool = Any

ToolCallWriter = Callable[[Any], None]

_tool_call_writer: ContextVar[ToolCallWriter | None] = ContextVar(
    "reactivegraph_tool_call_writer", default=None
)
"""Writer bound to the currently executing tool call.

Mirrors the upstream tool-execution runtime: a closure is installed per call,
so :meth:`ToolRuntime.emit_output_delta` can stream partial output without the
tool body threading a writer through its own signature.  The binding is
task-local and therefore safe under concurrency.
"""


@contextlib.contextmanager
def tool_call_writer(writer: ToolCallWriter) -> Iterator[None]:
    """Bind *writer* for the duration of one tool call."""
    token = _tool_call_writer.set(writer)
    try:
        yield
    finally:
        _tool_call_writer.reset(token)


@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    """A model call request, including the runtime and tool definitions."""

    model: BaseChatModel
    messages: list[AnyMessage]
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: Any | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        model: BaseChatModel,
        messages: list[AnyMessage],
        system_message: SystemMessage | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        tools: list[BaseTool | dict[str, Any]] | None = None,
        response_format: Any | None = None,
        state: AgentState[Any] | None = None,
        runtime: Runtime[ContextT] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        if system_prompt is not None and system_message is not None:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)
        if system_prompt is not None:
            system_message = SystemMessage(content=system_prompt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = model
            self.messages = messages
            self.system_message = system_message
            self.tool_choice = tool_choice
            self.tools = tools if tools is not None else []
            self.response_format = response_format
            self.state = state if state is not None else {"messages": []}
            self.runtime = runtime  # type: ignore[assignment]
            self.model_settings = model_settings if model_settings is not None else {}

    @property
    def system_prompt(self) -> str | None:
        """Return the text of :attr:`system_message`, if one is set."""
        if self.system_message is None:
            return None
        return self.system_message.text

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "system_prompt":
            warnings.warn(
                "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
                "Use request.override(system_message=SystemMessage(...)) instead to create "
                "a new request with the modified system message.",
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                object.__setattr__(self, "system_message", None)
            else:
                object.__setattr__(self, "system_message", SystemMessage(content=value))
            return

        warnings.warn(
            f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
            f"Use request.override({name}=...) instead to create a new request "
            "with the modified attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Any) -> ModelRequest[ContextT]:
        """Return a copy with *overrides* applied, without mutating this request."""
        if "system_prompt" in overrides and "system_message" in overrides:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)
        if "system_prompt" in overrides:
            system_prompt = overrides.pop("system_prompt")
            overrides["system_message"] = (
                None if system_prompt is None else SystemMessage(content=system_prompt)
            )
        return replace(self, **overrides)


@dataclass
class ModelResponse(Generic[ResponseT]):
    """Model output plus an optional parsed structured response."""

    result: list[BaseMessage]
    structured_response: ResponseT | None = None


@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """A model response that also applies a state :class:`Command`."""

    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None


ModelCallResult = ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]


@dataclass
class OmitFromSchema:
    """Mark a state attribute as omitted from input and/or output schemas."""

    input: bool = True
    output: bool = True


OmitFromInput = OmitFromSchema(input=True, output=False)
OmitFromOutput = OmitFromSchema(input=False, output=True)
PrivateStateAttr = OmitFromSchema(input=True, output=True)


class AgentState(TypedDict, Generic[ResponseT]):
    """The default agent state schema."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]


class InputAgentState(TypedDict):
    """The state schema accepted by an agent at its boundary."""

    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]


class OutputAgentState(TypedDict, Generic[ResponseT]):
    """The state schema returned by an agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


class _DefaultAgentState(AgentState[Any]):
    """AgentMiddleware default state."""


_HOOK_NAMES = (
    "before_agent",
    "abefore_agent",
    "before_model",
    "abefore_model",
    "after_model",
    "aafter_model",
    "after_agent",
    "aafter_agent",
    "wrap_model_call",
    "awrap_model_call",
    "wrap_tool_call",
    "awrap_tool_call",
)
"""The full hook surface, sync and async, in upstream's declaration order."""

_UNSET = object()
_host_middleware_base: Any = _UNSET


def host_middleware_base() -> type | None:
    """LangChain's ``AgentMiddleware`` when it is importable, else ``None``.

    Hosts that install LangChain before us declare their middleware against
    *its* base. That class cannot become ours — importing it drags in
    LangGraph, which must stay optional — but a middleware written against it
    must still satisfy our ``isinstance``/``issubclass`` checks, because the
    harness's feature flags are typed on our base.

    Successful lookups are cached; failures are not, so a process that made
    LangGraph temporarily unimportable and later restored it still converges.
    """
    global _host_middleware_base
    if _host_middleware_base is not _UNSET:
        return _host_middleware_base
    try:
        from langchain.agents.middleware import AgentMiddleware as HostAgentMiddleware
    except ImportError:  # pragma: no cover - langchain is an optional host dep
        return None
    _host_middleware_base = HostAgentMiddleware
    return HostAgentMiddleware


class _MiddlewareMeta(type):
    """Accept host middleware classes as instances of ours.

    ``AgentMiddleware`` is our class, so a class deriving from LangChain's is
    not a real subclass. Type checks in the harness (feature flags, ``@Next``
    anchors, extension placement) must nevertheless accept both, otherwise a
    host middleware is silently dropped into the "not an instance" branch.
    """

    def __subclasscheck__(cls, subclass: Any) -> bool:
        if cls is not AgentMiddleware or type.__subclasscheck__(cls, subclass):
            # The bridge widens only the framework base. Leaking it into
            # every subclass would make ``issubclass(X, ClarificationMiddleware)``
            # true for any host middleware, so anchor resolution in the
            # harness would match the wrong middleware.
            return type.__subclasscheck__(cls, subclass)
        host = host_middleware_base()
        if host is None:
            return False
        return type.__subclasscheck__(host, subclass)

    def __instancecheck__(cls, instance: Any) -> bool:
        if cls is not AgentMiddleware or type.__instancecheck__(cls, instance):
            return type.__instancecheck__(cls, instance)
        host = host_middleware_base()
        if host is None:
            return False
        return type.__instancecheck__(host, instance)


_HostAgentMiddleware = host_middleware_base()
"""The host base to inherit from, or ``None`` when LangChain is absent."""


def _drop_shadowed_host_hooks() -> None:
    """Let the host's hook functions win through the real base class.

    The native hook definitions below are needed when LangChain is absent.
    When it is present, the class really inherits the host base, and keeping
    our byte-identical copies in the subclass ``__dict__`` would *shadow* it:
    a host-side patch or override (including the harness monkeypatching
    ``langchain.agents.middleware.AgentMiddleware.abefore_agent``) could never
    be reached from ``super()``. Deleting the shadowing entries keeps the
    native fallback and restores normal inheritance semantics.

    Identity against the host base is also what upstream's participation test
    (``m.__class__.hook is not AgentMiddleware.hook``) relies on; inheritance
    supplies it without copying function objects.
    """
    for hook in _HOOK_NAMES:
        if hook in AgentMiddleware.__dict__ and hasattr(_HostAgentMiddleware, hook):
            delattr(AgentMiddleware, hook)


class AgentMiddleware(
    *((_HostAgentMiddleware,) if _HostAgentMiddleware is not None else ()),  # type: ignore[misc]
    Generic[StateT, ContextT, ResponseT],
    metaclass=_MiddlewareMeta,
):
    """Base class for agent middleware.

    Hooks are no-ops by default.  ``wrap_*`` defaults raise
    :class:`NotImplementedError` with the upstream message, so a middleware
    that only implements the async hook cannot silently run in sync mode.
    """

    state_schema: type[StateT] = cast("type[StateT]", _DefaultAgentState)
    transformers: Sequence[Any] = ()

    @property
    def name(self) -> str:
        """The middleware instance name, defaulting to its class name."""
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Run before agent execution starts."""

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async form of :meth:`before_agent`."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Run before a model call."""

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async form of :meth:`before_model`."""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Run after a model call."""

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async form of :meth:`after_model`."""

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Run after agent execution completes."""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async form of :meth:`after_agent`."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelCallResult[ResponseT]:
        """Intercept a model call; sync implementations must opt in."""
        msg = (
            "Synchronous implementation of wrap_model_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_model_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelCallResult[ResponseT]:
        """Intercept a model call; async implementations must opt in."""
        msg = (
            "Asynchronous implementation of awrap_model_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_model_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept a tool call; sync implementations must opt in."""
        msg = (
            "Synchronous implementation of wrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_tool_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone sync function, or "
            # Upstream wording (including its sync/async mix-up) is an observable
            # contract; keep it byte-identical.
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept a tool call; async implementations must opt in."""
        msg = (
            "Asynchronous implementation of awrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_tool_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)


if _HostAgentMiddleware is not None:
    _drop_shadowed_host_hooks()


class _ToolCallRequestOverrides(TypedDict, total=False):
    tool_call: ToolCall
    tool: Any
    state: Any


def _host_directly_injected_tool_arg() -> type | None:
    """LangChain's ``_DirectlyInjectedToolArg`` marker, when installed.

    ``langchain_core.tools``' ``@tool`` decorator keeps a directly injected
    argument out of ``tool_call_schema`` only when the annotation is a
    subclass of *its own* marker. DeerFlow builds its builtins (``present_files``
    among them) with that decorator while annotating them with our
    ``ToolRuntime``, so without this base the runtime stays in the schema and
    ``bind_tools`` dies generating a JSON schema for ``stream_writer: Callable``.
    """
    try:
        from langchain_core.tools.base import (
            _DirectlyInjectedToolArg as HostDirectlyInjectedToolArg,
        )
    except ImportError:  # pragma: no cover - langchain is an optional host dep
        return None
    return HostDirectlyInjectedToolArg


HostDirectlyInjectedToolArg = _host_directly_injected_tool_arg()


class _DirectlyInjectedToolArg(
    *(  # type: ignore[misc]
        (HostDirectlyInjectedToolArg,)
        if HostDirectlyInjectedToolArg is not None
        else (object,)
    )
):  # type: ignore[misc, unused-ignore]
    """Base marker for tool arguments injected directly by type annotation."""


def _host_tool_runtime() -> type | None:
    """LangGraph's ``ToolRuntime``, when importable.

    ``langchain.tools.ToolRuntime`` is a re-export of
    ``langgraph.prebuilt.tool_node.ToolRuntime``, and LangChain validates an
    injected ``runtime`` argument against the *host* class. Subclassing it is
    what keeps ``isinstance`` and pydantic validation true for engine-produced
    runtimes; without the base, a host tool declared against the host class
    fails validation before its body runs. The import stays lazy so the engine
    remains usable without LangGraph.
    """
    try:
        from langgraph.prebuilt.tool_node import ToolRuntime as HostToolRuntime
    except ImportError:  # pragma: no cover - langgraph is an optional host dep
        return None
    return HostToolRuntime


HostToolRuntime = _host_tool_runtime()
_ToolRuntimeBase = HostToolRuntime or _DirectlyInjectedToolArg


class _ToolRuntimeMeta(type):
    """Accept host-created ``ToolRuntime`` objects as engine runtimes.

    Engine-produced runtimes are host subclasses, so host validators accept
    them. External callers (and DeerFlow's own tests) may still construct the
    host class directly and pass it into a tool declared against *our*
    annotation; pydantic's dataclass validator consults ``isinstance``, so
    widening only the framework class lets that direction through without
    accepting unrelated types.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        if HostToolRuntime is not None and isinstance(instance, HostToolRuntime):
            return True
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: type) -> bool:
        if (
            HostToolRuntime is not None
            and isinstance(subclass, type)
            and issubclass(subclass, HostToolRuntime)
        ):
            return True
        return super().__subclasscheck__(subclass)


@dataclass
class ToolCallRequest:
    """A tool call plus the state/runtime needed to execute it."""

    tool_call: ToolCall
    tool: Any | None
    state: Any
    runtime: ToolRuntime

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, "__dataclass_fields__") or not hasattr(self, name):
            object.__setattr__(self, name, value)
            return
        warnings.warn(
            f"Setting attribute '{name}' on ToolCallRequest is deprecated. "
            "Use the override() method instead to create a new instance with modified values.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ToolCallRequestOverrides]) -> ToolCallRequest:
        """Return a replacement request with *overrides* applied."""
        return replace(self, **cast("dict[str, Any]", overrides))


ToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]],
    ToolMessage | Command,
]


@dataclass
class ToolRuntime(
    _ToolRuntimeBase,  # type: ignore[misc, valid-type]
    Generic[ContextT, StateT],
    metaclass=_ToolRuntimeMeta,
):
    """Runtime context injected into a tool call.

    When LangGraph is installed this class really inherits the host's
    ``ToolRuntime``, so a host tool declared against that annotation accepts an
    engine-injected runtime instead of failing pydantic validation. The field
    list is re-declared to keep the engine's surface (and its JSON-schema-free
    ``Callable``/``Any`` annotations) stable with and without the host.
    """

    state: StateT
    context: ContextT
    config: dict[str, Any]
    stream_writer: StreamWriter | None
    tool_call_id: str | None
    store: Any | None
    tools: list[Any] = field(default_factory=list)
    execution_info: Any | None = None
    server_info: Any | None = None

    def emit_output_delta(self, delta: Any) -> None:
        """Stream a partial output chunk on the per-tool-call writer.

        Reads the writer installed by the tool execution runtime.  Silent no-op
        when the run did not request tool-output streaming, so tool authors can
        leave the call in place unconditionally.
        """
        writer = _tool_call_writer.get()
        if writer is None:
            return
        writer(delta)


def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach jump metadata used by the agent graph builder."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if can_jump_to is not None:
            func.__can_jump_to__ = can_jump_to  # type: ignore[attr-defined]
        return func

    return decorator


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def _make_state_hook(
    *,
    hook_name: str,
    async_hook_name: str,
    default_name: str,
    func: Callable[..., Any],
    state_schema: type[Any] | None,
    tools: list[Any] | None,
    can_jump_to: list[JumpTo] | None,
    name: str | None,
) -> AgentMiddleware[Any, Any, Any]:
    is_async = iscoroutinefunction(func)
    func_can_jump_to = (
        can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
    )

    if is_async:

        async def async_wrapped(
            _self: AgentMiddleware[Any, Any, Any], state: Any, runtime: Runtime[Any]
        ) -> Any:
            return await func(state, runtime)

        if func_can_jump_to:
            async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]
        wrapped: Callable[..., Any] = async_wrapped
        method_name = async_hook_name
    else:

        def sync_wrapped(
            _self: AgentMiddleware[Any, Any, Any], state: Any, runtime: Runtime[Any]
        ) -> Any:
            return func(state, runtime)

        if func_can_jump_to:
            sync_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]
        wrapped = sync_wrapped
        method_name = hook_name

    middleware_name = name or cast("str", getattr(func, "__name__", default_name))
    return cast(
        AgentMiddleware[Any, Any, Any],
        type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                method_name: wrapped,
            },
        )(),
    )


def _state_hook_decorator(
    *,
    hook_name: str,
    async_hook_name: str,
    default_name: str,
    func: Callable[..., Any] | None,
    state_schema: type[Any] | None,
    tools: list[Any] | None,
    can_jump_to: list[JumpTo] | None,
    name: str | None,
) -> Any:
    def decorator(inner: Callable[..., Any]) -> AgentMiddleware[Any, Any, Any]:
        return _make_state_hook(
            hook_name=hook_name,
            async_hook_name=async_hook_name,
            default_name=default_name,
            func=inner,
            state_schema=state_schema,
            tools=tools,
            can_jump_to=can_jump_to,
            name=name,
        )

    if func is not None:
        return decorator(func)
    return decorator


def before_model(
    func: Callable[..., Any] | None = None,
    *,
    state_schema: type[Any] | None = None,
    tools: list[Any] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware whose ``before_model`` hook is *func*."""
    return _state_hook_decorator(
        hook_name="before_model",
        async_hook_name="abefore_model",
        default_name="BeforeModelMiddleware",
        func=func,
        state_schema=state_schema,
        tools=tools,
        can_jump_to=can_jump_to,
        name=name,
    )


def after_model(
    func: Callable[..., Any] | None = None,
    *,
    state_schema: type[Any] | None = None,
    tools: list[Any] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware whose ``after_model`` hook is *func*."""
    return _state_hook_decorator(
        hook_name="after_model",
        async_hook_name="aafter_model",
        default_name="AfterModelMiddleware",
        func=func,
        state_schema=state_schema,
        tools=tools,
        can_jump_to=can_jump_to,
        name=name,
    )


def before_agent(
    func: Callable[..., Any] | None = None,
    *,
    state_schema: type[Any] | None = None,
    tools: list[Any] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware whose ``before_agent`` hook is *func*."""
    return _state_hook_decorator(
        hook_name="before_agent",
        async_hook_name="abefore_agent",
        default_name="BeforeAgentMiddleware",
        func=func,
        state_schema=state_schema,
        tools=tools,
        can_jump_to=can_jump_to,
        name=name,
    )


def after_agent(
    func: Callable[..., Any] | None = None,
    *,
    state_schema: type[Any] | None = None,
    tools: list[Any] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware whose ``after_agent`` hook is *func*."""
    return _state_hook_decorator(
        hook_name="after_agent",
        async_hook_name="aafter_agent",
        default_name="AfterAgentMiddleware",
        func=func,
        state_schema=state_schema,
        tools=tools,
        can_jump_to=can_jump_to,
        name=name,
    )


def dynamic_prompt(
    func: Callable[[ModelRequest[Any]], str | SystemMessage] | None = None,
) -> Any:
    """Create middleware that replaces the system message before a model call."""

    def decorator(
        inner: Callable[[ModelRequest[Any]], str | SystemMessage],
    ) -> AgentMiddleware[Any, Any, Any]:
        is_async = iscoroutinefunction(inner)

        def apply_prompt(
            request: ModelRequest[Any], prompt: str | SystemMessage
        ) -> ModelRequest[Any]:
            if isinstance(prompt, SystemMessage):
                return request.override(system_message=prompt)
            return request.override(system_message=SystemMessage(content=prompt))

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ModelRequest[Any],
                handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
            ) -> ModelResponse[Any] | AIMessage:
                async_inner = cast(
                    "Callable[[ModelRequest[Any]], Awaitable[str | SystemMessage]]",
                    inner,
                )
                prompt = await async_inner(request)
                return await handler(apply_prompt(request, prompt))

            wrapped: Callable[..., Any] = async_wrapped
            method_name = "awrap_model_call"
        else:

            def sync_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ModelRequest[Any],
                handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
            ) -> ModelResponse[Any] | AIMessage:
                prompt = inner(request)
                return handler(apply_prompt(request, prompt))

            async def async_from_sync(
                _self: AgentMiddleware[Any, Any, Any],
                request: ModelRequest[Any],
                handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
            ) -> ModelResponse[Any] | AIMessage:
                prompt = inner(request)
                return await handler(apply_prompt(request, prompt))

            wrapped = sync_wrapped
            method_name = "wrap_model_call"

        middleware_name = cast("str", getattr(inner, "__name__", "DynamicPromptMiddleware"))
        attrs: dict[str, Any] = {
            "state_schema": AgentState,
            "tools": [],
            method_name: wrapped,
        }
        if not is_async:
            attrs["awrap_model_call"] = async_from_sync
        return cast(
            AgentMiddleware[Any, Any, Any],
            type(middleware_name, (AgentMiddleware,), attrs)(),
        )

    if func is not None:
        return decorator(func)
    return decorator


def wrap_model_call(
    func: Callable[..., Any] | None = None,
    *,
    state_schema: type[Any] | None = None,
    tools: list[Any] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware from a ``(request, handler)`` model-call wrapper."""

    def decorator(inner: Callable[..., Any]) -> AgentMiddleware[Any, Any, Any]:
        is_async = iscoroutinefunction(inner)
        middleware_name = name or cast("str", getattr(inner, "__name__", "WrapModelCallMiddleware"))

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ModelRequest[Any],
                handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
            ) -> ModelCallResult[Any]:
                return await inner(request, handler)

            attrs: dict[str, Any] = {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "awrap_model_call": async_wrapped,
            }
        else:

            def sync_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ModelRequest[Any],
                handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
            ) -> ModelCallResult[Any]:
                return inner(request, handler)

            attrs = {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "wrap_model_call": sync_wrapped,
            }
        return cast(
            AgentMiddleware[Any, Any, Any],
            type(middleware_name, (AgentMiddleware,), attrs)(),
        )

    if func is not None:
        return decorator(func)
    return decorator


def wrap_tool_call(
    func: Callable[..., Any] | None = None,
    *,
    tools: list[Any] | None = None,
    name: str | None = None,
) -> Any:
    """Create middleware from a ``(request, handler)`` tool-call wrapper."""

    def decorator(inner: Callable[..., Any]) -> AgentMiddleware[Any, Any, Any]:
        is_async = iscoroutinefunction(inner)
        middleware_name = name or cast("str", getattr(inner, "__name__", "WrapToolCallMiddleware"))

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
            ) -> ToolMessage | Command[Any]:
                return await inner(request, handler)

            attrs = {
                "state_schema": AgentState,
                "tools": tools or [],
                "awrap_tool_call": async_wrapped,
            }
        else:

            def sync_wrapped(
                _self: AgentMiddleware[Any, Any, Any],
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
            ) -> ToolMessage | Command[Any]:
                return inner(request, handler)

            attrs = {
                "state_schema": AgentState,
                "tools": tools or [],
                "wrap_tool_call": sync_wrapped,
            }
        return cast(
            AgentMiddleware[Any, Any, Any],
            type(middleware_name, (AgentMiddleware,), attrs)(),
        )

    if func is not None:
        return decorator(func)
    return decorator

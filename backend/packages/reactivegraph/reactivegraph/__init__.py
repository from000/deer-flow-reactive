"""ReactiveGraph — reactive execution engine.

The Python package binds the Node Driver over the versioned RGP/1 protocol and
exposes the native reactive graph API. Build graphs with:

* high-level ``ReactiveGraph.build`` / ``GraphBuilder`` (task / computed / on /
  scope), with ``invoke`` executing through a Driver host or an in-process
  fallback;
* function/decorator workflows via ``functask`` / ``entrypoint`` /
  ``build_function_graph``.

State is transactional (``TrackedState``/``TrackedStateProxy``) and observable
through the scheduler trace. ``DriverHost`` manages the Node Driver subprocess.
"""

from reactivegraph._protocol_version import PROTOCOL_NAME, PROTOCOL_VERSION
from reactivegraph.checkpoint import (
    CheckpointStore,
    CheckpointTuple,
    MemoryCheckpointSaver,
    get_checkpointer,
    reset_checkpointer,
)
from reactivegraph.constants import (
    END,
    REMOVE_ALL_MESSAGES,
    START,
    TAG_HIDDEN,
    TAG_NOSTREAM,
)
from reactivegraph.errors import (
    ErrorCode,
    GraphBubbleUp,
    GraphInterrupt,
    GraphRecursionError,
    InvalidUpdateError,
    ParentCommand,
    create_error_message,
)
from reactivegraph.functional import (
    EntrypointPlan,
    FunctionTask,
    build_function_graph,
    entrypoint,
    functask,
)
from reactivegraph.graph import (
    ComputedDef,
    GraphBuilder,
    GraphBuildError,
    GraphDef,
    ReactiveGraph,
    TaskDef,
)
from reactivegraph.host import (
    DriverError,
    DriverHost,
    sanitize_traceback,
)
from reactivegraph.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
    add_usage,
    convert_to_messages,
    get_buffer_string,
    message_chunk_to_message,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
    subtract_usage,
)
from reactivegraph.prebuilt import (
    AgentIterationError,
    ReactiveAgent,
    ToolNode,
    ToolSpec,
    create_react_agent,
)
from reactivegraph.runtime import (
    Runtime,
    get_config,
    get_runtime,
    get_stream_writer,
    runtime_context,
)
from reactivegraph.state import TrackedState, TrackedStateProxy
from reactivegraph.store import (
    InvalidNamespaceError,
    Item,
    MemoryStore,
    SearchItem,
    get_store,
    reset_store,
)
from reactivegraph.types import Checkpointer, Command, Interrupt, Overwrite, Send

__version__ = "0.1.0"

__all__ = (
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "__version__",
    "GraphBuilder",
    "GraphDef",
    "ReactiveGraph",
    "TaskDef",
    "ComputedDef",
    "GraphBuildError",
    "Interrupt",
    "GraphInterrupt",
    "Command",
    "Overwrite",
    "Send",
    "Checkpointer",
    "ErrorCode",
    "GraphBubbleUp",
    "GraphRecursionError",
    "InvalidUpdateError",
    "ParentCommand",
    "create_error_message",
    "Runtime",
    "get_config",
    "get_runtime",
    "get_stream_writer",
    "runtime_context",
    "START",
    "END",
    "TAG_HIDDEN",
    "TAG_NOSTREAM",
    "REMOVE_ALL_MESSAGES",
    "FunctionTask",
    "EntrypointPlan",
    "functask",
    "entrypoint",
    "build_function_graph",
    "DriverHost",
    "DriverError",
    "sanitize_traceback",
    "ToolSpec",
    "ToolNode",
    "ReactiveAgent",
    "create_react_agent",
    "AgentIterationError",
    "TrackedState",
    "TrackedStateProxy",
    "AIMessage",
    "AIMessageChunk",
    "AnyMessage",
    "BaseMessage",
    "BaseMessageChunk",
    "ChatMessage",
    "HumanMessage",
    "HumanMessageChunk",
    "RemoveMessage",
    "SystemMessage",
    "SystemMessageChunk",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "add_usage",
    "convert_to_messages",
    "get_buffer_string",
    "message_chunk_to_message",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
    "subtract_usage",
    "CheckpointStore",
    "CheckpointTuple",
    "MemoryCheckpointSaver",
    "get_checkpointer",
    "reset_checkpointer",
    "InvalidNamespaceError",
    "Item",
    "MemoryStore",
    "SearchItem",
    "get_store",
    "reset_store",
)
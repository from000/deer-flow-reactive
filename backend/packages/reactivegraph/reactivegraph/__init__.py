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
    GraphInterrupt,
    Interrupt,
    ReactiveGraph,
    TaskDef,
)
from reactivegraph.host import (
    DriverError,
    DriverHost,
    sanitize_traceback,
)
from reactivegraph.prebuilt import (
    AgentIterationError,
    ReactiveAgent,
    ToolNode,
    ToolSpec,
    create_react_agent,
)
from reactivegraph.state import TrackedState, TrackedStateProxy

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
    "CheckpointStore",
    "CheckpointTuple",
    "MemoryCheckpointSaver",
    "get_checkpointer",
    "reset_checkpointer",
)
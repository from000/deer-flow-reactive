"""ReactiveGraph native Python API (Task 7 / docs/spec/native-api.md).

Mirror of packages/sdk-js/src/graph.ts: `GraphBuilder`, `ReactiveGraph`, `invoke`.
Python user functions run through the Driver callback host (Task 4); the graph
definition is compiled locally for validation, and event-routed tasks execute as
a single host callback that returns the resulting state (patches are produced by
the Driver on the callback result, exactly like `RUN` semantics).

Native semantics (event routes, pure skip, effect idempotency, conflicts) are
owned by the Driver scheduler; this module is the thin binding.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import functools
import json
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from reactivegraph.errors import GraphBuildError  # re-exported for importers
from reactivegraph.graph_calls import (
    MISSING,
    configs_for,
    resolve_batch_call,
    resolve_call,
)
from reactivegraph.graph_explain import ExplainMixin
from reactivegraph.host import DriverError, DriverHost
from reactivegraph.runtime import (
    Runtime,
    _PinnedIterator,
    get_config,
    get_runtime,
    runtime_context,
)
from reactivegraph.state import TrackedStateProxy

# Config keys that exist only on the Python side of the boundary. A ``Runtime``
# is not an RGP/1 canonical value (it carries callables), so it must be stripped
# before encoding or the codec raises ``CodecError``.
_PREGEL_RUNTIME_KEY = "__pregel_runtime"
_PYTHON_ONLY_CONFIGURABLE_KEYS = (_PREGEL_RUNTIME_KEY,)


def _pinned_runtime(config: dict | None) -> Runtime | None:
    """Return the ``Runtime`` a caller pinned into *config*, if any.

    LangGraph accepts ``config['configurable']['__pregel_runtime']`` as an
    internal channel; DeerFlow's run worker uses it because it drives the graph
    through ``astream(config=...)`` rather than the official ``context=``
    parameter. The object is Python-side only and never reaches the Driver.
    """
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    pinned = configurable.get(_PREGEL_RUNTIME_KEY)
    return pinned if isinstance(pinned, Runtime) else None


@contextlib.contextmanager
def _run_scope(config: dict | None) -> Iterator[None]:
    """Publish the run's config/runtime for the duration of a graph call.

    Node functions, tools and middleware read these through ``get_config()``
    and ``get_runtime()`` instead of receiving them as arguments. The ambient
    values the caller already set win over anything derived from *config*, so
    nesting a run inside another keeps the outer scope.
    """
    runtime = _pinned_runtime(config)
    if runtime is None:
        try:
            runtime = get_runtime()
        except RuntimeError:
            runtime = None
    if runtime is None:
        # Upstream ``get_runtime()`` always resolves inside a node: the pregel
        # loop injects a Runtime even when the caller supplied no config. Mint
        # an empty one so ``get_runtime().context`` is a valid (if empty) read
        # instead of an outside-a-run error.
        runtime = Runtime()
    if config is None:
        # Upstream ``get_config()`` always succeeds inside a node, even when
        # the caller passed no config at all; keep an outer run's config rather
        # than clobbering it with an empty one.
        try:
            config = get_config()
        except RuntimeError:
            config = {}
    with runtime_context(config=config, runtime=runtime):
        yield


def _thread_id_from_config(config: dict | None) -> str | None:
    """Return ``config.configurable.thread_id`` when the caller set one.

    The Driver keys its per-thread store by ``threadId``; without it every run
    falls back to the ``"default"`` store, so two concurrent invokes under
    different thread ids would share (and corrupt) one state. LangGraph's own
    contract is that ``config`` carries the thread id, and ``stream`` already
    forwards it explicitly — ``invoke`` has to do the same.
    """
    if not config:
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


def _wire_config(config: dict | None) -> dict:
    """Return *config* as an RGP/1-encodable value.

    ``configurable`` is the caller-facing escape hatch and may hold host
    objects; only the Python-only sentinel keys are dropped, everything else is
    passed through untouched.
    """
    if not config:
        return {}
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return dict(config)
    if not any(key in configurable for key in _PYTHON_ONLY_CONFIGURABLE_KEYS):
        return dict(config)
    trimmed = {
        key: value
        for key, value in configurable.items()
        if key not in _PYTHON_ONLY_CONFIGURABLE_KEYS
    }
    return {**config, "configurable": trimmed}


# Which task kind is executing on THIS thread. Set by the task callback
# wrapper around the user function and read by ``store_*`` so a pure task that
# reaches for the long-term store fails loudly instead of silently producing
# skip-invalid results (see ``ReactiveGraph.store_get``).
_current_task_kind: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reactivegraph_task_kind", default=None
)


def _is_public_state_key(key: Any) -> bool:
    """Whether a state key is observable in LangGraph's ``values`` stream.

    ``jump_to`` is an ephemeral routing directive and the ``__reactivegraph_*``
    namespace carries engine bookkeeping. Both may change on an otherwise
    no-op super-step, but neither is public channel state.
    """
    return key != "jump_to" and not (
        isinstance(key, str) and key.startswith("__reactivegraph_")
    )


def _fallback_fingerprint(state: dict, reads: tuple[str, ...] = ()) -> str | None:
    """in-process fallback 的 pure 任务输入指纹。

    粒度是**该任务实际依赖的输入**，不是整个 state：``reads`` 声明了任务的
    读集（支持 ``a.b`` 形式的下钻路径），只有这些路径上的值变化才让指纹失效。
    用整个 state 会让"改一个字段"使所有任务指纹全变——1000 任务图改 1 个字段
    要重跑 1000 次，与 Driver 路径的选择性执行行为不一致。

    ``reads`` 为空表示任务不声明依赖（宽松模式），此时退回整个 state：
    与旧的保守行为一致，不会误跳过。

    与 chain ``_fingerprint`` 同策：可 JSON 化才可缓存；不可序列化返回 None
    （调用方不应缓存）。``sort_keys`` 保证键序无关。
    """
    payload: Any = state
    if reads:
        payload = {
            read: ComputedDef._get(state, read)
            for read in reads
        }
    try:
        return json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        # Not JSON-serialisable: the callable cannot be reliably compared, so
        # the task must run (never cache on a lossy ``str()`` fingerprint).
        return None


def _runnable_base() -> tuple[type, ...]:
    """Return langchain-core's ``Runnable`` as an extra base, if installed.

    LangChain's composition (`a | b`) and its ``batch``/``astream`` helpers
    dispatch on ``isinstance(obj, Runnable)``, not on duck typing (see
    ``langchain_core.runnables.base.coerce_to_runnable``). Without the real base
    class our graph is rejected with "Expected a Runnable, callable or dict",
    so a host could not place a ReactiveGraph inside an existing LangChain
    pipeline.

    This mirrors the established pattern in ``reactivegraph.messages``:
    inherit upstream when it is importable, stay self-contained otherwise.
    The engine keeps its own ``invoke``/``stream``/``batch`` implementations —
    only the identity that LangChain checks comes from upstream.
    """
    try:
        from langchain_core.runnables import Runnable as _HostRunnable
    except ImportError:  # pragma: no cover - langchain-core is an optional dep
        return ()
    return (_HostRunnable,)


@dataclass(frozen=True)
class TaskOutcome:
    """A task result that also controls event routing.

    ``update`` is the state patch (same shapes ``TaskDef.fn`` may return);
    ``emits`` overrides the implicit ``<task_id>:written`` follow-up event.
    ``emits=()`` therefore stops this branch — that is how a model task ends an
    otherwise-cyclic ``model -> tools -> model`` loop. ``receipt`` records an
    idempotency receipt for effect tasks (same as the legacy
    ``(update, receipt)`` tuple).
    """

    update: Any = field(default_factory=dict)
    emits: tuple[str, ...] | None = None
    receipt: Any = None


@dataclass(frozen=True)
class TaskDef:
    """One declared task: function, triggers, reads, writes, kind."""
    id: str
    kind: str = "effect"  # "pure" | "effect" | "opaque"
    fn: Callable[[Any], Any] | None = None
    on: tuple[str, ...] = ()
    timeout_ms: int = 0
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    retry: dict | None = None
    scope: str | None = None
    max_runs: int = 1
    estimated_tokens_in: int = 0
    estimated_tokens_out: int = 0
    estimated_usd: float = 0.0


@dataclass(frozen=True)
class ComputedDef:
    """One declared computed value and its read set."""
    id: str
    selector: Callable[[dict], Any]
    reads: tuple[str, ...] = ()
    _cache: dict = field(default_factory=lambda: {"value": None, "hash": None})

    def evaluate(self, state: dict) -> Any:
        """Cached evaluation: recompute only when a read path's value changed."""
        h = self._read_hash(state)
        if self._cache["hash"] == h:
            return self._cache["value"]
        value = self.selector(state)
        self._cache["hash"] = h
        self._cache["value"] = value
        return value

    def _read_hash(self, state: dict) -> str:
        parts = [f"{r}:{self._get(state, r)}" for r in self.reads]
        parts.sort()
        return "|".join(parts)

    @staticmethod
    def _get(state: dict, path: str) -> Any:
        cur: Any = state
        for seg in path.split("."):
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return None
        return cur


@dataclass
class GraphDef:
    """Immutable graph definition produced by :class:`GraphBuilder`."""
    id: str
    tasks: list[TaskDef] = field(default_factory=list)
    computeds: list[ComputedDef] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    # Explicit entry event for the LangChain Runnable call form. ``None``
    # infers the event when the graph declares exactly one.
    entry_event: str | None = None
    _task_by_id: dict = field(default_factory=dict)
    _computed_by_id: dict = field(default_factory=dict)

    def task(self, td: TaskDef) -> None:
        """Declare a task function with its routing, reads and writes."""
        if td.id in self._task_by_id:
            hint = " — Hint: 每个任务 id 必须唯一,请检查重复的 task() 声明"
            raise GraphBuildError(f"duplicate task id: {td.id}{hint}")
        self._task_by_id[td.id] = td
        self.tasks.append(td)

    def computed(self, cd: ComputedDef) -> None:
        """Declare a lazily evaluated derived value."""
        if cd.id in self._task_by_id or cd.id in self._computed_by_id:
            hint = " — Hint: 每个 computed id 必须唯一,请检查重复的 computed() 声明"
            raise GraphBuildError(f"duplicate node id: {cd.id}{hint}")
        self._computed_by_id[cd.id] = cd
        self.computeds.append(cd)

    def route_for(self, event: str) -> list[str]:
        """Return the task ids routed by *event*."""
        return [t.id for t in self.tasks if event in t.on]


class GraphBuilder:
    """Fluent native builder (mirrors JS GraphBuilder)."""

    def __init__(self, graph_id: str = "graph") -> None:
        self._def = GraphDef(id=graph_id)

    def task(
        self,
        task_id: str,
        *,
        kind: str = "effect",
        fn: Callable[[Any], Any] | None = None,
        on: tuple[str, ...] = (),
        timeout_ms: int = 0,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        retry: dict | None = None,
        scope: str | None = None,
        max_runs: int = 1,
        estimated_tokens_in: int = 0,
        estimated_tokens_out: int = 0,
        estimated_usd: float = 0.0,
    ) -> GraphBuilder:
        """Declare a task function with its routing, reads and writes."""
        if not isinstance(max_runs, int) or isinstance(max_runs, bool) or max_runs < 1:
            raise GraphBuildError(
                f"task {task_id}: max_runs must be an integer >= 1 (got {max_runs!r})"
            )
        self._def.task(
            TaskDef(
                id=task_id,
                kind=kind,
                fn=fn,
                on=on,
                timeout_ms=timeout_ms,
                reads=reads,
                writes=writes,
                retry=retry,
                scope=scope,
                max_runs=max_runs,
                estimated_tokens_in=estimated_tokens_in,
                estimated_tokens_out=estimated_tokens_out,
                estimated_usd=estimated_usd,
            )
        )
        return self

    def computed(
        self, computed_id: str, selector: Callable[[dict], Any], reads: tuple[str, ...] = ()
    ) -> GraphBuilder:
        """Declare a lazily evaluated derived value."""
        self._def.computed(ComputedDef(id=computed_id, selector=selector, reads=reads))
        return self

    def on(self, event: str, task_id: str) -> GraphBuilder:
        """Subscribe this task to one or more trigger events."""
        td = self._def._task_by_id.get(task_id)
        if td is None:
            hint = " — Hint: 先 b.task(id, ...) 声明任务,再 b.on(event, id) 路由"
            raise GraphBuildError(f"route targets unknown task: {task_id}{hint}")
        self._def._task_by_id[task_id] = TaskDef(
            id=td.id,
            kind=td.kind,
            fn=td.fn,
            on=tuple(dict.fromkeys((*td.on, event))),
            timeout_ms=td.timeout_ms,
            reads=td.reads,
            writes=td.writes,
            retry=td.retry,
            scope=td.scope,
            estimated_tokens_in=td.estimated_tokens_in,
            estimated_tokens_out=td.estimated_tokens_out,
            estimated_usd=td.estimated_usd,
        )
        idx = next(i for i, t in enumerate(self._def.tasks) if t.id == task_id)
        self._def.tasks[idx] = self._def._task_by_id[task_id]
        return self

    def entry(self, event: str) -> GraphBuilder:
        """Declare the default entry event for the Runnable call form.

        ``graph.invoke(payload)`` (the shape LangChain's composition uses) has
        no place to name an event. When the graph routes exactly one event that
        one is used automatically; with several, this declaration picks one and
        the two-argument native form keeps working unchanged.
        """
        if not isinstance(event, str) or not event:
            raise GraphBuildError(
                f"entry event must be a non-empty string (got {event!r})"
            )
        self._def.entry_event = event
        return self

    def scope(self, name: str) -> GraphBuilder:
        """Restrict this task to a named sub-state namespace."""
        self._def.scopes.append(name)
        return self

    def build(self) -> GraphDef:
        """Materialise the declared graph definition."""
        return self._def


class ReactiveGraph(ExplainMixin, *_runnable_base()):  # type: ignore[misc]
    """Native graph handle: compile + invoke through the Driver host.

    When a `DriverHost` is attached the graph is compiled to the Driver
    (COMPILE_GRAPH) and executed by its reactive scheduler (RUN): task bodies
    run as host callbacks (TASK_INVOKE) and the Driver commits the returned
    patches — the callback never decides whether its own state changes commit.
    Without a host, a pure in-process fallback runs stateful tasks directly.
    """

    def __init__(
        self,
        graph_def: GraphDef,
        host: DriverHost | None = None,
        callback_id: str = "graph",
    ) -> None:
        self.definition = graph_def
        self.callback_id = callback_id
        self.host = host
        self._graph_id: str | None = None
        # in-process fallback（无 host）的 pure 任务指纹缓存：跨 invoke 共享
        # （同图对象复用跳过），deepcopy 存值防调用方污染；锁保证并发 ainvoke
        # 下同输入只执行一次（与 chain `_cache_lock` 语义等价）。
        self._fallback_cache: dict[str, tuple[str | None, dict]] = {}
        self._fallback_lock = threading.Lock()
        # pure 任务被指纹跳过时的可选回调（链侧统计/可观测用）；
        # 参数为被跳过的 task id。
        self._skip_callback: Callable[[str], None] | None = None
        # P3-3 可观测（附录 C）：最近一次执行的 trace 事件列表（fallback
        # 段级 / Driver run 级）。每次执行开始重置；条目
        # {"event": ...}——不阻断执行、异常安全。
        self._trace: list[dict] = []
        self._computed_invalidations: dict[str, tuple[str, ...]] = {}
        self._conflict: dict[str, Any] | None = None
        self._last_run_id: str | None = None
        # 编译/注册到 Driver 的并发锁：多线程首次并发 invoke/stream 同一
        # 图时防止重复 register/compile（duplicate graph id）——realworld
        # 并发线程隔离测试暴露（Task 5）。
        self._compile_lock = threading.Lock()

    @staticmethod
    def build(
        # The callback's return value is ignored. Accepting object keeps the
        # fluent ``lambda b: b.task(...).on(...)`` form type-correct for callers.
        build_fn: Callable[[GraphBuilder], object],
        host: DriverHost | None = None,
        graph_id: str = "graph",
    ) -> ReactiveGraph:
        """Materialise the declared graph definition."""
        builder = GraphBuilder(graph_id)
        build_fn(builder)
        return ReactiveGraph(builder.build(), host=host)

    # -- Driver compile -----------------------------------------------------

    def _ensure_compiled(self) -> str:
        """Compile the graph definition to the Driver once; register task
        callbacks. Returns the Driver-side graph id."""
        if self._graph_id is not None:
            return self._graph_id
        with self._compile_lock:  # 双检锁：并发首跑只编译一次
            if self._graph_id is not None:
                return self._graph_id
            if self.host is None:
                raise RuntimeError("ReactiveGraph has no DriverHost")
            for td in self.definition.tasks:
                if td.fn is not None:
                    self.host.register_callback(td.id, self._task_success_wrapper(td))
            for cd in self.definition.computeds:
                self.host.register_callback(
                    f"computed:{cd.id}", self._computed_success_wrapper(cd)
                )
            self._graph_id = self.host.compile_graph(self._to_wire_spec())
            return self._graph_id

    def _to_wire_spec(self) -> dict:
        """Serialize the GraphDef to the cross-language GraphSpec."""
        tasks: list[dict] = []
        routes: list[dict] = []
        for td in self.definition.tasks:
            task: dict[str, Any] = {
                "id": td.id,
                "kind": td.kind,
                "on": list(td.on),
                "timeoutMs": td.timeout_ms,
                "callbackId": td.id,
            }
            if td.reads:
                # Declared read set: the Driver uses it to scope the pure-task
                # fingerprint, so changing one unrelated field does not
                # invalidate every task (mirrors the in-process fallback).
                task["reads"] = list(td.reads)
            if td.writes:
                task["writes"] = list(td.writes)
            if td.retry:
                task["retry"] = td.retry
            if td.scope:
                task["scope"] = td.scope
            task["maxRuns"] = td.max_runs
            tasks.append(task)
            for ev in td.on:
                routes.append({"event": ev, "taskId": td.id})
        return {
            "id": self.definition.id,
            "tasks": tasks,
            "routes": routes,
            "scopes": list(self.definition.scopes),
            # RGP/1 computed wire：selector 作为 `computed:{id}` 回调由 Driver
            # 调度（Scheduler.compute 缓存/失效 + Transaction 提交）。
            "computeds": [
                {
                    "id": cd.id,
                    "reads": list(cd.reads),
                    "callbackId": f"computed:{cd.id}",
                }
                for cd in self.definition.computeds
            ],
        }

    @staticmethod
    def _split_task_result(raw: Any) -> tuple[Any, Any, list[str] | None]:
        """Normalize a task return value into ``(update, receipt, emits)``.

        ``emits is None`` means "no explicit routing" — the Driver then uses the
        implicit ``<task_id>:written`` follow-up event. An empty list is a
        deliberate loop stop and must not be collapsed into ``None``.
        """
        if isinstance(raw, TaskOutcome):
            emits = None if raw.emits is None else [str(e) for e in raw.emits]
            return raw.update, raw.receipt, emits
        update, receipt = ReactiveGraph._split_update_and_receipt(raw)
        return update, receipt, None

    @staticmethod
    def _split_update_and_receipt(raw: Any) -> tuple[Any, Any]:
        """Split a task result into (update, receipt). A task may return
        (update_dict, receipt) to record an idempotency receipt for this exact
        input; anything else is treated as a plain update with no receipt."""
        if (
            isinstance(raw, tuple)
            and len(raw) == 2
            and isinstance(raw[0], (dict, TrackedStateProxy))
        ):
            return raw[0], raw[1]
        return raw, None

    @staticmethod
    def _task_success_wrapper(td: TaskDef):
        """Wrap a task fn so its result becomes a TaskSuccess payload the Driver
        validates and commits.

        The input is wrapped in a `TrackedStateProxy`: reads the task performs
        on it are reported as `reads` (feeds conflict detection and the durable
        transaction log). The task may return either a plain update dict
        (keys become `set` patches), a `(update, receipt)` tuple, or the proxy
        itself after direct mutation (its recorded patches are reported
        as-is)."""

        def callback(input_: Any) -> dict:
            model = TrackedStateProxy(input_ or {})
            token = _current_task_kind.set(td.kind)
            try:
                raw = td.fn(model) if td.fn is not None else {}
            finally:
                _current_task_kind.reset(token)
            # Streaming callback: sync/async generators are passed through to
            # the host layer, which consumes them, streams each token/chunk as
            # `stream_chunks`, and uses the generator's return value as the
            # result — which MUST be a full TaskSuccess shape
            # ({patches, reads, writes, return_value, external_receipts}).
            if isinstance(raw, Iterator) or hasattr(raw, "__anext__"):
                return raw  # type: ignore[return-value]
            result, receipt, emits = ReactiveGraph._split_task_result(raw)
            if isinstance(result, TrackedStateProxy):
                patches = [dict(p, taskId=td.id) for p in result.patches()]
                writes = sorted({str(p["path"][0]) for p in patches}) if patches else []
            elif isinstance(result, dict):
                writes = list(result.keys())
                patches = [
                    {"path": [k], "operation": "set", "value": v, "taskId": td.id}
                    for k, v in result.items()
                ]
            else:
                writes, patches = [], []
            payload = {
                "reads": model.read_paths(),
                "patches": patches,
                "writes": writes,
                "return_value": result,
                "external_receipts": [{"receipt": receipt}] if receipt is not None else [],
            }
            if emits is not None:
                payload["emits"] = emits
            return payload

        return callback

    @staticmethod
    def _computed_success_wrapper(cd: ComputedDef):
        """Wrap a computed selector so its result becomes a TaskSuccess payload
        the Driver commits (patch path = `[computed.id]`), exactly like a task
        callback. The Driver owns when to invoke it (read-set invalidation),
        the caching (`Scheduler.compute`), and the atomic commit."""

        def callback(input_: Any) -> dict:
            state = input_ if isinstance(input_, dict) else {}
            value = cd.selector(state)
            return {
                "reads": list(cd.reads),
                "patches": [
                    {
                        "path": [cd.id],
                        "operation": "set",
                        "value": value,
                        "taskId": f"computed:{cd.id}",
                    }
                ],
                "writes": [cd.id],
                "return_value": value,
                "external_receipts": [],
            }

        return callback

    def _execute_fallback(
        self, event: str, payload: Any, on_event: Callable[[dict], None] | None
    ) -> dict:
        """In-process fallback execution (no Driver host attached).

        事件传播执行：从触发 ``event`` 开始，任务执行后按约定发出
        ``f"{tid}:written"`` 事件，``route_for`` 找到订阅者继续（注册顺序=
        传播顺序、串行；真实 Driver 为并行/反应式调度）。``fn`` 契约与
        Driver 一致（收到普通 dict 浅拷贝）；pure 任务按输入指纹跳过（对齐
        Driver 选择性）、effect/opaque 不跳过；缓存跨 invoke 共享（同图对
        象）、deepcopy 防调用方污染。computeds 在执行结束后按读路径哈希惰
        性求值（值以 computed id 写入 state）。
        """
        # in-process fallback：事件传播执行（方案 C P1）。
        # 从触发 `event` 队列开始：任务执行后按约定发出 f"{tid}:written"
        # 事件，route_for 找到订阅者继续（注册顺序=传播顺序、串行；真实
        # Driver 为并行/反应式调度，差异见 docs/plans/*-strengthening.md）。
        # fn 契约与 Driver 一致（收到普通 dict 浅拷贝）；pure 任务按输入
        # 指纹跳过（对齐 Driver 选择性）、effect/opaque 不跳过；缓存跨
        # invoke 共享（同图对象）、deepcopy 防调用方污染。computeds 在
        # 执行结束后按读路径哈希惰性求值（值以 computed id 写入 state）。
        with self._fallback_lock:
            state: dict = dict(payload or {})
            if on_event is not None:
                # P3-1 Task 3：初始帧对齐 Driver（runtime.ts emitValues 首帧）
                on_event(
                    {
                        "eventType": "values",
                        "payload": {"state": copy.deepcopy(state)},
                    }
                )
            self._propagate_fallback(event, state, on_event)
            self._evaluate_computeds(state, on_event)
        return state

    def _run_fallback_task(
        self,
        tid: str,
        state: dict,
        on_event: Callable[[dict], None] | None,
    ) -> tuple[Any, list[str] | None]:
        """Execute one fallback task, returning ``(update, emits)``.

        Pure tasks consult the cross-invoke fingerprint cache; effect/opaque
        tasks always run. Trace/task_start/task_error framing stays here so the
        propagation loop only has to deal with control flow.
        """
        td = self.definition._task_by_id[tid]
        self._trace.append({"event": f"task:{tid}:start", "task": tid})
        if on_event is not None:
            on_event({"eventType": "task_start", "task": tid})
        try:
            update: Any = None
            emits: list[str] | None = None
            if td.kind == "pure":
                fp = _fallback_fingerprint(state, td.reads)
                cached = self._fallback_cache.get(tid)
                if fp is not None and cached is not None and cached[0] == fp:
                    update = cached[1]
                    self._trace.append({"event": "cache_hit", "task": tid})
                    if self._skip_callback is not None:
                        self._skip_callback(tid)
                else:
                    update, _, emits = ReactiveGraph._split_task_result(
                        td.fn(dict(state)) or {}
                    )
                    if isinstance(update, dict) and fp is not None:
                        self._fallback_cache[tid] = (fp, copy.deepcopy(update))
            else:
                update, _, emits = ReactiveGraph._split_task_result(
                    td.fn(dict(state)) or {}
                )
        except BaseException:
            if on_event is not None:
                on_event({"eventType": "task_error", "task": tid})
            self._trace.append({"event": f"task:{tid}", "task": tid, "error": True})
            raise
        self._trace.append({"event": f"task:{tid}", "task": tid})
        if isinstance(update, dict):
            state.update(update)
        return update, emits

    def _emit_task_result(
        self,
        tid: str,
        update: Any,
        state: dict,
        on_event: Callable[[dict], None] | None,
    ) -> None:
        """Emit the values/task_end frames for a completed fallback task."""
        if on_event is None:
            return
        has_public_write = isinstance(update, dict) and any(
            _is_public_state_key(key) for key in update
        )
        if has_public_write:
            # LangGraph emits at most one values frame per super-step that
            # committed public channel state. Empty/private-only updates remain
            # observable as task_start/task_end but must not masquerade as a
            # public state transition.
            on_event(
                {
                    "eventType": "values",
                    "payload": {"state": copy.deepcopy(state)},
                }
            )
        on_event(
            {
                "eventType": "task_end",
                "task": tid,
                "writes": dict(update) if isinstance(update, dict) else {},
            }
        )

    def _propagate_fallback(
        self,
        event: str,
        state: dict,
        on_event: Callable[[dict], None] | None,
    ) -> None:
        """Run the fallback event-propagation loop until the queue drains."""
        # 重入预算：每个任务每次 invoke 至多执行 max_runs 次（默认 1，
        # 与 Driver 的 at-most-once 环截断一致）。显式声明 max_runs>1
        # 的任务允许在事件环中重入（model→tools→model）。
        runs: dict[str, int] = {}
        pending: deque[str] = deque([event])
        guard = 0
        while pending:
            ev = pending.popleft()
            for tid in self.definition.route_for(ev):
                budget = self.definition._task_by_id[tid].max_runs
                if runs.get(tid, 0) >= max(1, budget):
                    continue
                runs[tid] = runs.get(tid, 0) + 1
                guard += 1
                if guard > 10000:
                    raise GraphBuildError(
                        "fallback event propagation exceeded 10000 steps "
                        "(cycle?) — Hint: 检查 on() 订阅是否成环"
                    )
                if not self.definition._task_by_id[tid].fn:
                    continue
                update, emits = self._run_fallback_task(tid, state, on_event)
                self._emit_task_result(tid, update, state, on_event)
                # 约定事件：任务写入后触发下游订阅；任务可用
                # TaskOutcome.emits 覆盖（含空元组 = 终止本分支）。
                if emits is None:
                    pending.append(f"{tid}:written")
                else:
                    pending.extend(emits)

    def _evaluate_computeds(
        self, state: dict, on_event: Callable[[dict], None] | None
    ) -> None:
        """Evaluate every computed after propagation, recording invalidations."""
        for cd in self.definition.computeds:
            previous_hash = cd._cache.get("hash")
            current_hash = cd._read_hash(state)
            self._trace.append({"event": f"computed:{cd.id}:hit", "computed_id": cd.id})
            if on_event is not None:
                on_event({"eventType": "computed", "computed_id": cd.id})
            state[cd.id] = cd.evaluate(state)
            if previous_hash != current_hash and cd.reads:
                self._computed_invalidations[cd.id] = cd.reads


    def _execute(
        self, event: str, payload: Any, on_event: Callable[[dict], None] | None = None
    ) -> dict:
        """Route an event to its tasks and run them (through the Driver host).

        `on_event`（P3-1 流式）：fallback 执行期间的可观测钩子，事件形状
        task_start/task_end/task_error/values/computed——供 graph.stream 产出
        （host 模式经 host.run_stream 转发，不经过此钩子）。
        """
        self._trace = []  # P3-3：每次执行开始重置 trace
        self._conflict = None
        task_ids = self.definition.route_for(event)
        if not task_ids and not (self.host is None and self.definition.computeds):
            # fallback 下"无任务路由但存在 computeds"的图允许（只求值派生键）
            hint = " — Hint: 用 b.on(event, task_id) 或 task(..., on=(event,)) 声明路由"
            raise GraphBuildError(f"no task routes for event {event}{hint}")
        if self.host is None:
            return self._execute_fallback(event, payload, on_event)
        graph_id = self._ensure_compiled()
        self._trace = [{"event": "run:start"}]
        result = self.host.run(payload, graph_id=graph_id, event=event)
        # 保留完整 RunResult（含 runId/interrupted）：中断工作流经
        # self._last_run["runId"] 调 resume（P2-2）。
        self._last_run = result if isinstance(result, dict) else {}
        self._last_run_id = self._last_run.get("runId")
        self._trace.append(
            {
                "event": "run:end",
                "interrupted": bool(self._last_run.get("interrupted")),
            }
        )
        return self._last_run.get("state", {})

    def invoke(
        self,
        event: Any,
        payload: Any = MISSING,
        *,
        config: dict | None = None,
    ) -> dict:
        """Run the graph synchronously.

        Two call shapes are accepted on one object:

        * native — ``invoke(event, payload, config=...)`` names the entry event;
        * Runnable — ``invoke(input, config)`` is what LangChain's composition
          helpers call positionally (``a | graph``, ``.with_retry()``, ...).

        A graph that routes a single event infers it for the Runnable shape;
        with several events declare ``b.entry(event)``.
        """
        event, payload, config = resolve_call(
            self.definition, event, payload, config=config
        )
        if self.host is not None:
            from reactivegraph.protocol import make_id

            graph_id = self._ensure_compiled()
            # The Driver keys TASK_INVOKE by run id; generating it here (rather
            # than letting the Driver mint one) is what lets the host pin the
            # caller's context to the run before the first callback arrives.
            run_id = make_id()
            self._trace = [{"event": "run:start"}]
            with _run_scope(config):
                result = self.host.run(
                    payload,
                    graph_id=graph_id,
                    event=event,
                    run_id=run_id,
                    thread_id=_thread_id_from_config(config),
                    config=_wire_config(config),
                )
            # ``_last_run`` is graph-wide observability state for ``resume``.
            # It is intentionally shared, so return the local result: two
            # concurrent invokes must never race here and hand one caller the
            # other run's state.
            run_result = result if isinstance(result, dict) else {}
            self._last_run = run_result
            self._last_run_id = run_result.get("runId")
            self._trace.append({"event": "run:end"})
            return run_result.get("state", {})
        # in-process fallback: publish the same ambient config/runtime so a
        # host's middleware behaves identically with and without a Driver.
        with _run_scope(config):
            return self._execute(event, payload)

    # -- async 面 ----------------------------------------------------------
    # 底层 Driver 本身就是异步引擎;Python 端经 executor 线程跑同步内核,
    # 让调用方的事件循环不被阻塞(并发 invoke 由线程池承载)。

    async def ainvoke(
        self,
        event: Any,
        payload: Any = MISSING,
        *,
        config: dict | None = None,
    ) -> dict:
        """异步 invoke:在 executor 线程执行同步内核,不阻塞当前 event loop。

        `run_in_executor` 不继承调用方 ContextVar（`get_config`/`get_runtime`
        在任务体里读的就是它），因此在调用线程上先 copy_context，再把
        context 带进 executor 线程重入。

        `config` 与同步 `invoke` 同义：它必须透传到内核，否则异步运行里的
        `get_config()["configurable"]` 会是空的，中间件读不到调用方传入的
        `thread_id`（DeerFlow 的 ThreadDataMiddleware 正是这样读的）。
        """
        event, payload, config = resolve_call(
            self.definition, event, payload, config=config
        )
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            None, ctx.run, functools.partial(self.invoke, event, payload, config=config)
        )

    def batch(
        self,
        event: Any,
        payloads: Any = MISSING,
        *,
        config: dict | Sequence[dict] | None = None,
        return_exceptions: bool = False,
    ) -> list[dict]:
        """Run one event against several payloads, one result each, in order.

        LangChain's ``Runnable`` contract (``batch(inputs, configs)``) and the
        native ``batch(event, payloads)`` shape both work. Order is preserved
        and each payload is an independent run, so a batch cannot leak state
        between items.
        """
        event, payloads, config = resolve_batch_call(
            self.definition, event, payloads, config=config
        )
        configs = configs_for(config, len(payloads))
        results: list[Any] = []
        for payload, item_config in zip(payloads, configs, strict=True):
            try:
                results.append(self.invoke(event, payload, config=item_config))
            except Exception as exc:  # noqa: BLE001 - reported per item
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def abatch(
        self,
        event: Any,
        payloads: Any = MISSING,
        *,
        config: dict | Sequence[dict] | None = None,
        return_exceptions: bool = False,
    ) -> list[dict]:
        """Async twin of :meth:`batch` (sequential, deterministic ordering)."""
        event, payloads, config = resolve_batch_call(
            self.definition, event, payloads, config=config
        )
        configs = configs_for(config, len(payloads))
        results: list[Any] = []
        for payload, item_config in zip(payloads, configs, strict=True):
            try:
                results.append(await self.ainvoke(event, payload, config=item_config))
            except Exception as exc:  # noqa: BLE001 - reported per item
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def astream(
        self,
        event: Any,
        payload: Any = MISSING,
        *,
        config: dict | None = None,
    ) -> AsyncIterator[dict]:
        """异步 stream:同步生成器在 executor 线程迭代,事件经 asyncio.Queue 传出。

        `config` 与同步 `stream` 同义透传(Driver 侧 trace/concurrency 等)。
        """
        event, payload, config = resolve_call(
            self.definition, event, payload, config=config
        )
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        # 生产者线程不继承调用方 ContextVar；在协程帧内（仍在调用方
        # context 中）取快照，让任务体里的 get_config/get_runtime 可用。
        ctx = contextvars.copy_context()

        def _produce() -> None:
            try:
                for ev in self.stream(event, payload, config=config):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the async generator
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, sentinel)

        threading.Thread(target=ctx.run, args=(_produce,), daemon=True).start()
        while True:
            item = await q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def stream(
        self,
        event: Any,
        payload: Any = MISSING,
        *,
        thread_id: str | None = None,
        trace: bool = False,
        run_id: str | None = None,
        config: dict | None = None,
    ) -> Iterator[dict]:
        """Run the event route with streaming, publishing ``config`` for the
        duration of the run (upstream sets the child-runnable config for
        ``stream`` just like it does for ``invoke``).

        ``stream(input, config)`` (the ``Runnable`` shape) and the native
        ``stream(event, payload, ...)`` both work. Resolution is eager so a
        graph with an ambiguous entry event fails at the call, not on the
        first ``next()``.
        """
        event, payload, config = resolve_call(
            self.definition, event, payload, config=config
        )

        def _generate() -> Generator[dict, Any, Any]:
            with _run_scope(config):
                yield from self._stream_impl(
                    event,
                    payload,
                    thread_id=thread_id,
                    trace=trace,
                    run_id=run_id,
                    config=config,
                )

        return _PinnedIterator(_generate())

    def _stream_impl(
        self,
        event: str,
        payload: Any,
        *,
        thread_id: str | None = None,
        trace: bool = False,
        run_id: str | None = None,
        config: dict | None = None,
    ) -> Iterator[dict]:
        """Streaming implementation, already inside the run scope.

        Requires a DriverHost. Without a host, yields the in-process
        segment-boundary event sequence (task_start/task_end/task_error/
        values/computed + terminal). Driver failures are raised to the
        caller, not swallowed.

        `thread_id`（P3-1 Task 3）：host 模式透传给 Driver 的 threadId——
        新 thread 每次从空 store 起步，values 首帧与 fallback（每 run 全新
        state）严格同构；fallback 无 thread 概念，忽略该参数。
        """
        import queue
        import threading

        from reactivegraph.protocol import make_id

        task_ids = self.definition.route_for(event)
        if not task_ids:
            hint = " — Hint: 用 b.on(event, task_id) 或 task(..., on=(event,)) 声明路由"
            raise GraphBuildError(f"no task routes for event {event}{hint}")
        if self.host is None:
            yield from self._stream_fallback(event, payload)
            return
        graph_id = self._ensure_compiled()
        run_id = run_id or make_id()
        self._last_run_id = run_id
        events_q: queue.Queue[dict | BaseException | None] = queue.Queue()

        def _feed(ev: dict) -> None:
            events_q.put(ev)

        def _run() -> None:
            host = self.host
            assert host is not None  # guarded above
            try:
                host.run_stream(
                    payload,
                    _feed,
                    graph_id=graph_id,
                    event=event,
                    run_id=run_id,
                    thread_id=thread_id,
                    config={
                        **_wire_config(config),
                        **({"trace": True} if trace else {}),
                    },
                )
            except DriverError as exc:  # 段错误帧（review：仅 DriverError 补发 task_error）
                # P3-1 Task 4：Driver 错误帧上抛前补发 task_error 事件
                # （Driver 错误不携带任务名，task 置 None——与 fallback 的
                # 段级 task_error 形状对齐到 eventType 层）。
                events_q.put({"eventType": "task_error", "task": None})
                events_q.put(exc)
            except BaseException as exc:  # 基础设施异常（进程退出/超时）不伪装 task_error
                events_q.put(exc)
            finally:
                events_q.put(None)

        # 同上：`_run` 线程负责 host.run_stream，回调池 worker 的 run-scoped
        # context 由 host 从这一帧捕获。
        ctx = contextvars.copy_context()
        threading.Thread(target=ctx.run, args=(_run,), daemon=True).start()
        self._trace = []  # P3-3 Task 3：stream 开始重置（host 分支不经 _execute）
        self._conflict = None
        while True:
            ev = events_q.get()
            if ev is None:
                return
            if isinstance(ev, BaseException):
                if isinstance(ev, DriverError) and ev.meta is not None:
                    conflict = ev.meta.get("conflict")
                    if isinstance(conflict, dict):
                        self._conflict = conflict
                raise ev
            self._consume_custom_span(ev)
            yield ev

    def _stream_fallback(self, event: str, payload: Any) -> Iterator[dict]:
        """Yield fallback segment events, replaying queue order, then terminal.

        P3-1：fallback 经 ``_execute`` 的 on_event 钩子产出段边界事件
        （task_start/end/error + values/computed），terminal 收尾——与 Driver
        （host.run_stream）事件流对齐（design §3）。异常先让调用方消费已入队
        的 task_error 事件，再原样上抛。
        """
        import queue

        fallback_q: queue.Queue[dict | None] = queue.Queue()

        def _fallback_feed(ev: dict) -> None:
            fallback_q.put(ev)

        exc: BaseException | None = None
        try:
            self._execute(event, payload, on_event=_fallback_feed)
        except BaseException as err:
            exc = err
        finally:
            fallback_q.put(None)
        while True:
            fb_ev = fallback_q.get()
            if fb_ev is None:
                break
            yield fb_ev
        if exc is not None:
            raise exc
        yield {"eventType": "terminal"}

    def _consume_custom_span(self, ev: dict) -> None:
        """Translate a Driver custom span frame into ``_trace`` entries.

        P3-3 Task 3：custom span 帧（payload.event="span:{kind}"）消费为
        ``_trace``（span 级事件，不阻断；custom 帧本身仍透传给调用方）。
        """
        if ev.get("eventType") != "custom":
            return
        cpayload = ev.get("payload") or {}
        cname = cpayload.get("event") or ""
        if not cname.startswith("span:"):
            return
        sp = cpayload.get("payload") or {}
        sname = sp.get("name") or ""
        attrs = sp.get("attributes") or {}
        tid = attrs.get("taskId", "")
        cid = attrs.get("computedId", "")
        if sname.startswith("task:") and tid:
            if sname.endswith(":skip"):
                reason = str(attrs.get("reason", "unknown"))
                event = "cache_hit" if reason == "fingerprint_unchanged" else "skip"
                self._trace.append({"event": event, "task": tid, "reason": reason})
            else:
                self._trace.append({"event": f"{sname}:start", "task": tid})
                end: dict = {"event": sname, "task": tid}
                if sp.get("status") == "error":
                    end["error"] = True
                self._trace.append(end)
        elif sname.startswith("computed:") and cid and sname.endswith(":invalidate"):
            self._computed_invalidations[cid] = tuple(attrs.get("reads") or [])
        elif sname.startswith("computed:") and cid:
            # computed 求值/缓存命中 → 对齐 fallback trace
            # （computed:{id}:hit 命中、computed:{id} 求值）
            suffix = "" if sname.endswith(":hit") else ":hit"
            self._trace.append({"event": f"{sname}{suffix}", "computed_id": cid})

    @staticmethod
    def _causal_trace_event(item: dict) -> dict[str, Any]:
        """Normalize a Driver/fallback trace entry to causal-trace v1."""
        event_name = str(item.get("event", ""))
        task_id = item.get("task")
        computed_id = item.get("computed_id")
        reason = item.get("reason")
        seq = item.get("seq") or item.get("sequence")
        if event_name == "run:start":
            kind = "trigger"
        elif event_name == "run:end":
            kind = "span_end"
        elif event_name in {"cache_hit", "skip"}:
            kind = "cache_hit"
        elif event_name.endswith(":start"):
            kind = "span_start"
        elif event_name.endswith(":hit"):
            kind = "cache_hit"
        elif event_name.startswith("computed:"):
            kind = "patch"
        elif item.get("error"):
            kind = "span_end"
        else:
            kind = "span_end"
        out: dict[str, Any] = {"seq": int(seq or 0), "kind": kind}
        if isinstance(task_id, str):
            out["taskId"] = task_id
        if isinstance(computed_id, str):
            out["selector"] = computed_id
        if isinstance(reason, str):
            out["reason"] = reason
        if item.get("error"):
            out["reason"] = "error"
        return out

    def resume(self, run_id: str, interrupt_response: Any, thread_id: str | None = None) -> dict:
        """Resume an interrupted run with the human response (requires a DriverHost)."""
        if self.host is None:
            hint = " — Hint: ReactiveGraph.build(build_fn, host=host) 传入 DriverHost"
            raise GraphBuildError(f"resume requires a DriverHost{hint}")
        return self.host.resume(run_id, interrupt_response, thread_id=thread_id)

    # -- checkpoint / long-term store (require a DriverHost) ----------------

    def _require_host(self) -> Any:
        if self.host is None:
            hint = " — Hint: checkpoint/store 操作需 host=DriverHost(...),见 05-durability.md"
            raise GraphBuildError(f"this operation requires a DriverHost{hint}")
        return self.host

    def get_state(self, thread_id: str = "default") -> dict:
        """Latest committed thread state (GET_STATE)."""
        return self._require_host().get_state(thread_id)

    def list_checkpoints(self, thread_id: str = "default") -> list:
        """Thread checkpoint history (checkpoint op `list`)."""
        return self._require_host().checkpoint_op("list", thread_id=thread_id)

    def get_checkpoint(self, thread_id: str = "default", checkpoint_id: str | None = None) -> Any:
        """A single thread checkpoint (checkpoint op `get`)."""
        return self._require_host().checkpoint_op(
            "get", thread_id=thread_id, checkpoint_id=checkpoint_id
        )

    def delete_thread(self, thread_id: str = "default") -> None:
        """Delete a thread's checkpoints (checkpoint op `delete_thread`)."""
        self._require_host().checkpoint_op("delete_thread", thread_id=thread_id)

    def restore_thread(self, thread_id: str = "default", checkpoint_id: str | None = None) -> dict:
        """Time travel / fork: roll the thread state back to a historical
        checkpoint (checkpoint op `restore`). The restored snapshot becomes a
        NEW committed state — a subsequent invoke() continues from it.
        Returns the restored state."""
        return self._require_host().checkpoint_op(
            "restore", thread_id=thread_id, checkpoint_id=checkpoint_id
        )

    def _reject_store_access_from_pure_task(self, operation: str) -> None:
        """Fail loudly when a ``pure`` task touches the long-term store.

        A pure task is skipped when its declared ``reads`` are unchanged, and
        those reads can only name *state* keys. The store lives outside the
        state graph, so its contents are invisible to that fingerprint: a pure
        task that reads it can be skipped even though an earlier run wrote new
        data, silently returning stale results (this happened in the DeerFlow
        memory port). Declaring the task ``opaque`` is the correct fix —
        opaque tasks are never skipped and never receipt-gated.
        """
        if _current_task_kind.get() == "pure":
            raise GraphBuildError(
                f"store_{operation} called from a task declared kind='pure'. "
                "A pure task is skipped when its declared reads are unchanged, "
                "and the long-term store is not part of that fingerprint — so "
                "the task could silently observe stale data. Declare the task "
                "kind='opaque' instead."
            )

    def store_get(self, namespace: Sequence[str], key: str) -> Any:
        """Read one item's value from the long-term store (None if absent)."""
        self._reject_store_access_from_pure_task("get")
        result = self._require_host().store_op("get", namespace=namespace, key=key)
        return result.get("value") if isinstance(result, dict) else result

    def store_put(self, namespace: Sequence[str], key: str, value: Any) -> None:
        """Write one item to the long-term store."""
        self._reject_store_access_from_pure_task("put")
        self._require_host().store_op("put", namespace=namespace, key=key, value=value)

    def store_search(
        self,
        namespace: Sequence[str],
        filter_: dict | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list:
        """Search items in a store namespace."""
        return self._require_host().store_op(
            "search", namespace=namespace, key="", filter_=filter_, limit=limit, offset=offset
        )

    def store_delete(self, namespace: Sequence[str], key: str) -> None:
        """Delete one item from the long-term store."""
        self._reject_store_access_from_pure_task("delete")
        self._require_host().store_op("delete", namespace=namespace, key=key)

    def list_namespaces(self) -> list:
        """List store namespaces."""
        return self._require_host().store_op("list_namespaces", namespace=[], key="")


__all__ = (
    "ComputedDef",
    "GraphBuildError",
    "GraphBuilder",
    "GraphDef",
    "ReactiveGraph",
    "TaskDef",
)

"""ReactiveGraph Python host: launches the bundled Node Driver as a child process
and speaks RGP/1 over stdio.

Responsibilities (Task 4):
- one long-lived bundled Driver child process per host instance,
- handshake negotiation before any graph work,
- register Python callbacks by stable IDs; a callback result/exception/cancellation
  correlates to exactly one invocation,
- preserve sanitized traceback metadata in task failure events,
- guarantee no orphan child processes on close/crash.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import os
import shutil
import subprocess
import threading
import traceback
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from reactivegraph.protocol import (
    FrameDecoder,
    encode_frame,
    encode_value,
)
from reactivegraph.wire import from_wire_value, to_wire_value

# Discover the bundled Driver entry: environment override, then package-relative.
_DRIVER_ENTRY_ENV = "REACTIVEGRAPH_DRIVER_ENTRY"


def default_driver_entry() -> str:
    """Resolve the bundled Driver entry script path."""
    env = os.environ.get(_DRIVER_ENTRY_ENV)
    if env:
        return env
    here = Path(__file__).resolve().parents[3]  # repo root
    candidate = here / "packages" / "driver" / "dist" / "main.js"
    if candidate.exists():
        return str(candidate)
    # fall back to a globally installed CLI if present
    resolved = shutil.which("reactivegraph-driver")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "bundled Driver entry not found; set REACTIVEGRAPH_DRIVER_ENTRY "
        "to packages/driver/dist/main.js"
    )


def default_node_bin() -> str:
    """Resolve the Node executable used to spawn the Driver."""
    env = os.environ.get("REACTIVEGRAPH_NODE_BIN")
    if env:
        return env
    resolved = shutil.which("node")
    if resolved:
        return resolved
    raise FileNotFoundError("node executable not found")


Callback = Callable[[Any], Any]
AsyncCallback = Callable[[Any], Awaitable[Any]]


class DriverError(Exception):
    """A task failed on the Driver side; carries sanitized metadata."""

    def __init__(self, message: str, meta: dict | None = None) -> None:
        super().__init__(message)
        self.meta = meta


class DriverHost:
    """Manages the Driver child process and the RGP/1 session to it."""

    def __init__(
        self,
        node_bin: str | None = None,
        driver_entry: str | None = None,
        env: dict | None = None,
        node: bool = True,
        checkpoint_backend: str | None = None,
        db_path: str | None = None,
        pg_dsn: str | None = None,
        redis_url: str | None = None,
    ) -> None:
        """DriverHost 会话。

        `checkpoint_backend`/`db_path`/`pg_dsn`/`redis_url`（P3-4）为便捷
        参数：优先合并进 env（`REACTIVEGRAPH_DB`/`REACTIVEGRAPH_PG_DSN`/
        `REACTIVEGRAPH_REDIS_URL`），Driver 侧 main.ts buildPersistence 已
        支持 sqlite/pg/redis/memory；`checkpoint_backend="memory"` 时强制
        移除 DB env（不注入任何持久后端）。

        注意（语义）：`redis_url` 与 checkpoint 持久化无关——Driver 侧仅
        用于构建 `RedisCache`（缓存层）；内存/持久化的 checkpoint 由
        `db_path`（sqlite）/`pg_dsn`（postgres）驱动。与 checkpoint 参数
        同族仅因共用环境变量机制。
        """
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        if checkpoint_backend == "memory":
            self.env.pop("REACTIVEGRAPH_DB", None)
            self.env.pop("REACTIVEGRAPH_PG_DSN", None)
            self.env.pop("REACTIVEGRAPH_REDIS_URL", None)
        else:
            if db_path is not None:
                self.env["REACTIVEGRAPH_DB"] = db_path
            if pg_dsn is not None:
                self.env["REACTIVEGRAPH_PG_DSN"] = pg_dsn
            if redis_url is not None:
                self.env["REACTIVEGRAPH_REDIS_URL"] = redis_url
        # Resolve entry/node from the effective env so callers can point tests at
        # a fake driver or custom node without mutating the process environment.
        self.node_bin = (
            node_bin or self.env.get("REACTIVEGRAPH_NODE_BIN") or default_node_bin()
        )
        self.driver_entry = (
            driver_entry or self.env.get("REACTIVEGRAPH_DRIVER_ENTRY") or default_driver_entry()
        )
        self._process: subprocess.Popen | None = None
        self._decoder = FrameDecoder()
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._callbacks: dict[str, Callback] = {}
        self._handshake_done = False
        self._closed = False
        self._reader: threading.Thread | None = None
        self._exit_code: int | None = None
        # Stream subscribers: a run may register one sink for STREAM_EVENTs.
        self._stream_sink: dict[str, Callable[[dict], None]] = {}  # runId -> sink
        # Run-scoped ambient context (``get_config``/``get_runtime``). Task
        # bodies execute on ``_callback_pool`` workers, and a ContextVar does
        # not travel across a thread boundary; the caller's context is captured
        # here (still on the caller's thread) and re-entered on the worker,
        # keyed by the run id the Driver echoes back on TASK_INVOKE.
        self._run_context: dict[str, contextvars.Context] = {}  # runId -> Context
        self._run_context_lock = threading.Lock()
        # P3-2 并行兑现：TASK_INVOKE 回调在线程池并行执行（reader 线程只
        # 做分发；串行批在途仅一个 TASK_INVOKE，天然保序；并行批 runAll
        # 并发执行回调）。worker 数经 env REACTIVEGRAPH_CALLBACK_WORKERS
        # 配置（默认 4，最小 1）。用户回调需自行保证线程安全（并发语义
        # 与 runAll 一致）。
        workers = int(self.env.get("REACTIVEGRAPH_CALLBACK_WORKERS", "4"))
        self._callback_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, workers),
            thread_name_prefix="rg-cb",
        )

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the Driver process and complete the handshake."""
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            [self.node_bin, self.driver_entry],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        try:
            while True:
                # NOTE: must use read1(), not read(): read(n) on a pipe blocks
                # until the buffer fills (or EOF), while read1() returns whatever
                # is available now. The Driver writes short frames and keeps the
                # pipe open, so read() would stall forever.
                stdout = self._process.stdout
                assert stdout is not None
                chunk = stdout.read1(4096)  # type: ignore[attr-defined]
                if not chunk:
                    break
                with self._read_lock:
                    for frame in self._decoder.push(chunk):
                        self._handle_frame(frame)
        except Exception as exc:  # pragma: no cover - defensive
            # Log instead of silently dying: a decode failure must be visible.
            import sys

            print(f"[host] reader loop error: {exc}", file=sys.stderr)
        finally:
            self._on_peer_exit()

    def _handle_frame(self, frame: bytes) -> None:
        from reactivegraph.protocol import decode_value

        envelope = from_wire_value(decode_value(frame))
        kind = envelope.get("kind")
        eid = envelope.get("id")
        method = envelope.get("method")
        if kind == "response":
            pending = self._pending.pop(eid, None)
            if pending is None:
                return
            payload = envelope.get("payload") or {}
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                pending["set_exception"](
                    DriverError(
                        f"{err.get('type', 'Error')}: {err.get('message', 'unknown')}",
                        meta=err.get("meta") if isinstance(err, dict) else None,
                    )
                )
            else:
                pending["set_result"](payload)
        elif kind == "request" and method == "TASK_INVOKE":
            self._handle_task_invoke(envelope)
        elif kind == "event":
            self._handle_event(envelope)

    def _handle_event(self, envelope: dict) -> None:
        """Dispatch STREAM_EVENT frames to the sink registered for the run."""
        method = envelope.get("method")
        if method != "STREAM_EVENT":
            return
        payload = envelope.get("payload") or {}
        run_id = payload.get("runId")
        sink = self._stream_sink.get(run_id) if run_id else None
        if sink is None:
            return
        try:
            sink(payload)
        except Exception:  # noqa: BLE001 - a misbehaving sink must not kill the reader
            pass

    def _handle_task_invoke(self, envelope: dict) -> None:
        payload = envelope.get("payload") or {}
        invocation_id = payload.get("invocationId")
        if not isinstance(invocation_id, str):
            self._send_response(
                envelope["id"],
                "TASK_INVOKE",
                {"error": {"type": "ProtocolError", "message": "missing task invocationId"}},
            )
            return
        callback_id = payload.get("callbackId")
        if not isinstance(callback_id, str):
            callback_id = None
        input_value = payload.get("input")
        callback = self._callbacks.get(callback_id or "") if callback_id else None
        if callback is None:
            self._send_response(
                envelope["id"],
                "TASK_INVOKE",
                {
                    "error": {
                        "type": "UnknownCallback",
                        "message": f"unregistered callback id {callback_id!r}",
                    }
                },
            )
            return
        # Re-enter the *caller's* context on the worker: a ContextVar set
        # before ThreadPoolExecutor.submit does not travel to the worker, and
        # task bodies read the run-scoped config/runtime from it.
        ctx = self._run_context_for(envelope)
        if ctx is None:
            self._callback_pool.submit(
                self._execute_callback, invocation_id, envelope["id"], callback, input_value,
            )
        else:
            self._callback_pool.submit(
                ctx.run,
                self._execute_callback,
                invocation_id,
                envelope["id"],
                callback,
                input_value,
            )

    def _execute_callback(
        self, invocation_id: str, req_id: str, callback: Callable, input_value: Any,
    ) -> None:
        """Execute one TASK_INVOKE callback off the reader thread (P3-2 并行
        兑现：回调在 `_callback_pool` worker 执行，reader 线程只做分发；
        `_send_task_result` 发送经写锁，线程安全。串行批在途仅一个
        TASK_INVOKE，天然保序；并行批（runAll）并发执行回调）。"""
        try:
            result = callback(input_value)
            if hasattr(result, "__anext__"):
                # Async-generator callback: consume tokens/chunks and stream
                # them back as the TASK_INVOKE response's `stream_chunks`,
                # before the final committed result.
                def _run_async_stream() -> None:
                    async def _consume() -> tuple[Any, list[dict[str, Any]]]:
                        chunks: list[dict[str, Any]] = []
                        value: Any = None
                        while True:
                            try:
                                item = await result.__anext__()
                            except StopAsyncIteration as stop:
                                value = getattr(stop, "value", None)
                                break
                            chunks.append(self._as_stream_chunk(item))
                        return value, chunks

                    try:
                        value, chunks = asyncio_run(_consume())
                        self._send_task_result(invocation_id, req_id, value, None, chunks)
                    except Exception as exc:  # noqa: BLE001
                        self._send_task_result(invocation_id, req_id, None, exc)

                threading.Thread(target=_run_async_stream, daemon=True).start()
            elif hasattr(result, "__await__"):

                def _run_async() -> None:
                    try:
                        value = asyncio_run(result)
                        self._send_task_result(invocation_id, req_id, value, None)
                    except Exception as exc:  # noqa: BLE001
                        self._send_task_result(invocation_id, req_id, None, exc)

                threading.Thread(target=_run_async, daemon=True).start()
            elif isinstance(result, Iterator):
                # Sync-generator callback: same contract, driven on a worker
                # thread; each yielded token/chunk becomes a stream_chunk.
                def _run_stream() -> None:
                    try:
                        chunks: list[dict[str, Any]] = []
                        value: Any = None
                        gen = iter(result)
                        while True:
                            try:
                                item = next(gen)
                            except StopIteration as stop:
                                value = stop.value
                                break
                            chunks.append(self._as_stream_chunk(item))
                        self._send_task_result(invocation_id, req_id, value, None, chunks)
                    except Exception as exc:  # noqa: BLE001
                        self._send_task_result(invocation_id, req_id, None, exc)

                threading.Thread(target=_run_stream, daemon=True).start()
            else:
                self._send_task_result(invocation_id, req_id, result, None)
        except Exception as exc:  # noqa: BLE001
            self._send_task_result(invocation_id, req_id, None, exc)

    @staticmethod
    def _as_stream_chunk(item: Any) -> dict[str, Any]:
        """Normalize a generator yield into a wire stream chunk.

        A dict with ``type``/``payload`` passes through; anything else (e.g. a
        token string) becomes a transient ``messages`` chunk.
        """
        if isinstance(item, dict) and "type" in item and "payload" in item:
            return {"type": item["type"], "payload": item["payload"]}
        return {"type": "messages", "payload": {"role": "assistant", "content": str(item)}}

    def _send_task_result(
        self,
        invocation_id: str,
        req_id: str,
        value: Any,
        exc: BaseException | None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        payload: dict[str, Any]
        if exc is not None:
            payload = {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": sanitize_traceback(exc),
                }
            }
        elif (
            isinstance(value, dict)
            and "patches" in value
            and "return_value" in value
        ):
            # Callback already returned TaskSuccess shape (reads/patches/
            # return_value/external_receipts) — the Driver validates and
            # commits the patches; it never trusts the callback blindly.
            payload = {
                "reads": value.get("reads", []),
                "patches": value.get("patches", []),
                "writes": value.get("writes", []),
                "return_value": value.get("return_value"),
                "external_receipts": value.get("external_receipts", []),
            }
            # 显式 follow-up 事件（RGP/1 §6 扩展）：缺失 = 隐式
            # `<task_id>:written`；`[]` = 终止本分支（循环收敛）。
            if "emits" in value:
                payload["emits"] = value["emits"]
        else:
            payload = {
                "reads": [],
                "patches": [],
                "return_value": value,
                "external_receipts": [],
            }
        if chunks:
            payload["stream_chunks"] = chunks
        self._send_response(req_id, "TASK_INVOKE", payload)

    def _send_response(self, req_id: str, method: str, payload: Any) -> None:
        envelope = {
            "version": 1,
            "id": req_id,
            "kind": "response",
            "method": method,
            "payload": payload,
        }
        self._write_frame(envelope)

    def _on_peer_exit(self) -> None:
        code = None
        if self._process is not None:
            code = self._process.poll()
        self._exit_code = code
        for pending in list(self._pending.values()):
            pending["set_exception"](
                DriverError(f"driver process exited (code={code}); session terminated")
            )
        self._pending.clear()

    # -- requests -----------------------------------------------------------

    def _write_frame(self, envelope: dict) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        with self._write_lock:
            self._process.stdin.write(encode_frame(encode_value(to_wire_value(envelope))))
            self._process.stdin.flush()

    def request(self, method: str, payload: Any, timeout_ms: int = 30000) -> Any:
        """Send one request and await its matching response frame."""
        if self._closed or self._process is None:
            raise DriverError("driver host is closed")
        from reactivegraph.protocol import make_id

        request_id = make_id()
        event = threading.Event()
        result_holder: dict[str, Any] = {"value": None, "error": None}

        def _set_result(v: Any) -> None:
            result_holder["value"] = v
            event.set()

        def _set_exception(e: BaseException) -> None:
            result_holder["error"] = e
            event.set()

        # Publish the callbacks BEFORE writing the frame so a fast response from
        # the reader thread can never race ahead of registration.
        with self._write_lock:
            if self._closed or self._process is None:
                raise DriverError("driver host is closed")
            self._pending[request_id] = {"set_result": _set_result, "set_exception": _set_exception}
            assert self._process.stdin is not None
            self._process.stdin.write(
                encode_frame(
                    encode_value(
                        to_wire_value(
                            {
                                "version": 1,
                                "id": request_id,
                                "kind": "request",
                                "method": method,
                                "payload": payload,
                            }
                        )
                    )
                )
            )
            self._process.stdin.flush()

        if not event.wait(timeout_ms / 1000):
            with self._write_lock:
                self._pending.pop(request_id, None)
            raise DriverError(f"request {method} timed out after {timeout_ms}ms")
        if result_holder["error"] is not None:
            raise result_holder["error"]
        return result_holder["value"]

    def handshake(self, sdk_version: str = "0.1.0", timeout_ms: int = 10000) -> dict:
        """Exchange protocol versions and capabilities."""
        result = self.request(
            "DRIVER_HELLO",
            {"sdkVersion": sdk_version, "protocolVersions": [1], "featureFlags": []},
            timeout_ms=timeout_ms,
        )
        self._handshake_done = True
        return result

    def run(self, input_value: Any, callback_id: str = "default", **kwargs: Any) -> Any:
        # ``runId`` must reach the Driver: it is the key the Driver echoes back
        # on every TASK_INVOKE, and therefore what ties a callback worker back
        # to the caller's run-scoped context.
        """Invoke a graph by id and return the run result."""
        payload = {"graphId": kwargs.get("graph_id"), "input": input_value,
                   "threadId": kwargs.get("thread_id"), "callbackId": callback_id,
                   "event": kwargs.get("event"), "config": kwargs.get("config"),
                   "runId": kwargs.get("run_id")}
        run_id = kwargs.get("run_id")
        with self._pin_run_context(run_id):
            return self.request("RUN", payload)

    def run_stream(self, input_value: Any, on_event: Callable[[dict], None],
                   callback_id: str = "default", **kwargs: Any) -> dict:
        """RUN with streaming: STREAM_EVENT payloads are delivered to
        `on_event` as they arrive (from the reader thread), then the final
        run result (state) is returned. The caller must provide a `run_id`
        so events can be routed back to this call."""
        run_id = kwargs.get("run_id")
        if not run_id:
            raise DriverError("run_stream requires run_id for event routing")
        self._stream_sink[run_id] = on_event
        try:
            payload = {
                "graphId": kwargs.get("graph_id"),
                "input": input_value,
                "threadId": kwargs.get("thread_id"),
                "callbackId": callback_id,
                "event": kwargs.get("event"),
                "runId": run_id,
                "stream": True,
                # P3-3 Task 3：config 透传（trace=True 时 Driver 开启 span
                # 桥接，custom span 帧经 STREAM_EVENT 送回 Python _trace）。
                "config": kwargs.get("config") or {},
            }
            with self._pin_run_context(run_id):
                result = self.request("RUN", payload)
            return result.get("state", {}) if isinstance(result, dict) else {}
        finally:
            self._stream_sink.pop(run_id, None)

    @contextlib.contextmanager
    def _pin_run_context(self, run_id: str | None) -> Iterator[None]:
        """Publish this thread's ambient run context for *run_id*.

        The Driver echoes the run id on every ``TASK_INVOKE``, which lets the
        callback worker look up the context its caller was running under. The
        entry is removed when the run returns, so a finished run never leaks
        context into a later one.
        """
        if not run_id:
            yield
            return
        snapshot = contextvars.copy_context()
        with self._run_context_lock:
            self._run_context[run_id] = snapshot
        try:
            yield
        finally:
            with self._run_context_lock:
                self._run_context.pop(run_id, None)

    def _run_context_for(self, envelope: dict) -> contextvars.Context | None:
        """Return a fresh copy of the run's captured context.

        A ``Context`` may only be entered by one thread at a time, and the
        Driver dispatches parallel batches (``runAll``) concurrently — so every
        callback gets its own copy instead of sharing one.
        """
        trace = envelope.get("trace")
        run_id = trace.get("runId") if isinstance(trace, dict) else None
        if not isinstance(run_id, str):
            return None
        with self._run_context_lock:
            captured = self._run_context.get(run_id)
            # Copied under the lock: the stored context is never entered, so
            # copying it is always legal and stays race-free.
            return captured.copy() if captured is not None else None

    def compile_graph(self, spec: dict, schema_hash: str | None = None,
                      timeout_ms: int = 30000) -> str:
        """COMPILE_GRAPH: register a graph definition with the Driver.

        `spec` is the cross-language GraphSpec (tasks/routes/scopes). Returns
        the compiled graph id.
        """
        if schema_hash is not None:
            spec = {**spec, "schemaHash": schema_hash}
        result = self.request("COMPILE_GRAPH", spec, timeout_ms=timeout_ms)
        return result["graphId"]

    def release_graph(self, graph_id: str, timeout_ms: int = 30000) -> dict:
        """Drop the cached graph id for this session."""
        return self.request("RELEASE_GRAPH", {"graphId": graph_id}, timeout_ms=timeout_ms)

    def get_state(self, thread_id: str = "default", timeout_ms: int = 30000) -> dict:
        """Read thread state from the Driver."""
        result = self.request("GET_STATE", {"config": {"threadId": thread_id}},
                              timeout_ms=timeout_ms)
        return result.get("values", {})

    def checkpoint_op(self, op: str, thread_id: str = "default",
                      checkpoint_id: str | None = None, record: Any = None,
                      timeout_ms: int = 30000) -> Any:
        """CHECKPOINT_OP: get | list | put | delete_thread on the Driver's
        checkpoint saver for a thread."""
        payload: dict[str, Any] = {"threadId": thread_id, "op": op}
        if checkpoint_id is not None:
            payload["checkpointId"] = checkpoint_id
        if record is not None:
            payload["record"] = record
        result = self.request("CHECKPOINT_OP", payload, timeout_ms=timeout_ms)
        return result.get("result") if isinstance(result, dict) else result

    def store_op(self, op: str, namespace: list[str] | tuple[str, ...] = (),
                 key: str = "", value: Any = None, filter_: dict | None = None,
                 limit: int | None = None, offset: int | None = None,
                 timeout_ms: int = 30000) -> Any:
        """STORE_OP: get | put | search | delete | list_namespaces on the
        Driver's long-term store."""
        payload: dict[str, Any] = {"namespace": list(namespace), "key": key, "op": op}
        if value is not None:
            payload["value"] = value
        if filter_ is not None:
            payload["filter"] = filter_
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset
        result = self.request("STORE_OP", payload, timeout_ms=timeout_ms)
        return result.get("result") if isinstance(result, dict) else result

    def vector_upsert(
        self,
        vector: list[float],
        namespace: Sequence[str] | None = None,
        id: str = "v",
        metadata: Any = None,
    ) -> None:
        """Persist a vector point (VECTOR_UPSERT)."""
        self.request(
            "VECTOR_UPSERT",
            {
                "namespace": list(namespace or []),
                "id": id,
                "vector": list(vector),
                "metadata": metadata,
            },
        )

    def vector_search(
        self,
        query: list[float],
        namespace: Sequence[str] | None = None,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """Cosine nearest-neighbour search (VECTOR_SEARCH)."""
        result = self.request(
            "VECTOR_SEARCH",
            {
                "namespace": list(namespace or []),
                "query": list(query),
                "limit": limit,
                "minScore": min_score,
            },
        )
        return result.get("results", []) if isinstance(result, dict) else []

    def request_internal(self, method: str, payload: Any,
                         timeout_ms: int = 30000) -> Any:
        """Send an RGP/1 request without a typed wrapper.

        Reserved for Driver-side read-only introspection endpoints whose wire
        method is accepted by both protocol peers but not yet promoted to the
        stable public Host API.
        """
        return self.request(method, payload, timeout_ms=timeout_ms)

    def export_dot(self, graph_id: str = "graph") -> str:
        """Render a compiled graph as Graphviz DOT (EXPORT_DOT round trip)."""
        result = self.request("EXPORT_DOT", {"graphId": graph_id})
        return result.get("dot", "") if isinstance(result, dict) else ""

    def resume(self, run_id: str, interrupt_response: Any, thread_id: str | None = None,
               timeout_ms: int = 30000) -> dict:
        """Resume an interrupted run with new values."""
        result = self.request("RESUME", {
            "runId": run_id,
            "threadId": thread_id,
            "interruptResponse": interrupt_response,
        }, timeout_ms=timeout_ms)
        state = result.get("state", {}) if isinstance(result, dict) else {}
        # 多段 HITL：resume 期间任务可再次挂起（Driver 报告 interrupted）——
        # 返回 state 视图并注入 interrupted 标志（向后兼容：既有调用方
        # 拿到的仍是 state 键值，A5/D3 语义不变）。
        if isinstance(result, dict) and result.get("interrupted") is True:
            return {**state, "interrupted": True}
        return state

    def register_callback(self, callback_id: str, callback: Callback) -> None:
        """Register a Python callable invoked by the Driver."""
        self._callbacks[callback_id] = callback

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        """Gracefully shut down the Driver; idempotent; never leaks a child."""
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        try:
            self.request("SHUTDOWN", {}, timeout_ms=3000)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self._process.kill()
                except Exception:  # noqa: BLE001
                    pass
            # Close the three pipes explicitly. ``Popen`` keeps them alive
            # until GC otherwise, which surfaces as ``PytestUnraisableException
            # Warning: Exception ignored in: <_io.FileIO ...>`` under
            # ``-W error`` and leaks descriptors in long-lived hosts.
            for stream in (
                self._process.stdin,
                self._process.stdout,
                self._process.stderr,
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001 - best-effort close
                        pass
            self._process = None
        # 在途回调不等待（与 async 回调线程同语义）：进程已终止，后续
        # 响应无人消费；wait=False 防止 close 被慢回调阻塞。
        self._callback_pool.shutdown(wait=False)

    def __enter__(self) -> DriverHost:
        self.start()
        self.handshake()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def sanitize_traceback(exc: BaseException) -> dict[str, Any]:
    """Return sanitized traceback metadata: last frame file/line/name only."""
    tb = traceback.extract_tb(exc.__traceback__)
    last = tb[-1] if tb else None
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "file": last.filename if last else None,
        "line": last.lineno if last else None,
        "function": last.name if last else None,
    }


def asyncio_run(coro: Awaitable[Any]) -> Any:
    """Run an async callback to completion (sync host bridge)."""
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


# Re-export for convenience
__all__ = (
    "DriverError",
    "DriverHost",
    "default_driver_entry",
    "default_node_bin",
    "sanitize_traceback",
)
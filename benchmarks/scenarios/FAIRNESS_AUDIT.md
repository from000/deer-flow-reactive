审计日期：2026-10-15
对照版本：fork = deer-flow-reactive；upstream = deer-flow @ 4582a41347fa5680238e8dd5c8a7d0468ad6c606

# 真实 Agnes LLM 对照测试：公平性代码审计

## 审计范围与方法

- **目的**：确保即将进行的「真实 Agnes LLM 对照测试」对 fork（ReactiveGraph 执行底座）与 upstream（LangGraph 执行底座）公平，重点评估对 `engine_s` 与 `llm_calls` 两个指标的影响。
- **范围**：两仓库 `backend/packages/harness/deerflow/agents/**`、两个 venv 中的 `langchain` / `langchain-openai` / `openai` 源码，以及 fork 的 `backend/packages/reactivegraph/reactivegraph/**`。
- **方法**：纯代码审计。逐文件 diff + 逐行引用，不运行任何 LLM 调用，不修改任何业务代码；环境版本用 `importlib.metadata.version()` 与文件 sha1 校验。
- **仓库标签约定**：
  - `[fork]` = `deer-flow-reactive`（工作区 `/Users/zhangxiqiang/code/git/deer-flow-reactive`）
  - `[upstream]` = `deer-flow` @ `4582a41347fa5680238e8dd5c8a7d0468ad6c606`（工作区 `/Users/zhangxiqiang/code/git/deer-flow`）
- **本次审计未执行 LLM**：所有结论只来自源码、依赖版本与文件哈希。

---

## Q1. `create_deerflow_agent` 的 middleware 链默认差异

### 结论

**存在 1 处实质差异：fork 的 `create_deerflow_agent` 默认链比 upstream 多挂 2 个 middleware（`DurableContextMiddleware` + `SystemMessageCoalescingMiddleware`）。** `sandbox` / `memory` / `summarization` / `subagent` / `vision` / `auto_title` / `guardrail` / `loop_detection` / `token_budget` 的 `RuntimeFeatures` 默认值两边完全一致。另有一处仅当调用方显式 `token_budget=True` 时才生效的潜在差异（本次默认路径不触发）。

### 证据

**(1) `RuntimeFeatures` 默认值：两边逐字段一致。**

- `[upstream] backend/packages/harness/deerflow/agents/features.py:30-41`
  ```python
  30    sandbox: bool | AgentMiddleware = True
  31    memory: bool | AgentMiddleware = False
  32    # Explicit memory config for direct create_deerflow_agent(features=...) callers.
  33    # The lead-agent AppConfig path passes resolved_app_config.memory directly.
  34    memory_config: MemoryConfig | None = None
  35    summarization: Literal[False] | AgentMiddleware = False
  36    subagent: bool | AgentMiddleware = False
  37    vision: bool | AgentMiddleware = False
  38    auto_title: bool | AgentMiddleware = False
  39    guardrail: Literal[False] | AgentMiddleware = False
  40    loop_detection: bool | AgentMiddleware = True
  41    token_budget: bool | AgentMiddleware = False
  ```
- `[fork] backend/packages/harness/deerflow/agents/features.py:30-41`：同一段字段与默认值逐字一致；唯一差别是文件头的 middleware 基类导入（`[upstream] :11 from langchain.agents.middleware import AgentMiddleware` vs `[fork] :11 from reactivegraph.middleware import AgentMiddleware`），不影响默认链。

**(2) fork 在默认链中额外追加 2 个 middleware。**

- `[fork] backend/packages/harness/deerflow/agents/factory.py:272-289`
  ```python
  272    # --- [5a] DurableContext (always) ---
  ...
  286    chain.append(DurableContextMiddleware(pii_redaction_config=pii_redaction_config))
  287    # DurableContext adds its authority contract as a second SystemMessage; strict backends
  288    # (vLLM, SGLang, Qwen, Anthropic) reject that, so merge them into one leading message.
  289    chain.append(SystemMessageCoalescingMiddleware())
  ```
- `[upstream] backend/packages/harness/deerflow/agents/factory.py:244-247`：`ToolErrorHandlingMiddleware()`（`:245`）之后直接进入 `# --- [6] Summarization ---`（`:247`），**没有** `DurableContextMiddleware` 与 `SystemMessageCoalescingMiddleware`。
- 说明：这两个 middleware 在 upstream 只挂在 **lead_agent 应用路径**，未进入 SDK factory：
  - `[upstream] backend/packages/harness/deerflow/agents/lead_agent/agent.py:359-365`（DurableContext）、`:425-427`（SystemMessageCoalescing）
  - `[fork] backend/packages/harness/deerflow/agents/lead_agent/agent.py:604-612`（DurableContext）、`:695-697`（SystemMessageCoalescing）

**(3) `token_budget=True` 时的潜在差异（默认不触发）。**

- `[upstream] backend/packages/harness/deerflow/agents/factory.py:343`
  ```python
  343            chain.append(TokenBudgetMiddleware.from_config(TokenBudgetConfig()))
  ```
- `[fork] backend/packages/harness/deerflow/agents/factory.py:435-436`
  ```python
  435            # ``enabled`` defaults to False for config.yaml; ``token_budget=True`` is the opt-in.
  436            chain.append(TokenBudgetMiddleware.from_config(TokenBudgetConfig(enabled=True)))
  ```
- `TokenBudgetConfig.enabled` 默认 `False`：`[upstream] backend/packages/harness/deerflow/config/token_budget_config.py:6-9`。
- 该分支只在 `feat.token_budget is not False`（即显式传 `token_budget=True`）时进入；本次 benchmark 使用默认 `token_budget=False`，**两边都跳过**，不构成实际差异。

**(4) 其余链段逐条对应（顺序一致）。**

- `[fork] factory.py:249-439` 与 `[upstream] factory.py:222-346`：
  sandbox（fork `:249-257` / up `:222-232`）、DanglingToolCall（fork `:260` / up `:235`）、ToolErrorHandling（fork `:270` / up `:245`）、Summarization（fork `:291-298` / up `:247-252`）、Todo（fork `:298-302` / up `:254-258`）、AutoTitle（fork `:304-311` / up `:260-267`）、Memory（fork `:313-353` / up `:269-297`）、Vision（fork `:355-362` / up `:299-306`）、Subagent（fork `:369-415` / up `:313-322`）、LoopDetection（fork `:417-425` / up `:325-333`）、TokenBudget（fork `:427-436` / up `:335-343`）、Clarification（fork `:439` / up `:346`）。

**(5) 两个额外 middleware 在本次场景下近似 no-op（若无 summary/delegation/skill/system 注入）。**

- `[fork] backend/packages/harness/deerflow/agents/middlewares/durable_context_middleware.py:296-334`
  ```python
  296    def _inject(self, request: ModelRequest) -> ModelRequest:
  ...
  305        if not data_block:
  306            return request
  ```
- `[fork] backend/packages/harness/deerflow/agents/middlewares/system_message_coalescing_middleware.py:62-79`
  ```python
  77        in_msg_systems = [m for m in request.messages if isinstance(m, SystemMessage)]
  78        if not in_msg_systems:
  79            return None
  ```

### 影响评估

- **对 `llm_calls`**：无影响。两个额外 middleware 都不改变模型调用次数；在无 summary/delegation/skill/SystemMessage 注入时它们直接透传请求。
- **对 `engine_s`**：方向对 fork 不利（保守）。fork 每轮多执行 2 个 middleware 的 hook 节点，属于内在本地开销，只会抬高 fork 的 `engine_s`。因此若 fork 仍显著更快，结论是稳健的；但**两边 middleware 链并非逐项对等，必须在报告中披露**。

---

## Q2. 真实 `ChatOpenAI` 实例处理对称性

### 结论

**大体对称，但存在 4 处不对称**：

1. fork 每图仅 `bind_tools` 一次并缓存结果；upstream **每次模型调用都重新 `bind_tools` / `bind`**。
2. fork 每次模型调用多做一次消息「dict 序列化 → 反序列化」往返。
3. fork `_BoundModel.bind()` 不传 `tool_choice` / `model_settings`（本次 benchmark 未设置这些参数，故当前等价）。
4. 没有 fake-model / real-model 专用分支——真实模型与 fake 模型在两边都走同一条执行路径；也没有 `with_structured_output` / `response_format` 注入。

### 证据

**(1) fork 复用了缓存后的 bound model，且不传 `tool_choice/model_settings`。**

- `[fork] backend/packages/reactivegraph/reactivegraph/agent_recursion.py:183-213`
  ```python
  183    class _BoundModel:
  ...
  190        def __init__(self, model: Any, tools: Sequence[Any]) -> None:
  191            self._model = model
  192            self._tools = list(tools)
  193            self._bound: Any = None
  ...
  195        def bind(self) -> Any:
  ...
  197            if self._tools and hasattr(self._model, "bind_tools"):
  198                self._bound = self._model.bind_tools(self._tools)
  199            elif hasattr(self._model, "bind"):
  200                self._bound = self._model.bind()
  ...
  205        def invoke(self, messages: list[AnyMessage]) -> AIMessage:
  206            """Run *event* (or the graph) synchronously and return the result."""
  207            bound = self._bound if self._bound is not None else self.bind()
  ...
  212            payload = [message_to_foreign_dict(message) for message in messages]
  213            return convert_to_messages([bound.invoke(payload)])[0]
  ```

**(2) upstream 每次模型执行都重新 bind。**

- `[upstream] backend/.venv/lib/python3.12/site-packages/langchain/agents/factory.py:1272-1404`（`_get_bound_model` 内 `:1399` `request.model.bind_tools(...)`、`:1404` `request.model.bind(**request.model_settings)`）
- `[upstream] 同文件 :1406-1419`
  ```python
  1406    def _execute_model_sync(request: ModelRequest[ContextT]) -> ModelResponse:
  ...
  1413        # Get the bound model (with auto-detection if needed)
  1414        model_, effective_response_format = _get_bound_model(request)
  ...
  1419        output = model_.invoke(messages)
  ```
  `_get_bound_model` 在 `:1414` 每次调用时被执行，因此 bind 不缓存。

**(3) 无 structured output / response_format 注入。**

- `[fork] backend/packages/reactivegraph/reactivegraph/create_agent.py:1386-1396`
  ```python
  1386    _reject_unimplemented(
  1387        {
  1388            "response_format": response_format,
  ...
  1393            "cache": cache,
  ...
  1396        raise NotImplementedError("response_format")
  ```
- 两边 `create_deerflow_agent` 都未向底层 `create_agent` 传 `response_format`：`[upstream] factory.py:168-176`、`[fork] factory.py:189-197`。

**(4) 无真实模型专用分支。**

- `[fork] agent_recursion.py:190-203` 仅按 `hasattr(model,"bind_tools"/"bind")` 分支，不检查模型类型（fake / real 同一路径）。
- `[upstream] langchain/agents/factory.py:1272-1404` 同样按能力分支，无 fake/real 专用逻辑。

**(5) 工具集对称（保证两边都会 bind_tools）。**

- `[upstream] factory.py:347`：`extra_tools.append(ask_clarification_tool)`
- `[fork] factory.py:440`：`extra_tools.append(ask_clarification_tool)`
- 因此即使调用方不传 tools，两边 `effective_tools` 也非空，都会走 `bind_tools` 路径——这部分公平。

### 影响评估

- **对 `llm_calls`**：无影响。bind 行为、序列化往返都不改变实际 HTTP 调用次数。
- **对 `engine_s`**：两处偏差方向相反、量级都小。fork 省下每轮重复 bind，却多付每轮 message↔dict 序列化；两者均是本地 CPU，相比真实 LLM 秒级延迟可忽略，但对 fake model 毫秒级场景会显现。**这是引擎实现差异，属于真实差异而非"白得加速"，必须在报告中如实说明。**
- **(3) 的语义边界**：本 benchmark 未设置 `tool_choice` / `model_settings`，故当前等价；若未来业务设置这些参数，fork 会丢参。此类场景不得计入「fork 更强」。

---

## Q3. loop_detection scope 差异对单次单轮的影响

### 结论

**不会改变 `llm_calls`。** 本对照场景每个 thread 独立、每次 invoke 单轮：`model_only` 无 tool_call，`tool_call` 只有 1 个重复哈希，远低于 `warn=3`/`hard=5`，两边都不会 warn、不会 hard-stop，模型调用次数一致。

### 证据

**(1) 两边 scope 键不同。**

- `[upstream] backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py:432-444`
  ```python
  432        thread_id = self._get_thread_id(runtime)
  ...
  437            if thread_id in self._history:
  438                self._history.move_to_end(thread_id)
  439            else:
  440                self._history[thread_id] = []
  ...
  443            history = self._history[thread_id]
  444            history.append(call_hash)
  ```
- `[fork] backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py:463-465` 与 `:557-570`
  ```python
  463    def _run_scope_key(self, runtime: Runtime) -> _RunScopeKey:
  464        """Return the shared tracking key for the current thread/run."""
  465        return self._get_thread_id(runtime), self._get_run_id(runtime)
  ...
  557        scope_key = self._run_scope_key(runtime)
  558        thread_id, run_id = scope_key
  ...
  563            if scope_key in self._history:
  564                self._history.move_to_end(scope_key)
  565            else:
  566                self._history[scope_key] = []
  ...
  569            history = self._history[scope_key]
  570            history.append(call_hash)
  ```

**(2) 两边阈值相同。**

- `[upstream] loop_detection_middleware.py:77-78`
  ```python
  77    _DEFAULT_WARN_THRESHOLD = 3  # inject warning after 3 identical calls
  78    _DEFAULT_HARD_LIMIT = 5  # force-stop after 5 identical calls
  ```
- `[fork] loop_detection_middleware.py:92-93`：常量与注释逐字一致。

**(3) 新鲜 thread + 单轮 → 哈希计数 ≤ 1，未达阈值。**

- 单轮 `tool_call` 场景（本 benchmark 的 `tool_call`）每个 thread 只会向 scope 历史追加 1 个 `call_hash`，`count=1 < warn=3 < hard=5`；`model_only` 更无 tool_call。
- fork 的 `before_agent` 只保留拓扑、不污染计数：`[fork] loop_detection_middleware.py:833-843`。

### 影响评估

- **对 `llm_calls`**：单次单轮场景公平、无差异；两边都不会触发 loop 保护的额外模型调用。
- **对多轮场景的边界（必须披露）**：同一 thread 连续重复相同 tool_call 到第 5 轮时，两边行为真实分叉——upstream thread 级计数会 hard-stop，fork run 级隔离继续执行。此前落盘数据 `benchmarks/scenarios/loop_detection_results.json` 记录：upstream `message_counts=[10,2,2,2,2,2]`、`total_model_calls=10`；fork `message_counts=[10,10,10,10,10,10]`、`total_model_calls=30`。这是行为/语义差异（upstream 自动止损），不是速度差异；该场景**不得**与单轮场景混入同一性能结论。

---

## Q4. 版本与 max_retries / 429 处理一致性

### 结论

**429 与重试行为应当一致**：两边 `langchain-openai`、`openai`、`httpx` 版本完全相同，`langchain_openai/chat_models/base.py` 文件 sha1 完全相同，重试路径一致。但**测量侧有一个必须修正的问题**：`real_llm.py` 在修订前会把约 61 秒的限流等待算进 `engine_s`；该文件已于 2026-10-15 07:54 修正，但修正后**尚未重跑**，因此旧的 `real_llm_results.json` 不得直接引用。

### 证据

**(1) 依赖版本对比（两 venv 实测）。**

| 包 | [upstream] | [fork] |
|---|---|---|
| langchain-openai | 1.2.1 | 1.2.1 |
| openai | 2.32.0 | 2.32.0 |
| httpx | 0.28.1 | 0.28.1 |
| langchain | 1.3.14 | 1.3.14 |
| langgraph | 1.2.9 | 1.2.9 |
| pydantic | 2.13.3 | 2.13.3 |
| langchain-core | 1.4.9 | **1.6.10（不同）** |

- 说明：`langchain-core` 版本不同，但不参与 OpenAI HTTP / 429 重试逻辑；重试由 `langchain-openai` + `openai` 完成。

**(2) `langchain_openai/chat_models/base.py` 两边逐字节一致。**

- sha1：
  - `[upstream] backend/.venv/lib/python3.12/site-packages/langchain_openai/chat_models/base.py` → `d365e737d9fe67ee5bb7496e8d527dadfd247c1b`
  - `[fork] .../langchain_openai/chat_models/base.py` → `d365e737d9fe67ee5bb7496e8d527dadfd247c1b`

**(3) `max_retries` 走同一路径。**

- `[upstream] langchain_openai/chat_models/base.py:697`：`max_retries: int | None = None`
- `[upstream] 同文件 :1183-1184`
  ```python
  1183        if self.max_retries is not None:
  1184            client_params["max_retries"] = self.max_retries
  ```
- 测试脚本显式 `max_retries=0` 传给 `ChatOpenAI`：`[fork] benchmarks/scenarios/scripts/real_llm.py` 的 `build_model()`（`TimedModel(..., max_retries=0, timeout=120)`）。

**(4) 429 记账问题（测量侧）。**

- 旧结果 `[fork] benchmarks/scenarios/real_llm_results.json`（2026-10-15 07:32）：
  - reactivegraph：`wall_s=64.7394`、`llm_s=3.2944`、`rate_limit_wait_s=0.3936`、`engine_s=61.0514`
  - upstream：`wall_s=82.404`、`llm_s=20.8672`、`rate_limit_wait_s=0.3855`、`engine_s=61.1512`
  - `engine_s≈61s` 与 Agnes 60 秒限流窗口吻合，说明约 61 秒等待被错误计入 engine_s。
- 修订后的脚本 `[fork] benchmarks/scenarios/scripts/real_llm.py`（07:54）在 429 分支同时累计「失败请求耗时」与 `time.sleep(nap)`：
  ```python
  if status == 429 or "RateLimit" in name or "429" in str(e)[:200]:
      RATE_LIMIT_WAITS.append(time.perf_counter() - t0)  # 失败请求本身的耗时
      if attempt < 3:
          nap = 61 + attempt * 5
          t_sleep = time.perf_counter()
          time.sleep(nap)
          RATE_LIMIT_WAITS.append(time.perf_counter() - t_sleep)
  ```
  并在 `engine_s = wall - sum(LLM_TIMES) - waits` 中扣除。
- 结论：**修订后必须重跑真实 Agnes 测试**，否则 engine_s 不具可比性。

### 影响评估

- **重试语义**：公平。两边库版本、`base.py`、`max_retries=0` 设置一致。
- **对 `engine_s`**：旧数据不可用；新数据需剔除 `rate_limit_wait_s` 并交叉验证（wall / llm / waits 三项独立上报）。
- **对 `llm_calls`**：无影响（`len(LLM_TIMES)` 只统计成功调用）。
- **Agnes 约束**：账号级 1 req/min；`tool_call` 场景第 2 次模型调用大概率触发一次 429。脚本靠串行 + sleep 规避，但报告须一并披露限流窗口与重试记账方式。

---

## Q5. 其它可能造成不公平的点

### 结论

- recursion_limit：两边一致（9999）。
- 默认 checkpointer：两边均为 `None`，两边都不在 benchmark 中写 checkpoint。
- 默认 cache：两边均关闭（fork 对非 None cache 直接拒绝）。
- 残余不公平点：Q1（fork 多 2 middleware）与 Q2（fork 多消息序列化 / upstream 每轮重 bind）；另有 fork `SandboxMiddleware` 体量远大于 upstream。

### 证据

**(1) 默认无持久化。**

- `[upstream] factory.py:76`：`checkpointer: BaseCheckpointSaver | None = None`；`:174` 透传 `checkpointer=checkpointer`
- `[fork] factory.py:79`：同上默认；`:195` 透传
- `[fork] backend/packages/reactivegraph/reactivegraph/create_agent.py:398-402`：仅当 `self.checkpointer is not None and not self._checkpointer_is_inherit` 时才构造 `ThreadCheckpointer`；默认 `None` 不会写 checkpoint。

**(2) recursion_limit 一致。**

- `[fork] backend/packages/reactivegraph/reactivegraph/agent_state.py:16`
  ```python
  16    DEFAULT_RECURSION_LIMIT = 9_999
  ```
- `[upstream] langchain/agents/factory.py:1778-1780`
  ```python
  1778    # Set recursion limit to 9_999
  1779    # https://github.com/langchain-ai/langgraph/issues/7313
  1780    config: RunnableConfig = {"recursion_limit": 9_999}
  ```

**(3) cache 默认一致关闭。**

- `[upstream] langchain/agents/factory.py:755`：`cache: BaseCache[Any] | None = None`
- `[fork] create_agent.py:1386-1393`：非 None `cache` 直接 `NotImplementedError`；默认 `None` 两边都不缓存。

**(4) fork `SandboxMiddleware` 体量更大。**

- `[fork] backend/packages/harness/deerflow/sandbox/middleware.py` diff（vs `[upstream]` 同名文件）：fork 新增 lease / authz / network-policy 等数百行逻辑；在真实 tool 场景会产生可测量的额外本地开销。

**(5) 工具集对称。**

- 两边 `_assemble_from_features` 都无条件追加 `ask_clarification_tool`：`[upstream] factory.py:347`、`[fork] factory.py:440`；`model_only` 场景 effective_tools 非空，两边行为一致。

### 影响评估

- **对 `llm_calls`**：无影响。
- **对 `engine_s`**：残余差异方向均为「fork 做更多本地工作」（Q1/Q2-序列化/Q5-sandbox），整体对 fork 保守；但两边构件组合不完全对等，必须在报告显式声明。

---

## 公平性裁决表

| 维度 | 裁决 | 依据 |
|---|---|---|
| `llm_calls`（单次单轮、独立 thread） | ✅ 可公平比较 | 无 fake/real 分支；loop scope 差异在新鲜 thread + 单轮下不触发；工具集对称；`max_retries` 与库版本一致 |
| `engine_s`（fake model / 毫秒级） | ⚠️ 可比但链不对等 | fork 多 2 middleware + per-call 消息序列化，Sandbox 逻辑更大 → fork 实测更快属**保守结论**；须披露构成差异 |
| `engine_s`（真实 Agnes / 秒级） | ⚠️ 可比性弱 | LLM 秒级延迟主导，`engine_s` 是扣除后的残差；旧 61s 数据因限流记账错误不可用，须重跑 |
| 多轮 / 同 thread 重复 tool_call | ❌ 不可直接比 | loop scope 差异导致行为分叉（upstream 第 5 轮 hard-stop），属语义差异而非性能差异 |

---

## 必须披露的清单

1. fork 的 `create_deerflow_agent` 比 upstream 多 2 个 middleware（`DurableContextMiddleware` + `SystemMessageCoalescingMiddleware`）；两边 middleware 链并非逐项对等。
2. fork 每次模型调用多一次 message↔dict 序列化往返；upstream 每次模型调用多一次 `bind_tools`/`bind`。两者量级小，但属于引擎实现差异。
3. `engine_s`（尤其真实 LLM）是 `wall − llm − 限流等待` 的残差，对限流/负载敏感；`benchmarks/scenarios/real_llm_results.json`（07:32）的 `engine_s≈61s` 已被 07:54 修订脚本覆盖，**必须重跑后引用**。
4. `langchain-core` 版本不同（upstream 1.4.9 / fork 1.6.10）；`langchain-openai`(1.2.1) / `openai`(2.32.0) / `httpx`(0.28.1) / `langchain`(1.3.14) / `langgraph`(1.2.9) 一致。
5. 单轮独立 thread 场景两边等价；同 thread 重复 tool_call 场景两边行为不同（upstream 自动止损、fork 持续执行），不可混入性能结论。
6. `token_budget=True` 时 fork 构造 `TokenBudgetConfig(enabled=True)`、upstream 使用默认 `enabled=False`；默认路径（`token_budget=False`）不触发。
7. 本审计未运行任何真实 LLM 调用；真实 Agnes 对照结论尚待重跑后的脚本输出验证。

---

## 审计后状态更新（2026-10-16）

本审计当时列出的 1–3 号前置项**均已执行完毕**，结论与数据见
`benchmarks/scenarios/real_llm_results.json` 与 `REAL_SCENARIOS.md`「真实 LLM 场景」章节：

- **已重跑**（Agnes，限流等待已从 residual 中剔除）：
  `model_only` n=6、`tool_call` n=6、`multi_turn` n=3（同 thread 5 轮）、`big_state` n=3（64KB prompt + 4 轮）。
- **端到端墙钟**：四个场景互有胜负，**无加速优势**（模型/网络主导）。
- **引擎侧 residual_cpu**：fork 低 1.38–1.68×（multiple-scenario）。
- **3 号建议的落实**：`multi_turn` 使用**每轮不同文本**（`Turn i: reply with one short sentence.`），
  不含重复 tool_call，因此不会触发 upstream 的 thread 级 hard-stop；该场景两侧行为路径可比。
  本审计第 246 行记录的「同 thread 重复 tool_call 分叉」场景仍然是**不可混入性能**的语义差异，
  其单独复现数据仍以 `loop_detection_results.json` 为准。

---

## 建议

1. **以默认链跑 benchmark，并在报告里披露 Q1/Q2 差异**：显式传 `features=RuntimeFeatures(sandbox=False)` 之类的「对齐 middleware」虽能消除差异，但会偏离真实 SDK 默认路径；更稳妥的是保持默认链 + 披露。
2. **重跑真实 Agnes 测试**（修订后的 `benchmarks/scenarios/scripts/real_llm.py`），确认 `rate_limit_wait_s` 已从 `engine_s` 剔除；必要时把 `REAL_LLM_SLEEP` 提高到 70–75 秒以稳定跨过 60 秒限流窗口。
3. **单轮 vs 多轮分开成两份结论**：多轮对照组若使用同 thread 重复 tool_call，须要么显式关掉 loop detection，要么把 upstream hard-stop 的行为差异单独列出，不与性能数字混排。
4. **记录 `langchain-core` 版本差异**：在最终报告的环境章节写明 1.4.9 vs 1.6.10，并说明其不参与 429/重试路径。
5. **必要时做一次「对齐 middleware」的二次实验**：在 fork 上显式 `features=RuntimeFeatures(sandbox=False, loop_detection=False)` 等，验证「去掉额外构件后 fork 是否仍更快」，作为对 Q1 结论的稳健性交叉检查（可作为附注，不替代默认链结果）。

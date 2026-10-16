# 真实 Agnes LLM 对比数据：披露与解读边界

日期：2026-10-15

> 本文约束 `benchmarks/scenarios/real_llm.py` 产出的真实 LLM 对比数据。它不是结果报告；修订脚本重跑并落盘完整指标前，不得发布真实 LLM 加速结论。

## A. 指标口径：四个数各自是什么
**事实**
- `wall_s`：`graph.invoke()` 的墙钟时间；不含模块导入与 `create_deerflow_agent()` 构图时间。
- `llm_s`：所有**成功** `_generate()` 调用的墙钟时间之和；429 失败尝试与显式 sleep 不计入。
- `rate_limit_wait_s`：429 失败请求本身的耗时 + 为跨过 Agnes 限流窗口执行的 `time.sleep`。
- `residual_s = wall_s - llm_s - rate_limit_wait_s`：扣除 LLM 与限流等待后的残差，包含 middleware、消息编解码、序列化、状态提交、工具执行、GC、调度及其它本地工作；**不等于纯引擎耗时**。
- 另有 `residual_cpu_s = cpu_s - llm_cpu_s`，用 `time.process_time()` 排除部分调度/挂起噪声，但同样不能直接等同于“引擎净耗时”。
**证据（文件:行号）**
- `benchmarks/scenarios/scripts/real_llm.py:51-54`：LLM 时间、CPU 时间、事件与 429 等待分账数组。
- `benchmarks/scenarios/scripts/real_llm.py:66-100`：只包住 `_generate`；失败 429 的请求耗时与 sleep 进入等待数组，成功调用才进入 `LLM_TIMES`。
- `benchmarks/scenarios/scripts/real_llm.py:117-126`、`:134-152`：`model_only`/`tool_call` 两块场景均用 `perf_counter` 测 `invoke`，并计算 `residual_s` / `residual_cpu_s`。
- `benchmarks/scenarios/scripts/real_llm.py:183-233`：`multi_turn`（同 thread 5 轮）/ `big_state`（64KB system prompt + 4 轮）场景，均显式挂 checkpointer（`:118-123` 名为 `Saver` 的选择逻辑：fork 用 `reactivegraph.checkpoint.MemoryCheckpointSaver`、上游用 `langgraph.checkpoint.memory.InMemorySaver`）。不挂 checkpointer 时 factory 是 ephemeral 的，同一 thread 不会累积历史——这是 2026-10-17 修正的测量缺陷。除整段残差外还记录逐轮 median；`round_residual_cpu_median_s` 是每轮扣掉该轮内 LLM CPU 后的轮内残差。
- `benchmarks/scenarios/scripts/real_llm.py:124`、`:150`、`:227-230`：残差公式明确为 `wall - sum(llm) - waits`。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:371-372`：真实 Agnes 的 `engine_s` 被定性为扣除后的残差，可比性弱。
**对结论的影响**
不得把 `residual_s` 差异直接写成“ReactiveGraph 引擎优势”；也不得把构图/导入时间算进该指标。跨机、负载变化、GC、异步调度都可能污染残差。

## B. 两边执行栈不对等：fork 默认多 2 个 middleware
**事实**
- fork 的 `create_deerflow_agent` 默认链比 upstream 多挂 `DurableContextMiddleware` 和 `SystemMessageCoalescingMiddleware`。
- 这两个 middleware 在上游仅出现在 lead_agent 应用路径，没有进入 SDK factory 默认链。
- 在本 benchmark 的无 summary/delegation/skill/SystemMessage 注入场景下，二者近似透传，但 hook 节点与请求处理仍然执行。
**证据（文件:行号）**
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:22`：结论段落列出额外 2 个 middleware。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:47-56`：fork `factory.py:272-289` 追加挂载，upstream `factory.py:244-247` 没有对应项。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:57-59`：说明上游这两个 middleware 只在 lead_agent 路径出现。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:80-99`：本次场景近似 no-op，但审计仍判定 fork 每轮多执行 hook，方向对 fork 保守。
**对结论的影响**
若 fork 在同等或更重本地工作下仍更快，结论相对稳健；但必须披露两边链并非逐项对等，不能把差异归因于“纯 ReactiveGraph 核心”。

## C. 模型调用层差异：缓存 bind vs 每次重 bind，以及多一次 dict 往返
**事实**
- fork 每图只 `bind_tools()` 一次并缓存 `_bound`；上游每次模型执行都重新调用 `bind_tools()` / `bind()`。
- fork 每次模型调用多做一次 message ↔ foreign dict 的序列化/反序列化往返。
- 本次 benchmark 未设置 `tool_choice` / `model_settings`，因此这部分当前等价；若未来设置，fork 的 `_BoundModel.bind()` 会丢参，不得计入 fork 优势。
**证据（文件:行号）**
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:107-112`：四项不对称的总结。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:118-154`：fork `agent_recursion.py:183-213` 缓存 bind；上游 `langchain/agents/factory.py:1272-1404` 每次执行 `_get_bound_model`。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:156-168`：无 structured output / `response_format` 注入，避免另一条混淆路径。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:181-185`：两处偏差方向相反、量级都小，但属于真实实现差异。
**对结论的影响**
这些是本地 CPU 差异：fork 省下重复 bind，却多付序列化。真实 LLM 秒级延迟下通常很小，fake-model 毫秒级场景会显现；只能作为“实现差异披露”，不能包装成无成本的纯加速。

## D. loop_detection scope 差异：不能混入性能结论
**事实**
- fork 的 loop 历史键为 `(thread_id, run_id)`；上游为 `thread_id`。两边阈值相同：warn=3、hard=5。
- 单轮独立 thread 场景中，tool_call 哈希计数最多为 1，两边等价，不会触发 loop 保护。
- 同一 thread 连续重复相同 tool_call 到第 5 轮时行为分叉：上游会 hard-stop，fork 每个 run 都有完整预算而继续执行。
**证据（文件:行号）**
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:189-193`：单次单轮不影响 `llm_calls` 的结论。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:199-227`：fork `_run_scope_key` 与上游 `thread_id` scope 的源码差异。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:229-246`：阈值相同；多轮分叉数据为 upstream `[10,2,2,2,2,2]`、fork `[10,10,10,10,10,10]`。
- `REAL_SCENARIOS.md:42-65`：同一现象的行为学复现与 trade-off 说明。
**对结论的影响**
单轮独立 thread 可以比较性能；同 thread 重复 tool_call 的差异是语义/行为差异，不是速度差异。该场景不得与单轮数字合并成一个“加速比”。

## E. Agnes 约束：限流、重试与样本量上限
**事实**
- Agnes 限制为账号级 1 请求/分钟；脚本串行执行，两侧之间默认 sleep 65 秒。
- `TimedModel` 显式设置 `max_retries=0`，429 不被内部重试掩盖；429 后脚本最多尝试 4 次，sleep 为 `61 + attempt * 5` 秒。
- `REAL_LLM_REPEATS` 默认 1；样本量受配额限制。修订后测量必须剔除 `rate_limit_wait_s`。
- 旧 `real_llm_results.json` 中约 61 秒的“engine_s”是错误记账，不得引用。
- 现实样本量上限：`model_only` 每 paired sample 2 次成功调用，理论上限约 30 pairs/hour，65 秒间隔下现实约 27 pairs/hour；`tool_call` 每 paired sample 4 次成功调用，理论上限约 15 pairs/hour；`multi_turn`（5 轮）每 paired sample 10 次、`big_state`（4 轮，64KB prompt）每 paired sample 8 次，二者更烧配额，429 等待与启动/load 检查会进一步降低实际值。
**证据（文件:行号）**
- `benchmarks/scenarios/scripts/real_llm.py:13-14`、`:21`：账号级 1 req/min、串行/限流设计。
- `benchmarks/scenarios/scripts/real_llm.py:35-39`：`REAL_LLM_REPEATS` 与 `REAL_LLM_SLEEP` 默认值。
- `benchmarks/scenarios/scripts/real_llm.py:66-92`：429 判定、显式重试、失败请求与 sleep 的等待记账。
- `benchmarks/scenarios/scripts/real_llm.py:107-111`：`TimedModel(..., max_retries=0, timeout=120)`。
- `benchmarks/scenarios/scripts/real_llm.py:217-224`：每侧串行运行，侧间 sleep 65 秒。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:254`、`:288-305`：修订前 61 秒残差问题、修订后必须重跑。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:309-312`：重试语义公平，但限流窗口与记账方式必须披露。
**对结论的影响**
限流等待必须从残差中剔除；样本量不足时只能报告“探索性单机结果”，不得宣称稳定显著加速。任何未重跑的旧 JSON 都不能作为证据。

## F. 未覆盖路径：异步与 streaming 不在当前探针内
**事实**
- 计时探针只覆写了同步 `_generate()`，场景调用也是同步 `graph.invoke()`。
- 脚本没有覆写 `_agenerate()`、`_stream()` 或 `_astream()`；异步 `ainvoke()`、streaming `astream()` / `astream_events()` 的引擎路径未被当前真实 Agnes 数据覆盖。
- 若未来模型调用走异步或流式路径，当前 `llm_s` 可能缺少对应计时，残差也可能被系统性高估。
**证据（文件:行号）**
- `benchmarks/scenarios/scripts/real_llm.py:56-100`：`TimedModel` 唯一覆写的是 `_generate`。
- `benchmarks/scenarios/scripts/real_llm.py:116-233`：四个场景都调用同步 `graph.invoke()`。
- `benchmarks/scenarios/scripts/real_llm.py`（全文检索 `_agenerate` / `_stream` / `_astream` / `ainvoke` / `astream` 无匹配）：没有异步/流式覆写或调用。
**对结论的影响**
本轮真实 LLM 结论只适用于同步 `invoke` 路径；不得外推为“所有 ReactiveGraph 执行路径都更快”。若产品默认使用 async/streaming，必须增加对应探针后另行复测。

## G. 两类证据不可混用：fake-model 与真实 LLM
**事实**
- fake-model 场景为零推理延迟，衡量的是引擎/编排 CPU 开销，单位是毫秒；其比值不能代表真实 LLM 端到端加速比。
- 真实 Agnes 场景以网络 + 推理的秒级时间为主；`llm_s` 通常远大于 `residual_s`，残差差异容易被负载噪声淹没。
- `REAL_SCENARIOS.md` 的约 2× 延迟 / 3–4× 并发数据来自 fake model 与受控 harness，不能与真实 LLM 的 `residual_s` 直接做除法。
**证据（文件:行号）**
- `REAL_SCENARIOS.md:7-21`：fake-model 的同机、同 factory、输出等价前提与毫秒级结果。
- `REAL_SCENARIOS.md:34-40`：测量机负载、2×/3–4× 优势的范围说明。
- `REAL_SCENARIOS.md`：真实 LLM（Agnes）四个场景（model_only / tool_call / multi_turn / big_state）的表与诚实结论；端到端墙钟互有胜负，residual_cpu 优势 1.45–2.34×（其中 multi_turn/big_state 为 2026-10-17 checkpointer 修正后的数据，采样期 load1 40–230，低置信度）。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:183-185`：本地 CPU 差异在真实 LLM 延迟下可忽略，但 fake-model 毫秒级会显现。
- `benchmarks/scenarios/FAIRNESS_AUDIT.md:369-373`：审计裁决区分 fake-model `engine_s`、真实 Agnes `engine_s` 与多轮语义场景。
**对结论的影响**
必须把“引擎 CPU 开销”和“真实 LLM 端到端残差”分成两类证据；可以并列展示，但不得合并成一个加速倍数。

## 报告写作规则
1. 不得说 `residual_s` 差异 = 引擎优势；残差包含 middleware、编解码、序列化、状态提交、GC、调度等，且不含构图/导入时间。
2. 不得把 fake-model 的毫秒级比值与真实 Agnes 的 `wall_s` / `llm_s` / `residual_s` 混算成同一个“X 倍加速”。
3. 不得隐去 fork 多 2 个 middleware、缓存 bind vs 每次重 bind、message↔dict 额外往返等执行栈差异。
4. 不得用单轮独立 thread 数据推断同 thread 重复 tool_call 的性能；loop_detection scope 分叉是语义差异，必须单列。
5. 任何真实 LLM 结论必须同时给出样本量、`wall_s`/`llm_s`/`rate_limit_wait_s`/`residual_s`、失败/429 次数与机器负载；未按修订脚本重跑的数据不得发布。

# 真实场景效率验证（DeerFlow fork vs 上游 langgraph）

日期：2026-10-15 · 状态：本机实测，非 release gate 断言；differential discipline（median of ≥3）。

## 结论摘要

在**同一台机器、同一 DeerFlow agent factory、同一 fake model、输出逐条等价**的前提下，
`from000/deer-flow-reactive`（ReactiveGraph 执行底座）对上游 `bytedance/deer-flow`（langgraph 1.2.9）：

| 场景 | ReactiveGraph | 上游 langgraph | 比值 |
|---|---:|---:|---:|
| 单轮 model-only agent（50 次/轮，median of 3 runs） | **3.01 ms** | **7.09 ms** | **2.35×** |
| 单轮 tool-call round trip（50 次/轮，median of 3 runs） | **6.15 ms** | **11.81 ms** | **1.92×** |
| 多轮对话累积（同 thread 10 轮，挂 checkpointer，median of 3 rounds） | **8.69 ms** | **19.22 ms** | **2.21×** |
| 大 state（64KB prompt + 30 轮累积，挂 checkpointer，median of 3 rounds） | **14.19 ms** | **21.92 ms** | **1.54×** |
| 64 会话并发 / 8 workers（吞吐，median of 3 paired rounds） | **117.3 sessions/s** | **29.2 sessions/s** | **4.02×** |

上表由 `scripts/summarize.py` 从落盘 JSON 直接算出（可一键复算，杜绝手抄）。

**诚实范围**：优势稳定在 **约 1.5–2.4×（单场景延迟）到 4×（并发吞吐）**，不是数量级差距。
输出等价性已逐条断言（`model_only` 与 `tool_call` 两条路径的最终 message 列表完全一致）。

### 勘误（2026-10-17）：多轮/大 state 曾测成"重复单轮"

早期版本的 `run_more.py` / `real_llm.py` 有两个测量缺陷，已在落盘 JSON 与文档中修正：

1. **没挂 checkpointer**：`create_deerflow_agent` 不传 `checkpointer` 时是 ephemeral 的，
   同一 thread 的后续 invoke 不读累积历史。HTTP 层实测：5 轮 multi_turn 每次请求只带
   **1 条消息**（正确应递增为 1→3→5→7→9）。因此早期"多轮"数字（fake 2.32×、
   real-LLM 1.68×）测的是**重复单轮**，不是多轮累积；`big_state` 同理（1.38×）。
2. **big_state 复用同一 `HumanMessage` 对象**：上游 `add_messages` 按 id 去重。
   修复第 1 点（挂上 checkpointer）后、修复本条之前，实测 message_count 为
   **上游 34 / fork 66**（30 轮只留 1 条 human），两侧工作量不对等；
   无 checkpointer 的原始版本两侧都是 2（根本没累积），掩盖了这个缺陷。

修正后两侧消息数完全对齐（multi_turn 20/20、big_state 66/66），数字见上文表格
（fake 2.21× / 1.54×；real-LLM 2.33× / 2.34×）。旧数据已从 JSON 删除，不可再引用。

## 方法

- **执行底座**：fork 的 `create_deerflow_agent` → `reactivegraph.create_agent`（纯 Python 进程内引擎，无 LangGraph 执行路径）；上游同 factory → langgraph 1.2.9。
- **对照公平性**：
  - 两边同一个 fake `FakeMessagesListChatModel`（确定性，无网络噪声）；
  - tool_call 场景两边都用真实 `@tool`；
  - **每个样本独立 `thread_id`**，避免跨轮状态污染；
  - **输出等价性先断言再计时**（`parity_check.py`）。
- **采样**：每场景 50 采样取 median；每个场景重复 3 轮/3 次再取 median。
- **多轮必须挂 checkpointer**：`create_deerflow_agent` 不传 `checkpointer` 时是 **ephemeral** 的——
  同一 `thread_id` 多次 invoke **不会累积历史**，每次都是从空 state 开始的单轮。
  因此"多轮累积 / 大 state"场景显式挂 checkpointer（两侧各用自家实现：
  fork `reactivegraph.checkpoint.MemoryCheckpointSaver`、上游 `langgraph.checkpoint.memory.InMemorySaver`）。
  否则测出来的不是多轮，而是"重复单轮"（早期版本即踩此坑，见上方勘误）。
- **每轮新消息对象**：`big_state` 每轮构造新的 `HumanMessage`；复用同一对象时 LangGraph 的
  `add_messages` 会按 id 去重（30 轮只剩 1 条 human），造成两侧工作量不对等（早期版本踩此坑，见上方勘误）。
- **配对交替**：fork 与上游**同轮相邻执行**（A/B/B/A 式轮流），避免"先整段跑完一侧"的
  系统负载偏置；每轮记录 load average 一并落盘（`*_results.json` 的 `load1_at_run`）。
- **环境**：Python 3.12 / Darwin x86_64。测量机为共享开发机：
  - fake-model 主表（model-only/tool-call/并发）测量期 load1 3.7–4.5；
  - fake-model 的 multi_turn/big_state 重测（2026-10-17）load1 9.8–13，中等偏高；
  - 真实 LLM 的 multi_turn/big_state 重测同日本机有大量并发重负载，load1 高达 40–230。
    因此这两个真实 LLM 场景**只有 `residual_cpu`（进程 CPU 时间）可引用**，
    `residual_s`（墙钟减法）与端到端墙钟被负载严重污染，仅归档不作结论。

## 决定性发现 1：语义对齐后差距是 ~2×，不是宣传的"千倍"

早期文档里的"1480×"测的是**裸 Driver `Scheduler`**，不含 Python 主机与 Node 子进程边界；
换成真实 fork agent factory 端到端后，优势是 **约 2×**。这与 `docs/competitive-proof.md §1b` 的口径更正一致。

## 决定性发现 2：同 thread 重复模式下，上游会跨 invoke 累积 loop 预算

两边都带 DeerFlow 的 `loop_detection_middleware`（warn=3 / hard=5），单次 run 内的
失控循环两边都能正确 hard-stop（实测各 5 次 model call 后兜底）。

差异出现在**同一 thread 连续多次 invoke**：`scripts/loop_detection_repro.py` 让模型每次
invoke 只发同一个 tool_call（name+args 相同、消息 id 唯一），同一 thread 连续 invoke 6 次：

```text
reactivegraph:      message_counts=[10, 10, 10, 10, 10, 10]   total_model_calls=30
langgraph_upstream: message_counts=[10,  2,  2,  2,  2,  2]   total_model_calls=10
```

- **fork**：loop 预算按 `(thread_id, run_id)` 作用域，每次 invoke 是新 run ⇒ 每次都有完整预算，
  行为可预期；
- **上游**：history 只按 `thread_id` 累积，预算跨 invoke 共享 ⇒ 第 1 次 invoke 耗尽预算后，
  **第 2 次起立即 hard-stop**，同一 thread 的后续 invoke 行为不一致、会被静默截断。

**归因诚实**：这个差异来自 fork **移植 middleware 时改的 scope key**
（`migration commit 27f2f93a`），**不是 ReactiveGraph 引擎本身的性能属性**。
`_hash_tool_calls` 只取 name+args，两边算法一致；差别只在 history 的 key。

**不是纯赢**：上游的累积语义省 model 调用（10 vs 30），对"防死循环 + 省钱"是优点；
fork 的每-run 语义换来的是行为一致性。属于**语义 trade-off，需按业务场景判断**。

## 覆盖矩阵：显式启停各 RuntimeFeatures（fake model，带有效 config.yaml）

回答"DeerFlow 的能力是不是都试了"：SDK factory 层可独立启停的 feature 开关全部可构图、
可 invoke（8 组合 × 20 次），数字来自 `scripts/feature_matrix.py`（**必须提供有效
`config.yaml`**，否则 memory/auto_title/vision 会走 `FileNotFoundError` 失败路径，
曾测出 6.7–7.8× 的假优势——那不代表功能更快）。

| feature 组合 | fork | 上游 | 比值 |
|---|---:|---:|---:|
| default | 6.11 ms | 13.96 ms | 2.28× |
| no_sandbox | 2.92 ms | 8.01 ms | 2.75× |
| with_subagent | 5.01 ms | 17.40 ms | 3.47× |
| with_memory | 7.72 ms | 13.66 ms | 1.77× |
| with_vision | 5.37 ms | 15.44 ms | 2.88× |
| with_auto_title | 5.46 ms | 12.77 ms | 2.34× |
| with_token_budget | 5.65 ms | 16.22 ms | 2.87× |
| all_off | 2.24 ms | 4.41 ms | 1.97× |

**结论**：启用各 feature 后 fork 仍快 1.8–3.5×，与默认链口径一致，不是"只测了空壳"。

**仍不在覆盖范围**（需要外部资源或真实 LLM，非 fake-model 可测）：
- 真实 sandbox 容器 / MCP server / 文件上传 / Postgres·Redis 后端
- summarization / guardrail 的真实抽取路径（需要真实 LLM）
- 前端 / 网关 / TUI / CLI / 子代理编排（subagent 开关已覆盖 SDK 层，真实子代理链路未测）

## 真实 LLM 场景（Agnes，agnes-2.5-flash）

前置说明：这部分与上面的 fake-model 结果**不可混用**。fake model 衡量引擎 CPU 开销（毫秒级）；
真实 LLM 的墙钟由**网络 + 推理**主导（本次实测 1.6–7.5 s/次），引擎开销只是其中很小一部分。

- **脚本**：`scripts/real_llm.py`（每侧新进程、交替执行、记录 `load1_at_run`；
  `model_only`/`tool_call` 为单次 invoke，`multi_turn` 为同 thread 5 轮（挂 checkpointer），
  `big_state` 为 64KB system prompt + 4 轮（挂 checkpointer）；均可通过 `REAL_LLM_SCENARIOS` 选择）
- **约束**：Agnes 账号级限流 1 请求/分钟；脚本显式等待并**单独记账**，不混入 residual；
  `max_retries=0`（429 直接暴露）；`REAL_LLM_MAX_LOAD` 负载门槛 + 等待。
- **指标**：`residual_cpu_s = cpu_s − Σllm_cpu_s`，即扣除 LLM 调用后本进程真正消耗的 CPU 时间。
  它比 `wall − llm` 的墙钟残差更抗调度噪声，但仍**不等于纯引擎耗时**。
- **负载披露**：multi_turn/big_state 重测（2026-10-17）与其它重负载任务重叠，
  `load1_at_run` 达 40–230（早期 model_only/tool_call 样本为 4–26）。
  这也导致重测期间出现 1 次 fork 侧 120s 客户端超时（模型单次调用达 31s+）；
  超时样本已删除。因 `residual_cpu` 是进程 CPU 时间，受调度延迟影响远小于墙钟，
  但仍请把 multi_turn/big_state 视为**同机低置信度样本（n=3）**，不要跨机引用。

| 场景 | fork residual_cpu (median) | 上游 residual_cpu (median) | 比值 |
|---|---:|---:|---:|
| model_only（n=6） | **16.9 ms**（12.0–24.7） | **24.6 ms**（17.3–48.6） | **1.45×** |
| tool_call（n=6，2 次 LLM 调用） | **23.6 ms**（19.4–31.4） | **34.9 ms**（24.6–49.4） | **1.48×** |
| multi_turn（n=3，5 轮同 thread，挂 checkpointer） | **73.8 ms**（72.8–77.7） | **172.1 ms**（150.9–178.3） | **2.33×** |
| big_state（n=3，64KB prompt + 4 轮，挂 checkpointer） | **41.7 ms**（38.9–43.4） | **97.6 ms**（82.3–100.3） | **2.34×** |

**关键诚实结论：**

1. **真实 LLM 端到端不会更快。** 墙钟由模型/网络主导，而 Agnes 的网络/推理耗时波动极大
   （单次调用 0.9–29 s），因此端到端墙钟**样本内互有胜负**：
   model_only fork 中位 3.92 s vs 上游 1.83 s（上游更快）；
   tool_call 4.09 s vs 3.81 s（接近）；multi_turn 20.78 s vs 16.82 s（上游更快）；
   big_state 17.75 s vs 19.99 s（fork 略快）。
   任何"用 ReactiveGraph 让真实 LLM 应用快 1.5 倍"的说法都是错的。
2. **引擎侧 CPU 开销仍然更低**：residual_cpu 中位数 1.45×（model_only）/ 1.48×（tool_call）/
   2.33×（multi_turn，5 轮同 thread）/ 2.34×（big_state，64KB prompt + 4 轮）。
   单轮场景（model_only/tool_call）的比值比 fake model 的 ~2–4× 明显收窄——真实调用链路
   引入的固定本地成本（消息编解码、schema、工具执行、middleware hook）稀释了引擎差异；
   多轮场景比例恢复到 ~2.3×，因为每次 invoke 都要重放/累积全部历史，引擎的状态处理差距
   随轮数放大。绝对差：单轮 **8–11 ms**（不可感知），多轮合计 **56–98 ms**（仍然很小）。
   - 注意 multi_turn/big_state 的 residual_cpu 是**整段多次 invoke 的合计**，
     与 model_only/tool_call 的单次 invoke 不可直接比大小；比值（同场景两侧）才可比。
   - 逐轮口径：multi_turn 轮内 residual_cpu 中位 11.1 vs 29.9 ms（2.69×）；big_state 7.3 vs 21.4 ms（2.93×）。
3. **`tool_messages` 数量在真实 LLM 下不稳定**（fork 见过 1 和 2，上游也见过 1 和 2）。
   确定性 fake model 下 `parity_check.py` 断言两边**逐条一致**
   （`tool_call`: 4 条消息 / 1 个 ToolMessage，`identical=True`），
   因此真实 LLM 的数量波动是**模型随机行为**（模型偶尔返回多个 tool_call），不是引擎差异。
   故真实 LLM 数据不得用于比较"工具轨迹"，只能比较调用次数与 residual。

### 控制实验：排除"冷启动造成的假象"

`real_llm.py` 每侧都在新进程里跑短会话（`model_only`/`tool_call` 单次 invoke，
`multi_turn`/`big_state` 则是新进程内多次 invoke）；若 fork 的优势只是热路径缓存，
cold 形态（新进程+单次）下应当消失。
`scripts/fresh_process_control.py`（fake model，免费可复现）复刻同一形态：

| 形态 | fork | 上游 | 比值 |
|---|---:|---:|---:|
| 冷启动（新进程 + 单次 invoke，wall） | 12.48 ms | 17.02 ms | 1.36× |
| 冷启动（residual_cpu） | 9.55 ms | 14.31 ms | 1.50× |
| 热路径（同进程 20 次，median of 后 15 次，wall） | 2.89 ms | 13.62 ms | 4.72× |

结论：**冷启动不是原因**——同样新进程形态下 fork 仍更快；真实 LLM 中 residual 差异收窄，
是因为真实调用链路引入了大量与引擎无关的固定本地开销。

## 诚实边界（必须读）

1. **不是 token 省钱**：fork 的 `create_deerflow_agent` 把 model/tools 节点注册为默认
   `kind="effect"`（有副作用语义），**不做跨 run 指纹跳过**。因此 agent 路径
   **不会**因重复输入而少调用 LLM。实测（`scripts/model_call_count.py`）：30 轮相同输入，
   fork 与上游均为 **30 次调用 / 0 次跳过**。
   - **HTTP 级证据（确定性，可复现）**：`scripts/model_cost_parity.py` 给两侧各起一个本地
     OpenAI 兼容端点，逐请求 capture 完整 HTTP body 并逐字节比对。`model_only` / `tool_call` /
     `multi_turn`(4 轮) 的每个请求 body **字节完全一致**（multi_turn 消息数按 1→3→5→7 累积，
     两侧逐条相同；`tool_call` 4 个请求含 tools schema 全等），脚本以退出码 0/1 断言。
     即：相同输入下，fork 与上游发给模型的 prompt / tools 完全相同 ⇒ **token 数相同、调用次数相同，
     因此模型成本相同**。"换 ReactiveGraph 更省钱"在 agent 路径上**不成立**。
   - 想省钱必须显式声明 `kind="pure"` + `reads`（如 core 的 `deerflow_real` 基准里那样，3 个 pure 任务同输入跳过 3/3）。
2. **上游 loop-detection 退化是行为差异，不是纯性能**：对某些工作负载这可能是好事（自动止损），需按场景判断。
3. **机器敏感 / 测量前提**：所有数值为单机 median，非跨机基准。本机是共享开发机，
   测量期间常有其它编译/测试进程（load1 3.7–4.5）。脚本用配对交替降低负载偏置并记录
   load1，但仍不能替代独占机器；跨机复现应以本机同环境同负载重跑为准。
   （反例：无配对、机器 load1≈26 时，上游 multi_turn 曾从 ~10ms 漂到 64ms。）
4. **真实 LLM 已跑，但结论有限**：Agnes（agnes-2.5-flash）实测可用（DeepSeek 余额不足 402、
   DashScope 欠费、SiliconFlow 未实名）。真实 LLM 下**端到端墙钟无优势**（模型/网络主导），
   四个场景（model_only / tool_call / multi_turn / big_state）的引擎侧 residual_cpu
   优势为 1.45–2.34×，样本量 n=6/6/3/3；其中 multi_turn/big_state 为高负载窗口内的
   低置信度样本（load1 40–230），引用时须带此限定。详见上文"真实 LLM 场景"章节。
5. **fake model 的局限**：fake model 零推理延迟，因此这里衡量的是**引擎/编排开销**，
   这正是框架层该比的量；但它不代表真实 LLM 场景的端到端加速比。

## 复现

```bash
# 需同时存在两个 checkout（各自装好 backend/.venv）：
#   /path/to/deer-flow-reactive   (fork)
#   /path/to/deer-flow            (上游 bytedance/deer-flow)

export FORK_ROOT=/path/to/deer-flow-reactive
export UPSTREAM_ROOT=/path/to/deer-flow

cd benchmarks/scenarios/scripts      # 也可从仓库根目录直接执行；输出 JSON 固定写回 benchmarks/scenarios/

# 1. 输出等价性（计时前提，必须先跑）
python3 parity_check.py              # 期望两条 identical=True

# 2. 端到端三次重复（50 样本/轮，各 3 轮取 median）
python3 run_triple.py

# 3. 多轮对话累积 / 64KB 大 state
python3 run_more.py

# 4. 并发吞吐（64 会话 / 8 workers）
python3 run_concurrency.py

# 5. loop-detection 行为差异复现（决定性发现 2）
python3 loop_detection_repro.py

# 6. 验证 agent 路径不做跨 run 跳过（反"省钱"夸大）
python3 model_call_count.py

# 7. 控制实验：冷启动 vs 热路径（fake model，免费）
python3 fresh_process_control.py

# 8. feature 覆盖矩阵（fake model；需有效 config.yaml）
python3 feature_matrix.py

# 8b. 模型成本对等性：两侧请求体逐字节比对（本地端点，免费）
python3 model_cost_parity.py

# 9. 真实 LLM（Agnes；受 1 req/min 限流，耗时长）
export CUSTOM_LLM_URL=... CUSTOM_LLM_API_KEY=... CUSTOM_MODEL=agnes-2.5-flash
# 单次 invoke 基线
REAL_LLM_REPEATS=2 REAL_LLM_SCENARIOS=model_only,tool_call python3 real_llm.py
# 多轮同 thread（默认 5 轮）/ 64KB 大 state（默认 4 轮）
REAL_LLM_SCENARIOS=multi_turn REAL_LLM_MULTI_TURN_ROUNDS=5 python3 real_llm.py
REAL_LLM_SCENARIOS=big_state REAL_LLM_BIG_STATE_ROUNDS=4 REAL_LLM_BIG_STATE_KB=64 python3 real_llm.py
```

脚本依赖两个环境变量指向本地 checkout；结果 JSON 通过脚本自身路径定位写回
`benchmarks/scenarios/`，与调用时的 cwd 无关。

## 数据来源（原始 JSON）

- `triple_results.json` — 端到端 3 次重复
- `more_results.json` — multi_turn / big_state
- `concurrency_results.json` — 64 会话并发
- `final_results2.json` — 早间非配对口径的历史存档（含 message_count），不用于本报告数字
- `loop_detection_results.json` — 同一 thread 重复 tool_call 的 message 数量（决定性发现 2）
- `model_call_count_results.json` — 30 轮重复输入下的真实模型调用次数（诚实边界 1）
- `real_llm_results.json` — 真实 Agnes LLM 逐次测量（含 llm_events / message_sequence / load1）
- `fresh_process_control_results.json` — 冷启动 vs 热路径控制实验
- `feature_matrix_results.json` — 8 组 feature 开关 × 20 次的引擎开销
- `FAIRNESS_AUDIT.md` / `REAL_LLM_DISCLOSURES.md` — 两边执行栈差异与解读边界审计

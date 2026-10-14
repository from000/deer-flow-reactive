# 真实场景效率验证（DeerFlow fork vs 上游 langgraph）

日期：2026-10-15 · 状态：本机实测，非 release gate 断言；differential discipline（median of ≥3）。

## 结论摘要

在**同一台机器、同一 DeerFlow agent factory、同一 fake model、输出逐条等价**的前提下，
`from000/deer-flow-reactive`（ReactiveGraph 执行底座）对上游 `bytedance/deer-flow`（langgraph 1.2.9）：

| 场景 | ReactiveGraph | 上游 langgraph | 比值 |
|---|---:|---:|---:|
| 单轮 model-only agent（50 次/轮，median of 3 runs） | **3.01 ms** | **7.09 ms** | **2.35×** |
| 单轮 tool-call round trip（50 次/轮，median of 3 runs） | **6.15 ms** | **11.81 ms** | **1.92×** |
| 多轮对话累积（同 thread 10 轮，median of 3 rounds） | **2.48 ms** | **5.75 ms** | **2.32×** |
| 大 state（64KB system prompt，median of 3 rounds） | **1.95 ms** | **5.61 ms** | **2.87×** |
| 64 会话并发 / 8 workers（吞吐，median of 3 paired rounds） | **117.3 sessions/s** | **29.2 sessions/s** | **4.02×** |

上表由 `scripts/summarize.py` 从落盘 JSON 直接算出（可一键复算，杜绝手抄）。

**诚实范围**：优势稳定在 **约 2×（单次延迟）到 3–4×（并发吞吐）**，不是数量级差距。
输出等价性已逐条断言（`model_only` 与 `tool_call` 两条路径的最终 message 列表完全一致）。

## 方法

- **执行底座**：fork 的 `create_deerflow_agent` → `reactivegraph.create_agent`（纯 Python 进程内引擎，无 LangGraph 执行路径）；上游同 factory → langgraph 1.2.9。
- **对照公平性**：
  - 两边同一个 fake `FakeMessagesListChatModel`（确定性，无网络噪声）；
  - tool_call 场景两边都用真实 `@tool`；
  - **每个样本独立 `thread_id`**，避免跨轮状态污染；
  - **输出等价性先断言再计时**（`parity_check.py`）。
- **采样**：每场景 50 采样取 median；每个场景重复 3 轮/3 次再取 median。
- **配对交替**：fork 与上游**同轮相邻执行**（A/B/B/A 式轮流），避免"先整段跑完一侧"的
  系统负载偏置；每轮记录 load average 一并落盘（`*_results.json` 的 `load1_at_run`）。
- **环境**：Python 3.12 / Darwin x86_64。测量机为共享开发机，本轮 6 次测量 load1 稳定在
  3.7–4.5，两侧背景一致。

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

## 诚实边界（必须读）

1. **不是 token 省钱**：fork 的 `create_deerflow_agent` 把 model/tools 节点注册为默认
   `kind="effect"`（有副作用语义），**不做跨 run 指纹跳过**。因此 agent 路径
   **不会**因重复输入而少调用 LLM。实测（`scripts/model_call_count.py`）：30 轮相同输入，
   fork 与上游均为 **30 次调用 / 0 次跳过**。
   - 想省钱必须显式声明 `kind="pure"` + `reads`（如 core 的 `deerflow_real` 基准里那样，3 个 pure 任务同输入跳过 3/3）。
2. **上游 loop-detection 退化是行为差异，不是纯性能**：对某些工作负载这可能是好事（自动止损），需按场景判断。
3. **机器敏感 / 测量前提**：所有数值为单机 median，非跨机基准。本机是共享开发机，
   测量期间常有其它编译/测试进程（load1 3.7–4.5）。脚本用配对交替降低负载偏置并记录
   load1，但仍不能替代独占机器；跨机复现应以本机同环境同负载重跑为准。
   （反例：无配对、机器 load1≈26 时，上游 multi_turn 曾从 ~10ms 漂到 64ms。）
4. **真实 LLM 未纳入**：DeepSeek 账户余额不足（HTTP 402），真实模型端到端未能跑；
   真实网络/推理延迟会淹没引擎差异（通常 LLM 单次 >500ms，引擎差异 ~5ms）。
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

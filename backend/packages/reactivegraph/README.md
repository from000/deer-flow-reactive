# ReactiveGraph

**反应式图执行引擎**——以响应式更新为内核的智能体/工作流运行时。上游对标
`langchain-ai/langgraph`,但执行模型不同:我们不做"每次全量重跑",
而是编译期建依赖图、运行期只执行受影响的任务。

[![CI](https://github.com/from000/reactive-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/from000/reactive-graph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 为什么是反应式

LangGraph 的 Pregel 在每次 `invoke` 时全量执行所有节点；ReactiveGraph 依赖
分析 + 缓存失效，只执行受影响的节点。同一确定性图、同一机器：

| 场景 | upstream langgraph | ReactiveGraph |
|---|---|---|
| 对话图 invoke（warm 中位） | ~1.5 ms | ~0.004 ms |
| 1000 节点 fan-out 改 1 字段 | ~11602 ms | ~15.3 ms |
| 1000 节点 / 10 受影响，改 k0 | ~4669 ms | ~6.9 ms |
| **no-gain：1000 节点全依赖链** | **~6428 ms** | **~8.7 ms** |

> ⚠️ 上表数字来自早期一次未入库的临时本机测量，**仅供量级参考，
> 不可复现**。权威数字以 `benchmarks/differential/`（真实 langgraph 1.2.11 vs
> 原生 Driver，同图同输入、输出一致断言 + median）为准，见
> `benchmarks/differential/README.md`。

## 核心能力

* **选择性执行** — 事件路由决定 effect 任务资格；computed 按 read-set 失效；
  pure 任务输入指纹不变即跳过；effect 任务 receipt 幂等，确认即不重放。
* **事务性状态** — `transaction` 携带读写集 + patches 原子提交，冲突有明确
  策略（reducer / priority / 串行），不是"全量重算后合并"。
* **computed 缓存** — selector 声明 read-set，只读集变化才重算。
* **scope 子状态** — 命名子状态命名空间，天然模块化。
* **持久化执行** — durable event log（SQLite/内存）+ checkpoint + 恢复，
  重启后从日志精确回放。
* **因果 trace** — 调度每个决策（start/done/retry/skip/invalidate）有序
  可导出，调试与审计不用猜。
* **流式与中断** — `stream` 值流/自定义流；`interrupt`/`resume` 支持
  human-in-the-loop。

## 快速开始

```bash
# Python 原生 API
pip install reactivegraph

# CLI（单独发行）
pip install reactivegraph-cli
```

```python
from reactivegraph import ReactiveGraph

def build(b):
    b.task("greet", kind="effect", fn=lambda i: {"msg": f"hi {i['name']}"},
           on=("visit",))          # 或 b.on("visit", "greet")

g = ReactiveGraph.build(build)
out = g.invoke("visit", {"name": "Ada"})   # 实例方法，返回 state 字典
print(out["msg"])  # "hi Ada"
```

TypeScript 原生 API（`packages/sdk-js`）使用对象式签名：

```ts
import { ReactiveGraph, invoke } from "@reactivegraph/sdk-js";

const graph = ReactiveGraph.build((b) => {
  b.task({
    id: "greet",
    kind: "effect",
    handler: (input) => ({
      reads: ["name"], writes: ["msg"],
      patches: [{ path: ["msg"], operation: "set", value: `hi ${(input as { name: string }).name}` }],
    }),
  });
  b.on("visit", "greet");
});
const out = await invoke(graph, "visit", { name: "Ada" });
// out.state.msg === "hi Ada"
```

## 命令

| 命令 | 用途 |
|---|---|
| `make test` | Python 原生测试 |
| `pnpm typecheck` / `pnpm test` | TS 全部包 |
| `uv run --directory python/reactivegraph pytest -q` | Python 原生 |
| `pnpm benchmark` | 选择性更新基准 |

## 仓库布局

```
packages/                 TS：protocol / driver / sdk-js / devtools-protocol / testkit
python/                   Python：reactivegraph / reactivegraph_sdk / reactivegraph_cli
benchmarks/               benchmark harness 与 workloads
docs/                     规范、基准、性能对比
```

## 基准

性能数据与复现方法见 `docs/benchmarks.md`（选择性更新 + 与真实 upstream
langgraph 的差分测量，含 no-gain 场景的诚实记录）。

## 安全

见 `SECURITY.md`（gateway 认证 / 字段级权限 / 状态脱敏 / 只读 importer）。

## 开源贡献

见 `CONTRIBUTING.md` 与 `CODE_OF_CONDUCT.md`。我们欢迎 issue 与 PR。

## License

MIT —— 见 `LICENSE`。

## 第三方声明

兼容层（`messages` / `tools` / `ToolNode` / `channels` / `middleware` /
`checkpoint` / `store` / `message_utils` 等）在接口与行为上适配自
`langchain-ai/langgraph` 与 `langchain-ai/langchain`（MIT）；`uuid6` 生成实现
改编自 `oittaa/uuid6-python`（MIT）。完整的版权声明、参考版本与许可证全文见
[`THIRD_PARTY_NOTICES.md`](https://github.com/from000/reactive-graph/blob/main/THIRD_PARTY_NOTICES.md)。

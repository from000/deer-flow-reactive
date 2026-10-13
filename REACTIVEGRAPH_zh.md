# deer-flow-reactive

[English](./REACTIVEGRAPH.md) | 中文

这是一个真实的 DeerFlow 后端迁移项目，执行底座换成
[ReactiveGraph](https://github.com/from000/reactive-graph)。

它不是简单的 import 替换。智能体执行底座、真实中间件链、运行时 worker、
流式桥接、checkpointer、store、子智能体执行器和客户端构建路径都跑在
ReactiveGraph 上。

## 已验证状态

在已验证的本地版本上：

数据来自 CI（`.github/workflows/backend-unit-tests.yml`，会拉起真实的
`postgres:17` 与 `redis:7-alpine` 服务，因此集成测试是真正执行而非 skip）：

```text
backend unit tests:     19,101 passed, 82 skipped, 0 failed（4 个 shard 合计）
blocking I/O suite:     154 passed
engine suite:            51 passed
no-LangGraph probes:      6 passed
live-model tests:         之前失败的四个用例重试，4 passed
```

不带外部服务的本地运行会跳过这些集成测试，报告 19,098 passed / 168 skipped；
以 CI 数字为准。

与 upstream 的真实 full-factory 对比产出了归一化输出一致性：

| 指标 | upstream LangGraph | ReactiveGraph |
|---|---:|---:|
| 首次 invoke | 6.975 ms | 0.971 ms |
| 重复 invoke | 3.765 ms | 0.466 ms |
| 实测加速 | — | 7.19x / 8.07x |

> **测量口径说明**：`ReactiveGraph` 列跑的是 `create_agent(...).AgentGraph`
> 的**进程内**执行——`create_agent` 构建的 `ReactiveGraph` 没有 `DriverHost`，
> 因此本表对比的是 Python fallback 执行器与完整 LangGraph 栈，不含原生
> TypeScript Driver、RGP/1 往返与 Node 子进程边界。7.19x/8.07x 应理解为
> 「Python 执行器 vs 完整 LangGraph 栈」；Driver 端到端数字见
> [reactive-graph 的 competitive proof §1b](https://github.com/from000/reactive-graph/blob/main/docs/competitive-proof.md)
> （单位数倍数；且 2026-10-12 修复了一处 O(n²) 输入合并开销）。

用于对比的 upstream fixture 为：

```text
/tmp/reactive-realworld/deer-flow
commit 5d8a9492eea97e2f06f28d46df69c29355acf4da
```

## 为什么代码里还有 LangChain

这是预期且有意为之的。

ReactiveGraph 替换的是**编排运行时**：图执行、依赖追踪、选择性执行、
事务性状态、调度、trace、恢复和 checkpoint。

`langchain_core` 保留为消息、模型接口、工具 schema 和既有生态集成的
互操作协议：

- `BaseMessage`、`HumanMessage`、`AIMessage`、`ToolMessage`
- `BaseTool`
- `BaseChatModel`
- 各家 provider 兼容的消息与工具 schema

一句话：

> 引擎是 ReactiveGraph。`langchain_core` 是模型与工具的行业兼容线协议。

这类似于使用 OpenAI 兼容的 HTTP 接口，同时自己拥有执行引擎和状态模型。

仍有一小组 `langgraph` **类型**（不是执行代码）保留用于 wire 与存储兼容：网关构造 checkpoint payload 或 update 时会用到来自 `langgraph.checkpoint.base` 与 `langgraph.types` 的 `empty_checkpoint`/`Command`/`Overwrite`，另有仅用于 `TYPE_CHECKING` 的注解。这些只是 native saver 接受的数据结构，不会实例化任何 `langgraph` graph、调度器或 prebuilt agent。

## 已替换的部分

- `create_deerflow_agent`
- 真实 lead-agent 构建
- 中间件执行
- worker 运行时
- 流式桥接
- run store
- checkpointer
- SQLite 持久化 checkpoint/store
- 子智能体执行器
- 内嵌客户端智能体构建

## 有意留在本次迁移之外的部分

- 前端行为对比
- 生产规模下的 channel、MCP、sandbox 压测
- goal continuation
- `custom` stream 模式

Postgres checkpoint/store 路径**不在此列**：harness 会把
`config.type: postgres` 解析到原生 ReactiveGraph provider
（`reactivegraph.checkpoint.PostgresCheckpointSaver` 与
`reactivegraph.store.PostgresStore`），并且 CI 用真实 Postgres 17 服务
（`.github/workflows/backend-unit-tests.yml` 的 `TEST_POSTGRES_URI`）
跑这些套件。

这些是已记录的边界，不是隐藏的缺陷。

## 复现

```bash
cd backend
uv sync --locked
make test
```

upstream 对比与基准方法论位于 ReactiveGraph 仓库的
`benchmarks/deerflow_real/`。

## 试点招募

ReactiveGraph 正在招募少量外部试点用户。请阅读
[试点邀请](https://github.com/from000/reactive-graph/blob/main/.github/PILOT_INVITATION.md)，
并通过 [Pilot user issue](https://github.com/from000/reactive-graph/issues/new?template=pilot_user.md)
提交一个真实工作负载，与维护者一起完成试点。

# deer-flow-reactive

This is a real DeerFlow backend migration onto [ReactiveGraph](https://github.com/from000/reactive-graph).

It is not a thin import swap. The agent execution base, real middleware chain,
runtime worker, stream bridge, checkpointer, store, subagent executor, and client
construction path run on ReactiveGraph.

## Verified state

At the verified local revision:

```text
backend make test:      19,098 passed, 168 skipped, 0 failed
blocking I/O suite:     154 passed
engine suite:           51 passed
no-LangGraph probes:    6 passed
live-model tests:       previously failing four tests retried, 4 passed
```

A real full-factory comparison against upstream produced normalized output parity:

| Metric | upstream LangGraph | ReactiveGraph |
|---|---:|---:|
| first invoke | 6.975 ms | 0.971 ms |
| repeated invoke | 3.765 ms | 0.466 ms |
| observed speed-up | — | 7.19x / 8.07x |

The upstream fixture used for comparison is:

```text
/tmp/reactive-realworld/deer-flow
commit 5d8a9492eea97e2f06f28d46df69c29355acf4da
```

## Why LangChain still appears in the code

This is expected and intentional.

ReactiveGraph replaces the **orchestration runtime**: graph execution, dependency
tracking, selective execution, transactional state, scheduling, trace, recovery,
and checkpoints.

`langchain_core` remains as an interoperability protocol for messages, model
interfaces, tool schemas, and existing ecosystem integrations:

- `BaseMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`
- `BaseTool`
- `BaseChatModel`
- provider-compatible message and tool schemas

In short:

> The engine is ReactiveGraph. `langchain_core` is the industry-compatible wire
> protocol for models and tools.

This is analogous to using OpenAI-compatible HTTP while owning the execution
engine and state model.

## What was replaced

- `create_deerflow_agent`
- real lead-agent construction
- middleware execution
- worker runtime
- stream bridge
- run store
- checkpointer
- SQLite durable checkpoint/store
- subagent executor
- embedded client agent construction

## What remains intentionally outside this migration

- frontend behavior comparison
- production-scale channel, MCP, and sandbox load testing
- goal continuation
- `custom` stream mode

The Postgres checkpoint/store path is **not** on this list: the harness resolves
`config.type: postgres` to the native ReactiveGraph providers
(`reactivegraph.checkpoint.PostgresCheckpointSaver` and
`reactivegraph.store.PostgresStore`), and CI runs those suites against a real
Postgres 17 service (`TEST_POSTGRES_URI` in
`.github/workflows/backend-unit-tests.yml`).

These are documented boundaries, not hidden limitations.

## Reproduction

```bash
cd backend
uv sync --locked
make test
```

The upstream comparison and benchmark methodology live in the ReactiveGraph
repository under `benchmarks/deerflow_real/`.

# deer-flow-reactive

This is a real DeerFlow backend migration onto [ReactiveGraph](https://github.com/from000/reactive-graph).

It is not a thin import swap. The agent execution base, real middleware chain,
runtime worker, stream bridge, checkpointer, store, subagent executor, and client
construction path run on ReactiveGraph.

## Verified state

At the verified revision, from CI (`.github/workflows/backend-unit-tests.yml`,
which provisions real `postgres:17` and `redis:7-alpine` services so the
integration tests execute instead of skipping):

```text
backend unit tests:     19,101 passed, 82 skipped, 0 failed  (4 shards combined)
blocking I/O suite:     154 passed
engine suite:           51 passed
no-LangGraph probes:    6 passed
live-model tests:       previously failing four tests retried, 4 passed
```

A DSN-less local run skips the external-service tests and therefore reports
19,098 passed / 168 skipped; the CI figure is the authoritative one.

### Reproducing the verification locally

The exact CI invocation is reproducible from a checkout of this repository:
start `postgres:17` (user/password/db = `deerflow`/`deerflow`/`deerflow_test`)
and `redis:7-alpine` on their default ports, then run

```bash
export TEST_POSTGRES_URI='postgresql://deerflow:deerflow@localhost:5432/deerflow_test?sslmode=disable'
export DEDUPE_TEST_POSTGRES_URL="$TEST_POSTGRES_URI"
cd backend
uv sync --group dev --extra postgres
for shard in 1 2 3 4; do make test-shard SPLITS=4 GROUP=$shard; done
```

Alternatively, recreate both services with Docker:

```bash
docker run -d --name rg-deerflow-pg -p 5432:5432 \
  -e POSTGRES_USER=deerflow -e POSTGRES_PASSWORD=deerflow \
  -e POSTGRES_DB=deerflow_test postgres:17
docker run -d --name rg-deerflow-redis -p 6379:6379 redis:7-alpine
```

All four shards must report `0 failed`; the combined pass/skip counts are the
ones quoted above.

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

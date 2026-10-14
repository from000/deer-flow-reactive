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

> **Measurement scope.** The `ReactiveGraph` column runs
> `create_agent(...).AgentGraph` **in process**: `create_agent` builds its
> `ReactiveGraph` with no `DriverHost`, so this table covers the Python fallback
> executor, not the native TypeScript Driver, RGP/1, or the Node child-process
> boundary that a Driver-backed deployment pays. Treat 7.19x/8.07x as a
> Python-executor comparison against the complete LangGraph stack, and see
> [`reactive-graph` competitive proof §1b](https://github.com/from000/reactive-graph/blob/main/docs/competitive-proof.md)
> for the Driver-backed end-to-end numbers (a single-digit multiple, and after
> a 2026-10-12 fix to an O(n^2) input-merge cost).

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

A smaller set of `langgraph` **types** (not execution code) remains for wire and
storage compatibility: `empty_checkpoint`/`Command`/`Overwrite` from
`langgraph.checkpoint.base` and `langgraph.types` when the gateway constructs a
checkpoint payload or an update, plus `TYPE_CHECKING`-only annotations. These
are data structures accepted by the native savers; they never instantiate a
`langgraph` graph, scheduler, or prebuilt agent.

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

## Dependency security (2026-10-14)

The fork publishes Dependabot data. As of the final dependency sweep the
repository is down to **one** open advisory, and it is one this repository
cannot fix:

- `braces <= 3.0.3` (high, stack-exhaustion DoS in deeply nested patterns) is
  reachable through `micromatch@4.0.8` (`fast-glob` -> `micromatch`), which is
  the newest 4.x release and declares `braces: ^3.0.3`. Verified against the
  npm registry: `braces`' newest published version **is** 3.0.3, and the
  advisory's `first_patched_version` is empty — no patched release exists yet
  for any major line. There is nothing to upgrade to, so this one is tracked
  upstream rather than silenced or force-fixed with a semver-violating bump.

Everything else was remediated in-tree: `pnpm.overrides` pin the patched
transitive versions (sharp, dompurify, source-map-js, brace-expansion,
image-size, js-yaml, fflate, `@humanfs/node`, mermaid, katex, uuid, langsmith,
lodash-es, minimatch, flatted), the unused `nuxt-og-image` dependency was
removed (which eliminated its vite/unhead/esbuild chains), and the backend
lockfile pins `langchain-anthropic>=1.4.6,<1.5`, `oauthlib>=4.0.0` and
`soupsieve>=2.9.0`. `langchain-anthropic` is deliberately capped below 1.5:
1.5+ pulls `anthropic` 1.x, whose `_format_messages` signature drops the
no-model call form this gateway relies on (`tests/test_tool_call_args.py` pins
that contract).

## Reproduction

```bash
cd backend
uv sync --locked
make test
```

The upstream comparison and benchmark methodology live in the ReactiveGraph
repository under `benchmarks/deerflow_real/`.

## Pilot with us

ReactiveGraph is recruiting a small group of external pilots. Read the
[pilot invitation](https://github.com/from000/reactive-graph/blob/main/.github/PILOT_INVITATION.md)
and [open a Pilot user issue](https://github.com/from000/reactive-graph/issues/new?template=pilot_user.md)
to run one real workload with direct maintainer support.

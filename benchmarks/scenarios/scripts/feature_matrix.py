"""Feature 覆盖矩阵：各 RuntimeFeatures 组合下的引擎开销（fake model）。

**必须提供有效 `config.yaml`**：否则 `memory=True` / `auto_title=True` 等组合会走
`FileNotFoundError` 错误路径，测出的是"失败路径更便宜"（曾实测出 6.7–7.8× 的假优势）。
本脚本默认用各仓库自带的 `config.example.yaml`（`DEER_FLOW_CONFIG_PATH` 指向它），
如需其他配置可用 `FEATURE_MATRIX_CONFIG_FORK/UPSTREAM` 覆盖。

不在覆盖范围内（见 REAL_SCENARIOS.md"覆盖矩阵"）：
- 外部资源路径：真实 sandbox 容器、MCP server、文件上传、Postgres/Redis
- 真实 LLM 路径：summarization / guardrail / memory 抽取（fake model 下不触发真实推理）
- 前端 / 网关 / TUI / CLI 层

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path

from _common import fork_root, upstream_root

OUT = Path(__file__).resolve().parent.parent / "feature_matrix_results.json"
ROUNDS = int(os.environ.get("FEATURE_MATRIX_ROUNDS", "20"))

TEMPLATE = r'''
import json, os, statistics, sys, time, uuid
sys.path.insert(0, "__HARNESS__"); sys.path.insert(0, "__BACKEND__")
os.environ["DEER_FLOW_CONFIG_PATH"] = "__CONFIG__"
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from deerflow.agents.features import RuntimeFeatures
from deerflow.agents.factory import create_deerflow_agent

class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self

@tool("echo")
def echo_tool(value: str) -> str:
    """Echo."""
    return "echo:" + value

FEATURE_SETS = {
    "default": {},
    "no_sandbox": {"sandbox": False},
    "with_subagent": {"subagent": True},
    "with_memory": {"memory": True},
    "with_vision": {"vision": True},
    "with_auto_title": {"auto_title": True},
    "with_token_budget": {"token_budget": True},
    "all_off": {"sandbox": False, "loop_detection": False, "subagent": False, "memory": False},
}
ROUNDS = __ROUNDS__

out = {}
for name, kwargs in FEATURE_SETS.items():
    times, err = [], None
    for _ in range(ROUNDS):
        try:
            model = M(responses=[AIMessage(id="a", content="done")] * 5)
            g = create_deerflow_agent(model, tools=[echo_tool], features=RuntimeFeatures(**kwargs))
            cfg = {"configurable": {"thread_id": "fm-" + uuid.uuid4().hex}}
            t0 = time.perf_counter()
            g.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
            times.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:80]}"
            break
    out[name] = {"median_ms": round(statistics.median(times), 3) if times else None,
                 "n": len(times), "error": err}
print(json.dumps(out))
'''


def _config_for(root: Path, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    candidate = root / "config.example.yaml"
    if not candidate.exists():
        raise SystemExit(f"缺少配置文件：{candidate}（或用 {env_var} 指定）")
    return str(candidate)


def run_side(root: Path, config_path: str) -> dict:
    code = (TEMPLATE
            .replace("__ROUNDS__", str(ROUNDS))
            .replace("__HARNESS__", str(root / "backend" / "packages" / "harness"))
            .replace("__BACKEND__", str(root / "backend"))
            .replace("__CONFIG__", config_path))
    py = root / "backend" / ".venv" / "bin" / "python"
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return {"__error__": r.stderr[-500:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"__error__": (r.stdout[-200:] + r.stderr[-200:])}


if __name__ == "__main__":
    fork, up = fork_root(), upstream_root()
    out = {
        "reactivegraph": run_side(fork, _config_for(fork, "FEATURE_MATRIX_CONFIG_FORK")),
        "langgraph_upstream": run_side(up, _config_for(up, "FEATURE_MATRIX_CONFIG_UPSTREAM")),
    }
    print(json.dumps(out, indent=2))
    ratios = {}
    for name, fa in out["reactivegraph"].items():
        ua = out["langgraph_upstream"].get(name, {})
        a, b = fa.get("median_ms"), ua.get("median_ms")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ratios[name] = {"fork_ms": a, "upstream_ms": b, "ratio": round(b / a, 2)}
    out["ratios"] = ratios
    print("ratios:", json.dumps(ratios, indent=2))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

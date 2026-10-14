"""多会话并发吞吐对照（64 sessions / 8 workers，每个 session 独立 thread）。

每边重复 ``REPEATS`` 次，**交替执行**（fork / upstream 轮流），避免"先跑完一边再跑
另一边"的系统负载偏置；同时记录每轮的 load average 供读者判断测量背景。
需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from _common import fork_root, run_code, upstream_root

OUT = Path(__file__).resolve().parent.parent / "concurrency_results.json"
REPEATS = 3

TEMPLATE = r'''
import json, statistics, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from deerflow.agents.factory import create_deerflow_agent
class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
def one_session(_i):
    model = M(responses=[AIMessage(id=f"a{_i}", content="ok")] * 5)
    graph = create_deerflow_agent(model)
    cfg = {"configurable": {"thread_id": f"s-{uuid.uuid4().hex}"}}
    t0 = time.perf_counter(); graph.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
    return (time.perf_counter()-t0)*1000
def bench(sessions=64, workers=8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        times = list(ex.map(one_session, range(sessions)))
    wall = (time.perf_counter()-t0)*1000
    return {"sessions": sessions, "workers": workers, "wall_ms": round(wall,3),
            "throughput_sessions_per_s": round(sessions/(wall/1000),1),
            "median_ms": round(statistics.median(times),3),
            "p95_ms": round(sorted(times)[int(0.95*(len(times)-1))],3)}
print(json.dumps(bench()))
'''

def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover - platform dependent
        return float("nan")


if __name__ == "__main__":
    roots = {"reactivegraph": fork_root(), "langgraph_upstream": upstream_root()}
    out = {label: [] for label in roots}
    for i in range(REPEATS):
        # 交替：同一轮内两个 label 相邻，系统负载背景尽量接近
        for label in ("reactivegraph", "langgraph_upstream"):
            d = run_code(roots[label], TEMPLATE)
            d = dict(d)
            d["load1_at_run"] = _load1()
            out[label].append(d)
            print(label, f"run{i}", json.dumps(d))
    for label in roots:
        tps = [r["throughput_sessions_per_s"] for r in out[label] if "throughput_sessions_per_s" in r]
        if tps:
            out[f"{label}_summary"] = {
                "throughput_min": min(tps), "throughput_median": round(statistics.median(tps), 1),
                "throughput_max": max(tps), "runs": len(tps),
                "load1_runs": [r.get("load1_at_run") for r in out[label]],
            }
            print(label, "summary", json.dumps(out[f"{label}_summary"]))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

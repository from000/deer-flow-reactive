"""多轮对话累积 + 大 state 场景对照。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from _common import fork_root, run_code, upstream_root


OUT = Path(__file__).resolve().parent.parent / "more_results.json"

TEMPLATE = r'''
import json, statistics, sys, time, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from deerflow.agents.factory import create_deerflow_agent
class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
def multi_turn(rounds=10):
    model = M(responses=[AIMessage(id=f"r{i}", content=f"reply {i}") for i in range(200)])
    graph = create_deerflow_agent(model)
    cfg = {"configurable": {"thread_id": "mt-" + uuid.uuid4().hex}}
    times = []
    for i in range(rounds):
        t0 = time.perf_counter()
        out = graph.invoke({"messages": [HumanMessage(content=f"turn {i}")]}, cfg)
        times.append((time.perf_counter()-t0)*1000)
    return {"median_ms": round(statistics.median(times),3),
            "p95_ms": round(sorted(times)[int(0.95*(len(times)-1))],3),
            "message_count_last": len(out["messages"])}
def big_state(rounds=30, doc_kb=64):
    doc = "x" * (doc_kb * 1024)
    model = M(responses=[AIMessage(id=f"s{i}", content="ok") for i in range(rounds+50)])
    graph = create_deerflow_agent(model, system_prompt="You are a helpful assistant. " + doc)
    cfg = {"configurable": {"thread_id": "bs-" + uuid.uuid4().hex}}
    payload = {"messages": [HumanMessage(content="summarize")]}
    for _ in range(3): graph.invoke(payload, cfg)
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter(); out = graph.invoke(payload, cfg); times.append((time.perf_counter()-t0)*1000)
    return {"median_ms": round(statistics.median(times),3),
            "p95_ms": round(sorted(times)[int(0.95*(len(times)-1))],3),
            "message_count": len(out["messages"]), "doc_kb": doc_kb}
print(json.dumps({"multi_turn": multi_turn(), "big_state": big_state()}))
'''

def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover - platform dependent
        return float("nan")


if __name__ == "__main__":
    out = {}
    roots = {"reactivegraph": fork_root(), "langgraph_upstream": upstream_root()}
    # 交替执行两次（奇数轮口径），两侧共享相近的系统负载背景
    for label in roots:
        out[label] = []
    for _round in range(3):
        for label, root in roots.items():
            d = dict(run_code(root, TEMPLATE))
            d["load1_at_run"] = _load1()
            out[label].append(d)
            mt = d.get("multi_turn", {}); bs = d.get("big_state", {})
            print(label, f"round{_round}", "multi_turn median=", mt.get("median_ms"),
                  "big_state median=", bs.get("median_ms"), d.get("error", ""))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

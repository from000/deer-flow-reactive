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

# 关键：create_deerflow_agent 不传 checkpointer 时是 ephemeral 的，同一 thread_id
# 多次 invoke **不会累积历史**（每次都是从空 state 开始的单轮）。要测"多轮累积"
# 必须显式挂 checkpointer。两侧各用自家的 checkpointer 实现，保证 API 对等。
try:  # fork: ReactiveGraph native checkpointer
    from reactivegraph.checkpoint import MemoryCheckpointSaver as Saver
except ImportError:  # upstream: LangGraph checkpointer
    from langgraph.checkpoint.memory import InMemorySaver as Saver

class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
def multi_turn(rounds=10):
    model = M(responses=[AIMessage(id=f"r{i}", content=f"reply {i}") for i in range(200)])
    graph = create_deerflow_agent(model, checkpointer=Saver())
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
    graph = create_deerflow_agent(model, system_prompt="You are a helpful assistant. " + doc, checkpointer=Saver())
    cfg = {"configurable": {"thread_id": "bs-" + uuid.uuid4().hex}}
    # 每轮必须新建 HumanMessage：复用同一个 message 对象时，LangGraph 的 add_messages
    # 会按 id 去重（33 轮只留 1 条 human），而 fork 逐条追加，两侧工作量会失真。
    def turn_payload(i):
        return {"messages": [HumanMessage(content=f"summarize {i}")]}
    for i in range(3): graph.invoke(turn_payload(i), cfg)
    times = []
    for i in range(rounds):
        t0 = time.perf_counter()
        out = graph.invoke(turn_payload(3 + i), cfg)
        times.append((time.perf_counter()-t0)*1000)
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
            mt = d.get("multi_turn", {})
            bs = d.get("big_state", {})
            print(label, f"round{_round}", "multi_turn median=", mt.get("median_ms"),
                  "big_state median=", bs.get("median_ms"), d.get("error", ""))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

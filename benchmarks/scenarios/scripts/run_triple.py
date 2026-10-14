"""端到端 agent 对照（median of 3 runs，每轮 50 样本，独立 thread）。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from _common import fork_root, run_code, upstream_root


OUT = Path(__file__).resolve().parent.parent / "triple_results.json"

TEMPLATE = r'''
import json, statistics, sys, time, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from deerflow.agents.factory import create_deerflow_agent
class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
@tool("echo")
def echo_tool(value: str) -> str:
    """Echo."""
    return "echo:"+value
RESPONSES = [AIMessage(id="a", content="", tool_calls=[{"name":"echo","args":{"value":"world"},"id":"c1","type":"tool_call"}]),
             AIMessage(id="b", content="done")]
def bench(tool_call, rounds=50):
    times = []
    for _ in range(rounds):
        model = M(responses=(RESPONSES if tool_call else [AIMessage(id="a", content="done")]) * 30)
        graph = create_deerflow_agent(model, tools=[echo_tool] if tool_call else None)
        cfg = {"configurable": {"thread_id": "b-" + uuid.uuid4().hex}}
        payload = {"messages": [HumanMessage(content="same")]}
        t0 = time.perf_counter(); graph.invoke(payload, cfg); times.append((time.perf_counter()-t0)*1000)
    return round(statistics.median(times), 3)
print(json.dumps({"model_only": bench(False), "tool_call": bench(True)}))
'''

def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover - platform dependent
        return float("nan")


if __name__ == "__main__":
    fork, up = fork_root(), upstream_root()
    res = {"reactivegraph": [], "langgraph_upstream": []}
    for i in range(3):
        # 交替执行：同一轮内两侧相邻，系统负载背景尽量接近
        for label, root in (("reactivegraph", fork), ("langgraph_upstream", up)):
            d = dict(run_code(root, TEMPLATE))
            d["load1_at_run"] = _load1()
            res[label].append(d)
            print(f"run{i} {label}: {d}")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

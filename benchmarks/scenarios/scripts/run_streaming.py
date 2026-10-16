"""流式执行（astream）场景对照：fork vs 上游 LangGraph。

DeerFlow 生产路径的默认执行形态是 `agent.astream(stream_mode=["values"])`
（见 `deerflow/runtime/runs/worker.py` 与 `runtime/stream_modes.py`），不是同步 invoke。
因此"真实场景"还应覆盖 astream，否则只测了 SDK 层非生产路径。

本脚本用 fake model 测两侧 `astream(values)` / `astream(messages)` 的逐 chunk 墙钟
与进程 CPU 开销。与 sync 场景一样：先断言 chunk 序列等价，再计时。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from _common import fork_root, run_code, upstream_root

OUT = Path(__file__).resolve().parent.parent / "streaming_results.json"
SAMPLES = int(os.environ.get("STREAM_SAMPLES", "20"))
ROUNDS = int(os.environ.get("STREAM_ROUNDS", "3"))

TEMPLATE = r'''
import asyncio, json, statistics, sys, time, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from deerflow.agents.factory import create_deerflow_agent

SAMPLES = __SAMPLES__

class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self

def run_stream(mode):
    model = M(responses=[AIMessage(id=f"r{i}", content=f"reply {i}") for i in range(300)])
    graph = create_deerflow_agent(model)
    times, cpus, chunk_counts = [], [], []
    for s in range(SAMPLES):
        cfg = {"configurable": {"thread_id": f"st-{mode}-{uuid.uuid4().hex}"}}
        t0 = time.perf_counter(); c0 = time.process_time()
        chunks = 0
        seq = []
        async def one():
            nonlocal chunks, seq
            async for ev in graph.astream(
                {"messages": [HumanMessage(content="hi")]}, cfg,
                stream_mode=[mode] if mode == "values" else ["messages"],
            ):
                chunks += 1
                seq.append(str(ev)[:40])
        asyncio.run(one())
        times.append((time.perf_counter() - t0) * 1000)
        cpus.append((time.process_time() - c0) * 1000)
        chunk_counts.append(chunks)
    return {"mode": mode, "n": len(times),
            "median_ms": round(statistics.median(times), 3),
            "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
            "cpu_ms_median": round(statistics.median(cpus), 3),
            "chunks_median": round(statistics.median(chunk_counts), 0),
            "first_chunk_seq": seq[:3]}

out = {"values": run_stream("values"), "messages": run_stream("messages")}
print(json.dumps(out))
'''

def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover - platform dependent
        return float("nan")


if __name__ == "__main__":
    roots = {"reactivegraph": fork_root(), "langgraph_upstream": upstream_root()}
    out = {label: [] for label in roots}
    for _round in range(ROUNDS):
        for label, root in roots.items():
            tpl = TEMPLATE.replace("__SAMPLES__", str(SAMPLES))
            d = dict(run_code(root, tpl))
            d["load1_at_run"] = _load1()
            out[label].append(d)
            print(label, f"round{_round}", "values=", d.get("values", {}).get("median_ms"),
                  "messages=", d.get("messages", {}).get("median_ms"), d.get("error", ""))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

"""控制实验：全新进程 + 单次 invoke 的引擎开销（fake model，无 LLM 噪声）。

用途：验证"真实 LLM 测量中两边 residual 几乎相同"不是冷启动造成的假象。

real_llm.py 每侧都在**新进程**里只 invoke 一次，因此若 fork 的优势来自"热路径缓存"，
在同样的新进程形态下应当消失。本脚本用 fake model 复刻该形态（每次全新解释器、
单次 invoke），并额外测量同进程多次 invoke 的"热路径"作为对照。

实测结论（见 REAL_SCENARIOS.md 真实 LLM 章节）：
- 冷启动形态 fork 仍快约 2x（fork 5–17ms vs 上游 11–20ms wall）
- 热路径形态 fork 快约 2.5–4x
  => 真实 LLM 下 residual 差异缩小，**不是**冷启动造成的，而是 LLM 路径带来的
     固定本地开销（消息编解码、schema、工具执行）稀释了引擎差异。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import fork_root, run_code, upstream_root

OUT = Path(__file__).resolve().parent.parent / "fresh_process_control_results.json"

TEMPLATE = r'''
import json, statistics, sys, time, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from deerflow.agents.factory import create_deerflow_agent

class M(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self

LLM_WALL, LLM_CPU = [], []

class TimedModel(M):
    def _generate(self, *a, **k):
        t0, c0 = time.perf_counter(), time.process_time()
        out = super()._generate(*a, **k)
        LLM_WALL.append(time.perf_counter() - t0)
        LLM_CPU.append(time.process_time() - c0)
        return out

def one_invoke(graph):
    cfg = {"configurable": {"thread_id": "fp-" + uuid.uuid4().hex}}
    payload = {"messages": [HumanMessage(content="same")]}
    t0, c0 = time.perf_counter(), time.process_time()
    graph.invoke(payload, cfg)
    return (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000

# 冷启动形态：与 real_llm.py 一致——新进程、单次 invoke
cold_wall, cold_cpu = one_invoke(create_deerflow_agent(TimedModel(responses=[AIMessage(id="a", content="done")] * 5)))

# 热路径对照：同进程多次 invoke（跳过前 5 次 warmup）
warm_model = TimedModel(responses=[AIMessage(id=f"a{i}", content="done") for i in range(60)])
warm_graph = create_deerflow_agent(warm_model)
warm = [one_invoke(warm_graph) for _ in range(20)]
warm_wall = statistics.median(w for w, _ in warm[5:])
warm_cpu = statistics.median(c for _, c in warm[5:])

print(json.dumps({
    "cold_wall_ms": round(cold_wall, 3), "cold_cpu_ms": round(cold_cpu, 3),
    "cold_llm_cpu_ms": round(sum(LLM_CPU[:1]) * 1000, 3),
    "cold_residual_cpu_ms": round((cold_cpu / 1000 - sum(LLM_CPU[:1])) * 1000, 3),
    "warm_wall_ms": round(warm_wall, 3), "warm_cpu_ms": round(warm_cpu, 3),
}))
'''

if __name__ == "__main__":
    out = {}
    for label, root in (("reactivegraph", fork_root()), ("langgraph_upstream", upstream_root())):
        out[label] = run_code(root, TEMPLATE)
        print(label, json.dumps(out[label]))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

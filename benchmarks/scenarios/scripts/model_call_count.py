"""验证：fork 的 agent 路径（create_deerflow_agent）是否对重复输入做跨 run 跳过。

结论（诚实披露）：**不跳过**。model/tools 节点在 `create_deerflow_agent` 里以默认
`kind="effect"` 注册（有副作用语义），因此同一 thread 重复相同输入时模型仍被逐次调用。

对照组：core 仓库 `benchmarks/deerflow_real/deerflow_reactive.py` 里显式声明
`kind="pure"` + `reads` 的任务，同输入可被指纹跳过。本脚本只测量 agent 路径，
用来防止"省钱"类夸大宣传。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import fork_root, run_code, upstream_root

OUT = Path(__file__).resolve().parent.parent / "model_call_count_results.json"

TEMPLATE = r'''
import json, sys, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from deerflow.agents.factory import create_deerflow_agent

CALLS = {"n": 0}

class CountingModel(FakeMessagesListChatModel):
    def _generate(self, *args, **kwargs):
        CALLS["n"] += 1
        return super()._generate(*args, **kwargs)
    def bind_tools(self, tools, **kwargs): return self

TURNS = 30
model = CountingModel(responses=[AIMessage(id=f"r{i}", content="ok") for i in range(TURNS + 5)])
graph = create_deerflow_agent(model)
cfg = {"configurable": {"thread_id": "count-" + uuid.uuid4().hex}}
for _ in range(TURNS):
    graph.invoke({"messages": [HumanMessage(content="identical input")]}, cfg)

print(json.dumps({"turns": TURNS, "model_calls": CALLS["n"],
                  "skipped": TURNS - CALLS["n"]}))
'''

if __name__ == "__main__":
    out = {}
    for label, root in (("reactivegraph", fork_root()), ("langgraph_upstream", upstream_root())):
        out[label] = run_code(root, TEMPLATE)
        print(label, json.dumps(out[label]))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

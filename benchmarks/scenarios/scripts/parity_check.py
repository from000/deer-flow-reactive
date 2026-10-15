"""输出等价性验证：同输入下 fork 与上游的最终 state 是否逐条一致。

这是任何计时对比的前提——先证明两边语义等价，再比较性能。
需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json

from _common import fork_root, run_code, upstream_root

TEMPLATE = r'''
import json, sys, uuid
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
def run(tool_call):
    model = M(responses=(RESPONSES if tool_call else [AIMessage(id="a", content="done")]) * 30)
    graph = create_deerflow_agent(model, tools=[echo_tool] if tool_call else None)
    out = graph.invoke({"messages": [HumanMessage(content="same")]},
                       {"configurable": {"thread_id": "p-" + uuid.uuid4().hex}})
    seq = []
    for m in out["messages"]:
        entry = {"type": type(m).__name__, "content": getattr(m, "content", None)}
        tc = getattr(m, "tool_calls", None)
        if tc:
            entry["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")} for c in tc]
        if type(m).__name__ == "ToolMessage":
            entry["name"] = getattr(m, "name", None)
        seq.append(entry)
    return seq
print(json.dumps({"model_only": run(False), "tool_call": run(True)}))
'''

if __name__ == "__main__":
    a = run_code(fork_root(), TEMPLATE)
    b = run_code(upstream_root(), TEMPLATE)
    for sc in ("model_only", "tool_call"):
        fa, ub = a.get(sc), b.get(sc)
        same = fa == ub
        print(f"{sc}: identical={same}")
        # 单独报告"消息序列 + 工具轨迹"摘要：真实 LLM 下工具轨迹可能因模型随机性不同，
        # 但 fake model（确定性）下两边必须逐条一致，这是计时比较的前提。
        for side, seq in (("fork", fa), ("up ", ub)):
            if isinstance(seq, list):
                types = [e.get("type") for e in seq]
                tool_msgs = sum(1 for e in seq if e.get("type") == "ToolMessage")
                print(f"  {side} messages={len(seq)} tool_messages={tool_msgs} seq={types}")
        if not same:
            print("  fork:", json.dumps(fa, ensure_ascii=False)[:300])
            print("  up  :", json.dumps(ub, ensure_ascii=False)[:300])

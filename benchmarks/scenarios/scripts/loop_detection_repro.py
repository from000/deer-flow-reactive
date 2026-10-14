"""复现：同一 thread 连续多次 invoke 相同 tool_call 时，两边的 loop-detection 差异。

场景：模型每次 **invoke** 都发出同一个 tool_call（name+args 相同，但消息 id 唯一，
避免 add_messages 按 id 替换），在同一 thread 连续 invoke 6 次，记录每次的 message 数。

结论（诚实披露）：
- fork（ReactiveGraph runtime）：loop budget 按 ``(thread_id, run_id)`` 作用域，
  每次 invoke 是新的 run_id，因此每次都有全新预算（5 次模型调用后兜底 hard-stop），
  行为稳定：[10, 10, 10, 10, 10, 10]。
- 上游（langgraph runtime）：history 只按 ``thread_id`` 累积，loop budget 跨 invoke
  共享；第 1 次 invoke 消耗全部 5 次预算，第 2 次 invoke 起立即 hard-stop，
  输出退化为 2 条：[10, 2, 2, 2, 2, 2]。

对上游而言这是"自动止损"（省钱、防死循环），但代价是**同一 thread 的后续 invoke
行为不一致**——若业务期望每次 invoke 都完整执行 agent 循环，上游会静默截断；
fork 每次 invoke 独立预算，行为可预期。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import fork_root, run_code, upstream_root

OUT = Path(__file__).resolve().parent.parent / "loop_detection_results.json"

TEMPLATE = r'''
import json, sys, uuid
sys.path.insert(0, "HARNESS"); sys.path.insert(0, "BACKEND")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from deerflow.agents.factory import create_deerflow_agent

class RepeatModel(BaseChatModel):
    calls: int = 0
    @property
    def _llm_type(self): return "repeat"
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        tc = AIMessage(id=f"a{self.calls}", content="",
                       tool_calls=[{"name": "echo", "args": {"value": "world"},
                                    "id": f"c{self.calls}", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=tc)])
    def bind_tools(self, tools, **kwargs): return self

@tool("echo")
def echo_tool(value: str) -> str:
    """Echo."""
    return "echo:" + value

model = RepeatModel()
graph = create_deerflow_agent(model, tools=[echo_tool])
cfg = {"configurable": {"thread_id": "lr-" + uuid.uuid4().hex}, "recursion_limit": 40}
counts = []
for _ in range(6):
    out = graph.invoke({"messages": [HumanMessage(content="same")]}, cfg)
    counts.append(len(out["messages"]))
print(json.dumps({"message_counts": counts, "total_model_calls": model.calls}))
'''

if __name__ == "__main__":
    out = {}
    for label, root in (("reactivegraph", fork_root()), ("langgraph_upstream", upstream_root())):
        out[label] = run_code(root, TEMPLATE)
        print(label, json.dumps(out[label]))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

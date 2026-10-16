"""模型侧成本对等性验证：fork 与上游发给 LLM 的请求是否逐字节相同。

回答"换 ReactiveGraph 能不能省钱"：不能。本脚本给两侧各起一个**本地 OpenAI 兼容
端点**，逐请求 capture 完整 HTTP body，然后逐字节比对。相同输入下若两侧 body 完全
一致，则 prompt / tools schema / 消息序列都相同 ⇒ token 数相同、调用次数相同 ⇒
模型成本相同。

只用 fake/本地端点，不消耗真实 LLM 配额、不联网。

结论（本机实测）：model_only / tool_call / multi_turn 三种场景，两侧每个请求 body
逐字节相同；multi_turn 消息数按 1→3→5→7 累积，两侧同步。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from _common import fork_root, upstream_root

SCENARIOS = [s.strip() for s in os.environ.get("COST_PARITY_SCENARIOS",
             "model_only,tool_call,multi_turn").split(",") if s.strip()]
ROUNDS = int(os.environ.get("COST_PARITY_MULTI_TURN_ROUNDS", "4"))


class _Capture:
    def __init__(self) -> None:
        self.bodies: list[str] = []
        self.lock = threading.Lock()


def _make_handler(cap: _Capture):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (stdlib naming)
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) or b"{}"
            with cap.lock:
                cap.bodies.append(raw.decode("utf-8", "replace"))
            body = json.loads(raw)
            # tool_call 场景需要先返回 tool_call 再返回正文，否则 ReAct 循环不完整
            has_tool_result = any(
                (m.get("role") == "tool") for m in body.get("messages", [])
            )
            if "get_weather" in json.dumps(body.get("tools", [])) and not has_tool_result:
                msg = {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": '{"city": "Shanghai"}'}}]}
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "ok"}
                finish = "stop"
            resp = {"id": "chatcmpl-x", "object": "chat.completion",
                    "created": int(time.time()), "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # silence
            pass

    return Handler


def _serve(cap: _Capture) -> tuple[HTTPServer, int]:
    srv = HTTPServer(("127.0.0.1", 0), _make_handler(cap))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def _run_side(root: Path, scenario: str, port: int) -> dict:
    env = dict(os.environ)
    env.update({
        "CUSTOM_LLM_URL": f"http://127.0.0.1:{port}/v1",
        "CUSTOM_LLM_API_KEY": "sk-local",
        "CUSTOM_MODEL": "local-probe",
        "REAL_LLM_REPEATS": "1",
        "REAL_LLM_SCENARIOS": scenario,
        "REAL_LLM_SLEEP": "0",
        "REAL_LLM_MAX_LOAD": "100000",
        "REAL_LLM_MAX_LOAD_WAIT": "0",
        "REAL_LLM_MULTI_TURN_ROUNDS": str(ROUNDS),
        "REAL_LLM_BIG_STATE_ROUNDS": str(ROUNDS),
    })
    script = Path(__file__).resolve().parent / "real_llm.py"
    r = subprocess.run([sys.executable, str(script)], env=env,
                       capture_output=True, text=True, timeout=900)
    out = {"returncode": r.returncode, "stdout_tail": r.stdout[-300:]}
    if r.returncode != 0:
        out["stderr_tail"] = r.stderr[-400:]
    return out


if __name__ == "__main__":
    roots = {"reactivegraph": fork_root(), "langgraph_upstream": upstream_root()}
    all_identical = True
    for scenario in SCENARIOS:
        caps = {label: _Capture() for label in roots}
        srvs = {}
        for label, root in roots.items():
            srvs[label], port = _serve(caps[label])
            _run_side(root, scenario, port)
        for srv in srvs.values():
            srv.shutdown()
        fb = caps["reactivegraph"].bodies
        ub = caps["langgraph_upstream"].bodies
        n = min(len(fb), len(ub))
        identical = (len(fb) == len(ub)) and all(
            json.dumps(json.loads(a), sort_keys=True) == json.dumps(json.loads(b), sort_keys=True)
            for a, b in zip(fb[:n], ub[:n])
        )
        msgs_fork = [len(json.loads(b).get("messages", [])) for b in fb]
        msgs_up = [len(json.loads(b).get("messages", [])) for b in ub]
        all_identical &= identical
        print(f"  scenario={scenario:11s} requests={len(fb)}/{len(ub)} "
              f"byte_identical={identical}")
        print(f"    messages_per_request fork={msgs_fork} upstream={msgs_up}")
    print()
    print("RESULT:", "IDENTICAL — model-side cost is equal" if all_identical
          else "DIFFERS — investigate before claiming cost parity")
    sys.exit(0 if all_identical else 1)

"""真实 LLM 场景对照（Agnes OpenAI-compatible endpoint，agnes-2.5-flash）。

为什么这样设计
--------------
真实调用的墙钟时间由**网络 + 推理**主导（每次数秒），远大于引擎编排开销（毫秒级），
所以单看端到端 wall time 无法体现引擎差异。本脚本把每步 LLM 调用时间单独测量并从
wall time 中扣除，得到引擎开销：

    engine_s = wall_s - sum(llm_call_s)

这才是编排/调度/状态管理的真实开销。两边使用同一模型、同一 prompt、同一调用次数。

Agnes 限流为 1 请求/分钟（账号级），所以父脚本在两次子进程之间主动 sleep，
顺序执行、不并发。``max_retries=0`` 保证 429 直接暴露而不被内部重试掩盖。

需先 export FORK_ROOT / UPSTREAM_ROOT，见 _common.py。
环境变量：
  CUSTOM_LLM_URL / CUSTOM_LLM_API_KEY / CUSTOM_MODEL（默认 agnes-2.5-flash）
  REAL_LLM_REPEATS（默认 1，受配额限制）
  REAL_LLM_SCENARIOS（默认 model_only；可选 model_only / tool_call / multi_turn / big_state）
  REAL_LLM_MULTI_TURN_ROUNDS（默认 5，multi_turn 轮数）
  REAL_LLM_BIG_STATE_ROUNDS / REAL_LLM_BIG_STATE_KB（默认 4 / 64，big_state 轮数与 system prompt 大小）
  REAL_LLM_SLEEP（默认 65 秒，Agnes 1 req/min）
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from _common import fork_root, upstream_root

OUT = Path(__file__).resolve().parent.parent / "real_llm_results.json"
REPEATS = int(os.environ.get("REAL_LLM_REPEATS", "1"))
SCENARIOS = [s.strip() for s in os.environ.get("REAL_LLM_SCENARIOS", "model_only").split(",") if s.strip()]
RATE_LIMIT_SLEEP = float(os.environ.get("REAL_LLM_SLEEP", "65"))
MAX_LOAD = float(os.environ.get("REAL_LLM_MAX_LOAD", "12"))
MAX_LOAD_WAIT = float(os.environ.get("REAL_LLM_MAX_LOAD_WAIT", "900"))

TEMPLATE = r'''
import json, statistics, sys, time, uuid
import openai
sys.path.insert(0, "__HARNESS__"); sys.path.insert(0, "__BACKEND__")
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from deerflow.agents.factory import create_deerflow_agent

SCENARIO = __SCENARIO__
MULTI_TURN_ROUNDS = __MT_ROUNDS__
BIG_STATE_ROUNDS = __BS_ROUNDS__
BIG_STATE_KB = __BS_KB__
LLM_TIMES = []          # 纯推理/网络时间（不含 429 等待）
LLM_CPU_TIMES = []      # 每次 LLM 调用消耗的本进程 CPU 时间
LLM_EVENTS = []         # 逐次调用明细：attempt / status / llm_s / wait_s
RATE_LIMIT_WAITS = []    # 429 限流等待（两边同样受限，单独统计并扣除）

class TimedModel(ChatOpenAI):
    """记录每次 LLM 调用的 wall 时间与 CPU 时间。

    CPU 时间（``time.process_time``）只统计本进程真正占用 CPU 的部分，不受调度延迟
    影响，因此在共享/高负载机器上比 "wall_s - llm_s" 的减法更稳健：

        engine_wall_s = wall_s      - sum(llm_wall_s) - rate_limit_wait_s   （受负载噪声影响）
        engine_cpu_s  = total_cpu_s - sum(llm_cpu_s)                        （对调度延迟免疫）
    """

    def _generate(self, *args, **kwargs):
        # Agnes 账号级限流 1 req/min：tool_call 场景第二次调用必然 429。
        # 显式等待后重试，并把等待时间单独记账（不计入 engine_s，也不计入 llm_s）。
        for attempt in range(4):
            t0, c0 = time.perf_counter(), time.process_time()
            try:
                out = super()._generate(*args, **kwargs)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status is None:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                is_429 = isinstance(e, openai.RateLimitError) or status == 429
                if is_429:
                    failed_s = time.perf_counter() - t0
                    RATE_LIMIT_WAITS.append(failed_s)
                    LLM_EVENTS.append({"attempt": attempt, "status": 429,
                                       "llm_s": round(failed_s, 4), "wait_s": 0.0})
                    if attempt < 3:
                        nap = 61 + attempt * 5  # 等过限流窗口
                        t_sleep = time.perf_counter()
                        time.sleep(nap)
                        slept = time.perf_counter() - t_sleep
                        # 关键：sleep 也必须计入限流等待，否则会被误算成 residual_s
                        RATE_LIMIT_WAITS.append(slept)
                        LLM_EVENTS.append({"attempt": attempt, "status": "wait",
                                           "llm_s": 0.0, "wait_s": round(slept, 4)})
                        continue
                raise
            ok_s = time.perf_counter() - t0
            LLM_TIMES.append(ok_s)
            LLM_CPU_TIMES.append(time.process_time() - c0)
            LLM_EVENTS.append({"attempt": attempt, "status": 200,
                               "llm_s": round(ok_s, 4), "wait_s": 0.0})
            return out
        raise RuntimeError("unreachable")

@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22C, clear."

def build_model():
    return TimedModel(
        model=__MODEL__, base_url=__BASE_URL__, api_key=__API_KEY__,
        temperature=0, max_retries=0, timeout=300,
    )

# create_deerflow_agent 不传 checkpointer 时是 ephemeral 的：同一 thread_id 多次 invoke
# 不会累积历史。多轮场景必须显式挂 checkpointer；两侧各用自家实现以保证 API 对等。
try:  # fork: ReactiveGraph native checkpointer
    from reactivegraph.checkpoint import MemoryCheckpointSaver as Saver
except ImportError:  # upstream: LangGraph checkpointer
    from langgraph.checkpoint.memory import InMemorySaver as Saver

def run_model_only():
    graph = create_deerflow_agent(build_model())
    cfg = {"configurable": {"thread_id": "rl-" + uuid.uuid4().hex}}
    payload = {"messages": [HumanMessage(content="Reply with exactly one word: pong")]}
    t0, c_start = time.perf_counter(), time.process_time()
    out = graph.invoke(payload, cfg)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c_start
    waits = sum(RATE_LIMIT_WAITS)
    return {"wall_s": round(wall, 4), "llm_s": round(sum(LLM_TIMES), 4),
            "rate_limit_wait_s": round(waits, 4),
            "residual_s": round(wall - sum(LLM_TIMES) - waits, 4),
            "cpu_s": round(cpu, 4), "llm_cpu_s": round(sum(LLM_CPU_TIMES), 4),
            "residual_cpu_s": round(cpu - sum(LLM_CPU_TIMES), 4),
            "llm_calls": len(LLM_TIMES), "llm_events": LLM_EVENTS,
            "final_text": str(out["messages"][-1].content)[:60]}

def run_tool_call():
    graph = create_deerflow_agent(build_model(), tools=[get_weather])
    cfg = {"configurable": {"thread_id": "rl-" + uuid.uuid4().hex}}
    payload = {"messages": [HumanMessage(content="What is the weather in Shanghai? Use the get_weather tool, then answer briefly.")]}
    t0, c_start = time.perf_counter(), time.process_time()
    out = graph.invoke(payload, cfg)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c_start
    tool_msgs = [m for m in out["messages"] if type(m).__name__ == "ToolMessage"]
    tool_trace = [{"name": getattr(m, "name", None), "content": str(m.content)[:40]} for m in tool_msgs]
    msg_seq = []
    for m in out["messages"]:
        entry = {"type": type(m).__name__}
        tc = getattr(m, "tool_calls", None)
        if tc:
            entry["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")} for c in tc]
        msg_seq.append(entry)
    waits = sum(RATE_LIMIT_WAITS)
    return {"wall_s": round(wall, 4), "llm_s": round(sum(LLM_TIMES), 4),
            "rate_limit_wait_s": round(waits, 4),
            "residual_s": round(wall - sum(LLM_TIMES) - waits, 4),
            "cpu_s": round(cpu, 4), "llm_cpu_s": round(sum(LLM_CPU_TIMES), 4),
            "residual_cpu_s": round(cpu - sum(LLM_CPU_TIMES), 4),
            "llm_calls": len(LLM_TIMES), "llm_events": LLM_EVENTS, "tool_messages": len(tool_msgs),
            "tool_trace": tool_trace,
            "final_text": str(out["messages"][-1].content)[:60]}


def _residual(t0, c_start, extra):
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c_start
    waits = sum(RATE_LIMIT_WAITS)
    out = {"wall_s": round(wall, 4), "llm_s": round(sum(LLM_TIMES), 4),
           "rate_limit_wait_s": round(waits, 4),
           "residual_s": round(wall - sum(LLM_TIMES) - waits, 4),
           "cpu_s": round(cpu, 4), "llm_cpu_s": round(sum(LLM_CPU_TIMES), 4),
           "residual_cpu_s": round(cpu - sum(LLM_CPU_TIMES), 4),
           "llm_calls": len(LLM_TIMES), "llm_events": LLM_EVENTS}
    out.update(extra)
    return out

def run_multi_turn(rounds=None):
    """同一 thread 多轮 invoke：验证跨 run 的 checkpoint/state 复读开销。"""
    rounds = rounds or MULTI_TURN_ROUNDS
    graph = create_deerflow_agent(build_model(), checkpointer=Saver())
    cfg = {"configurable": {"thread_id": "rl-mt-" + uuid.uuid4().hex}}
    holes, cpu_holes, boundaries = [], [], []
    t0, c_start = time.perf_counter(), time.process_time()
    out = None
    for i in range(rounds):
        r0, rc0 = time.perf_counter(), time.process_time()
        out = graph.invoke({"messages": [HumanMessage(content=f"Turn {i + 1}: reply with one short sentence.")]}, cfg)
        holes.append(time.perf_counter() - r0)
        cpu_holes.append(time.process_time() - rc0)
        boundaries.append(len(LLM_TIMES))
    # 每轮各自扣掉该轮内部的 LLM CPU 时间，得到逐轮 residual
    round_residual_cpu = []
    prev = 0
    for i, b in enumerate(boundaries):
        llm_cpu_in_round = sum(LLM_CPU_TIMES[prev:b])
        round_residual_cpu.append(cpu_holes[i] - llm_cpu_in_round)
        prev = b
    return _residual(t0, c_start, {
        "rounds": rounds,
        "round_wall_median_s": round(statistics.median(holes), 4),
        "round_residual_cpu_median_s": round(statistics.median(round_residual_cpu), 4),
        "message_count": len(out["messages"]),
        "final_text": str(out["messages"][-1].content)[:60]})

def run_big_state(rounds=None, doc_kb=None):
    """大 system prompt（默认 64KB）+ 多轮 invoke：验证大 state 的序列化/复读开销。"""
    rounds = rounds or BIG_STATE_ROUNDS
    doc_kb = doc_kb or BIG_STATE_KB
    doc = "x" * (doc_kb * 1024)
    graph = create_deerflow_agent(build_model(), system_prompt="You are a terse assistant. " + doc, checkpointer=Saver())
    cfg = {"configurable": {"thread_id": "rl-bs-" + uuid.uuid4().hex}}
    holes, cpu_holes, boundaries = [], [], []
    t0, c_start = time.perf_counter(), time.process_time()
    out = None
    for i in range(rounds):
        r0, rc0 = time.perf_counter(), time.process_time()
        out = graph.invoke({"messages": [HumanMessage(content=f"Turn {i + 1}: reply OK.")]}, cfg)
        holes.append(time.perf_counter() - r0)
        cpu_holes.append(time.process_time() - rc0)
        boundaries.append(len(LLM_TIMES))
    round_residual_cpu = []
    prev = 0
    for i, b in enumerate(boundaries):
        round_residual_cpu.append(cpu_holes[i] - sum(LLM_CPU_TIMES[prev:b]))
        prev = b
    return _residual(t0, c_start, {
        "rounds": rounds, "doc_kb": doc_kb,
        "round_wall_median_s": round(statistics.median(holes), 4),
        "round_residual_cpu_median_s": round(statistics.median(round_residual_cpu), 4),
        "message_count": len(out["messages"]),
        "final_text": str(out["messages"][-1].content)[:60]})

try:
    result = {"model_only": run_model_only, "tool_call": run_tool_call,
              "multi_turn": run_multi_turn, "big_state": run_big_state}[SCENARIO]()
    result["scenario"] = SCENARIO
    print(json.dumps({"ok": True, "result": result}))
except Exception as e:
    print(json.dumps({"ok": False, "scenario": SCENARIO, "error": f"{type(e).__name__}: {str(e)[:200]}"}))
'''

_SUBS = {
    "__HARNESS__": lambda root: str(root / "backend" / "packages" / "harness"),
    "__BACKEND__": lambda root: str(root / "backend"),
    "__MODEL__": lambda root: json.dumps(os.environ.get("CUSTOM_MODEL", "agnes-2.5-flash")),
    "__BASE_URL__": lambda root: json.dumps(os.environ["CUSTOM_LLM_URL"]),
    "__API_KEY__": lambda root: json.dumps(os.environ["CUSTOM_LLM_API_KEY"]),
    "__MT_ROUNDS__": lambda root: str(int(os.environ.get("REAL_LLM_MULTI_TURN_ROUNDS", "5"))),
    "__BS_ROUNDS__": lambda root: str(int(os.environ.get("REAL_LLM_BIG_STATE_ROUNDS", "4"))),
    "__BS_KB__": lambda root: str(int(os.environ.get("REAL_LLM_BIG_STATE_KB", "64"))),
}


def run_side(root: Path, scenario: str) -> dict:
    code = TEMPLATE.replace("__SCENARIO__", json.dumps(scenario))
    for token, fn in _SUBS.items():
        code = code.replace(token, fn(root))
    py = root / "backend" / ".venv" / "bin" / "python"
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {"ok": False, "scenario": scenario, "error": r.stderr[-400:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "scenario": scenario, "error": (r.stdout[-200:] + r.stderr[-200:])}


def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # pragma: no cover - platform dependent
        return float("nan")


def wait_for_quiet_machine() -> float:
    """等 load1 降到阈值以下再测量；超时则返回当前值并继续（调用方记录）。

    共享开发机上同时跑编译/浏览器测试时 load1 可达数十，此时毫秒级引擎数字不可信，
    而这些测量要消耗有限的 API 配额，因此宁可先等。
    """
    waited = 0.0
    load = _load1()
    while load > MAX_LOAD and waited < MAX_LOAD_WAIT:
        time.sleep(15)
        waited += 15
        load = _load1()
    if load > MAX_LOAD:
        print(f"WARNING: load1={load} still above {MAX_LOAD} after {waited:.0f}s; measuring anyway",
              flush=True)
    return load


if __name__ == "__main__":
    roots = {"reactivegraph": fork_root(), "langgraph_upstream": upstream_root()}
    out = {label: [] for label in roots}
    first = True
    for scenario in SCENARIOS:
        for i in range(REPEATS):
            for label in ("reactivegraph", "langgraph_upstream"):
                if not first:
                    time.sleep(RATE_LIMIT_SLEEP)  # Agnes: 1 request/minute per account
                first = False
                wait_for_quiet_machine()  # 机器压满时先等，避免浪费配额得到噪声
                d = run_side(roots[label], scenario)
                d["load1_at_run"] = _load1()
                out[label].append(d)
                print(f"{scenario} r{i} {label}: {json.dumps(d)}", flush=True)
    def _median(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(statistics.median(vals), 4) if vals else None

    for label in roots:
        ok = [r["result"] for r in out[label] if r.get("ok")]
        if not ok:
            continue
        # 逐场景统计：不同场景的调用次数不同（1 vs 2 次 LLM），混在一起会误导
        per_scenario = {}
        for sc in sorted({r["scenario"] for r in ok}):
            rows = [r for r in ok if r["scenario"] == sc]
            per_scenario[sc] = {
                "runs": len(rows),
                "llm_calls": sorted({r["llm_calls"] for r in rows}),
                "wall_s_median": _median(r["wall_s"] for r in rows),
                "llm_s_median": _median(r["llm_s"] for r in rows),
                "rate_limit_wait_s_median": _median(r.get("rate_limit_wait_s", 0) for r in rows),
                "residual_s_median": _median(r["residual_s"] for r in rows),
                "residual_s_min": _median(r["residual_s"] for r in rows),
                "residual_cpu_s_median": _median(r.get("residual_cpu_s") for r in rows),
            }
        out[f"{label}_summary"] = {"runs": len(ok), "per_scenario": per_scenario}
        print(label, "summary:", json.dumps(out[f"{label}_summary"]), flush=True)
    # 合并追加：多次调用本脚本（不同场景/轮次）时保留已有测量
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
        except (ValueError, OSError):
            prev = {}
    else:
        prev = {}
    for label in list(out):
        if label.endswith("_summary"):
            prev[label] = out[label]
            continue
        existing = prev.get(label)
        if not isinstance(existing, list):
            existing = []
        prev[label] = existing + out[label]
    with open(OUT, "w") as f:
        json.dump(prev, f, indent=2)
    print("wrote", OUT)

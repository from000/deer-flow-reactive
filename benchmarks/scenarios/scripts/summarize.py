"""从原始 JSON 计算报告数字（median / 比值 / 区间），避免手抄。

用法：python3 summarise.py —— 只读 benchmarks/scenarios/*.json，不重新测量。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    with open(ROOT / name) as f:
        return json.load(f)


def med(records, *keys):
    vals = []
    for r in records:
        v = r
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        if isinstance(v, (int, float)):
            vals.append(v)
    return round(statistics.median(vals), 3) if vals else None


def rng(records, *keys):
    vals = []
    for r in records:
        v = r
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        if isinstance(v, (int, float)):
            vals.append(v)
    if not vals:
        return None
    return (round(min(vals), 1), round(max(vals), 1))


def ratios(out, a, b, *keys):
    ra, rb = med(out[a], *keys), med(out[b], *keys)
    return None if not ra or not rb else round(rb / ra, 2)


if __name__ == "__main__":
    triple = load("triple_results.json")
    more = load("more_results.json")
    conc = load("concurrency_results.json")
    loop = load("loop_detection_results.json")
    mcc = load("model_call_count_results.json")

    print("=== 端到端（median of 3 runs, 50 samples) ===")
    for sc in ("model_only", "tool_call"):
        ra = med(triple["reactivegraph"], sc)
        rb = med(triple["langgraph_upstream"], sc)
        print(f"  {sc:11s} fork={ra:>7} ms  upstream={rb:>7} ms  ratio={rb/ra:>5.2f}x")

    print("=== multi_turn / big_state (median of 3 rounds) ===")
    for sc in ("multi_turn", "big_state"):
        ra = med(more["reactivegraph"], sc, "median_ms")
        rb = med(more["langgraph_upstream"], sc, "median_ms")
        print(f"  {sc:11s} fork={ra:>7} ms  upstream={rb:>7} ms  ratio={rb/ra:>5.2f}x")

    print("=== concurrency (64 sessions / 8 workers, median of 3 paired rounds) ===")
    ra = med(conc["reactivegraph"], "throughput_sessions_per_s")
    rb = med(conc["langgraph_upstream"], "throughput_sessions_per_s")
    # 吞吐越高越好，比值取 fork / upstream
    print(f"  throughput fork={ra}  upstream={rb}  ratio={ra/rb:.2f}x")
    print(f"  fork range={rng(conc['reactivegraph'],'throughput_sessions_per_s')}"
          f"  upstream range={rng(conc['langgraph_upstream'],'throughput_sessions_per_s')}")

    print("=== loop_detection ===")
    print("  fork    ", loop["reactivegraph"]["message_counts"], "model_calls=", loop["reactivegraph"]["total_model_calls"])
    print("  upstream", loop["langgraph_upstream"]["message_counts"], "model_calls=", loop["langgraph_upstream"]["total_model_calls"])

    print("=== model_call_count (30 identical turns) ===")
    print("  fork    ", mcc["reactivegraph"])
    print("  upstream", mcc["langgraph_upstream"])

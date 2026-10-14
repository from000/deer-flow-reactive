"""共享：从环境变量解析 fork / 上游 checkout 路径。"""
from __future__ import annotations

import os
from pathlib import Path


def fork_root() -> Path:
    p = Path(os.environ.get("FORK_ROOT", "")).expanduser()
    if not (p / "backend").is_dir():
        raise SystemExit("FORK_ROOT 未设置或无效（需指向 deer-flow-reactive checkout）")
    return p


def upstream_root() -> Path:
    p = Path(os.environ.get("UPSTREAM_ROOT", "")).expanduser()
    if not (p / "backend").is_dir():
        raise SystemExit("UPSTREAM_ROOT 未设置或无效（需指向 bytedance/deer-flow checkout）")
    return p


def run_code(root: Path, template: str):
    """在上游/fork 的 venv 里执行内联脚本，返回末尾一行 JSON。"""
    import json
    import subprocess

    py = root / "backend" / ".venv" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"未找到 venv: {py}（先在 {root}/backend 跑 uv sync --group dev）")
    code = (template
            .replace("HARNESS", str(root / "backend" / "packages" / "harness"))
            .replace("BACKEND", str(root / "backend")))
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return {"error": r.stderr[-800:]}
    return json.loads(r.stdout.strip().splitlines()[-1])

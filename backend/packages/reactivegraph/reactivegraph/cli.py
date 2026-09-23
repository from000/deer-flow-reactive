"""`reactivegraph new <app>` — scaffold a minimal runnable project (DX).

Generates `<app>/` with a runnable `main.py` (fallback mode, no Driver
needed), a `pyproject.toml` declaring the `reactivegraph` dependency, and a
README with run instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MAIN_TEMPLATE = '''"""最小可跑 ReactiveGraph 应用(脚手架生成)。

运行:
    uv run python main.py
"""

from reactivegraph import GraphBuilder, ReactiveGraph


def build(b: GraphBuilder) -> None:
    b.task("greet", kind="pure", fn=lambda x: {"msg": f"hi {x.get('name')}"}).on("visit", "greet")


if __name__ == "__main__":
    g = ReactiveGraph.build(build)
    out = g.invoke("visit", {"name": "Ada"})
    print("state:", out)
'''

PYPROJECT_TEMPLATE = '''[project]
name = "{app}"
version = "0.1.0"
description = "Scaffolded ReactiveGraph application"
requires-python = ">=3.10"
dependencies = ["reactivegraph"]
'''

README_TEMPLATE = """# {app}

由 `reactivegraph new` 生成的 ReactiveGraph 最小工程。

```bash
uv run python main.py        # 输出 state: {{'msg': 'hi Ada'}}
```

继续学习:https://github.com/from000/reactive-graph 的 `docs/tutorials/`
(从"为什么反应式"到 1000 节点应用,每篇配可跑代码)。
"""


def scaffold(app: str, dest: Path) -> None:
    root = dest / app
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"目标已存在且非空: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(MAIN_TEMPLATE)
    (root / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.format(app=app))
    (root / "README.md").write_text(README_TEMPLATE.format(app=app))
    print(f"created {app}/  (run: cd {app} && uv run python main.py)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reactivegraph", description="ReactiveGraph 脚手架")
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="生成一个新的最小可跑工程")
    new.add_argument("app", help="应用名(同时是目录名)")
    args = parser.parse_args(argv)
    if args.command == "new":
        scaffold(args.app, Path.cwd())
    return 0


if __name__ == "__main__":
    sys.exit(main())
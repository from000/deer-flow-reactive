"""DeerFlow's own agent surface must not *require* LangGraph to be importable.

The five ``create_agent`` call sites now build their graphs with
``reactivegraph.create_agent``. That is necessary but not sufficient: the
middleware classes DeerFlow hands to the engine are still declared as
subclasses of ``langchain.agents.middleware.AgentMiddleware``. Importing that
module executes ``langchain/agents/__init__.py`` -> ``langchain/agents/factory.py``
-> ``from langgraph._internal._runnable import RunnableCallable``, which pulls
in the entire LangGraph package (measured: 132 modules, from a single line).

So today ``import deerflow.agents.lead_agent.agent`` raises ImportError on a
machine without LangGraph installed, even though the engine it runs on is ours.
These tests pin the end state: the base class comes from ReactiveGraph, and the
harness imports cleanly with ``langgraph`` made unimportable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "packages" / "harness"


def _run_without_langgraph(module: str) -> subprocess.CompletedProcess[str]:
    """Import *module* in a fresh interpreter where ``langgraph`` cannot load."""
    code = textwrap.dedent(
        f"""
        import importlib.abc, sys

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "langgraph":
                    raise ImportError("langgraph is not installed")
                return None

        sys.meta_path.insert(0, _Blocker())
        import {module} as m
        print(m.__name__)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_HARNESS),
    )


@pytest.mark.parametrize(
    "module",
    [
        "deerflow.agents.features",
        "deerflow.agents.middlewares.thread_data_middleware",
        "deerflow.agents.middlewares.loop_detection_middleware",
        "deerflow.agents.middlewares.tool_error_handling_middleware",
        # The real product path, not a leaf module: this transitively imports the
        # middleware stack the lead agent actually builds.
        "deerflow.agents.lead_agent.agent",
    ],
)
def test_harness_agent_modules_import_without_langgraph(module: str) -> None:
    result = _run_without_langgraph(module)
    assert result.returncode == 0, (
        f"{module} still needs langgraph:\n{result.stderr[-2000:]}"
    )


def test_middleware_base_is_reactivegraph() -> None:
    """DeerFlow middleware must inherit *our* base, not LangChain's."""
    from reactivegraph.middleware import AgentMiddleware as ReactiveBase

    from deerflow.agents.features import RuntimeFeatures  # noqa: F401
    from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware

    assert issubclass(ThreadDataMiddleware, ReactiveBase)

"""Function/decorator workflows, Python side (Task 8).

Mirror of packages/driver/src/functional.ts: `functask`/`entrypoint` let plain
Python functions become graph tasks. Task persistence (fingerprint skip,
idempotency receipt) is owned by the Driver scheduler; here we provide the
declarative building blocks and the graph-building convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from reactivegraph.graph import GraphBuilder


@dataclass(frozen=True)
class FunctionTask:
    id: str
    fn: Callable[[Any], Any]
    kind: str = "pure"
    on: tuple[str, ...] = ()
    timeout_ms: int = 0


@dataclass(frozen=True)
class EntrypointPlan:
    graph_id: str
    task_ids: tuple[str, ...]
    routes: tuple[tuple[str, str], ...] = ()


@overload
def functask(
    fn: Callable[[Any], Any],
    *,
    task_id: str | None = None,
    kind: str = "pure",
    on: tuple[str, ...] = (),
) -> FunctionTask: ...


@overload
def functask(
    fn: None = None,
    *,
    task_id: str | None = None,
    kind: str = "pure",
    on: tuple[str, ...] = (),
) -> Callable[[Callable[[Any], Any]], FunctionTask]: ...


def functask(
    fn: Callable[[Any], Any] | None = None,
    *,
    task_id: str | None = None,
    kind: str = "pure",
    on: tuple[str, ...] = (),
) -> Callable[[Callable[[Any], Any]], FunctionTask] | FunctionTask:
    """Decorator turning a plain function into a FunctionTask.

    Usage::

        @functask(kind="effect", on=("visit",))
        def greet(input_):
            return {"msg": f"hi {input_['name']}"}
    """

    def wrap(f: Callable[[Any], Any]) -> FunctionTask:
        return FunctionTask(id=task_id or f.__name__, fn=f, kind=kind, on=on)

    if fn is None:
        return wrap
    return wrap(fn)


def entrypoint(
    graph_id: str,
    tasks: list[FunctionTask],
    routes: list[tuple[str, str]] | None = None,
) -> EntrypointPlan:
    """Plan a graph from task defs (build via build_function_graph)."""
    return EntrypointPlan(
        graph_id=graph_id,
        task_ids=tuple(t.id for t in tasks),
        routes=tuple(routes or ()),
    )


def build_function_graph(
    graph_id: str,
    tasks: list[FunctionTask],
    routes: list[tuple[str, str]] | None = None,
) -> GraphBuilder:
    """Build a GraphBuilder from FunctionTask defs."""
    builder = GraphBuilder(graph_id)
    for t in tasks:
        builder.task(t.id, kind=t.kind, fn=t.fn, on=t.on, timeout_ms=t.timeout_ms)
    for event, task_id in (routes or []):
        builder.on(event, task_id)
    return builder


__all__ = ("EntrypointPlan", "FunctionTask", "build_function_graph", "entrypoint", "functask")
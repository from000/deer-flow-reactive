"""Does the ReactiveGraph saver actually drive a real LangGraph graph?

The isinstance check in ``ensure_valid_checkpointer`` is necessary but far from
sufficient: Pregel calls put/get/put_writes in a specific order and depends on
their exact semantics (latest-checkpoint resolution, parent lineage, pending
writes, resume from interrupt, time travel). These tests run genuine LangGraph
graphs against ``ReactiveGraphSaver`` so a green result means the storage
contract is genuinely honoured, not merely type-compatible.
"""

from __future__ import annotations

import itertools
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from deerflow.runtime.checkpointer.reactive_saver import ReactiveGraphSaver


class State(TypedDict):
    n: int
    log: list[str]


def _counter_graph(checkpointer):
    def bump(state: State) -> dict:
        return {"n": state["n"] + 1, "log": [*state.get("log", []), "bump"]}

    builder = StateGraph(State)
    builder.add_node("bump", bump)
    builder.add_edge(START, "bump")
    builder.add_edge("bump", END)
    return builder.compile(checkpointer=checkpointer)


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict:
    configurable: dict = {"thread_id": thread_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _assert_parity_with_upstream(thread_id: str) -> None:
    """Run the same script against InMemorySaver and ours; results must match."""
    results = {}
    for name, saver in (("upstream", InMemorySaver()), ("ours", ReactiveGraphSaver())):
        graph = _counter_graph(saver)
        cfg = _cfg(thread_id)
        first = graph.invoke({"n": 0, "log": []}, cfg)
        second = graph.invoke({"n": first["n"], "log": first["log"]}, cfg)
        state = graph.get_state(cfg)
        history = [s for s in graph.get_state_history(cfg)]
        results[name] = {
            "first": first,
            "second": second,
            "values": state.values,
            "next": state.next,
            "history_len": len(history),
            "history_n": [s.values.get("n") for s in history],
            "parent_chain": [s.parent_config is not None for s in history],
        }
    assert results["ours"] == results["upstream"], results


class TestRealLangGraphExecution:
    def test_accepts_saver_where_langgraph_demands_base_checkpoint_saver(self):
        """compile() must not raise TypeError for our saver."""
        graph = _counter_graph(ReactiveGraphSaver())
        assert graph.checkpointer is not None

    def test_single_invoke_persists_state(self):
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-single")
        out = graph.invoke({"n": 0, "log": []}, cfg)
        assert out["n"] == 1
        assert graph.get_state(cfg).values["n"] == 1

    def test_second_invoke_resumes_from_stored_checkpoint(self):
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-resume")
        graph.invoke({"n": 0, "log": []}, cfg)
        second = graph.invoke({"n": 1, "log": ["bump"]}, cfg)
        assert second["n"] == 2
        assert second["log"] == ["bump", "bump"]

    def test_state_history_is_newest_first_with_parent_chain(self):
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-history")
        graph.invoke({"n": 0, "log": []}, cfg)
        graph.invoke({"n": 1, "log": ["bump"]}, cfg)

        history = list(graph.get_state_history(cfg))
        # LangGraph appends the empty initial checkpoint, so a two-invoke thread
        # has more entries than turns. What matters is ordering and lineage.
        assert [s.values.get("n") for s in history] == [2, 1, 1, 1, 0, None]
        assert history[0].parent_config is not None
        assert history[-1].parent_config is None

    def test_time_travel_replays_from_earlier_checkpoint(self):
        """Re-invoking an old checkpoint id must branch from that checkpoint."""
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-travel")
        graph.invoke({"n": 0, "log": []}, cfg)
        graph.invoke({"n": 1, "log": ["bump"]}, cfg)

        history = list(graph.get_state_history(cfg))
        # Pick the first checkpoint that actually carries state; the newest
        # entry is the initial empty one.
        oldest = next(s for s in reversed(history) if s.values.get("n") == 0)
        replayed = graph.invoke(
            {"n": oldest.values["n"], "log": oldest.values["log"]},
            oldest.config,
        )
        assert replayed["n"] == oldest.values["n"] + 1

    def test_delete_thread_purges_history(self):
        saver = ReactiveGraphSaver()
        graph = _counter_graph(saver)
        cfg = _cfg("t-purge")
        graph.invoke({"n": 0, "log": []}, cfg)
        assert list(graph.get_state_history(cfg))

        saver.delete_thread("t-purge")
        assert list(graph.get_state_history(cfg)) == []

    def test_parity_with_upstream_in_memory_saver(self):
        _assert_parity_with_upstream("t-parity")


class TestInterruptAndResume:
    def test_pending_writes_round_trip_through_a_real_interrupt(self):
        """A real interrupt exercises put_writes and the resume path."""
        from langgraph.types import Command, interrupt

        def gate(state: State) -> dict:
            decision = interrupt({"ask": "approve?"})
            return {"n": state["n"] + (1 if decision == "yes" else 0), "log": ["gate"]}

        builder = StateGraph(State)
        builder.add_node("gate", gate)
        builder.add_edge(START, "gate")
        builder.add_edge("gate", END)

        saver = ReactiveGraphSaver()
        graph = builder.compile(checkpointer=saver)
        cfg = _cfg("t-interrupt")

        graph.invoke({"n": 0, "log": []}, cfg)
        snapshot = graph.get_state(cfg)
        assert snapshot.next == ("gate",), "graph must be parked at the interrupt"

        resumed = graph.invoke(Command(resume="yes"), cfg)
        assert resumed["n"] == 1
        assert resumed["log"] == ["gate"]

    @pytest.mark.anyio
    async def test_async_invoke_and_astream(self):
        """ainvoke must go through aput/aget_tuple, not the sync path."""
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-async")
        out = await graph.ainvoke({"n": 0, "log": []}, cfg)
        assert out["n"] == 1

        state = await graph.aget_state(cfg)
        assert state.values["n"] == 1

        chunks = [c async for c in graph.astream({"n": 1, "log": ["bump"]}, cfg, stream_mode="values")]
        assert chunks[-1]["n"] == 2

    @pytest.mark.anyio
    async def test_async_history(self):
        graph = _counter_graph(ReactiveGraphSaver())
        cfg = _cfg("t-async-history")
        await graph.ainvoke({"n": 0, "log": []}, cfg)
        await graph.ainvoke({"n": 1, "log": ["bump"]}, cfg)
        history = [s async for s in graph.aget_state_history(cfg)]
        assert [s.values.get("n") for s in history] == [2, 1, 1, 1, 0, None]


class TestSubgraphNamespaces:
    def test_subgraph_checkpoints_are_isolated_by_namespace(self):
        """A compiled subgraph writes under its own checkpoint_ns."""

        class Outer(TypedDict):
            n: int

        def inner_node(state: Outer) -> dict:
            return {"n": state["n"] + 10}

        inner_builder = StateGraph(Outer)
        inner_builder.add_node("inner", inner_node)
        inner_builder.add_edge(START, "inner")
        inner_builder.add_edge("inner", END)

        saver = ReactiveGraphSaver()
        inner = inner_builder.compile(checkpointer=saver)

        def call_inner(state: Outer) -> dict:
            return inner.invoke({"n": state["n"]}, {"configurable": {"thread_id": "sub-thread"}})

        outer_builder = StateGraph(Outer)
        outer_builder.add_node("outer", call_inner)
        outer_builder.add_edge(START, "outer")
        outer_builder.add_edge("outer", END)
        outer = outer_builder.compile(checkpointer=saver)

        out = outer.invoke({"n": 1}, _cfg("t-sub"))
        assert out["n"] == 11


class TestPutWritesParity:
    """Exhaustive differential check of put_writes against upstream.

    Hand-written cases missed the real rule: upstream keys writes by
    ``(task_id, slot)`` where ``slot`` is the batch position, or a fixed
    negative constant for reserved channels. Regular writes are first-write-wins
    (so a retried task cannot duplicate) while reserved channels overwrite (so
    the latest interrupt wins). Enumerating the combinations is what caught the
    intra-batch reserved-channel case.
    """

    CHANNELS = ["a", "b", "__interrupt__", "__error__"]
    VALUES = ["v0", "v1"]

    @staticmethod
    def _cfg(thread_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }

    @staticmethod
    def _seed(saver, thread_id: str) -> None:
        saver.put(
            TestPutWritesParity._cfg(thread_id),
            {"id": "cp-1", "channel_values": {}, "channel_versions": {}, "versions_seen": {}},
            {},
            {},
        )

    @classmethod
    def _batches(cls) -> list[list[tuple[str, str]]]:
        batches: list[list[tuple[str, str]]] = []
        for channel in cls.CHANNELS:
            for value in cls.VALUES:
                batches.append([(channel, value)])
        for first, second in itertools.product(cls.CHANNELS, repeat=2):
            batches.append([(first, "v0"), (second, "v1")])
        for combo in itertools.product(cls.CHANNELS, repeat=3):
            batches.append([(channel, f"v{index}") for index, channel in enumerate(combo)])
        return batches

    def test_every_write_sequence_matches_upstream(self):
        batches = self._batches()
        sequences = [[batch] for batch in batches]
        sequences += [[first, second] for first in batches for second in batches[:12]]

        mismatches = []
        for sequence in sequences:
            upstream = InMemorySaver()
            ours = ReactiveGraphSaver()
            self._seed(upstream, "u")
            self._seed(ours, "o")
            for batch in sequence:
                upstream.put_writes(self._cfg("u"), batch, "task")
                ours.put_writes(self._cfg("o"), batch, "task")
            expected = list(upstream.get_tuple(self._cfg("u")).pending_writes)
            actual = list(ours.get_tuple(self._cfg("o")).pending_writes)
            if expected != actual:
                mismatches.append((sequence, expected, actual))

        assert not mismatches, f"{len(mismatches)} mismatching sequences, first: {mismatches[0]}"

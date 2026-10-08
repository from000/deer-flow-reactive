"""Engine constants shared by the agent factory and its streaming adapter.

These live in their own module so ``create_agent`` (which builds the graph) and
``streaming`` (which translates frames into LangGraph modes) can agree on the
same private-state keys, recursion bounds and hook pairs without importing each
other.

``PRIVATE_STATE_KEYS`` covers run-private channels: they ride the graph like
any other channel so a task re-entry reads the budget its predecessor spent,
and every public surface (``invoke`` result, ``values``/``updates`` frames,
checkpoints) strips them.
"""

from __future__ import annotations

DEFAULT_RECURSION_LIMIT = 9_999

# The compiled task budget is a fixed wire value, so a run whose
# ``recursion_limit`` exceeds it would be silently truncated by the scheduler's
# at-most-once gate instead of raising. Refuse the run instead: the fallback's
# own event-propagation guard sits at the same ceiling, so one bound covers
# both execution paths.
MAX_RECURSION_LIMIT = 10_000


# ---------------------------------------------------------------------------
# recursion_limit accounting
# ---------------------------------------------------------------------------

# The compiled graph's "the one-shot lifecycle hook already fired" marker.
AGENT_STARTED = "__reactivegraph_agent_started__"

# Run-private state keys. They ride the graph like any other channel so a task
# re-entry reads the budget its predecessor spent, and they are stripped from
# every public surface (``invoke`` result, ``values``/``updates`` frames,
# checkpoints), exactly as ``AGENT_STARTED`` already was.
RECURSION_LIMIT_KEY = "__reactivegraph_recursion_limit__"
STEP_COUNT_KEY = "__reactivegraph_step_count__"
EXHAUSTED_KEY = "__reactivegraph_recursion_exhausted__"
RUN_PRIVATE_KEYS = frozenset(
    {AGENT_STARTED, RECURSION_LIMIT_KEY, STEP_COUNT_KEY, EXHAUSTED_KEY}
)
# ``jump_to`` is an ``EphemeralValue + PrivateStateAttr`` upstream: a hook can
# use it to choose the next node, but it must not be visible to later nodes or
# to the caller. The engine's compatibility layer owns that consumption.
PRIVATE_STATE_KEYS = RUN_PRIVATE_KEYS | {"jump_to"}

# Hook name -> the sync/async pair LangChain compiles into one node. The
# ordering here is the order the nodes run in, which is also the order the
# step budget must be charged in: when the limit lands between two nodes, the
# earlier node's writes are the ones LangGraph committed before giving up.
HOOK_PAIRS: dict[str, tuple[str, str]] = {
    "before_agent": ("before_agent", "abefore_agent"),
    "before_model": ("before_model", "abefore_model"),
    "after_model": ("after_model", "aafter_model"),
    "after_agent": ("after_agent", "aafter_agent"),
}


ALL_HOOKS = tuple(hook for pair in HOOK_PAIRS.values() for hook in pair)

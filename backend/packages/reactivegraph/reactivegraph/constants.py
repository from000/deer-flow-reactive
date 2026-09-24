"""Sentinel constants shared by graph builders and message reducers.

The values are part of the wire/checkpoint contract: they are written into
checkpoints, matched against serialized configs and compared by identity in hot
paths. They are therefore interned, exactly as upstream does.

``REMOVE_ALL_MESSAGES`` lives here rather than in a ``graph.message`` module
because this package exposes :mod:`reactivegraph.graph` as a module, not a
package; the sentinel is the only part of that module the engine needs.
"""

from __future__ import annotations

import sys

__all__ = (
    "END",
    "REMOVE_ALL_MESSAGES",
    "START",
    "TAG_HIDDEN",
    "TAG_NOSTREAM",
)

START = sys.intern("__start__")
"""The (virtual) entry node of a graph."""

END = sys.intern("__end__")
"""The (virtual) terminal node of a graph."""

TAG_NOSTREAM = sys.intern("nostream")
"""Tag that disables streaming for a node or chat model call."""

TAG_HIDDEN = sys.intern("langsmith:hidden")
"""Tag that hides a node/edge from tracing and streaming surfaces."""

REMOVE_ALL_MESSAGES = sys.intern("__remove_all__")
"""``RemoveMessage`` id that clears the accumulated message list."""

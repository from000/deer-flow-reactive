"""LangGraph-compatible module path for the binary-operator channel.

DeerFlow's ``deerflow.checkpoint_patches`` imports
``BinaryOperatorAggregate`` from ``langgraph.channels.binop`` to patch its
``update`` method at import time. The engine's own channel already implements
the patched semantics, so the module exists only so that import can be
rewritten mechanically to ``reactivegraph.channels.binop``.
"""

from reactivegraph.channels import BinaryOperatorAggregate

__all__ = ("BinaryOperatorAggregate",)

"""Warning categories for deprecated ReactiveGraph APIs.

Deprecations get a dedicated :class:`DeprecationWarning` subclass so callers can
turn *our* deprecations into errors without silencing unrelated ones. The
message rendering mirrors the upstream categories (``message. Deprecated in
VX.Y to be removed in VZ.W.``) so deprecation logs stay comparable after the
engine swap.
"""

from __future__ import annotations

__all__ = (
    "ReactiveGraphDeprecationWarning",
    "ReactiveGraphDeprecatedSinceV10",
)


class ReactiveGraphDeprecationWarning(DeprecationWarning):
    """Base category for deprecated ReactiveGraph APIs."""

    message: str
    since: tuple[int, int]
    expected_removal: tuple[int, int]

    def __init__(
        self,
        message: str,
        *args: object,
        since: tuple[int, int],
        expected_removal: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(message, *args)
        self.message = message.rstrip(".")
        self.since = since
        self.expected_removal = (
            expected_removal if expected_removal is not None else (since[0] + 1, 0)
        )

    def __str__(self) -> str:
        return (
            f"{self.message}. Deprecated in V{self.since[0]}.{self.since[1]}"
            f" to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}."
        )


class ReactiveGraphDeprecatedSinceV10(ReactiveGraphDeprecationWarning):
    """Deprecated since the 1.0 API surface."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(1, 0), expected_removal=(2, 0))

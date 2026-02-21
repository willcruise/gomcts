"""Deprecated shim for backwards compatibility.

The canonical implementation lives in `gomcts.core.rules`.
"""

from gomcts.core.rules import (  # noqa: F401
    neighbors,
    collect_group_and_liberties,
    simulate_place_and_capture,
    is_suicide_after,
    simple_ko_forbidden,
    positional_superko_forbidden,
    capture_aware_score,
    final_margin,
)

__all__ = [
    "neighbors",
    "collect_group_and_liberties",
    "simulate_place_and_capture",
    "is_suicide_after",
    "simple_ko_forbidden",
    "positional_superko_forbidden",
    "capture_aware_score",
    "final_margin",
]


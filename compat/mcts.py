"""Deprecated shim for backwards compatibility.

The canonical implementation lives in `gomcts.core.mcts`.
"""

from gomcts.core.mcts import ScoreAwareMCTS, temperature_schedule

__all__ = ["ScoreAwareMCTS", "temperature_schedule"]


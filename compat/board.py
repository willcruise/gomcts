"""Deprecated shim for backwards compatibility.

The canonical implementation lives in `gomcts.core.board`.
"""

from gomcts.core.board import Board, Board9x9

__all__ = ["Board", "Board9x9"]


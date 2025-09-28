from __future__ import annotations

from typing import Generator, Iterable, List, Optional, Set, Tuple

import numpy as np


def neighbors(size: int, r: int, c: int) -> Generator[Tuple[int, int], None, None]:
    """Yield board-adjacent coordinates within bounds (up, down, left, right)."""
    if r > 0:
        yield r - 1, c
    if r < size - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < size - 1:
        yield r, c + 1


def collect_group_and_liberties(grid: np.ndarray, r: int, c: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Given a grid and a starting stone at (r, c), return the connected group set
    and its liberties set.
    """
    color = int(grid[r, c])
    assert color != 0
    size = int(grid.shape[0])
    stack: List[Tuple[int, int]] = [(r, c)]
    visited: Set[Tuple[int, int]] = {(r, c)}
    libs: Set[Tuple[int, int]] = set()
    while stack:
        rr, cc = stack.pop()
        for nr, nc in neighbors(size, rr, cc):
            v = int(grid[nr, nc])
            if v == 0:
                libs.add((nr, nc))
            elif v == color and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return visited, libs


def simulate_place_and_capture(grid: np.ndarray,
                               r: int,
                               c: int,
                               color: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]], bool]:
    """
    Simulate placing a stone of `color` at (r,c) on a COPY of grid.
    Returns (new_grid, captures, captured_any)
    - new_grid: grid after removing captured opponent stones
    - captures: list of (rr, cc, captured_color)
    - captured_any: True if at least one opponent group was captured
    """
    assert int(color) in (-1, 1)
    N = int(grid.shape[0])
    new_grid = grid.copy()
    new_grid[r, c] = int(color)
    opp = -int(color)
    captures: List[Tuple[int, int, int]] = []
    captured_any = False
    for nr, nc in neighbors(N, r, c):
        if int(new_grid[nr, nc]) == opp:
            group, libs = collect_group_and_liberties(new_grid, nr, nc)
            if len(libs) == 0:
                captured_any = True
                for rr, cc in group:
                    captures.append((rr, cc, opp))
                    new_grid[rr, cc] = 0
    return new_grid, captures, captured_any


def is_suicide_after(grid: np.ndarray, r: int, c: int, color: int, captured_any: bool) -> bool:
    """
    After a hypothetical placement (and applying captures if any), return True
    if the placed stone's group has no liberties AND no captures occurred
    (classic suicide definition under rulesets that forbid suicide).
    """
    new_group, new_libs = collect_group_and_liberties(grid, r, c)
    return (not captured_any) and (len(new_libs) == 0)


def simple_ko_forbidden(prev_hash: bytes, new_hash: bytes, history: List[bytes]) -> bool:
    """Return True if simple ko forbids the move (immediate repetition)."""
    if len(history) >= 2:
        return new_hash == history[-2]
    return False


def positional_superko_forbidden(new_poshash: bytes, seen: Set[bytes]) -> bool:
    """Return True if positional superko forbids the move (position already seen)."""
    return new_poshash in seen



def capture_aware_score(grid: np.ndarray,
                        captures_black: int,
                        captures_white: int) -> float:
    """
    Return simple capture-aware score margin from Black's perspective.

    score = (black stones on board) - (white stones on board)
            + (stones captured by Black) - (stones captured by White)
    """
    black = int((grid == 1).sum())
    white = int((grid == -1).sum())
    margin = (black - white) + int(captures_black) - int(captures_white)
    return float(margin)


def final_margin(grid: np.ndarray,
                 captures_black: int,
                 captures_white: int,
                 use_capture_aware: bool = True,
                 komi: float = 0.0) -> float:
    """
    Compute final score margin from Black's perspective.

    - If use_capture_aware is True, use capture-aware proxy: (B-W) + (capsB - capsW).
    - Else, use simple stones-only difference: (B-W).
    - Always subtract komi from the margin (komi > 0 favors White).
    """
    if use_capture_aware:
        base = capture_aware_score(grid, captures_black, captures_white)
    else:
        black = int((grid == 1).sum())
        white = int((grid == -1).sum())
        base = float(black - white)
    return float(base) - float(komi)
"""MCTS builders/wrappers used across entrypoints.

This centralizes the wiring between:
- `Board` state transitions
- the policy/value network inference
- score-aware MCTS configuration
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from gomcts.core.board import Board
from gomcts.core.mcts import ScoreAwareMCTS
from gomcts.core import rules
from gomcts.neural.policy import (
    MLPPolicyValue,
    infer_policy_value,
    infer_policy_value_from_features_batch_torch,
)


# Defaults (kept here to avoid config-scatter / "magic numbers" in call sites)
DEFAULT_SCORE_WEIGHT: float = 0.35
DEFAULT_MAX_CHILDREN_PER_NODE: int = 16


def _current_player_fn(b: Board) -> int:
    # Map +1 black -> 0, -1 white -> 1
    return 0 if int(b.turn) == 1 else 1


def _score_estimator_fn(b: Board) -> float:
    return float(
        rules.capture_aware_score(
            b.grid,
            getattr(b, "captures_black", 0),
            getattr(b, "captures_white", 0),
        )
    )


def build_mcts_standard(
    net: MLPPolicyValue,
    size: int,
    c_puct: float = 2.0,
    dirichlet_alpha: Optional[float] = None,
    dirichlet_frac: float = 0.25,
    dirichlet_c0: float = 10.0,
    *,
    score_weight: float = DEFAULT_SCORE_WEIGHT,
    max_children_per_node: int = DEFAULT_MAX_CHILDREN_PER_NODE,
    enable_policy_value_cache: bool = True,
    enable_legal_moves_cache: bool = True,
) -> ScoreAwareMCTS:
    """Build a score-aware MCTS with optional caching and batched inference."""

    # ---- Legal moves callback (optional cache) ----
    _legal_cache: Dict[Tuple[bytes, Optional[bytes]], np.ndarray] = {}

    def legal_actions_fn(b: Board):
        if not enable_legal_moves_cache:
            return b.legal_moves()
        try:
            prev_hash = b._position_history[-1] if getattr(b, "_position_history", None) else None
            key = (b._hash_from(b.grid, b.turn), prev_hash)
        except (AttributeError, IndexError, TypeError):
            return b.legal_moves()
        cached = _legal_cache.get(key)
        if cached is not None:
            return cached
        res = b.legal_moves()
        _legal_cache[key] = res
        return res

    def next_state_fn(b: Board, action: int):
        nb = b.clone()
        nb.play(int(action))
        return nb

    def is_terminal_fn(b: Board):
        return b.is_terminal()

    # ---- Policy/value callback (optional transposition cache) ----
    _pv_cache: Dict[bytes, Tuple[np.ndarray, float]] = {}

    def policy_value_fn(b: Board):
        if not enable_policy_value_cache:
            return infer_policy_value(net, b)
        try:
            key = b._hash_from(b.grid, b.turn)
        except (AttributeError, TypeError):
            return infer_policy_value(net, b)
        cached = _pv_cache.get(key)
        if cached is not None:
            return cached
        pri, val = infer_policy_value(net, b)
        _pv_cache[key] = (pri, val)
        return pri, val

    num_actions = int(size) * int(size) + 1
    mcts = ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        policy_value_batch_fn=lambda feats_batch, A=num_actions: infer_policy_value_from_features_batch_torch(net, feats_batch, A),
        current_player_fn=_current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=float(score_weight),
        score_norm_scale=None,
        score_estimator_fn=_score_estimator_fn,
        use_inplace_simulation=True,
        max_children_per_node=int(max_children_per_node),
    )

    # Propagate optional batched inference tuning knobs from net (if user set them)
    bs = getattr(net, "_mcts_batch_size", None)
    if bs is not None:
        setattr(mcts, "batch_size", int(bs))
    fl_ms = getattr(net, "_mcts_flush_ms", None)
    if fl_ms is not None:
        setattr(mcts, "flush_timeout_ms", float(fl_ms))

    return mcts


def build_mcts_no_pass(
    net: MLPPolicyValue,
    size: int,
    c_puct: float = 2.0,
    dirichlet_alpha: Optional[float] = None,
    dirichlet_frac: float = 0.25,
    dirichlet_c0: float = 10.0,
) -> ScoreAwareMCTS:
    """Build MCTS that disallows PASS (useful for some toy settings)."""

    def legal_actions_fn(b: Board):
        legal = b.legal_moves()
        pass_idx = int(b.pass_index)
        return [int(a) for a in legal if int(a) != pass_idx]

    def next_state_fn(b: Board, action: int):
        nb = b.clone()
        nb.play(int(action))
        return nb

    def is_terminal_fn(b: Board):
        return b.is_terminal() if hasattr(b, "is_terminal") else bool((b.grid != 0).all())

    def policy_value_fn(b: Board):
        priors, value = infer_policy_value(net, b)
        pass_idx = int(b.pass_index)
        if 0 <= pass_idx < int(priors.shape[0]):
            priors = priors.copy()
            priors[pass_idx] = 0.0
            s = float(priors.sum())
            if s > 0.0:
                priors /= s
        return priors, float(value)

    num_actions = int(size) * int(size) + 1
    return ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        policy_value_batch_fn=lambda feats_batch, A=num_actions: infer_policy_value_from_features_batch_torch(net, feats_batch, A),
        current_player_fn=_current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=float(DEFAULT_SCORE_WEIGHT),
        score_norm_scale=None,
        score_estimator_fn=None,
        use_inplace_simulation=True,
        max_children_per_node=int(DEFAULT_MAX_CHILDREN_PER_NODE),
    )


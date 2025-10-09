from typing import Optional

import rules
from board import Board
from mcts import ScoreAwareMCTS
from policyneural import MLPPolicyValue, infer_policy_value, infer_policy_value_from_features_batch_torch


def build_mcts_standard(net: MLPPolicyValue,
                        size: int,
                        c_puct: float,
                        dirichlet_alpha: Optional[float],
                        dirichlet_frac: float,
                        dirichlet_c0: float) -> ScoreAwareMCTS:
    def legal_actions_fn(b: Board):
        # Optional cache keyed by (turn-aware pos hash, previous hash) to be safe with simple ko
        key = None
        try:
            prev_hash = b._position_history[-1] if hasattr(b, '_position_history') and b._position_history else None
            key = (b._hash_from(b.grid, b.turn), prev_hash)
        except Exception:
            key = None
        cache = getattr(legal_actions_fn, "_cache", None)
        if cache is None:
            cache = {}
            setattr(legal_actions_fn, "_cache", cache)
        if key is not None and key in cache:
            return cache[key]
        res = b.legal_moves()
        if key is not None:
            cache[key] = res
        return res

    def next_state_fn(b: Board, action: int):
        nb = b.clone()
        nb.play(action)
        return nb

    def is_terminal_fn(b: Board):
        return b.is_terminal()

    def current_player_fn(b: Board):
        return 0 if b.turn == 1 else 1

    # Simple transposition cache keyed by Zobrist turn-aware position hash
    _pv_cache = {}
    def policy_value_fn(b: Board):
        try:
            key = (b._hash_from(b.grid, b.turn))
        except Exception:
            key = None
        if key is not None and key in _pv_cache:
            return _pv_cache[key]
        pri, val = infer_policy_value(net, b)
        if key is not None:
            _pv_cache[key] = (pri, val)
        return pri, val

    def score_estimator_fn(b: Board) -> float:
        return float(rules.capture_aware_score(
            b.grid,
            getattr(b, 'captures_black', 0),
            getattr(b, 'captures_white', 0)
        ))

    num_actions = size * size + 1
    mcts = ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        policy_value_batch_fn=lambda feats_batch, A=num_actions: infer_policy_value_from_features_batch_torch(net, feats_batch, A),
        current_player_fn=current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=0.35,
        score_norm_scale=None,
        score_estimator_fn=score_estimator_fn,
        use_inplace_simulation=True,
        max_children_per_node=16,
    )
    # Attach a back-reference so nodes can see tree settings (for top-K)
    setattr(mcts.root if mcts.root is not None else mcts, 'tree', mcts)
    return mcts


def build_mcts_no_pass(net: MLPPolicyValue,
                       size: int,
                       c_puct: float,
                       dirichlet_alpha: Optional[float],
                       dirichlet_frac: float,
                       dirichlet_c0: float) -> ScoreAwareMCTS:
    def legal_actions_fn(b: Board):
        legal = b.legal_moves()
        pass_idx = b.pass_index
        return [int(a) for a in legal if int(a) != int(pass_idx)]

    def next_state_fn(b: Board, action: int):
        nb = b.clone()
        nb.play(action)
        return nb

    def is_terminal_fn(b: Board):
        return b.is_terminal() if hasattr(b, 'is_terminal') else bool((b.grid != 0).all())

    def current_player_fn(b: Board):
        return 0 if b.turn == 1 else 1

    def policy_value_fn(b: Board):
        priors, value = infer_policy_value(net, b)
        pass_idx = b.pass_index
        if 0 <= pass_idx < priors.shape[0]:
            priors = priors.copy()
            priors[pass_idx] = 0.0
            s = float(priors.sum())
            if s > 0.0:
                priors /= s
        return priors, value

    num_actions = size * size + 1
    mcts = ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        policy_value_batch_fn=lambda feats_batch, A=num_actions: infer_policy_value_from_features_batch_torch(net, feats_batch, A),
        current_player_fn=current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=0.35,
        score_norm_scale=None,
        score_estimator_fn=None,
        use_inplace_simulation=True,
        max_children_per_node=16,
    )
    setattr(mcts.root if mcts.root is not None else mcts, 'tree', mcts)
    return mcts



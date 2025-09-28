from typing import Optional

import rules
from board import Board
from mcts import ScoreAwareMCTS
from policyneural import MLPPolicyValue, infer_policy_value


def build_mcts_standard(net: MLPPolicyValue,
                        size: int,
                        c_puct: float,
                        dirichlet_alpha: Optional[float],
                        dirichlet_frac: float,
                        dirichlet_c0: float) -> ScoreAwareMCTS:
    def legal_actions_fn(b: Board):
        return b.legal_moves()

    def next_state_fn(b: Board, action: int):
        nb = b.clone()
        nb.play(action)
        return nb

    def is_terminal_fn(b: Board):
        if len(b.history) >= 2 and b.history[-1] == b.pass_index and b.history[-2] == b.pass_index:
            return True
        return False

    def current_player_fn(b: Board):
        return 0 if b.turn == 1 else 1

    def policy_value_fn(b: Board):
        return infer_policy_value(net, b)

    def score_estimator_fn(b: Board) -> float:
        return float(rules.capture_aware_score(
            b.grid,
            getattr(b, 'captures_black', 0),
            getattr(b, 'captures_white', 0)
        ))

    num_actions = size * size + 1
    return ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        current_player_fn=current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=0.35,
        score_norm_scale=None,
        score_estimator_fn=score_estimator_fn,
    )


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
        return bool((b.grid != 0).all())

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
    return ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        current_player_fn=current_player_fn,
        c_puct=float(c_puct),
        root_dirichlet_alpha=(None if dirichlet_alpha is None else float(dirichlet_alpha)),
        root_dirichlet_frac=float(dirichlet_frac),
        root_dirichlet_c0=float(dirichlet_c0),
        use_score_utility=True,
        score_weight=0.35,
        score_norm_scale=None,
        score_estimator_fn=None,
    )



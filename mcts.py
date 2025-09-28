import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

def temperature_schedule(move_number: int, t0: float = 1.0, t_min: float = 0.1, decay: float = 0.995) -> float:
    """Simple exponential temperature decay schedule useful for self-play."""
    t = max(t_min, t0 * (decay ** move_number))
    return float(t)



class _ScoreNode:
    """
    Node for score-aware MCTS that tracks both winrate and score statistics per action.

    - N[a]: visit count
    - W_win[a], Q_win[a]: accumulated/mean winrate value in [-1,1]
    - W_score[a], Q_score[a]: accumulated/mean normalized score utility in [-1,1]
    - P[a]: prior
    """

    def __init__(self, to_play: int, parent: Optional["_ScoreNode"] = None):
        self.parent: Optional[_ScoreNode] = parent
        self.to_play: int = to_play
        self.children: Dict[int, _ScoreNode] = {}
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W_win: Dict[int, float] = {}
        self.Q_win: Dict[int, float] = {}
        self.W_score: Dict[int, float] = {}
        self.Q_score: Dict[int, float] = {}

    def is_expanded(self) -> bool:
        return len(self.P) > 0

    def expand(self, legal_actions: List[int], priors: np.ndarray) -> None:
        priors = np.asarray(priors, dtype=np.float64)
        p = np.array([max(0.0, float(priors[a])) for a in legal_actions], dtype=np.float64)
        s = float(np.sum(p))
        if s <= 0.0:
            p = np.ones_like(p) / max(1, len(p))
        else:
            p = p / s
        for a, pa in zip(legal_actions, p):
            self.P[a] = float(pa)
            self.N[a] = 0
            self.W_win[a] = 0.0
            self.Q_win[a] = 0.0
            self.W_score[a] = 0.0
            self.Q_score[a] = 0.0

    def best_action(self, c_puct: float, blend_q: Callable[[float, float], float]) -> int:
        total_visits = 1 + sum(self.N.values())
        best_score = -1e18
        best_a = None
        for a in self.P.keys():
            q_blend = blend_q(self.Q_win[a], self.Q_score[a])
            u = c_puct * self.P[a] * math.sqrt(total_visits) / (1 + self.N[a])
            score = q_blend + u
            if score > best_score:
                best_score = score
                best_a = a
        return int(best_a)


class ScoreAwareMCTS:
    """
    MCTS variant that blends winrate and score utility, inspired by KataGo's score integration.

    API matches `MCTS` so it can be used as a drop-in alternative without changing existing workflows.
    You can control blending via constructor flags.
    """

    def __init__(
        self,
        num_actions: int,
        legal_actions_fn: Callable[[Any], List[int]],
        next_state_fn: Callable[[Any, int], Any],
        is_terminal_fn: Callable[[Any], bool],
        policy_value_fn: Callable[[Any], Tuple[np.ndarray, float]],
        current_player_fn: Optional[Callable[[Any], int]] = None,
        c_puct: float = 1.5,
        root_dirichlet_alpha: Optional[float] = None,
        root_dirichlet_frac: float = 0.25,
        root_dirichlet_c0: float = 10.0,
        rng: Optional[np.random.RandomState] = None,
        # Score-utility controls
        use_score_utility: bool = True,
        score_weight: float = 0.25,
        score_norm_scale: Optional[float] = None,
        score_estimator_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        self.num_actions = int(num_actions)
        self.legal_actions_fn = legal_actions_fn
        self.next_state_fn = next_state_fn
        self.is_terminal_fn = is_terminal_fn
        self.policy_value_fn = policy_value_fn
        self.current_player_fn = current_player_fn
        self.c_puct = float(c_puct)
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_frac = float(root_dirichlet_frac)
        self.root_dirichlet_c0 = float(root_dirichlet_c0)
        self.rng = rng if rng is not None else np.random.RandomState()

        # Score utility configuration
        self.use_score_utility = bool(use_score_utility)
        self.score_weight = float(np.clip(score_weight, 0.0, 1.0))
        self._score_norm_scale_static = None if score_norm_scale is None else float(max(1e-6, score_norm_scale))
        self._score_estimator_fn = score_estimator_fn

        self.root: Optional[_ScoreNode] = None

    def run(self, root_state: Any, num_simulations: int) -> None:
        to_play = self._current_player(root_state, default=0)
        self.root = _ScoreNode(to_play=to_play)

        # Expand root
        legal = self.legal_actions_fn(root_state)
        priors, _ = self.policy_value_fn(root_state)
        if priors.shape[0] != self.num_actions:
            raise ValueError(f"priors length {priors.shape[0]} != num_actions {self.num_actions}")
        self.root.expand(legal, priors)

        # Optional Dirichlet noise at root for exploration
        alpha = self.root_dirichlet_alpha
        if len(legal) > 0:
            if alpha is None or alpha <= 0:
                c0 = getattr(self, "root_dirichlet_c0", 10.0)
                alpha = float(c0) / float(len(legal))
            noise = self.rng.dirichlet([alpha] * len(legal))
            for a, n in zip(legal, noise):
                self.root.P[a] = (1 - self.root_dirichlet_frac) * self.root.P[a] + self.root_dirichlet_frac * float(n)

        for _ in range(int(num_simulations)):
            self._simulate(root_state)

    def get_action_probs(self, temp: float = 1.0) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("run() must be called before get_action_probs().")
        visits = np.zeros(self.num_actions, dtype=np.float64)
        for a, n in self.root.N.items():
            visits[a] = n
        if temp <= 1e-6:
            if float(visits.sum()) <= 0.0:
                pi = np.zeros_like(visits)
                legal_actions = list(self.root.P.keys())
                if len(legal_actions) == 0:
                    return np.ones_like(visits) / len(visits)
                pi[legal_actions] = 1.0 / float(len(legal_actions))
                return pi
            pi = np.zeros_like(visits)
            a = int(np.argmax(visits))
            pi[a] = 1.0
            return pi
        x = visits ** (1.0 / temp)
        s = np.sum(x)
        if s <= 0:
            pi = np.zeros_like(x)
            legal_actions = list(self.root.P.keys())
            if len(legal_actions) == 0:
                return np.ones_like(x) / len(x)
            pi[legal_actions] = 1.0 / float(len(legal_actions))
            return pi
        return x / s

    def choose_action(self, temp: float = 1e-3) -> int:
        pi = self.get_action_probs(temp=temp)
        return int(np.random.choice(self.num_actions, p=pi))

    # ----- Internal helpers -----
    def _blend_q(self, q_win: float, q_score: float) -> float:
        if not self.use_score_utility or self.score_weight <= 0.0:
            return float(q_win)
        return float((1.0 - self.score_weight) * q_win + self.score_weight * q_score)

    def _simulate(self, root_state: Any) -> None:
        assert self.root is not None
        node = self.root
        state = root_state
        search_path: List[Tuple[_ScoreNode, int]] = []  # (node, action)

        while True:
            if self.is_terminal_fn(state):
                terminal_win = self._terminal_value(state, node.to_play)
                terminal_score = self._terminal_score_value(state, node.to_play)
                self._backup(search_path, terminal_win, terminal_score)
                return

            if not node.is_expanded():
                legal = self.legal_actions_fn(state)
                priors, v_win = self.policy_value_fn(state)
                if priors.shape[0] != self.num_actions:
                    raise ValueError(f"priors length {priors.shape[0]} != num_actions {self.num_actions}")
                node.expand(legal, priors)
                v_score = self._score_value_from_state(state, node.to_play)
                self._backup(search_path, float(v_win), float(v_score))
                return

            a = node.best_action(self.c_puct, self._blend_q)
            search_path.append((node, a))

            next_state = self.next_state_fn(state, a)
            next_to_play = 1 - node.to_play if self.current_player_fn is None else self._current_player(next_state, default=1 - node.to_play)
            if a not in node.children:
                node.children[a] = _ScoreNode(to_play=next_to_play, parent=node)

            state = next_state
            node = node.children[a]

    def _backup(self, search_path: List[Tuple[_ScoreNode, int]], v_win_leaf: float, v_score_leaf: float) -> None:
        w = float(v_win_leaf)
        s = float(v_score_leaf)
        for node, a in search_path:
            node.N[a] += 1
            node.W_win[a] += w
            node.Q_win[a] = node.W_win[a] / node.N[a]
            node.W_score[a] += s
            node.Q_score[a] = node.W_score[a] / node.N[a]
            w = -w
            s = -s

    def _current_player(self, state: Any, default: int) -> int:
        if self.current_player_fn is None:
            return default
        return int(self.current_player_fn(state))

    def _terminal_value(self, state: Any, to_play: int) -> float:
        try:
            if hasattr(state, "result"):
                res_black = float(state.result())
                v = res_black if int(to_play) == 0 else -res_black
                return float(np.clip(v, -1.0, 1.0))
        except Exception:
            pass
        return 0.0

    def _terminal_score_value(self, state: Any, to_play: int) -> float:
        # Use the same heuristic as non-terminal score but without extra work
        return self._score_value_from_state(state, to_play)

    def _score_value_from_state(self, state: Any, to_play: int) -> float:
        # Estimate expected score margin from BLACK perspective in points
        if self._score_estimator_fn is not None:
            try:
                score_pts_black = float(self._score_estimator_fn(state))
            except Exception:
                score_pts_black = 0.0
        else:
            score_pts_black = self._default_score_estimator(state)
        # Normalize to [-1,1]
        s_norm = self._normalize_score(score_pts_black, state)
        # Convert to the perspective of the player to move at the leaf
        s_persp = s_norm if int(to_play) == 0 else -s_norm
        return float(np.clip(s_persp, -1.0, 1.0))

    def _normalize_score(self, score_pts_black: float, state: Any) -> float:
        # Dynamic scale based on board size; tuned for 9x9 -> ~10.0, scales linearly with N
        N = int(getattr(state, "size", 9))
        scale = self._score_norm_scale_static if self._score_norm_scale_static is not None else (10.0 * max(1.0, float(N) / 9.0))
        return float(np.tanh(float(score_pts_black) / float(max(1e-6, scale))))

    @staticmethod
    def _default_score_estimator(state: Any) -> float:
        try:
            grid = getattr(state, "grid", None)
            if grid is None:
                return 0.0
            black = int((grid == 1).sum())
            white = int((grid == -1).sum())
            return float(black - white)
        except Exception:
            return 0.0

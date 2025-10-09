import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

def temperature_schedule(move_number: int, t0: float = 1.0, t_min: float = 0.1, decay: float = 0.995) -> float:
    """Simple exponential temperature decay schedule useful for self-play."""
    t = max(t_min, t0 * (decay ** move_number))
    return float(t)



class _ScoreNode:
    """
    Node for score-aware MCTS using fixed-size arrays for per-action stats to reduce
    Python overhead. Only legal actions at a node are considered during selection.
    """

    def __init__(self, num_actions: int, to_play: int, parent: Optional["_ScoreNode"] = None):
        self.parent: Optional[_ScoreNode] = parent
        self.to_play: int = to_play
        self.children: Dict[int, _ScoreNode] = {}
        self.num_actions: int = int(num_actions)
        self.P: np.ndarray = np.zeros((self.num_actions,), dtype=np.float32)
        self.N: np.ndarray = np.zeros((self.num_actions,), dtype=np.int32)
        self.W_win: np.ndarray = np.zeros((self.num_actions,), dtype=np.float32)
        self.Q_win: np.ndarray = np.zeros((self.num_actions,), dtype=np.float32)
        self.W_score: np.ndarray = np.zeros((self.num_actions,), dtype=np.float32)
        self.Q_score: np.ndarray = np.zeros((self.num_actions,), dtype=np.float32)
        self._legal: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._expanded: bool = False

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    def expand(self, legal_actions: List[int], priors: np.ndarray) -> None:
        priors = np.asarray(priors, dtype=np.float32)
        legal = np.asarray(list(map(int, legal_actions)), dtype=np.int32)
        if legal.size == 0:
            self._legal = np.zeros((0,), dtype=np.int32)
            self._expanded = True
            return
        p_raw = np.maximum(0.0, priors[legal])
        s = float(np.sum(p_raw))
        if s <= 0.0:
            p = np.ones_like(p_raw) / float(len(legal))
        else:
            p = (p_raw / s)
        # Optional top-K pruning based on prior
        max_k = None
        try:
            # Look through tree root reference if available
            root = self
            while getattr(root, 'parent', None) is not None:
                root = root.parent  # type: ignore[assignment]
            tree = getattr(root, 'tree', None)
            if tree is not None:
                max_k = getattr(tree, 'max_children_per_node', None)
        except Exception:
            max_k = None
        if max_k is not None and int(max_k) > 0 and legal.size > int(max_k):
            idx_sorted = np.argsort(-p)
            keep_idx = idx_sorted[: int(max_k)]
            keep_actions = legal[keep_idx]
            self.P[:] = 0.0
            self.P[keep_actions] = p[keep_idx].astype(np.float32)
            self._legal = keep_actions
        else:
            self.P[legal] = p.astype(np.float32, copy=False)
            self._legal = legal
        self._expanded = True

    def best_action(self, c_puct: float, blend_q: Callable[[float, float], float]) -> int:
        legal = self._legal
        if legal.size == 0:
            return 0  # fallback
        total_visits = 1 + int(self.N[legal].sum())
        inv_sqrt = 1.0 / math.sqrt(total_visits)
        # Gather arrays for legal indices
        P = self.P[legal]
        N = self.N[legal].astype(np.float32)
        Qw = self.Q_win[legal]
        Qs = self.Q_score[legal]
        # Blend Q per-action using provided blend function (no external dependencies)
        qb = np.array([blend_q(float(qwi), float(qsi)) for qwi, qsi in zip(Qw.tolist(), Qs.tolist())], dtype=np.float32)
        u = (c_puct * P * inv_sqrt) / (1.0 + N)
        scores = qb.astype(np.float32) + u.astype(np.float32)
        idx = int(np.argmax(scores))
        return int(legal[idx])


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
        policy_value_batch_fn: Optional[Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]]] = None,
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
        # Performance: if True and the state supports play()/undo(), run simulations
        # by mutating the root state in-place with full undo, avoiding clone()
        use_inplace_simulation: bool = True,
        # Limit number of children expanded per node by prior (progressive widening / prior pruning)
        max_children_per_node: Optional[int] = None,
    ) -> None:
        self.num_actions = int(num_actions)
        self.legal_actions_fn = legal_actions_fn
        self.next_state_fn = next_state_fn
        self.is_terminal_fn = is_terminal_fn
        self.policy_value_fn = policy_value_fn
        self.policy_value_batch_fn = policy_value_batch_fn
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
        self.use_inplace_simulation = bool(use_inplace_simulation)
        self.max_children_per_node = None if max_children_per_node is None else int(max(1, max_children_per_node))

        self.root: Optional[_ScoreNode] = None
        self._last_action: Optional[int] = None

    def run(self, root_state: Any, num_simulations: int) -> None:
        to_play = self._current_player(root_state, default=0)
        self.root = _ScoreNode(num_actions=self.num_actions, to_play=to_play)

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
            noise = self.rng.dirichlet([alpha] * len(legal)).astype(np.float32)
            lidx = np.asarray(legal, dtype=np.int32)
            self.root.P[lidx] = (1.0 - self.root_dirichlet_frac) * self.root.P[lidx] + self.root_dirichlet_frac * noise

        # If batch function available, run batched simulation path
        if self.policy_value_batch_fn is not None and int(num_simulations) > 1:
            self._run_batched(root_state, int(num_simulations))
        else:
            for _ in range(int(num_simulations)):
                self._simulate(root_state)

    def advance_root(self, action: int) -> None:
        """Advance the root to the selected child if available, preserving subtree.

        If the child doesn't exist, drop the tree (root=None) so the next run starts fresh.
        """
        if self.root is None:
            self._last_action = None
            return
        a = int(action)
        child = self.root.children.get(a)
        if child is None:
            # Drop tree when we don't have this branch
            self.root = None
            self._last_action = None
            return
        # Detach parent references to allow GC
        child.parent = None
        self.root = child
        self._last_action = a

    def get_action_probs(self, temp: float = 1.0) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("run() must be called before get_action_probs().")
        visits = self.root.N.astype(np.float64)
        legal_actions = self.root._legal.tolist()
        if temp <= 1e-6:
            if float(visits.sum()) <= 0.0:
                pi = np.zeros_like(visits)
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
        # In-place fast path if the board exposes play()/undo()
        if self.use_inplace_simulation and hasattr(root_state, "play") and hasattr(root_state, "undo"):
            b = root_state
            node = self.root
            search_path: List[Tuple[_ScoreNode, int]] = []
            moves_played: List[int] = []
            try:
                while True:
                    if self.is_terminal_fn(b):
                        terminal_win = self._terminal_value(b, node.to_play)
                        terminal_score = self._terminal_score_value(b, node.to_play)
                        self._backup(search_path, terminal_win, terminal_score)
                        return

                    if not node.is_expanded():
                        legal = self.legal_actions_fn(b)
                        priors, v_win = self.policy_value_fn(b)
                        if priors.shape[0] != self.num_actions:
                            raise ValueError(f"priors length {priors.shape[0]} != num_actions {self.num_actions}")
                        node.expand(legal, priors)
                        v_score = self._score_value_from_state(b, node.to_play)
                        self._backup(search_path, float(v_win), float(v_score))
                        return

                    a = node.best_action(self.c_puct, self._blend_q)
                    search_path.append((node, a))

                    # Create child node and advance state in-place
                    # Determine next_to_play from the resulting board after move
                    if a not in node.children:
                        # Play move to determine next player, then create child and continue
                        b.play(a)
                        moves_played.append(a)
                        next_to_play = 0 if int(getattr(b, "turn", -1)) == 1 else 1
                        node.children[a] = _ScoreNode(num_actions=self.num_actions, to_play=next_to_play, parent=node)
                        node = node.children[a]
                        continue
                    else:
                        b.play(a)
                        moves_played.append(a)
                        node = node.children[a]
            finally:
                # Ensure we restore root_state regardless of exit path
                for _ in range(len(moves_played)):
                    getattr(root_state, "undo")()
        else:
            # Fallback: clone-based state progression
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
                # Fast detect next_to_play using board.turn if available
                if self.current_player_fn is None and hasattr(next_state, 'turn'):
                    next_to_play = 0 if int(getattr(next_state, 'turn')) == 1 else 1
                else:
                    next_to_play = 1 - node.to_play if self.current_player_fn is None else self._current_player(next_state, default=1 - node.to_play)
                if a not in node.children:
                    node.children[a] = _ScoreNode(num_actions=self.num_actions, to_play=next_to_play, parent=node)

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

    # ----- Batched simulation -----
    def _run_batched(self, root_state: Any, num_simulations: int, batch_size: int = 16, flush_timeout_ms: float = 2.0) -> None:
        assert self.root is not None
        b = root_state
        leaves: List[Tuple[_ScoreNode, List[Tuple[_ScoreNode, int]], List[int], np.ndarray]] = []
        sims_done = 0
        while sims_done < int(num_simulations):
            leaves.clear()
            # Phase A: selection only, collect up to batch_size leaves, or flush on timeout
            import time
            deadline = time.time() + float(flush_timeout_ms) / 1000.0
            while len(leaves) < int(batch_size) and sims_done + len(leaves) < int(num_simulations):
                node = self.root
                search_path: List[Tuple[_ScoreNode, int]] = []
                moves_played: List[int] = []
                while True:
                    if self.is_terminal_fn(b):
                        terminal_win = self._terminal_value(b, node.to_play)
                        terminal_score = self._terminal_score_value(b, node.to_play)
                        self._backup(search_path, terminal_win, terminal_score)
                        # undo played moves for this descent
                        for _u in range(len(moves_played)):
                            getattr(b, "undo")()
                        break
                    if not node.is_expanded():
                        feats = b.to_features().astype(np.float32).reshape(1, -1)
                        leaves.append((node, search_path, moves_played.copy(), feats.reshape(-1)))
                        # Undo before next descent
                        for _u in range(len(moves_played)):
                            getattr(b, "undo")()
                        break
                    a = node.best_action(self.c_puct, self._blend_q)
                    search_path.append((node, a))
                    b.play(a)
                    moves_played.append(a)
                    if a not in node.children:
                        next_to_play = 0 if int(getattr(b, "turn", -1)) == 1 else 1
                        node.children[a] = _ScoreNode(num_actions=self.num_actions, to_play=next_to_play, parent=node)
                    node = node.children[a]
                # Flush early if timeout reached
                if time.time() >= deadline:
                    break

            if not leaves:
                sims_done += batch_size
                continue

            # Phase B: batched inference
            feats_batch = np.stack([fv for (_, _, _, fv) in leaves], axis=0)
            pri_batch, v_batch = self.policy_value_batch_fn(feats_batch, self.num_actions)  # type: ignore[misc]

            # Phase C: expand + backup
            for (node, search_path, moves_played, _fv), pri, v in zip(leaves, pri_batch, v_batch):
                # Re-enter leaf to compute legal
                for a in moves_played:
                    b.play(int(a))
                legal = self.legal_actions_fn(b)
                node.expand(legal, pri)
                v_score = self._score_value_from_state(b, node.to_play)
                self._backup(search_path, float(v), float(v_score))
                for _u in range(len(moves_played)):
                    getattr(b, "undo")()

            sims_done += len(leaves)

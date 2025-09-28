import argparse
from typing import List, Tuple, Optional

import numpy as np
import rules
import torch

from board import Board9x9, Board
from mcts import ScoreAwareMCTS, temperature_schedule
from mcts_wrappers import build_mcts_standard
from policyneural import MLPPolicyValue, infer_policy_value


 


def _build_mcts(net: MLPPolicyValue,
                size: int,
                c_puct: float,
                dirichlet_alpha: Optional[float],
                dirichlet_frac: float,
                dirichlet_c0: float) -> ScoreAwareMCTS:
    return build_mcts_standard(net, size, c_puct, dirichlet_alpha, dirichlet_frac, dirichlet_c0)


def self_play_game(net: MLPPolicyValue,
                   num_simulations: int,
                   komi: float,
                   size: int,
                   c_puct: float,
                   dirichlet_alpha: Optional[float],
                   dirichlet_frac: float,
                   dirichlet_c0: float,
                   temp_t0: float,
                   temp_min: float,
                   temp_decay: float,
                   min_game_len: int) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, float]]:
    """
    Play one self-play game, return training data and game outcome.
    Returns (feature_list, pi_list, (winner, score_margin))
      - feature_list: list of (4*N*N,) feature arrays (e.g., 324 on 9x9)
      - pi_list: list of (N*N+1,) visit distributions (e.g., 82 on 9x9)
      - winner: +1 black win, -1 white win, 0 draw
      - score_margin: (B - W) from Black's perspective (float)
    """
    mcts = _build_mcts(net,
                       size=size,
                       c_puct=c_puct,
                       dirichlet_alpha=dirichlet_alpha,
                       dirichlet_frac=dirichlet_frac,
                       dirichlet_c0=dirichlet_c0)
    # Enable standard Go-like rules during self-play to improve learning dynamics
    board = Board(size, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
    # Komi is not used for scoring or training targets anymore. Keep rule_info neutral.
    board.komi = float(komi)
    board.rule_info = 0.0
    features: List[np.ndarray] = []
    policies: List[np.ndarray] = []

    t = 0
    while True:
        mcts.run(board, num_simulations=num_simulations)
        temp = temperature_schedule(t, t0=float(temp_t0), t_min=float(temp_min), decay=float(temp_decay))
        pi = mcts.get_action_probs(temp=temp)

        # record training example
        features.append(board.to_features().copy())
        policies.append(pi.astype(np.float32))

        # Sample strictly from current legal moves to avoid selecting illegal actions
        legal = board.legal_moves().astype(int)
        pi_masked = np.zeros_like(pi)
        pi_masked[legal] = pi[legal]
        s = float(pi_masked.sum())
        if s <= 0.0:
            pi_masked = np.zeros_like(pi_masked)
            pi_masked[legal] = 1.0 / max(1, len(legal))
        else:
            pi_masked = pi_masked / s

        action = int(np.random.choice(len(pi_masked), p=pi_masked))
        # Print only the final selected move after simulations
        action_label = "PASS" if int(action) == int(board.pass_index) else f"{divmod(int(action), board.size)[0]},{divmod(int(action), board.size)[1]}"
        print(f"[mcts] move {t} -> {action_label}")
        board.play(action)

        # game end: two passes, but only if we've reached min_game_len plies
        if (
            len(board.history) >= 2
            and board.history[-1] == board.pass_index
            and board.history[-2] == board.pass_index
            and t >= int(min_game_len)
        ):
            break
        t += 1
        # game end: full board occupied
        if (board.grid != 0).all():
            break
        # game end: max moves reached (N*N)
        if t >= size * size:
            break

    # Compute final margin (capture-aware by default; komi handled later in targets)
    s_final = float(rules.final_margin(
        board.grid,
        getattr(board, 'captures_black', 0),
        getattr(board, 'captures_white', 0),
        use_capture_aware=True,
        komi=0.0,
    ))
    winner = 1 if s_final > 0 else (-1 if s_final < 0 else 0)
    print(f"[game] final capture-aware score (B-W): {s_final:.1f}; winner: {'Black' if winner == 1 else ('White' if winner == -1 else 'Draw')}")
    return features, policies, (winner, float(s_final))


def train_on_selfplay(net: MLPPolicyValue,
                     games: int = 10,
                     sims: int = 64,
                     komi: float = 6.5,
                     lr: float = 3e-3,
                     value_weight: float = 1.0,
                     l2: float = 1e-4,
                     size: int = 9,
                     quiet: bool = False,
                     log_every: int = 1,
                     c_puct: float = 2.0,
                     dirichlet_alpha: Optional[float] = 0.15,
                     dirichlet_frac: float = 0.25,
                     dirichlet_c0: float = 10.0,
                     temp_t0: float = 1.0,
                     temp_min: float = 0.25,
                     temp_decay: float = 0.995,
                     min_game_len: int = 50,
                     checkpoint_every: int = 1,
                     include_komi_in_margin: bool = False) -> None:
    """
    Generate `games` self-play games and update `net` after each game.
    This ensures later games use weights learned from earlier games.
    Feature size is dynamic; the network adapts input W1 at runtime.
    """
    for gi in range(games):
        feats, pis, outcome = self_play_game(
            net,
            num_simulations=sims,
            komi=komi,
            size=size,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            dirichlet_c0=dirichlet_c0,
            temp_t0=temp_t0,
            temp_min=temp_min,
            temp_decay=temp_decay,
            min_game_len=int(min_game_len),
        )
        # Unpack outcome and build continuous margin-based targets z in [-1,1]
        winner, s_final = outcome
        N = int(size)
        tau = max(1.0, 0.5 * N)
        margin = float(s_final) - (float(komi) if include_komi_in_margin else 0.0)
        z_black = float(np.tanh(margin / float(tau)))

        # assign outcomes z from current player's perspective at each step
        # board.turn starts at +1 (black). After k moves, player to move flips each time.
        # We recorded features before the move was played, so step index parity determines perspective.
        game_X: List[np.ndarray] = []
        game_pi: List[np.ndarray] = []
        game_z: List[np.ndarray] = []
        for k in range(len(feats)):
            to_move = 1 if (k % 2 == 0) else -1
            z = z_black if to_move == 1 else -z_black
            xk = feats[k].astype(np.float32)
            pik = pis[k]
            zk = np.array([z], dtype=np.float32)
            game_X.append(xk)
            game_pi.append(pik)
            game_z.append(zk)

        # Per-game loss + optimizer step (saves weights via net.step)
        if game_X:
            Xg = np.stack(game_X, axis=0)
            Pig = np.stack(game_pi, axis=0)
            Zg = np.stack(game_z, axis=0)
            _, _, cache_g = net.forward(Xg)
            loss_g, grads_g = net.backward(cache_g, Pig, Zg, l2=l2, c_v=value_weight)
            # Pass checkpoint frequency to net.step if supported
            try:
                net.step(grads_g, lr=lr, save_every=int(max(1, checkpoint_every)))
            except TypeError:
                net.step(grads_g, lr=lr)
            print(f"[train] weights updated after game {gi+1}")
            if (not quiet) and ((gi + 1) % max(1, int(log_every)) == 0):
                print(f"game {gi+1}/{games} complete: samples={Xg.shape[0]}, loss={loss_g:.6f}")
    # No final batch update; weights are updated per game and auto-saved in net.step


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training for NxN Go (toy)")
    parser.add_argument("--games", type=int, default=10, help="number of self-play games")
    parser.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument("--komi", type=float, default=6.5, help="komi for winner evaluation")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--value_weight", type=float, default=1.0, help="value loss weight")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--size", type=int, default=9, help="board size N (e.g., 9, 13, 19)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-game training logs")
    parser.add_argument("--log_every", type=int, default=1, help="print progress every N games (ignored if --quiet)")
    parser.add_argument("--c_puct", type=float, default=2.0, help="PUCT exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.0, help="Dirichlet alpha at root; 0 or negative = auto (c0/|legal|)")
    parser.add_argument("--dirichlet_frac", type=float, default=0.25, help="Dirichlet mixing fraction at root")
    parser.add_argument("--dirichlet_c0", type=float, default=10.0, help="Target total concentration for auto alpha (alpha0 ≈ c0)")
    parser.add_argument("--temp_t0", type=float, default=1.0, help="temperature schedule t0")
    parser.add_argument("--temp_min", type=float, default=0.25, help="temperature floor")
    parser.add_argument("--temp_decay", type=float, default=0.995, help="temperature exponential decay factor")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="save weights every N games (>=1)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="compute device selection")
    parser.add_argument("--require_gpu", action="store_true", help="error if CUDA is not available")
    parser.add_argument("--min_game_len", type=int, default=50, help="Minimum total plies before two-pass can end the game.")
    parser.add_argument("--include_komi_in_margin", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    # Rollout removed
    args = parser.parse_args()

    # Device selection with optional strict GPU requirement
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.require_gpu and device != "cuda":
        raise SystemExit("GPU required but not available: torch.cuda.is_available() is False")
    net = MLPPolicyValue(device=device)
    if not args.quiet:
        print("Using device:", device)
    # treat non-positive alpha as disabled (None)
    alpha = None if args.dirichlet_alpha is not None and args.dirichlet_alpha <= 0 else args.dirichlet_alpha

    # Optional determinism
    if args.seed is not None:
        import random
        try:
            random.seed(int(args.seed))
        except Exception:
            pass
        try:
            np.random.seed(int(args.seed))
        except Exception:
            pass
        try:
            torch.manual_seed(int(args.seed))
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
        except Exception:
            pass

    train_on_selfplay(net,
                      games=args.games,
                      sims=args.sims,
                      komi=args.komi,
                      lr=args.lr,
                      value_weight=args.value_weight,
                      l2=args.l2,
                      size=args.size,
                      quiet=args.quiet,
                      log_every=args.log_every,
                      c_puct=args.c_puct,
                      dirichlet_alpha=alpha,
                      dirichlet_frac=args.dirichlet_frac,
                      dirichlet_c0=args.dirichlet_c0,
                      temp_t0=args.temp_t0,
                      temp_min=args.temp_min,
                      temp_decay=args.temp_decay,
                      min_game_len=int(args.min_game_len),
                      checkpoint_every=max(1, int(args.checkpoint_every)),
                      include_komi_in_margin=bool(args.include_komi_in_margin))

    # Quick sanity inference after training on an empty board
    empty = Board(args.size)
    p, v = infer_policy_value(net, empty)
    print("trained on self-play.")
    print("empty board value estimate:", float(v))
    print("policy mass:", float(p.sum()))


if __name__ == "__main__":
    main()



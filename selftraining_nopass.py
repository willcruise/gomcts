import argparse
from typing import List, Tuple, Optional

import numpy as np
import rules
import torch

from board import Board9x9, Board
from mcts import ScoreAwareMCTS, temperature_schedule
from mcts_wrappers import build_mcts_no_pass
from policyneural import MLPPolicyValue, infer_policy_value


def _board_is_full(b: Board) -> bool:
    return bool((b.grid != 0).all())


def evaluate_winner_simple(board: Board9x9, komi: float = 6.5) -> int:
    black = int(np.sum(board.grid == 1))
    white = int(np.sum(board.grid == -1))
    score = float(black - white)
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def _build_mcts_no_pass(net: MLPPolicyValue,
                        size: int,
                        c_puct: float,
                        dirichlet_alpha: Optional[float],
                        dirichlet_frac: float,
                        dirichlet_c0: float) -> ScoreAwareMCTS:
    return build_mcts_no_pass(net, size, c_puct, dirichlet_alpha, dirichlet_frac, dirichlet_c0)


def self_play_game_no_pass(net: MLPPolicyValue,
                           num_simulations: int,
                           komi: float,
                           size: int,
                           c_puct: float,
                           dirichlet_alpha: Optional[float],
                           dirichlet_frac: float,
                           dirichlet_c0: float,
                           temp_t0: float,
                           temp_min: float,
                           temp_decay: float) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, float]]:
    mcts = _build_mcts_no_pass(net,
                               size=size,
                               c_puct=c_puct,
                               dirichlet_alpha=dirichlet_alpha,
                               dirichlet_frac=dirichlet_frac,
                               dirichlet_c0=dirichlet_c0)
    board = Board(size, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
    board.komi = float(komi)
    board.rule_info = 0.0

    features: List[np.ndarray] = []
    policies: List[np.ndarray] = []

    t = 0
    while not _board_is_full(board):
        mcts.run(board, num_simulations=num_simulations)
        temp = temperature_schedule(t, t0=float(temp_t0), t_min=float(temp_min), decay=float(temp_decay))
        pi = mcts.get_action_probs(temp=temp)

        # Mask pass always; renormalize.
        pass_idx = board.pass_index
        pi = pi.copy()
        if 0 <= pass_idx < pi.shape[0]:
            pi[pass_idx] = 0.0
            s = float(pi.sum())
            if s > 0.0:
                pi /= s
            else:
                # Fallback: uniform over point moves
                K = pass_idx
                pi = np.zeros_like(pi, dtype=np.float32)
                if K > 0:
                    pi[:K] = 1.0 / K

        # record training example BEFORE playing
        features.append(board.to_features().copy())
        policies.append(pi.astype(np.float32))

        action = int(np.random.choice(len(pi), p=pi))
        # Print only the final selected move after simulations (no pass moves here)
        action_label = f"{divmod(int(action), board.size)[0]},{divmod(int(action), board.size)[1]}"
        print(f"[mcts] move {t} -> {action_label}")
        board.play(action)
        t += 1

    # Game finished due to full board; compute unified final margin (komi handled later)
    s_final = float(rules.final_margin(
        board.grid,
        getattr(board, 'captures_black', 0),
        getattr(board, 'captures_white', 0),
        use_capture_aware=True,
        komi=0.0,
    ))
    winner = 1 if s_final > 0 else (-1 if s_final < 0 else 0)
    return features, policies, (winner, float(s_final))


def train_on_selfplay_no_pass(net: MLPPolicyValue,
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
                              checkpoint_every: int = 1,
                              include_komi_in_margin: bool = False) -> None:
    for gi in range(games):
        feats, pis, outcome = self_play_game_no_pass(
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
        )
        winner, s_final = outcome
        N = int(size)
        tau = max(1.0, 0.5 * N)
        margin = float(s_final) - (float(komi) if include_komi_in_margin else 0.0)
        z_black = float(np.tanh(margin / float(tau)))

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

        if game_X:
            Xg = np.stack(game_X, axis=0)
            Pig = np.stack(game_pi, axis=0)
            Zg = np.stack(game_z, axis=0)
            _, _, cache_g = net.forward(Xg)
            loss_g, grads_g = net.backward(cache_g, Pig, Zg, l2=l2, c_v=value_weight)
            try:
                net.step(grads_g, lr=lr, save_every=int(max(1, checkpoint_every)))
            except TypeError:
                net.step(grads_g, lr=lr)
            if (not quiet) and ((gi + 1) % max(1, int(log_every)) == 0):
                print(f"[no-pass] game {gi+1}/{games}: samples={Xg.shape[0]}, loss={loss_g:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training (no-pass, full-board end)")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--komi", type=float, default=6.5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--c_puct", type=float, default=2.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.0)
    parser.add_argument("--dirichlet_frac", type=float, default=0.25)
    parser.add_argument("--dirichlet_c0", type=float, default=10.0)
    parser.add_argument("--temp_t0", type=float, default=1.0)
    parser.add_argument("--temp_min", type=float, default=0.25)
    parser.add_argument("--temp_decay", type=float, default=0.995)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--include_komi_in_margin", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.require_gpu and device != "cuda":
        raise SystemExit("GPU required but not available: torch.cuda.is_available() is False")
    net = MLPPolicyValue(device=device)
    if not args.quiet:
        print("Using device:", device)
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
            import torch
            torch.manual_seed(int(args.seed))
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
        except Exception:
            pass

    train_on_selfplay_no_pass(
        net,
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
        checkpoint_every=max(1, int(args.checkpoint_every)),
        include_komi_in_margin=bool(args.include_komi_in_margin)
    )

    # quick inference sanity
    empty = Board(args.size)
    p, v = infer_policy_value(net, empty)
    print("trained on no-pass self-play.")
    print("empty board value estimate:", float(v))
    print("policy mass:", float(p.sum()))


if __name__ == "__main__":
    main()



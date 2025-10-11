import argparse
import os
import shutil
from typing import List, Tuple, Optional

import numpy as np
import rules
import torch

from board import Board9x9, Board
from mcts import ScoreAwareMCTS, temperature_schedule
from mcts_wrappers import build_mcts_standard
from policyneural import MLPPolicyValue, infer_policy_value


 


def _selfplay_worker(q, seed_off: int, games_n: int,
                     size: int, sims: int, komi: float,
                     c_puct: float,
                     dirichlet_alpha: Optional[float], dirichlet_frac: float, dirichlet_c0: float,
                     temp_t0: float, temp_min: float, temp_decay: float, min_game_len: int,
                     include_komi_in_margin: bool,
                     device_str: str) -> None:
    """Spawnable worker that generates self-play samples and sends (Xg, Pig, Zg).

    On Windows, this must be top-level to be picklable.
    """
    local_net = MLPPolicyValue(device=device_str)
    try:
        import random as _rnd
        _rnd.seed(int(seed_off))
        np.random.seed(int(seed_off))
        try:
            import torch as _torch
            _torch.manual_seed(int(seed_off))
        except Exception:
            pass
    except Exception:
        pass
    for _ in range(int(games_n)):
        feats, pis, outcome = self_play_game(
            local_net,
            num_simulations=int(sims),
            komi=float(komi),
            size=int(size),
            c_puct=float(c_puct),
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=float(dirichlet_frac),
            dirichlet_c0=float(dirichlet_c0),
            temp_t0=float(temp_t0),
            temp_min=float(temp_min),
            temp_decay=float(temp_decay),
            min_game_len=int(min_game_len),
        )
        if feats:
            winner, s_final = outcome
            # Apply komi to determine the actual winner for training
            # In Go, White gets komi compensation, so: Black_score - White_score - komi
            final_margin = float(s_final) - float(komi)
            z_black = 1.0 if final_margin > 0 else (-1.0 if final_margin < 0 else 0.0)
            Xg = np.stack([f.astype(np.float32) for f in feats], axis=0)
            Pig = np.stack([p for p in pis], axis=0)
            Z_list: List[np.ndarray] = []
            for k in range(len(feats)):
                to_move = 1 if (k % 2 == 0) else -1
                z = z_black if to_move == 1 else -z_black
                Z_list.append(np.array([z], dtype=np.float32))
            Zg = np.stack(Z_list, axis=0)
            q.put((Xg, Pig, Zg))
    q.put(None)
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
                     include_komi_in_margin: bool = False,
                     # Batched MCTS/runtime tuning
                     mcts_batch_size: int = 16,
                     mcts_flush_ms: float = 2.0,
                     use_cuda_graphs: bool = False,
                     # Parallel self-play
                     workers: int = 1,
                     worker_games: int = 0,
                     # Evaluation gate options
                     eval_every: int = 0,
                     eval_games: int = 200,
                     eval_threshold: float = 0.55,
                     eval_sims: Optional[int] = None,
                     eval_swap_colors: bool = True,
                     eval_random_opening: int = 0,
                     eval_dir: Optional[str] = None) -> None:
    """
    Generate `games` self-play games and update `net` after each game.
    This ensures later games use weights learned from earlier games.
    Feature size is dynamic; the network adapts input W1 at runtime.
    """
    # Propagate runtime knobs into net for wrappers to pick up
    try:
        setattr(net, "_mcts_batch_size", int(mcts_batch_size))
        setattr(net, "_mcts_flush_ms", float(mcts_flush_ms))
        setattr(net, "_use_cuda_graphs", bool(use_cuda_graphs))
    except Exception:
        pass

    # --- Evaluation helpers (closure uses same MCTS budget and rules) ---
    def _save_baseline_snapshot(path: str) -> None:
        # persist a baseline snapshot of current weights
        try:
            # default .pt path
            src = os.path.join(os.path.dirname(__file__), "weights.pt")
            if os.path.exists(src):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.copy2(src, path)
        except Exception:
            pass

    def _restore_weights(path: str) -> bool:
        try:
            dst = os.path.join(os.path.dirname(__file__), "weights.pt")
            if os.path.exists(path):
                shutil.copy2(path, dst)
                # Force reload into net for the correct policy_dim when next used
                # We reinitialize lazily on the next forward via infer wrapper
                return True
        except Exception:
            pass
        return False

    def _play_one_eval_game(net_black: MLPPolicyValue, net_white: MLPPolicyValue, sims_budget: int) -> int:
        mcts_black = _build_mcts(net_black,
                                 size=size,
                                 c_puct=c_puct,
                                 dirichlet_alpha=None,  # no noise during eval
                                 dirichlet_frac=0.0,
                                 dirichlet_c0=dirichlet_c0)
        mcts_white = _build_mcts(net_white,
                                 size=size,
                                 c_puct=c_puct,
                                 dirichlet_alpha=None,
                                 dirichlet_frac=0.0,
                                 dirichlet_c0=dirichlet_c0)
        b = Board(size, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
        b.komi = float(komi)
        b.rule_info = 0.0

        # Optional randomized opening: play k random legal moves alternating colors
        if int(eval_random_opening) > 0:
            import random
            for _ in range(int(eval_random_opening)):
                legal = b.legal_moves().astype(int)
                if len(legal) == 0:
                    break
                a = int(random.choice(list(map(int, legal))))
                b.play(a)
                if b.is_terminal():
                    break

        t = 0
        while True:
            player_to_move = 1 if b.turn == 1 else -1
            if player_to_move == 1:
                mcts_black.run(b, num_simulations=sims_budget)
                pi = mcts_black.get_action_probs(temp=1e-3)
            else:
                mcts_white.run(b, num_simulations=sims_budget)
                pi = mcts_white.get_action_probs(temp=1e-3)
            legal = b.legal_moves().astype(int)
            pi_masked = np.zeros_like(pi)
            pi_masked[legal] = pi[legal]
            s = float(pi_masked.sum())
            if s <= 0.0:
                pi_masked[legal] = 1.0 / max(1, len(legal))
            else:
                pi_masked = pi_masked / s
            action = int(np.random.choice(len(pi_masked), p=pi_masked))
            b.play(action)

            # end conditions
            if (
                len(b.history) >= 2 and b.history[-1] == b.pass_index and b.history[-2] == b.pass_index
            ) or (b.grid != 0).all() or (t >= size * size):
                break
            t += 1

        s_final = float(rules.final_margin(
            b.grid,
            getattr(b, 'captures_black', 0),
            getattr(b, 'captures_white', 0),
            use_capture_aware=True,
            komi=0.0,
        ))
        winner = 1 if s_final > 0 else (-1 if s_final < 0 else 0)
        return winner

    def _evaluate_against_baseline(candidate_w_path: str, games_n: int, threshold: float) -> Tuple[bool, float]:
        # Build a separate model instance for baseline to avoid interfering with `net`
        baseline = MLPPolicyValue(device=getattr(net, "_device", "cpu").type if hasattr(net, "_device") else "cpu")
        # Replace weights.pt with baseline snapshot and allow lazy reload
        ok = _restore_weights(candidate_w_path)
        if not ok:
            # If no snapshot provided, treat as accept to avoid blocking
            return True, 1.0

        # Candidate is the current `net` (already updated). Baseline is freshly loaded from snapshot path by infer wrapper.
        sims_budget = int(eval_sims) if eval_sims is not None and int(eval_sims) > 0 else int(sims)
        wins = 0.0
        total = 0
        for g in range(int(games_n)):
            if bool(eval_swap_colors) and (g % 2 == 1):
                w = _play_one_eval_game(baseline, net, sims_budget)  # baseline as Black
                # From candidate perspective, candidate is White in this game
                if w == -1:
                    wins += 1.0
                elif w == 0:
                    wins += 0.5
            else:
                w = _play_one_eval_game(net, baseline, sims_budget)  # candidate as Black
                if w == 1:
                    wins += 1.0
                elif w == 0:
                    wins += 0.5
            total += 1
        wr = float(wins / max(1, total))
        return (wr > float(threshold)), wr

    # Directory to store evaluation snapshots
    eval_store = eval_dir if eval_dir is not None else os.path.join(os.path.dirname(__file__), "eval_snapshots")
    os.makedirs(eval_store, exist_ok=True)

    # Keep one rolling baseline snapshot initially (before any training)
    baseline_path = os.path.join(eval_store, "baseline.pt")
    _save_baseline_snapshot(baseline_path)

    # Optional simple multi-worker: spawn N worker processes that generate games,
    # then apply SGD updates in the main process to a single weights file.
    # We keep this minimal to avoid changing workflow: workers only produce (X, pi, z) batches.
    if int(workers) > 1 and int(worker_games) > 0:
        print(f"[INFO] Spawning {int(workers)} worker processes, {int(worker_games)} games each...")
        import multiprocessing as mp
        q: "mp.Queue" = mp.Queue(maxsize=max(8, int(workers) * 2))
        procs: list = []
        per = int(worker_games)
        base_seed = int(np.random.randint(0, 2**31 - 1))
        device_str = getattr(net, "_device", torch.device("cpu")).type if hasattr(net, "_device") else "cpu"
        for wi in range(int(workers)):
            p = mp.Process(target=_selfplay_worker, args=(q, base_seed + wi, per, int(size), int(sims), float(komi),
                                                          float(c_puct), dirichlet_alpha, float(dirichlet_frac), float(dirichlet_c0),
                                                          float(temp_t0), float(temp_min), float(temp_decay), int(min_game_len),
                                                          bool(include_komi_in_margin), device_str))
            p.daemon = True
            p.start()
            print(f"[INFO] Started worker {wi+1}/{int(workers)} (PID: {p.pid})")
            procs.append(p)
        finished = 0
        gi = 0
        try:
            while finished < int(workers):
                item = q.get()
                if item is None:
                    finished += 1
                    continue
                Xg, Pig, Zg = item
                _, _, cache_g = net.forward(Xg)
                loss_g, grads_g = net.backward(cache_g, Pig, Zg, l2=l2, c_v=value_weight)
                try:
                    net.step(grads_g, lr=lr, save_every=int(max(1, checkpoint_every)))
                except TypeError:
                    net.step(grads_g, lr=lr)
                gi += 1
                if (not quiet) and ((gi) % max(1, int(log_every)) == 0):
                    print(f"games_processed {gi}: samples={Xg.shape[0]}, loss={loss_g:.6f}")
        finally:
            for p in procs:
                try:
                    p.join(timeout=1.0)
                except Exception:
                    try:
                        p.terminate()
                    except Exception:
                        pass
        return

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
        # Unpack outcome and build discrete outcome targets z in {-1, 0, +1} like AlphaZero
        winner, s_final = outcome
        # Apply komi to determine the actual winner for training
        # In Go, White gets komi compensation, so: Black_score - White_score - komi
        final_margin = float(s_final) - float(komi)
        z_black = 1.0 if final_margin > 0 else (-1.0 if final_margin < 0 else 0.0)

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

        # Evaluation gate
        if int(eval_every) > 0 and ((gi + 1) % int(eval_every) == 0):
            # Save candidate (already saved by step, but snapshot explicitly for clarity)
            cand_path = os.path.join(eval_store, "candidate.pt")
            try:
                src = os.path.join(os.path.dirname(__file__), "weights.pt")
                if os.path.exists(src):
                    shutil.copy2(src, cand_path)
            except Exception:
                pass

            # Evaluate candidate vs baseline
            accepted, wr = _evaluate_against_baseline(baseline_path, int(eval_games), float(eval_threshold))
            print(f"[eval] candidate win-rate vs baseline: {wr*100:.2f}% -> {'ACCEPT' if accepted else 'REJECT'}")
            if accepted:
                # Promote: baseline becomes candidate
                try:
                    shutil.copy2(cand_path, baseline_path)
                except Exception:
                    pass
            else:
                # Revert working weights back to baseline
                restored = _restore_weights(baseline_path)
                if restored:
                    print("[eval] reverted weights to previous accepted baseline")
                else:
                    print("[eval] failed to restore baseline weights; keeping current weights")
    # No final batch update; weights are updated per game and auto-saved in net.step


def main() -> None:
    # Force spawn method for multiprocessing compatibility in Docker
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description="Self-play training for NxN Go (toy)")
    parser.add_argument("--config", type=str, default=None, help="path to YAML config file (command-line args override config)")
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
    # Batched/runtime knobs
    parser.add_argument("--mcts_batch_size", type=int, default=16, help="batched MCTS leaf inference batch size")
    parser.add_argument("--mcts_flush_ms", type=float, default=2.0, help="flush timeout in ms for leaf batch")
    parser.add_argument("--use_cuda_graphs", action="store_true", help="enable CUDA Graphs for fixed-shape batched inference")
    # Simple multi-worker self-play
    parser.add_argument("--workers", type=int, default=1, help="number of self-play worker processes")
    parser.add_argument("--worker_games", type=int, default=0, help="games per worker (0 disables multi-process)")
    parser.add_argument("--seed", type=int, default=None)
    # Evaluation gate flags
    parser.add_argument("--eval_every", type=int, default=0, help="run evaluation every N games (0 disables)")
    parser.add_argument("--eval_games", type=int, default=200, help="number of head-to-head eval games per evaluation")
    parser.add_argument("--eval_threshold", type=float, default=0.55, help="promotion threshold win-rate (0-1)")
    parser.add_argument("--eval_sims", type=int, default=None, help="MCTS sims per move during eval (defaults to --sims)")
    parser.add_argument("--eval_no_swap", action="store_true", help="disable color swapping during eval")
    parser.add_argument("--eval_random_opening", type=int, default=0, help="random opening plies before eval games")
    parser.add_argument("--eval_dir", type=str, default=None, help="directory to store eval snapshots")
    # Rollout removed
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        import yaml
        import os
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply config values, but only for args that weren't explicitly set on command line
        # We detect explicit CLI args by comparing with defaults
        defaults = parser.parse_args([])  # Parse with no args to get defaults
        for key, value in config.items():
            if hasattr(args, key):
                # Only override if the current value matches the default (wasn't set via CLI)
                if getattr(args, key) == getattr(defaults, key):
                    setattr(args, key, value)
        
        if not args.quiet:
            print(f"[INFO] Loaded config from: {args.config}")

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
                      include_komi_in_margin=bool(args.include_komi_in_margin),
                      mcts_batch_size=int(args.mcts_batch_size),
                      mcts_flush_ms=float(args.mcts_flush_ms),
                      use_cuda_graphs=bool(args.use_cuda_graphs),
                      workers=int(args.workers),
                      worker_games=int(args.worker_games),
                      eval_every=int(args.eval_every),
                      eval_games=int(args.eval_games),
                      eval_threshold=float(args.eval_threshold),
                      eval_sims=(None if args.eval_sims is None or int(args.eval_sims) <= 0 else int(args.eval_sims)),
                      eval_swap_colors=(not args.eval_no_swap),
                      eval_random_opening=int(args.eval_random_opening),
                      eval_dir=args.eval_dir)

    # Quick sanity inference after training on an empty board
    empty = Board(args.size)
    p, v = infer_policy_value(net, empty)
    print("trained on self-play.")
    print("empty board value estimate:", float(v))
    print("policy mass:", float(p.sum()))


if __name__ == "__main__":
    main()



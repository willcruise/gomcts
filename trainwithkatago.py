import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure KataGo AppImage falls back to extract-and-run when FUSE utilities are missing
os.environ.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")

from board import Board
from mcts import ScoreAwareMCTS, temperature_schedule
from mcts_wrappers import build_mcts_standard
from policyneural import MLPPolicyValue, infer_policy_value
import rules


ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # 'I' skipped per Go convention


def _running_in_notebook() -> bool:
    """Best-effort detection for Jupyter/Studio notebooks."""
    if "IPKernelApp" not in sys.modules:
        return False
    try:
        from IPython import get_ipython  # type: ignore
    except ImportError:
        return False
    return get_ipython() is not None


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _scripts_dir() -> str:
    return os.path.join(_repo_root(), "scripts")


_AUTO_INSTALL_RAN: bool = False
_APPLIED_CONDA_PREFIX: bool = False
_DEFAULT_KATAGO_ENV: str = "katago-env"


def _repo_katago_dir() -> Path:
    return Path(_repo_root()) / "katago"


def _auto_install_katago(force: bool = False, env_name: Optional[str] = None) -> Path:
    script_path = os.path.join(_scripts_dir(), "install_katago.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            "scripts/install_katago.py not found – cannot auto-install KataGo assets."
        )
    global _AUTO_INSTALL_RAN
    if _AUTO_INSTALL_RAN and not force:
        return _repo_katago_dir()
    cmd = [sys.executable, script_path]
    archive_hint_dirs: List[str] = []
    repo_archive_dir = os.path.join(_repo_root(), "katago_archives")
    if os.path.isdir(repo_archive_dir):
        archive_hint_dirs.append(repo_archive_dir)
    legacy_archives_dir = os.path.join(_repo_katago_dir(), "archives")
    if os.path.isdir(legacy_archives_dir):
        archive_hint_dirs.append(legacy_archives_dir)
    for hint in archive_hint_dirs:
        cmd.extend(["--archive-dir", hint])
    if force:
        cmd.append("--force")
    if env_name:
        cmd.extend(["--env-name", env_name])
    print("[setup] installing KataGo assets (this may take a few minutes)…")
    try:
        subprocess.run(cmd, check=True)
        _AUTO_INSTALL_RAN = True
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Automatic KataGo installation failed. See output above for details."
        ) from exc
    return _repo_katago_dir()


def _maybe_apply_conda_prefix(prefix: Optional[Path]) -> None:
    global _APPLIED_CONDA_PREFIX
    if _APPLIED_CONDA_PREFIX:
        return
    if prefix is None:
        return
    prefix_path = Path(prefix).expanduser()
    if not prefix_path.exists():
        return

    new_path_entries: List[str] = []
    if os.name == "nt":
        scripts_dir = prefix_path / "Scripts"
        library_bin = prefix_path / "Library" / "bin"
        for candidate in (scripts_dir, library_bin, prefix_path / "bin"):
            if candidate.exists():
                new_path_entries.append(str(candidate))
    else:
        bin_dir = prefix_path / "bin"
        if bin_dir.exists():
            new_path_entries.append(str(bin_dir))
        lib_dir = prefix_path / "lib"
        if lib_dir.exists():
            ld_prev = os.environ.get("LD_LIBRARY_PATH")
            ld_components = [str(lib_dir)]
            if ld_prev:
                ld_components.append(ld_prev)
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_components)

    if new_path_entries:
        current_path = os.environ.get("PATH")
        path_components = new_path_entries.copy()
        if current_path:
            path_components.append(current_path)
        os.environ["PATH"] = os.pathsep.join(path_components)

    os.environ.setdefault("KATAGO_CONDA_PREFIX", str(prefix_path))
    _APPLIED_CONDA_PREFIX = True
    print(f"[env] kataGo runtime prefix applied: {prefix_path}")


def _load_katago_env_hints(katago_dir: Path) -> None:
    prefix_hint = os.getenv("KATAGO_CONDA_PREFIX")
    if prefix_hint:
        _maybe_apply_conda_prefix(Path(prefix_hint))
        return
    hint_file = Path(katago_dir) / "conda_prefix.txt"
    if hint_file.is_file():
        try:
            text = hint_file.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if text:
            _maybe_apply_conda_prefix(Path(text))


def _resolve_device(device_arg: str, require_gpu: bool) -> str:
    import torch

    if device_arg == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device_arg

    if require_gpu and resolved != "cuda":
        raise SystemExit("GPU required but not available: torch.cuda.is_available() is False")
    return resolved


def _seed_everything(seed: Optional[int], device: str) -> None:
    if seed is None:
        return
    try:
        import random
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


class GTPClient:
    def __init__(self, katago_exe: str, model_path: str, config_path: str):
        exe_abs = os.path.abspath(katago_exe)
        model_abs = os.path.abspath(model_path)
        cfg_abs = os.path.abspath(config_path)
        exe_dir = os.path.dirname(exe_abs)

        # Validate paths before launching to avoid silent hangs
        if not os.path.isfile(exe_abs):
            raise FileNotFoundError(f"KataGo exe not found: {exe_abs}")
        if not os.path.isfile(model_abs):
            raise FileNotFoundError(f"KataGo model not found: {model_abs}")
        if not os.path.isfile(cfg_abs):
            raise FileNotFoundError(f"KataGo cfg not found: {cfg_abs}")

        print(f"[katago] launching: {exe_abs} (model={model_abs}, cfg={cfg_abs})")

        self._proc = subprocess.Popen(
            [exe_abs, "gtp", "-model", model_abs, "-config", cfg_abs],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr for visibility
            text=True,
            bufsize=1,
            cwd=exe_dir,
        )

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read(self, timeout: float = 60.0) -> str:
        import time
        assert self._proc.stdout is not None
        lines: List[str] = []
        end = time.time() + float(timeout)
        saw_ok = False  # track if we saw an '=' acknowledgement line
        while True:
            # Timeout guard
            if time.time() > end:
                raise TimeoutError("No GTP reply from KataGo within timeout. Check exe/model/cfg.")

            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError("KataGo process ended unexpectedly.")
            s = line.strip()
            if s == "":
                # Break on blank line if we have any payload OR we saw an '=' ack
                if lines or saw_ok:
                    break
                continue
            if s.startswith("?"):
                # Read until blank line then raise
                err_lines = [s]
                while True:
                    more = self._proc.stdout.readline()
                    if more == "" or more.strip() == "":
                        break
                    err_lines.append(more.strip())
                raise RuntimeError("GTP error: " + " | ".join(err_lines))
            if s.startswith("="):
                saw_ok = True
                payload = s[1:].strip()
                if payload:
                    lines.append(payload)
            else:
                lines.append(s)
        return "\n".join(lines)

    def boardsize(self, n: int) -> None:
        self._send(f"boardsize {int(n)}")
        self._read()

    def komi(self, k: float) -> None:
        self._send(f"komi {float(k)}")
        self._read()

    def clear_board(self) -> None:
        self._send("clear_board")
        self._read()

    def play(self, color: str, move: str) -> None:
        self._send(f"play {color} {move}")
        self._read()

    def genmove(self, color: str, timeout: float = 300.0) -> str:
        print("[katago] generating move..")
        self._send(f"genmove {color}")
        return self._read(timeout=float(timeout)).strip()

    def close(self) -> None:
        try:
            if self._proc and self._proc.stdin:
                try:
                    self._send("quit")
                except Exception:
                    pass
            if self._proc:
                try:
                    self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
        finally:
            try:
                if self._proc and self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                if self._proc and self._proc.stdout:
                    self._proc.stdout.close()
            except Exception:
                pass
            try:
                if self._proc and self._proc.stderr:
                    self._proc.stderr.close()
            except Exception:
                pass


def _build_mcts(net: MLPPolicyValue,
                size: int,
                c_puct: float,
                dirichlet_alpha: Optional[float],
                dirichlet_frac: float,
                dirichlet_c0: float) -> ScoreAwareMCTS:
    return build_mcts_standard(net, size, c_puct, dirichlet_alpha, dirichlet_frac, dirichlet_c0)


def _index_to_gtp(action: int, size: int) -> str:
    if int(action) == size * size:
        return "pass"
    r, c = divmod(int(action), int(size))
    letters = ALPHABET[:size]
    return f"{letters[c]}{int(size) - int(r)}"


def _gtp_to_index(coord: str, size: int) -> int:
    s = coord.strip().upper()
    if s == "PASS":
        return size * size
    if s == "RESIGN":
        raise ValueError("RESIGN move provided; should be handled by caller.")
    letters = ALPHABET[:size]
    col = letters.index(s[0])
    row_num = int(s[1:])
    r = int(size) - int(row_num)
    c = int(col)
    return r * int(size) + c


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _expand_exe_candidates(path: Optional[str],
                           exe_names: List[str],
                           search_subdirs: bool = True) -> List[str]:
    if not path:
        return []
    path = os.path.abspath(path)
    candidates: List[str] = []
    if os.path.isdir(path):
        for name in exe_names:
            candidates.append(os.path.join(path, name))
        if search_subdirs:
            try:
                for entry in sorted(os.listdir(path)):
                    sub = os.path.join(path, entry)
                    if not os.path.isdir(sub):
                        continue
                    for name in exe_names:
                        candidates.append(os.path.join(sub, name))
            except OSError:
                pass
    else:
        candidates.append(path)
    return candidates


def _expand_model_candidates(path: Optional[str]) -> List[str]:
    if not path:
        return []
    path = os.path.abspath(path)
    if os.path.isdir(path):
        matches: List[str] = []
        for pattern in ("*.bin", "*.bin.gz"):
            matches.extend(sorted(glob.glob(os.path.join(path, pattern))))
        return matches
    return [path]


def _expand_cfg_candidates(path: Optional[str]) -> List[str]:
    if not path:
        return []
    path = os.path.abspath(path)
    if os.path.isdir(path):
        matches: List[str] = []
        for pattern in ("*.cfg", "*.cfg.in"):
            matches.extend(sorted(glob.glob(os.path.join(path, pattern))))
        return matches
    return [path]


def _take_first_existing(paths: List[Optional[str]]) -> Optional[str]:
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _resolve_katago_assets(katago_exe: Optional[str],
                           katago_model: Optional[str],
                           katago_cfg: Optional[str],
                           auto_install: bool = False,
                           force_install: bool = False) -> Tuple[str, str, str]:
    repo_dir = os.path.dirname(__file__)
    default_dir = os.path.join(repo_dir, "katago")

    import platform

    sys_is_windows = platform.system().lower().startswith("win")
    exe_names = ["katago.exe", "katago"] if sys_is_windows else ["katago", "katago.exe"]
    exe_candidates_raw: List[str] = []
    for raw in [katago_exe, os.getenv("KATAGO_EXE"), default_dir]:
        exe_candidates_raw.extend(_expand_exe_candidates(raw, exe_names))
    which_katago = shutil.which("katago")
    if which_katago:
        exe_candidates_raw.append(which_katago)

    exe_candidates = _dedupe_preserve_order(exe_candidates_raw)
    exe_path = _take_first_existing(exe_candidates)
    katago_dir: Optional[Path] = None
    if not exe_path and auto_install:
        katago_dir = _auto_install_katago(force=force_install, env_name=_DEFAULT_KATAGO_ENV)
        _load_katago_env_hints(katago_dir)
        exe_candidates = _dedupe_preserve_order(exe_candidates_raw)
        exe_candidates.extend(_expand_exe_candidates(str(katago_dir), exe_names))
        exe_path = _take_first_existing(exe_candidates)
    if not exe_path:
        raise FileNotFoundError(
            "KataGo executable not found. Provide --katago_exe, set KATAGO_EXE, or place the binary under <repo>/katago/."
        )

    exe_dir = os.path.dirname(exe_path)

    model_candidates_raw: List[str] = []
    for raw in [katago_model, os.getenv("KATAGO_MODEL")]:
        model_candidates_raw.extend(_expand_model_candidates(raw))
    for base in (exe_dir, default_dir):
        model_candidates_raw.extend(_expand_model_candidates(base))
    model_candidates = _dedupe_preserve_order(model_candidates_raw)
    model_path = _take_first_existing(model_candidates)
    if not model_path and auto_install:
        if katago_dir is None:
            katago_dir = _auto_install_katago(force=force_install, env_name=_DEFAULT_KATAGO_ENV)
            _load_katago_env_hints(katago_dir)
        model_candidates = _dedupe_preserve_order(model_candidates_raw)
        model_candidates.extend(_expand_model_candidates(str(katago_dir)))
        model_path = _take_first_existing(model_candidates)
    if not model_path:
        raise FileNotFoundError(
            "KataGo model file not found. Provide --katago_model, set KATAGO_MODEL, or place a model (.bin or .bin.gz) next to katago.exe."
        )

    cfg_candidates_raw: List[str] = []
    for raw in [katago_cfg, os.getenv("KATAGO_CFG")]:
        cfg_candidates_raw.extend(_expand_cfg_candidates(raw))
    repo_cfg = os.path.join(default_dir, "default_gtp.cfg")
    cfg_candidates_raw.extend(_expand_cfg_candidates(default_dir))
    cfg_candidates_raw.extend(_expand_cfg_candidates(exe_dir))
    if os.path.isfile(repo_cfg):
        cfg_candidates_raw.append(repo_cfg)
    for name in ("default_gtp.cfg", "gtp_example.cfg", "analysis_example.cfg"):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate):
            cfg_candidates_raw.append(candidate)
    cfg_candidates = _dedupe_preserve_order(cfg_candidates_raw)
    cfg_path = _take_first_existing(cfg_candidates)
    if not cfg_path and auto_install:
        if katago_dir is None:
            katago_dir = _auto_install_katago(force=force_install, env_name=_DEFAULT_KATAGO_ENV)
            _load_katago_env_hints(katago_dir)
        cfg_candidates = _dedupe_preserve_order(cfg_candidates_raw)
        cfg_candidates.extend(_expand_cfg_candidates(str(katago_dir)))
        cfg_path = _take_first_existing(cfg_candidates)
    if not cfg_path:
        raise FileNotFoundError(
            "KataGo GTP config not found. Provide --katago_cfg, set KATAGO_CFG, or copy a default_gtp.cfg next to katago.exe."
        )

    return exe_path, model_path, cfg_path


def _play_vs_katago_game(net: MLPPolicyValue,
                         gtp: GTPClient,
                         size: int,
                         sims: int,
                         our_color: str,
                         c_puct: float,
                         dirichlet_alpha: Optional[float],
                         dirichlet_frac: float,
                         dirichlet_c0: float,
                         temp_t0: float,
                         temp_min: float,
                         temp_decay: float,
                         min_game_len: int,
                         komi: float) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, float]]:
    # Align rules with KataGo's default 'tromp-taylor' (positional superko, no suicide)
    board = Board(size, enforce_rules=True, forbid_suicide=True, ko_rule='psk')
    board.rule_info = 0.0

    # Ensure KataGo is in a matching initial state
    # KataGo starts clear at initialization; still clear between games for safety.
    print(f"[katago] connecting (size={int(size)}, komi={float(komi)})...")
    gtp.boardsize(size)
    gtp.komi(float(komi))
    gtp.clear_board()

    features: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    our_moves_colors: List[int] = []  # +1 if we played as Black on that sample, -1 if White

    mcts = _build_mcts(net,
                       size=size,
                       c_puct=c_puct,
                       dirichlet_alpha=dirichlet_alpha,
                       dirichlet_frac=dirichlet_frac,
                       dirichlet_c0=dirichlet_c0)

    t = 0
    while True:
        to_move_color = 'b' if board.turn == 1 else 'w'
        if to_move_color == our_color:
            # Our agent selects a move via MCTS
            mcts.run(board, num_simulations=int(sims))
            temp = temperature_schedule(t, t0=float(temp_t0), t_min=float(temp_min), decay=float(temp_decay))
            pi = mcts.get_action_probs(temp=float(temp))
            

            # Record training example (from our turns only)
            features.append(board.to_features().copy())
            policies.append(pi.astype(np.float32))
            our_moves_colors.append(1 if our_color == 'b' else -1)

            # Sample legal move only
            legal = board.legal_moves().astype(int)
            pi_masked = np.zeros_like(pi)
            pi_masked[legal] = pi[legal]
            s = float(pi_masked.sum())
            if s <= 0.0:
                pi_masked = np.zeros_like(pi_masked)
                pi_masked[legal] = 1.0 / max(1, len(legal))
            else:
                pi_masked = pi_masked / s

            pi_working = pi_masked.copy()
            accepted_action: Optional[int] = None
            retries = 0
            while True:
                total = float(pi_working.sum())
                if total <= 0.0:
                    raise RuntimeError(
                        "No legal moves available after filtering illegal selections; check rule alignment."
                    )
                probs = pi_working / total
                candidate = int(np.random.choice(len(probs), p=probs))
                hist_before = len(board.history)
                board.play(candidate)
                if len(board.history) == hist_before:
                    retries += 1
                    pi_working[candidate] = 0.0
                    # Refresh legal mask to avoid stale entries if rules changed after KataGo response
                    current_legals = board.legal_moves().astype(int)
                    invalid = set(np.where(pi_working > 0.0)[0]) - set(current_legals)
                    for idx in invalid:
                        pi_working[int(idx)] = 0.0
                    print(
                        f"[warn] filtered illegal MCTS move {candidate} (attempt {retries}); resampling."
                    )
                    continue
                accepted_action = candidate
                break

            assert accepted_action is not None
            gtp_coord = _index_to_gtp(accepted_action, size)
            if accepted_action == board.pass_index:
                print(f"[mcts] move {t} (us {our_color}) -> pass [pass]")
            else:
                r, c = divmod(int(accepted_action), int(size))
                print(f"[mcts] move {t} (us {our_color}) -> {r},{c} [{gtp_coord}]")
            # Inform KataGo of our move to keep states in sync
            gtp.play('B' if our_color == 'b' else 'W', gtp_coord)
            # Root reuse: advance the MCTS root to the played action to preserve subtree
            try:
                mcts.advance_root(int(accepted_action))
            except Exception:
                pass
        else:
            # KataGo move via GTP
            move_str = gtp.genmove('B' if to_move_color == 'b' else 'W')
            print(f"[katago] move {t} ({to_move_color}) -> {move_str.strip()}")
            s_move = move_str.strip().upper()
            if s_move == "RESIGN":
                # If KataGo resigns on its turn, our side wins
                s_final = float(rules.final_margin(
                    board.grid,
                    getattr(board, 'captures_black', 0),
                    getattr(board, 'captures_white', 0),
                    use_capture_aware=True,
                    komi=0.0,
                ))
                winner = 1 if our_color == 'b' else -1
                return features, policies, (winner, float(s_final))
            try:
                action = _gtp_to_index(move_str, size)
            except Exception as e:
                # Treat unknown as pass to avoid desync, but warn for visibility
                print(f"[warn] malformed GTP move '{move_str.strip()}': {e}; mapping to PASS")
                action = board.pass_index
            board.play(action)

        # end conditions
        if (
            len(board.history) >= 2
            and board.history[-1] == board.pass_index
            and board.history[-2] == board.pass_index
        ):
            break

        # Continue until both players pass; do not cap at board size so games can
        # exceed N*N moves when captures reopen intersections.
        t += 1

    # Final margin via unified helper (komi handled at target stage)
    s_final = float(rules.final_margin(
        board.grid,
        getattr(board, 'captures_black', 0),
        getattr(board, 'captures_white', 0),
        use_capture_aware=True,
        komi=0.0,
    ))
    winner = 1 if s_final > 0 else (-1 if s_final < 0 else 0)
    return features, policies, (winner, float(s_final))


def train_vs_katago(net: MLPPolicyValue,
                    games: int = 10,
                    sims: int = 64,
                    size: int = 9,
                    lr: float = 3e-3,
                    value_weight: float = 1.0,
                    l2: float = 1e-4,
                    c_puct: float = 2.0,
                    dirichlet_alpha: Optional[float] = 0.15,
                    dirichlet_frac: float = 0.25,
                    dirichlet_c0: float = 10.0,
                    temp_t0: float = 1.0,
                    temp_min: float = 0.25,
                    temp_decay: float = 0.995,
                    min_game_len: int = 50,
                    checkpoint_every: int = 1,
                    katago_exe: Optional[str] = None,
                    katago_model: Optional[str] = None,
                    katago_cfg: Optional[str] = None,
                    auto_install_assets: bool = False,
                    force_install_assets: bool = False,
                    komi: float = 6.5,
                    include_komi_in_margin: bool = False) -> None:
    katago_exe, katago_model, katago_cfg = _resolve_katago_assets(
        katago_exe=katago_exe,
        katago_model=katago_model,
        katago_cfg=katago_cfg,
        auto_install=auto_install_assets,
        force_install=force_install_assets,
    )

    gtp = GTPClient(katago_exe=katago_exe, model_path=katago_model, config_path=katago_cfg)
    # Startup handshake to fail fast if engine is not responsive
    try:
        try:
            # Ask engine name
            from typing import Optional as _Optional
            _ = None  # placate linters about unused import
            gtp._send("name")
            name = gtp._read(timeout=180.0)
            print(f"[katago] engine name: {name}")
        except Exception as e:
            raise RuntimeError(f"KataGo did not respond to 'name': {e}")
        try:
            gtp._send("protocol_version")
            proto = gtp._read(timeout=180.0)
            print(f"[katago] protocol: {proto}")
        except Exception as e:
            raise RuntimeError(f"KataGo did not respond to 'protocol_version': {e}")
    except Exception as e:
        gtp.close()
        raise SystemExit(f"Failed to start KataGo: {e}")
    try:
        for gi in range(int(games)):
            our_color = 'b' if (gi % 2 == 0) else 'w'  # alternate colors
            feats, pis, outcome = _play_vs_katago_game(
                net=net,
                gtp=gtp,
                size=int(size),
                sims=int(sims),
                our_color=our_color,
                c_puct=float(c_puct),
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_frac=float(dirichlet_frac),
                dirichlet_c0=float(dirichlet_c0),
                temp_t0=float(temp_t0),
                temp_min=float(temp_min),
                temp_decay=float(temp_decay),
                min_game_len=int(min_game_len),
                komi=float(komi),
            )

            winner, s_final = outcome
            N = int(size)
            tau = max(1.0, 0.5 * N)
            margin = float(s_final) - (float(komi) if include_komi_in_margin else 0.0)
            z_black = float(np.tanh(margin / float(tau)))

            # Build per-sample targets from the perspective of our agent at each recorded step
            if feats:
                color_sign = 1.0 if our_color == 'b' else -1.0
                Xg = np.stack([f.astype(np.float32) for f in feats], axis=0)
                Pig = np.stack([p for p in pis], axis=0)
                Zg = np.stack([np.array([color_sign * z_black], dtype=np.float32) for _ in feats], axis=0)

                # One game update (saves inside step via checkpoint_every)
                _, _, cache_g = net.forward(Xg)
                loss_g, grads_g = net.backward(cache_g, Pig, Zg, l2=float(l2), c_v=float(value_weight))
                try:
                    net.step(grads_g, lr=float(lr), save_every=int(max(1, checkpoint_every)))
                except TypeError:
                    net.step(grads_g, lr=float(lr))
            print(f"[train] vs-KataGo weights updated after game {gi+1}; loss={loss_g:.6f}")
    finally:
        gtp.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train policy/value by playing vs KataGo (alternating colors)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--c_puct", type=float, default=2.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.15)
    parser.add_argument("--dirichlet_frac", type=float, default=0.25)
    parser.add_argument("--dirichlet_c0", type=float, default=10.0)
    parser.add_argument("--temp_t0", type=float, default=1.0)
    parser.add_argument("--temp_min", type=float, default=0.25)
    parser.add_argument("--temp_decay", type=float, default=0.995)
    parser.add_argument("--min_game_len", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--require_gpu", action="store_true", default=True)
    parser.add_argument(
        "--auto_install_assets",
        action="store_true",
        help="Automatically run scripts/install_katago.py when KataGo assets are missing",
    )
    parser.add_argument(
        "--force_install_assets",
        action="store_true",
        help="Force re-download of KataGo assets when auto-installing",
    )
    # KataGo paths (defaults to repo katago folder)
    parser.add_argument("--katago_exe", type=str, default=None, help="Path to KataGo binary (default: <repo>/katago/katago[.exe])")
    parser.add_argument("--katago_model", type=str, default=None, help="Path to KataGo model (default: <repo>/katago/model.bin)")
    parser.add_argument("--katago_cfg", type=str, default=None, help="Path to KataGo GTP config (default: <repo>/katago/default_gtp.cfg)")
    parser.add_argument("--komi", type=float, default=6.5)
    parser.add_argument("--include_komi_in_margin", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    device = _resolve_device(device_arg=args.device, require_gpu=bool(args.require_gpu))

    net = MLPPolicyValue(device=device)

    alpha = None if args.dirichlet_alpha is not None and args.dirichlet_alpha <= 0 else args.dirichlet_alpha

    # Optional determinism
    _seed_everything(seed=args.seed, device=device)

    train_vs_katago(
        net=net,
        games=args.games,
        sims=args.sims,
        size=args.size,
        lr=args.lr,
        value_weight=args.value_weight,
        l2=args.l2,
        c_puct=args.c_puct,
        dirichlet_alpha=alpha,
        dirichlet_frac=args.dirichlet_frac,
        dirichlet_c0=args.dirichlet_c0,
        temp_t0=args.temp_t0,
        temp_min=args.temp_min,
        temp_decay=args.temp_decay,
        min_game_len=int(args.min_game_len),
        checkpoint_every=max(1, int(args.checkpoint_every)),
        katago_exe=args.katago_exe,
        katago_model=args.katago_model,
        katago_cfg=args.katago_cfg,
        auto_install_assets=bool(args.auto_install_assets or (_running_in_notebook() and not (args.katago_exe and args.katago_model and args.katago_cfg))),
        force_install_assets=bool(args.force_install_assets),
        komi=float(args.komi),
        include_komi_in_margin=bool(args.include_komi_in_margin),
    )

    # quick sanity probe on empty board
    empty = Board(args.size)
    p, v = infer_policy_value(net, empty)
    print("trained vs KataGo.")
    print("empty board value estimate:", float(v))
    print("policy mass:", float(p.sum()))


if __name__ == "__main__":
    main()



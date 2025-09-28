#!/usr/bin/env python3
"""Utility to fetch KataGo CUDA assets for SageMaker-style environments.

This script automates the workflow described in the companion notebook
setup, including:

* ensuring a lightweight conda runtime environment with CUDA-friendly
  shared libraries (libzip, openssl, zlib, zstd)
* downloading the KataGo CUDA binary release
* relocating bundled libraries so that conda-provided libraries take
  precedence at runtime
* downloading a neural network model into `katago/models`
* writing a minimal default GTP configuration and conda prefix hint so
  other tools (e.g. `trainwithkatago.py`) can launch KataGo reliably

Run this on the target machine (e.g. AWS SageMaker notebook instance)
before invoking `trainwithkatago.py --auto_install_assets`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Iterable, Optional

try:
    from urllib.request import urlopen
except ImportError:  # pragma: no cover - very old Pythons
    urlopen = None  # type: ignore

try:
    import zipfile
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"zipfile module is required: {exc}")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KATAGO_DIR = REPO_ROOT / "katago"
DEFAULT_MODEL_URL = (
    "https://katagotraining.org/networks/kata1/"
    "kata1-b40c256-s5038979072-d1229425124.bin.gz"
)


class InstallError(RuntimeError):
    """Raised when a critical installation step fails."""


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _run(cmd: Iterable[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd_list = list(cmd)
    print(f"[cmd] {' '.join(cmd_list)}")
    try:
        return subprocess.run(cmd_list, check=check, text=True)
    except FileNotFoundError as exc:  # pragma: no cover - better diagnostics
        raise InstallError(f"Command not found: {cmd_list[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise InstallError(f"Command failed ({exc.returncode}): {' '.join(cmd_list)}") from exc


def _check_gpu() -> None:
    try:
        completed = subprocess.run(["nvidia-smi"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        print("[warn] nvidia-smi not available; GPU check skipped.")
        return
    if completed.returncode != 0:
        print("[warn] nvidia-smi exited with non-zero status; output follows:")
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    else:
        print("[info] nvidia-smi output:")
        sys.stdout.write(completed.stdout)


def _find_conda() -> Optional[Path]:
    candidates = []
    if "CONDA_EXE" in os.environ:
        candidates.append(Path(os.environ["CONDA_EXE"]))
    which = shutil.which("conda")
    if which:
        candidates.append(Path(which))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _conda_base(conda_exe: Path) -> Path:
    completed = subprocess.run(
        [str(conda_exe), "info", "--base"],
        check=True,
        capture_output=True,
        text=True,
    )
    base_path = completed.stdout.strip().splitlines()[-1]
    base = Path(base_path)
    if not base.exists():
        raise InstallError(f"Resolved conda base path does not exist: {base}")
    return base


def _ensure_mamba(conda_exe: Path) -> Path:
    mamba = shutil.which("mamba")
    if mamba:
        return Path(mamba).resolve()
    print("[setup] Installing mamba into base environment…")
    _run([str(conda_exe), "install", "-y", "-n", "base", "-c", "conda-forge", "mamba"])
    mamba = shutil.which("mamba")
    if not mamba:
        raise InstallError("mamba installation reported success but binary not found on PATH")
    return Path(mamba).resolve()


def _ensure_env(
    conda_exe: Path,
    mamba_exe: Path,
    env_name: str,
    packages: Iterable[str],
    force: bool,
) -> Path:
    base = _conda_base(conda_exe)
    prefix = base / "envs" / env_name
    if prefix.exists():
        if force:
            print(f"[setup] Removing existing env: {env_name}")
            _run([str(conda_exe), "env", "remove", "-y", "-n", env_name])
        else:
            print(f"[setup] Conda env '{env_name}' already exists: {prefix}")
            return prefix

    pkg_args = list(packages)
    cmd = [str(conda_exe), "run", "-n", "base", str(mamba_exe), "create", "-y", "-n", env_name]
    cmd.extend(pkg_args)
    try:
        _run(cmd)
    except InstallError as exc:
        # Fall back to invoking mamba directly if conda run is unavailable.
        print(f"[warn] {exc}. Retrying with direct mamba invocation…")
        direct_cmd = [str(mamba_exe), "create", "-y", "-n", env_name]
        direct_cmd.extend(pkg_args)
        _run(direct_cmd)
    if not prefix.exists():
        raise InstallError(f"Conda environment creation succeeded but prefix missing: {prefix}")
    return prefix


def _download_file(url: str, dest: Path) -> None:
    if urlopen is None:
        raise InstallError("urllib.request is unavailable; cannot download files")
    print(f"[download] Fetching {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urlopen(url) as response:  # type: ignore[arg-type]
            shutil.copyfileobj(response, tmp)
    tmp_path.replace(dest)


def _ensure_katago_binary(
    katago_dir: Path,
    version: str,
    force: bool,
) -> Path:
    katago_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "katago.exe" if sys.platform.startswith("win") else "katago"
    existing_candidates = sorted(katago_dir.rglob(exe_name))
    if existing_candidates and not force:
        exe_path = existing_candidates[0]
        print(f"[setup] KataGo binary already present: {exe_path}")
        return exe_path

    zip_name = f"katago-{version}-cuda-linux-x64.zip"
    url = f"https://github.com/lightvector/KataGo/releases/download/{version}/{zip_name}"
    archive_path = katago_dir / zip_name
    _download_file(url, archive_path)

    print(f"[setup] Extracting {archive_path} -> {katago_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(katago_dir)
    archive_path.unlink(missing_ok=True)

    exe_candidates = sorted(katago_dir.rglob(exe_name))
    if not exe_candidates:
        raise InstallError("KataGo executable not found after extraction")
    exe_path = exe_candidates[0]
    try:
        mode = exe_path.stat().st_mode
        exe_path.chmod(mode | 0o111)
    except PermissionError:
        pass

    exe_dir = exe_path.parent
    bundled = [p for p in exe_dir.glob("lib*") if p.exists()]
    if bundled:
        backup_dir = exe_dir / "bundled_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for item in bundled:
            target = backup_dir / item.name
            if target.exists():
                if force:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                else:
                    continue
            print(f"[setup] Moving bundled {item.name} -> {target}")
            shutil.move(str(item), str(target))

    print(f"[setup] KataGo executable ready: {exe_path}")
    return exe_path


def _ensure_model(models_dir: Path, model_url: str, force: bool) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_url).name
    model_path = models_dir / model_name
    if model_path.exists() and not force:
        print(f"[setup] KataGo model already present: {model_path}")
        return model_path
    _download_file(model_url, model_path)
    print(f"[setup] KataGo model downloaded: {model_path}")
    return model_path


def _ensure_default_cfg(katago_dir: Path, force: bool) -> Path:
    cfg_path = katago_dir / "default_gtp.cfg"
    if cfg_path.exists() and not force:
        print(f"[setup] GTP config already present: {cfg_path}")
        return cfg_path
    cfg_body = textwrap.dedent(
        """
        # Minimal GTP config suitable for CUDA benchmarks and Sabaki engines
        logFile = gtp.log
        maxVisits = 1000
        numSearchThreads = 4
        ponderingEnabled = false

        koRule = SIMPLE
        scoringRule = AREA
        taxRule = NONE
        multiStoneSuicideLegal = false
        whiteHandicapBonus = 0

        analysisPVLen = 15
        """
    ).strip() + "\n"
    cfg_path.write_text(cfg_body, encoding="utf-8")
    print(f"[setup] Default GTP config written: {cfg_path}")
    return cfg_path


def _write_prefix_hint(katago_dir: Path, prefix: Path) -> Path:
    hint_path = katago_dir / "conda_prefix.txt"
    hint_path.write_text(str(prefix), encoding="utf-8")
    print(f"[setup] Conda prefix hint stored at {hint_path}")
    return hint_path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Install KataGo CUDA assets and supporting conda environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--katago-dir", type=Path, default=DEFAULT_KATAGO_DIR,
                        help="Directory where KataGo assets will be stored")
    parser.add_argument("--version", default="v1.16.3", help="KataGo release tag to download")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                        help="URL to a KataGo neural network (.bin or .bin.gz)")
    parser.add_argument("--env-name", default="katago-env", help="Name of the conda environment to create")
    parser.add_argument("--force", action="store_true",
                        help="Re-download assets and recreate environments even if they exist")
    parser.add_argument("--skip-conda", action="store_true",
                        help="Skip conda environment provisioning (assumes PATH/LD_LIBRARY_PATH already configured)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    _print_header("KataGo CUDA Installer")
    _check_gpu()

    env_prefix: Optional[Path] = None
    conda_exe: Optional[Path] = None
    mamba_exe: Optional[Path] = None
    if not args.skip_conda:
        conda_exe = _find_conda()
        if not conda_exe:
            raise SystemExit("conda executable not found. Set CONDA_EXE or adjust PATH, or use --skip-conda.")
        print(f"[info] Using conda at {conda_exe}")
        mamba_exe = _ensure_mamba(conda_exe)
        print(f"[info] Using mamba at {mamba_exe}")
        env_prefix = _ensure_env(
            conda_exe,
            mamba_exe,
            args.env_name,
            packages=(
                "python=3.10",
                "libzip",
                "openssl>=3",
                "zlib",
                "zstd",
            ),
            force=args.force,
        )
        print(f"[setup] Conda environment ready at {env_prefix}")

    exe_path = _ensure_katago_binary(args.katago_dir, args.version, args.force)
    model_path = _ensure_model(args.katago_dir / "models", args.model_url, args.force)
    cfg_path = _ensure_default_cfg(args.katago_dir, args.force)
    if env_prefix is not None:
        _write_prefix_hint(args.katago_dir, env_prefix)

    print("\nInstallation complete.")
    print("Summary:")
    print(f"  KataGo executable : {exe_path}")
    print(f"  Model file        : {model_path}")
    print(f"  GTP config        : {cfg_path}")
    if env_prefix is not None:
        exports = textwrap.dedent(
            f"""
            To use KataGo within your current shell session, activate environment hints:

                export KATAGO_CONDA_PREFIX="{env_prefix}"
                export PATH="{env_prefix / 'bin'}"${{PATH:+:${{PATH}}}}
                export LD_LIBRARY_PATH="{env_prefix / 'lib'}"${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}

            Alternatively, rerun this installer on each fresh session with --skip-conda
            after sourcing the exports manually.
            """
        ).strip()
        print(exports)


if __name__ == "__main__":
    main()


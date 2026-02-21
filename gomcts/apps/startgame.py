"""Entry point for playing against the engine via the interactive CLI."""

from __future__ import annotations

import argparse

from .cli import cli as run_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against gomcts via interactive CLI")
    parser.add_argument("--size", type=int, default=9, help="initial board size")
    parser.add_argument("--sims", type=int, default=64, help="default MCTS simulations for genmove")
    parser.add_argument("--temp", type=float, default=1.0, help="default temperature for genmove")
    args = parser.parse_args()

    run_cli(size=int(args.size), default_sims=int(args.sims), default_temp=float(args.temp))


if __name__ == "__main__":
    main()




import argparse
from play_one_move import main as play_one_move_main
try:
    # Optional CLI: guard import so --cli fails gracefully if cli.py missing
    from cli import cli as run_cli
except Exception:
    run_cli = None


def main() -> None:
    parser = argparse.ArgumentParser(description="gomcts entrypoint")
    parser.add_argument("--cli", action="store_true", help="start interactive CLI")
    # For now, size is handled inside CLI/selftraining; play_one_move remains 9x9 demo.
    args = parser.parse_args()

    if args.cli:
        if run_cli is None:
            raise SystemExit("CLI not available; add cli.py or disable --cli")
        run_cli()
    else:
        play_one_move_main()


if __name__ == "__main__":
    main()




# Copy/Paste Commands

This file is intentionally terse: it’s a set of common commands you can copy/paste from the repo root.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train from SGF (supervised)

```bash
python -m gomcts.training.train_from_sgf \
  --sgf_dir ~/go_games/9x9 \
  --board_size 9 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.001
```

## Train vs KataGo

```bash
python -m gomcts.training.trainwithkatago \
  --games 500 \
  --sims 64 \
  --size 9 \
  --lr 0.001 \
  --auto_install_assets
```

## Self-play

```bash
python -m gomcts.training.selftraining --games 200 --sims 64 --size 9
```

## Play (interactive)

```bash
python -m gomcts.apps.startgame --size 9 --sims 64
```

## Artifacts

- Weights are saved to: `artifacts/weights.pt`


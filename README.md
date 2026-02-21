# gomcts

Toy Go engine + MCTS + policy/value net, with training entrypoints for:
- supervised learning from SGF files
- training vs KataGo
- pure self-play

This repo is intentionally minimal: **start here**, everything else is in the code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- On Jetson/ARM64, install PyTorch via NVIDIA/Jetson wheels (the `requirements.txt` intentionally avoids pinning torch there).

## Train (pick one)

### 1) Supervised learning from SGFs (fastest)

```bash
python -m gomcts.training.train_from_sgf \
  --sgf_dir ~/go_games/9x9 \
  --board_size 9 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.001
```

### 2) Train vs KataGo (strongest on limited compute)

```bash
python -m gomcts.training.trainwithkatago \
  --games 500 \
  --sims 64 \
  --size 9 \
  --lr 0.001 \
  --auto_install_assets
```

Or use the helper:

```bash
./scripts/train_jetson.sh 500
```

### 3) Pure self-play (slow, research-y)

```bash
python -m gomcts.training.selftraining --games 200 --sims 64 --size 9
```

You can also run config-driven self-play:

```bash
python -m gomcts.training.selftraining --config configs/train_single.yaml
```

## Play (interactive)

Launch the interactive CLI:

```bash
python -m gomcts.apps.startgame --size 9 --sims 64
```

Inside the CLI:
- `boardsize N`
- `showboard`
- `play b|w <coord|PASS>` (e.g. `play b D4`)
- `genmove [b|w] [sims] [temp]`
- `finalscore`
- `undo`
- `exit`

## Repo structure (where the real code lives)

- **Canonical implementation**: `gomcts/`
  - `gomcts/core/`: board, rules, MCTS
  - `gomcts/neural/`: policy/value network + inference helpers
- **Apps**: `gomcts/apps/` (interactive play)
- **Training**: `gomcts/training/` (SGF training, self-play, KataGo training)
- **Docs**: `docs/`
- **Examples**: `examples/` (sample SGFs + tiny demos)
- **Artifacts**: `artifacts/` (weights, eval snapshots, etc.)


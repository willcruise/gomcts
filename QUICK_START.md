# Quick Start: Training Your Go AI

## 🚀 Three Ways to Train (Pick One)

### 1️⃣ FASTEST: Train from Pro Games (2-6 hours) ⚡

```bash
# Step 1: Get SGF files (see SGF_TRAINING_GUIDE.md)
# Step 2: Train!
python train_from_sgf.py \
    --sgf_dir ~/go_games/9x9 \
    --epochs 10 \
    --board_size 9
```

**Best for**: Quick results, limited time

---

### 2️⃣ STRONGEST: Train vs KataGo (2-7 days) 🎯

```bash
# One command does it all (auto-installs KataGo)
./train_jetson.sh 500

# Or manually:
python trainwithkatago.py \
    --games 500 \
    --sims 64 \
    --size 9 \
    --auto_install_assets
```

**Best for**: Maximum strength on Jetson

---

### 3️⃣ RECOMMENDED: Hybrid Approach (3-4 days) 🏆

```bash
# Phase 1: Bootstrap fast (4 hours)
python train_from_sgf.py --sgf_dir ~/go_games --epochs 15 --board_size 9

# Phase 2: Get strong (3 days)
python trainwithkatago.py --games 1000 --sims 64 --size 9 --lr 0.0003
```

**Best for**: Optimal results with limited compute

---

## 🧪 Test Immediately (No Downloads)

```bash
# Test with provided sample files
python train_from_sgf.py \
    --sgf_dir ./sample_sgf_games \
    --epochs 3 \
    --board_size 9
```

---

## 🎮 Play Against Your AI

```bash
python startgame.py --sims 64
```

---

## 📚 Full Documentation

- `TRAINING_COMPARISON.md` - Compare all methods
- `SGF_TRAINING_GUIDE.md` - Detailed SGF training guide
- `TRAINING_ON_JETSON.md` - Jetson optimization tips

---

## ❓ Which Method Should I Use?

| If you have... | Use this... |
|----------------|-------------|
| 2-6 hours | SGF supervised learning |
| 2-7 days | Train vs KataGo |
| 3-4 days | Hybrid (SGF → KataGo) ← **BEST!** |
| Weeks | Pure self-play |
| Just testing | Sample SGF files |

---

## 💡 Key Insight

**You don't need millions of games like AlphaZero!**

By learning from expert games (SGF) or a strong teacher (KataGo), you can reach strong amateur level in **just a few days** on your Jetson! 🎉


# Copy-Paste Commands for SGF Training

## 🚀 Quick Commands (Just Copy & Run!)

### Setup: Create Directory

```bash
mkdir -p ~/go_games/kgs_9x9
cd ~/go_games/kgs_9x9
```

---

## Option 1: Download from KGS (Manual, Easiest)

1. **Visit**: https://www.gokgs.com/gameArchives.jsp
2. **Filter**: Board size = 9, Rank = 5d or higher
3. **Download**: Click games and save SGF files to `~/go_games/kgs_9x9/`
4. **Train**:

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 128
```

---

## Option 2: Test Immediately (No Downloads)

Use the sample files I already created:

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ./sample_sgf_games \
    --board_size 9 \
    --epochs 3 \
    --batch_size 32
```

---

## Option 3: Download from KGS (Automated)

```bash
cd ~/go_games/kgs_9x9

# Download games from a specific date range
# Replace YYYY-MM-DD with actual dates
wget -r -np -nd -A "*.sgf" \
  --accept-regex "gameArchives\.jsp.*" \
  "https://www.gokgs.com/gameArchives.jsp"
```

---

## Option 4: Clone GitHub Dataset

```bash
cd ~
git clone https://github.com/yenw/computer-go-dataset.git

cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py \
    --sgf_dir ~/computer-go-dataset \
    --board_size 9 \
    --epochs 10
```

---

## Verify Your Downloads

```bash
# Count SGF files
ls -1 ~/go_games/kgs_9x9/*.sgf | wc -l

# Check board sizes
grep "SZ\[" ~/go_games/kgs_9x9/*.sgf | grep -o "SZ\[[0-9]*\]" | sort | uniq -c

# View sample game
head -n 20 ~/go_games/kgs_9x9/*.sgf | head -n 30
```

---

## Full Training Pipeline

### Phase 1: SGF Training (2-4 hours)

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001
```

### Phase 2: Test Your Network

```bash
python startgame.py --sims 64
```

### Phase 3: Continue with KataGo (Optional, 2-3 days)

```bash
python trainwithkatago.py \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.0003 \
    --auto_install_assets
```

---

## Jetson-Optimized Training

```bash
cd /Users/williamcha/Desktop/gomcts

# Use smaller batch size for Jetson memory
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.001 \
    --device cuda
```

---

## Filter SGF Files by Board Size

```bash
# Extract only 9x9 games
mkdir -p ~/go_games/9x9_only

for file in ~/go_games/*/*.sgf; do
    if grep -q "SZ\[9\]" "$file"; then
        cp "$file" ~/go_games/9x9_only/
    fi
done

# Train on filtered games
python train_from_sgf.py \
    --sgf_dir ~/go_games/9x9_only \
    --board_size 9 \
    --epochs 10
```

---

## Monitor Training Progress

```bash
# In another terminal
watch -n 5 'tail -n 20 /Users/williamcha/Desktop/gomcts/training.log'

# Or check weights file updates
watch -n 10 'ls -lh /Users/williamcha/Desktop/gomcts/weights.pt'
```

---

## Resume Training

```bash
# If training was interrupted, just run again
# Loads existing weights.pt automatically
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 5 \
    --lr 0.0001  # Lower LR for fine-tuning
```

---

## Combine Multiple Sources

```bash
# Combine KGS + OGS + GoGoD
mkdir -p ~/go_games/all_9x9

cp ~/go_games/kgs_9x9/*.sgf ~/go_games/all_9x9/ 2>/dev/null || true
cp ~/go_games/ogs_9x9/*.sgf ~/go_games/all_9x9/ 2>/dev/null || true
cp ~/go_games/gogod_9x9/*.sgf ~/go_games/all_9x9/ 2>/dev/null || true

# Count total
ls -1 ~/go_games/all_9x9/*.sgf | wc -l

# Train on combined
python train_from_sgf.py \
    --sgf_dir ~/go_games/all_9x9 \
    --board_size 9 \
    --epochs 15
```

---

## Quick Comparison: Before vs After Training

```bash
# Before training (or with sample files)
cd /Users/williamcha/Desktop/gomcts
python startgame.py --sims 64

# Note how it plays

# After training with 1000 games
python startgame.py --sims 64

# Should be MUCH stronger!
```

---

## Check Training Results

```bash
cd /Users/williamcha/Desktop/gomcts

# Check weights file
ls -lh weights.pt

# Play a game
python startgame.py --sims 100

# Or test against itself
python selftraining.py --games 5 --sims 64
```

---

## Troubleshooting Commands

### No SGF files found

```bash
# Check if files exist
ls ~/go_games/kgs_9x9/

# Check file extensions
file ~/go_games/kgs_9x9/* | head -5

# Check path
pwd
echo $HOME
```

### Out of memory

```bash
# Reduce batch size
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --batch_size 32  # Reduced from 128
```

### Check GPU usage (Jetson)

```bash
# Monitor GPU
tegrastats

# Or
watch -n 1 'nvidia-smi'
```

---

## One-Liner Complete Setup

```bash
mkdir -p ~/go_games/kgs_9x9 && echo "Download SGF files from https://www.gokgs.com/gameArchives.jsp to ~/go_games/kgs_9x9/" && echo "Then run: cd /Users/williamcha/Desktop/gomcts && python train_from_sgf.py --sgf_dir ~/go_games/kgs_9x9 --board_size 9 --epochs 10"
```

---

## Summary: Three Quick Paths

### Path 1: Test Right Now (5 minutes)
```bash
cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py --sgf_dir ./sample_sgf_games --epochs 3 --board_size 9
```

### Path 2: Real Training, Manual Download (3-4 hours total)
```bash
# 1. Download from KGS (1-2 hours manual)
# Visit: https://www.gokgs.com/gameArchives.jsp
# Save to: ~/go_games/kgs_9x9/

# 2. Train (2 hours)
cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py --sgf_dir ~/go_games/kgs_9x9 --board_size 9 --epochs 15
```

### Path 3: Best Results (3-4 days)
```bash
# Phase 1: SGF (4 hours)
python train_from_sgf.py --sgf_dir ~/go_games/kgs_9x9 --board_size 9 --epochs 15

# Phase 2: KataGo (3 days)
python trainwithkatago.py --games 1000 --sims 64 --size 9 --lr 0.0003
```

---

**That's it! Pick a path and copy-paste the commands.** 🚀


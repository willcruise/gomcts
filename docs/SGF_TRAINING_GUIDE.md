# Training from Professional Go Games (SGF Files)

## 🎯 What is This Approach?

Instead of playing millions of self-play games or training against KataGo, you train your neural network to **imitate professional players** by learning from their game records.

This is called **supervised learning** or **imitation learning** - it's how the original AlphaGo was bootstrapped before they added reinforcement learning.

---

## 📊 Comparison: Training Methods

| Method | Data Needed | Jetson Time | Strength | Compute |
|--------|-------------|-------------|----------|---------|
| Self-play (AlphaZero) | 5M games | ~5000 days | Superhuman | ⚡⚡⚡⚡⚡ |
| Train vs KataGo | 500-2000 games | 2-7 days | Strong amateur | ⚡⚡⚡ |
| **SGF Supervised Learning** ⭐ | **1000 games** | **2-6 hours** | **Strong amateur** | **⚡** |

**Advantages of SGF Training:**
- ✅ **Extremely fast**: No MCTS during training (just forward/backward passes)
- ✅ **Sample efficient**: Learn from 50,000+ expert positions in hours
- ✅ **Jetson-friendly**: Low GPU memory, no simulation overhead
- ✅ **High quality**: Learn from top players, not your own mistakes
- ✅ **No opponent needed**: Just download game files and train

**Limitations:**
- ❌ Your network won't exceed the level of the data
- ❌ May not handle unusual positions well (needs diverse data)
- ❌ Value network less accurate (game outcomes only)

**Best Practice:** Combine approaches!
1. Start with SGF supervised learning (fast bootstrap)
2. Continue with vs-KataGo training (improves beyond imitation)

---

## 📥 Where to Get SGF Files

### Option 1: GoGoD Database (Recommended)
- **Website**: https://gogodonline.co.uk/
- **Content**: 100,000+ professional games
- **Cost**: ~$30-40 (one-time purchase)
- **Quality**: High - includes modern pro games
- **Board sizes**: Mostly 19x19, but has 9x9 and 13x13

### Option 2: Free Sources

#### a) KGS Go Server Archives
- **Website**: https://www.gokgs.com/gameArchives.jsp
- **Content**: Millions of online games (various skill levels)
- **Cost**: Free
- **Quality**: Mixed - filter for high-rank games (5d+)
- **How to download**:
  ```bash
  # Install KGS Archive Downloader
  # Or manually browse and download by date/player
  ```

#### b) OGS (Online Go Server)
- **Website**: https://online-go.com/
- **API**: Can download game records via API
- **Quality**: Mixed - filter by rank

#### c) Little Golem
- **Website**: http://www.littlegolem.net/
- **Content**: Correspondence games
- **Format**: SGF available

#### d) GitHub Collections
Search GitHub for "go sgf dataset" - many people have shared collections:
- https://github.com/yenw/computer-go-dataset
- https://github.com/featurecat/go-dataset

### Option 3: Generate from KataGo
If you have KataGo running, you can generate high-quality self-play games:
```bash
# KataGo can generate SGF files
katago gatekeeper -config your_config.cfg -model model.bin.gz
```

---

## 🚀 Quick Start

### Step 1: Download SGF Files

Create a directory and download some games:
```bash
mkdir -p ~/go_games/9x9
cd ~/go_games/9x9

# Example: Download from a free source
# (You'll need to find actual sources - see above)
```

### Step 2: Run Training

```bash
cd /Users/williamcha/Desktop/gomcts

# Train on your SGF collection
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001

# For Jetson with limited memory
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.001 \
    --device cuda
```

### Step 3: Monitor Progress

The script will show:
```
Epoch 1/10: Train Loss=2.1234, Val Loss=2.3456, Val Accuracy=35.2%
Epoch 2/10: Train Loss=1.8912, Val Loss=2.1234, Val Accuracy=42.8%
...
Epoch 10/10: Train Loss=1.2345, Val Loss=1.6789, Val Accuracy=58.3%
```

**What accuracy means:**
- **35%**: Network is learning basic patterns
- **45%**: Decent amateur level
- **55%**: Strong amateur (matches pro move >50% of time)
- **65%+**: Very strong (but hard to reach without huge dataset)

### Step 4: Test Your Network

```bash
# Play against your trained network
python -m gomcts.apps.startgame --sims 64
```

---

## 📈 Training Strategy

### Strategy 1: Pure Supervised Learning (Fastest)

```bash
# Just train on SGF files
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 20 \
    --board_size 9
```

**Time**: 2-6 hours  
**Result**: Strong amateur level  
**Best for**: Quick bootstrap

---

### Strategy 2: Supervised → Reinforcement (Recommended ⭐)

```bash
# Phase 1: Bootstrap from pro games (fast!)
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 10 \
    --board_size 9

# Phase 2: Improve with KataGo (slower but reaches higher level)
python -m gomcts.training.trainwithkatago \
    --games 500 \
    --sims 64 \
    --size 9 \
    --lr 0.0003  # Lower LR since network is already trained
```

**Time**: 3-4 hours (phase 1) + 2-3 days (phase 2)  
**Result**: Very strong amateur  
**Best for**: Maximum strength on Jetson

---

### Strategy 3: Supervised → Self-play

```bash
# Phase 1: Bootstrap from pro games
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 10 \
    --board_size 9

# Phase 2: Self-play refinement
python -m gomcts.training.selftraining \
    --games 500 \
    --sims 64 \
    --size 9 \
    --lr 0.0001
```

**Time**: 3-4 hours + 1-2 days  
**Result**: Strong amateur with unique style  
**Best for**: If you don't have KataGo

---

## 🔧 Advanced Tips

### Handling Different Board Sizes

If you have 19x19 games but want to train for 9x9:
- **Option 1**: Filter for 9x9 games only (smaller dataset)
- **Option 2**: Train on 19x19 first, then fine-tune on 9x9 self-play
- **Option 3**: Extract corner positions from 19x19 games (advanced)

### Data Augmentation

The board has 8-fold symmetry (4 rotations × 2 flips). You can augment your dataset:
```python
# In train_from_sgf.py, add data augmentation
# Each position can be rotated/flipped for 8x more data
```

### Curriculum Learning

Train progressively:
```bash
# Easy: Train on games from weaker players first
python -m gomcts.training.train_from_sgf --sgf_dir ~/go_games/kyu --epochs 5

# Hard: Then train on pro games
python -m gomcts.training.train_from_sgf --sgf_dir ~/go_games/pro --epochs 10
```

### Fine-tuning

If you already have trained weights:
```bash
# Continue training with lower learning rate
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 5 \
    --lr 0.0001  # 10x lower
```

---

## 📊 Expected Results

### After Training on Different Dataset Sizes

| SGF Games | Positions | Training Time | Val Accuracy | Play Strength |
|-----------|-----------|---------------|--------------|---------------|
| 100 | ~5,000 | 15 min | 35-40% | Weak amateur |
| 500 | ~25,000 | 1 hour | 42-48% | Decent amateur |
| 1,000 | ~50,000 | 2 hours | 48-55% | Strong amateur |
| 5,000 | ~250,000 | 8 hours | 55-60% | Very strong |
| 10,000+ | ~500,000+ | 15+ hours | 60-65%+ | Expert amateur |

**Note**: Accuracy plateaus around 60-65% even with massive datasets because:
- Pro players don't always agree on the "best" move
- Multiple moves can be equally good
- Your network may find reasonable alternatives

---

## 🐛 Troubleshooting

### "No SGF files found"
```bash
# Check your directory structure
ls -R ~/go_games/9x9

# Make sure files end in .sgf
```

### "No valid samples extracted"
- Your SGF files might be for different board sizes
- Try: `--board_size 19` if you have 19x19 games
- Check if SGF files are corrupted (open one in a text editor)

### Out of Memory on Jetson
```bash
# Reduce batch size
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --batch_size 32  # or even 16
```

### Training is Slow
- Supervised learning should be MUCH faster than MCTS-based training
- If it's slow, check if `--device cuda` is being used
- Monitor with `tegrastats` on Jetson

### Accuracy Not Improving
- Make sure you have enough diverse games (1000+ recommended)
- Try lower learning rate: `--lr 0.0003`
- Check validation loss - if it's increasing, you're overfitting

---

## 🎓 Theory: Why This Works

### What the Network Learns

**Policy Head**: 
- Learns which moves pros play in various positions
- Generalizes patterns: shapes, joseki, good moves
- Becomes a "style imitator"

**Value Head**:
- Learns to predict game outcomes
- Less accurate than MCTS-based training (only sees final result)
- Still useful for MCTS evaluation

### Comparison to AlphaGo's Approach

1. **AlphaGo (2016)**: 
   - Phase 1: Supervised learning on 30M positions from pro games
   - Phase 2: Reinforcement learning (self-play)
   - Phase 3: MCTS with learned policy/value

2. **AlphaGo Zero (2017)**:
   - Skipped supervised learning entirely
   - Pure self-play from random initialization
   - Needed massive compute (5M games)

3. **Your Approach**:
   - Hybrid: Supervised (fast) → Reinforcement (strong)
   - Best for limited compute!

---

## 🏆 Recommended Full Training Pipeline

```bash
# Step 1: Supervised learning (2-4 hours)
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 15 \
    --board_size 9 \
    --batch_size 128

# Step 2: Reinforcement with KataGo (2-3 days)
python -m gomcts.training.trainwithkatago \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.0003

# Step 3: Evaluate
python -m gomcts.apps.startgame --sims 100
```

**Total time**: ~3 days  
**Final strength**: Strong amateur (much better than pure self-play!)

---

## 📚 Additional Resources

- **SGF Format**: http://www.red-bean.com/sgf/
- **Go Databases**: https://senseis.xmp.net/?GoServers
- **AlphaGo Paper**: https://www.nature.com/articles/nature16961
- **AlphaGo Zero Paper**: https://www.nature.com/articles/nature24270

---

## 💡 Pro Tip

The best approach for Jetson is:
1. ⚡ **Supervised learning** (2-4 hours) ← Bootstrap quickly
2. 🎯 **Train vs KataGo** (2-3 days) ← Reach high level  
3. 🔄 **Periodic SGF updates** (1 hour) ← Stay current with new pro games

This gets you to strong amateur level in under a week!


# Go Neural Network Training: Method Comparison

## 🎯 Three Training Approaches for Your Jetson

You now have **three different ways** to train your Go neural network. Here's when to use each:

---

## Method 1: Supervised Learning from SGF Files ⚡

**File:** `gomcts/training/train_from_sgf.py`

### How it works:
- Load professional game records (SGF files)
- For each position, train network to predict the pro's move
- No MCTS needed during training (just forward/backward passes)

### Pros:
- ⚡ **Extremely fast**: 2-6 hours for 1000 games
- 💾 **Low memory**: No MCTS tree storage
- 🎓 **Learn from experts**: High quality training signal
- 📊 **Predictable**: Validation accuracy shows progress

### Cons:
- ❌ Can't exceed quality of training data
- ❌ Need to find/download SGF files first
- ❌ Value network less accurate (only sees final outcomes)

### When to use:
- ✅ **Bootstrap a new network quickly**
- ✅ When you have SGF files available
- ✅ Want to train overnight and have results by morning
- ✅ Limited compute time

### Example:
```bash
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 128
```

### Expected results:
- **Time**: 2-6 hours
- **Final strength**: Strong amateur (55%+ move accuracy)
- **Resources**: ~2GB GPU memory, ~1-2 cores

---

## Method 2: Training vs KataGo 🎯

**File:** `gomcts/training/trainwithkatago.py`

### How it works:
- Play games against KataGo (a superhuman AI)
- Your network chooses moves via MCTS
- Learn from both your moves AND the outcomes
- KataGo provides strong opponent feedback

### Pros:
- 🎯 **Very sample-efficient**: Learn from a strong teacher
- 💪 **Can exceed training data**: Reinforcement learning improves beyond imitation
- ⚖️ **Balanced learning**: Both policy and value networks improve
- 📈 **Continues improving**: No ceiling from training data

### Cons:
- ⏱️ Slower than supervised learning (2-7 days)
- 🔧 Requires KataGo installation
- 💻 More compute per game (MCTS simulations)

### When to use:
- ✅ **Best overall approach for Jetson**
- ✅ After supervised bootstrap (recommended!)
- ✅ Want to reach strong amateur/dan level
- ✅ Have 2-7 days of training time

### Example:
```bash
python -m gomcts.training.trainwithkatago \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.001 \
    --auto_install_assets
```

### Expected results:
- **Time**: 2-7 days for 500-2000 games
- **Final strength**: Strong amateur to low dan
- **Resources**: ~3-4GB GPU memory, MCTS overhead

---

## Method 3: Pure Self-Play 🔄

**File:** `gomcts/training/selftraining.py`

### How it works:
- Network plays against itself
- Learns from its own games
- Pure reinforcement learning (like AlphaGo Zero)

### Pros:
- 🎨 **Creative**: Develops unique playing style
- 🔬 **Self-sufficient**: No external data needed
- 📚 **Research-grade**: Follow AlphaGo Zero approach
- 🎮 **Fun to watch**: See your AI improve over time

### Cons:
- ❌ **Very slow**: Needs millions of games for pro level
- ❌ **Unpredictable**: Hard to know when it will improve
- ❌ **Compute-heavy**: Not ideal for Jetson
- ❌ **Can get stuck**: May develop bad habits

### When to use:
- ✅ After supervised + KataGo training (refinement)
- ✅ Experimenting with AlphaGo Zero approach
- ✅ Want unique playing style
- ✅ Have unlimited time

### Example:
```bash
python -m gomcts.training.selftraining \
    --games 500 \
    --sims 64 \
    --size 9 \
    --lr 0.003 \
    --eval_every 50
```

### Expected results:
- **Time**: Weeks to months for strong play
- **Final strength**: Depends on iterations (unpredictable)
- **Resources**: Similar to vs-KataGo but less efficient

---

## 📊 Direct Comparison

| Aspect | SGF Supervised ⚡ | vs KataGo 🎯 | Self-Play 🔄 |
|--------|------------------|--------------|--------------|
| **Speed** | ⚡⚡⚡⚡⚡ Fast | ⚡⚡⚡ Moderate | ⚡ Slow |
| **Strength** | Strong amateur | Strong amateur to dan | Variable |
| **Jetson Time** | 2-6 hours | 2-7 days | Weeks/months |
| **Sample Efficiency** | Excellent | Very good | Poor |
| **Setup Difficulty** | Easy (need SGF) | Moderate (need KataGo) | Easy |
| **GPU Memory** | Low (2GB) | Moderate (3-4GB) | Moderate (3-4GB) |
| **Predictability** | High | High | Low |
| **Ceiling** | Dataset quality | Very high | Unlimited |

---

## 🏆 Recommended Training Pipeline

### For Maximum Strength on Jetson:

```bash
# Phase 1: Bootstrap with supervised learning (2-4 hours) ⚡
python -m gomcts.training.train_from_sgf \
    --sgf_dir ~/go_games/9x9 \
    --epochs 15 \
    --board_size 9

# Check progress
python -m gomcts.apps.startgame --sims 64

# Phase 2: Improve with KataGo training (2-3 days) 🎯
python -m gomcts.training.trainwithkatago \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.0003  # Lower LR since already trained

# Check progress again
python -m gomcts.apps.startgame --sims 100

# Phase 3 (Optional): Self-play refinement (ongoing) 🔄
python -m gomcts.training.selftraining \
    --games 500 \
    --sims 64 \
    --eval_every 50 \
    --lr 0.0001
```

**Total time**: ~3-4 days  
**Final strength**: Strong amateur to low dan  
**This is the optimal strategy for Jetson!**

---

## 🚀 Quick Start Recommendations

### "I want results NOW!" (2-6 hours)
```bash
# Get SGF files, then:
python -m gomcts.training.train_from_sgf --sgf_dir ~/go_games --epochs 10 --board_size 9
```

### "I want the best quality" (3-4 days)
```bash
# Phase 1: SGF
python -m gomcts.training.train_from_sgf --sgf_dir ~/go_games --epochs 15 --board_size 9

# Phase 2: KataGo
python -m gomcts.training.trainwithkatago --games 1000 --sims 64 --size 9 --lr 0.0003
```

### "I want to follow AlphaGo Zero exactly" (weeks/months)
```bash
python -m gomcts.training.selftraining --games 10000 --sims 64 --size 9
# Warning: This will take a very long time!
```

### "I just want to test quickly" (15 minutes)
```bash
# Use provided sample files
python -m gomcts.training.train_from_sgf \
    --sgf_dir ./examples/sample_sgf_games \
    --epochs 3 \
    --batch_size 32 \
    --board_size 9
```

---

## 💡 Key Insights

1. **Supervised learning is MUCH faster** than self-play
   - AlphaGo used supervised learning first for good reason!

2. **KataGo training is best for strength**
   - Learn from a superhuman teacher vs learning from yourself

3. **Combine approaches for best results**
   - Supervised → KataGo → Self-play
   - Each phase builds on the previous

4. **Pure self-play is not practical for Jetson**
   - Would take months/years to reach good level
   - Better to bootstrap with supervised or KataGo

5. **Validation accuracy shows real progress**
   - 50%+ move accuracy = strong amateur
   - 60%+ = approaching dan level

---

## 📈 Expected Progression

### Starting from Random Weights:

| Method | After 1 Day | After 3 Days | After 1 Week |
|--------|-------------|--------------|--------------|
| **SGF** | Strong amateur ✅ | (done) | (done) |
| **KataGo** | Decent amateur | Strong amateur ✅ | Low dan |
| **Self-play** | Weak beginner | Weak amateur | Decent amateur |

### Starting from SGF-Trained Weights:

| Method | After 1 Day | After 3 Days | After 1 Week |
|--------|-------------|--------------|--------------|
| **KataGo** | Strong amateur | Low dan ✅ | Mid dan |
| **Self-play** | Strong amateur | Strong amateur | Low dan |

---

## 🎓 Theory Summary

- **Supervised Learning**: Learn to imitate → Fast but limited by data
- **Reinforcement (vs KataGo)**: Learn to win against strong opponent → Efficient and strong
- **Self-Play**: Discover strategies yourself → Slow but potentially unlimited

**Best practice**: Start with imitation (fast), then improve through competition (efficient), then refine with self-discovery (creative).

This is exactly what DeepMind did with AlphaGo (before Zero)!

---

## 📚 File Reference

- `gomcts/training/train_from_sgf.py` - Supervised learning from pro games
- `gomcts/training/trainwithkatago.py` - Train by playing against KataGo
- `gomcts/training/selftraining.py` - Pure self-play reinforcement learning
- `SGF_TRAINING_GUIDE.md` - Detailed guide for SGF training
- `TRAINING_ON_JETSON.md` - Jetson-specific optimization guide
- `examples/sample_sgf_games/` - Test files to try training immediately

---

**Bottom Line**: For Jetson with limited compute, **supervised learning followed by KataGo training** is by far the most efficient path to strong play! 🏆


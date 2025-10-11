# Training Go Neural Networks on Jetson (Limited Compute)

## Key Insight
AlphaZero needed ~5 million self-play games to reach pro level because it learned from scratch. **You can be much more efficient by learning from a strong teacher!**

## Strategy 1: Train Against KataGo (RECOMMENDED ⭐)

This is the **most sample-efficient** approach for limited compute.

### Why This Works
- **Self-play**: Agent learns from its own mistakes (slow when weak)
- **Training vs KataGo**: Learn from an expert's moves (fast improvement)
- You can reach strong amateur level with just **500-2000 games** vs KataGo

### How to Run
```bash
# On Jetson - uses the optimized config
python trainwithkatago.py --config configs/train_jetson_vs_katago.yaml

# Or manually specify parameters
python trainwithkatago.py \
    --games 500 \
    --sims 64 \
    --size 9 \
    --lr 0.001 \
    --checkpoint_every 10 \
    --auto_install_assets
```

### Expected Results
- **After 100 games**: Basic understanding of good moves
- **After 500 games**: Decent amateur play
- **After 2000 games**: Strong amateur level (way better than random)

### Resource Usage
- **Time**: ~2-5 minutes per game on Jetson (depends on game length)
- **Memory**: ~2-4 GB GPU memory
- **Storage**: Weights file ~1-5 MB

---

## Strategy 2: Hybrid Training (Advanced)

Alternate between KataGo training and self-play:

```bash
# Phase 1: Bootstrap from KataGo (200 games)
python trainwithkatago.py --games 200 --sims 64 --size 9

# Phase 2: Self-play to diversify (100 games, faster)
python selftraining.py --games 100 --sims 64 --size 9

# Phase 3: Back to KataGo (200 more games)
python trainwithkatago.py --games 200 --sims 64 --size 9
```

---

## Strategy 3: Cloud Training + Jetson Inference

If you have access to cloud GPUs (Google Colab, AWS, etc.):

1. **Train on Cloud**: Use higher sims (1000+) and more games
   ```bash
   # On cloud GPU
   python trainwithkatago.py --games 2000 --sims 200 --size 9
   ```

2. **Download Weights**: Copy `weights.pt` to your Jetson

3. **Continue Training**: Fine-tune on Jetson
   ```bash
   # On Jetson - continue from cloud weights
   python trainwithkatago.py --games 500 --sims 64 --size 9
   ```

4. **Inference Only**: Just play games on Jetson (very fast)
   ```bash
   python startgame.py --sims 64
   ```

---

## Strategy 4: Optimize Jetson Performance

### 1. Enable Maximum Performance Mode
```bash
# On Jetson, maximize clock speeds
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Max clock speeds
```

### 2. Monitor Resources
```bash
# Check GPU usage
tegrastats

# Watch temperature (throttles at 80°C)
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### 3. Use Smaller Networks (if needed)
Edit `policyneural.py` line 27:
```python
def __init__(self, hidden_size: int = 128, ...):  # Reduced from 256
```

### 4. Reduce MCTS Simulations During Training
```yaml
sims: 32  # Even lower if needed - you still learn from KataGo!
```

---

## Strategy 5: Curriculum Learning

Start small, then scale up:

```bash
# Phase 1: Tiny board (5x5) - fast training
python trainwithkatago.py --games 500 --sims 64 --size 5

# Phase 2: Medium board (7x7)
python trainwithkatago.py --games 500 --sims 64 --size 7

# Phase 3: Standard (9x9)
python trainwithkatago.py --games 1000 --sims 64 --size 9
```

---

## Strategy 6: Use Evaluation to Track Progress

Your code already has evaluation built-in! Use it:

```bash
python selftraining.py \
    --games 500 \
    --sims 64 \
    --eval_every 50 \
    --eval_games 20 \
    --eval_threshold 0.55 \
    --size 9
```

This will:
- Play 20 evaluation games every 50 training games
- Only keep weights if they beat the baseline 55%+ of the time
- Prevent regression

---

## Comparison: Compute Requirements

| Approach | Games Needed | Jetson Training Time | Quality |
|----------|--------------|---------------------|---------|
| AlphaZero Self-Play | 5,000,000 | ~5000 days 🚫 | Superhuman |
| Your Self-Play | 10,000 | ~14 days | Weak amateur |
| **vs KataGo** ⭐ | **500-2000** | **~2-7 days** | **Strong amateur** |
| Cloud + Jetson | 2000 (cloud) + 500 (Jetson) | 3-5 days total | Very strong |

---

## Monitoring Training Progress

### Check Current Strength
```bash
# Play against your trained agent
python startgame.py --sims 64

# Or have it play against KataGo
python trainwithkatago.py --games 10 --sims 64  # Watch win rate
```

### Expected Value Estimates
- **Random weights**: Value ~0 (no idea who's winning)
- **After 100 games vs KataGo**: Value properly tracks position
- **After 500 games**: Value closely matches KataGo's assessment

### Policy Quality
- **Early**: Policy spread across many moves
- **Improving**: Policy focuses on top KataGo moves
- **Strong**: Policy matches KataGo's top 3 moves >70% of time

---

## Troubleshooting

### Out of Memory
```yaml
mcts_batch_size: 16  # Reduce from 32
sims: 32             # Reduce from 64
```

### Too Slow
```yaml
sims: 32             # Fewer simulations
checkpoint_every: 25 # Save less often
```

### KataGo Not Found
```bash
python trainwithkatago.py --auto_install_assets --games 100
```

---

## Bottom Line for Jetson

✅ **DO**: Train against KataGo (500-2000 games)  
✅ **DO**: Use 32-64 MCTS sims  
✅ **DO**: Monitor progress with evaluation  
❌ **DON'T**: Try pure self-play for millions of games  
❌ **DON'T**: Use 1000+ sims on Jetson  

**Expected timeline**: Reach strong amateur level in 2-7 days of continuous training!


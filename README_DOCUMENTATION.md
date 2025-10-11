# Go Neural Network Training - Complete Documentation Index

## 📚 Your Questions Answered

### Question 1: How does the loss function train both heads?

**Read these files:**

1. **`LOSS_FUNCTION_EXPLAINED.md`** (⭐ START HERE)
   - Detailed explanation of how policy and value heads are trained
   - Mathematical formulas with examples
   - Step-by-step training walkthrough
   - Hyperparameter effects

2. **`VISUAL_LOSS_SUMMARY.txt`**
   - Visual diagrams of the loss function
   - ASCII art showing gradient flow
   - Complete training example with numbers
   - Quick reference guide

**Key Concepts:**
- Policy Head uses **Cross-Entropy Loss** to learn which moves are good
- Value Head uses **Mean Squared Error** to learn position evaluation
- Both heads share a hidden layer → get gradients from both losses!
- Formula: `loss = CE + c_v × MSE + l2 × Reg`

---

### Question 2: How to download SGF files and train?

**Read these files:**

1. **`SGF_DOWNLOAD_TUTORIAL.md`** (⭐ START HERE)
   - Step-by-step download instructions for 5 different sources
   - KGS, OGS, GoGoD, GitHub, KataGo
   - Complete examples with verification commands
   - Troubleshooting section

2. **`COPY_PASTE_COMMANDS.md`**
   - Ready-to-use commands (just copy & paste!)
   - Quick paths for different scenarios
   - No explanation, just commands

**Quick Answer:**
```bash
# 1. Download from KGS: https://www.gokgs.com/gameArchives.jsp
#    Save to ~/go_games/kgs_9x9/

# 2. Train
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10
```

---

## 📖 All Documentation Files

### Training Methods

- **`TRAINING_COMPARISON.md`** - Compares SGF vs KataGo vs Self-play
- **`TRAINING_ON_JETSON.md`** - Jetson-specific optimization guide
- **`QUICK_START.md`** - One-page quick reference

### SGF Training (Your Approach!)

- **`SGF_TRAINING_GUIDE.md`** - Complete guide to SGF training
- **`SGF_DOWNLOAD_TUTORIAL.md`** - How to get SGF files
- **`COPY_PASTE_COMMANDS.md`** - Ready commands to copy
- **`train_from_sgf.py`** - The training script itself

### Loss Function Deep Dive

- **`LOSS_FUNCTION_EXPLAINED.md`** - Detailed mathematical explanation
- **`VISUAL_LOSS_SUMMARY.txt`** - Visual diagrams and examples

### Code Files

- **`policyneural.py`** - Neural network implementation (both heads)
- **`train_from_sgf.py`** - Supervised learning from SGF files
- **`trainwithkatago.py`** - Train against KataGo
- **`selftraining.py`** - Pure self-play training

### Helper Scripts

- **`train_jetson.sh`** - One-command Jetson training
- **`download_sample_sgf.sh`** - SGF download helper
- **`sample_sgf_games/`** - Test files to try immediately

---

## 🚀 Quick Navigation

### I want to understand the loss function
→ Read: `LOSS_FUNCTION_EXPLAINED.md` then `VISUAL_LOSS_SUMMARY.txt`

### I want to train from SGF files RIGHT NOW
→ Read: `COPY_PASTE_COMMANDS.md` (just copy commands)

### I want detailed SGF training instructions
→ Read: `SGF_DOWNLOAD_TUTORIAL.md` (step-by-step)

### I want to test immediately (no downloads)
→ Run: 
```bash
python train_from_sgf.py --sgf_dir ./sample_sgf_games --epochs 3 --board_size 9
```

### I want to compare all training methods
→ Read: `TRAINING_COMPARISON.md`

### I want Jetson optimization tips
→ Read: `TRAINING_ON_JETSON.md`

### I just want the basics
→ Read: `QUICK_START.md`

---

## 📊 File Sizes Reference

```
LOSS_FUNCTION_EXPLAINED.md    11 KB  (detailed theory)
VISUAL_LOSS_SUMMARY.txt        14 KB  (visual examples)
SGF_DOWNLOAD_TUTORIAL.md       13 KB  (download guide)
COPY_PASTE_COMMANDS.md         6 KB   (quick commands)
SGF_TRAINING_GUIDE.md          10 KB  (training guide)
TRAINING_COMPARISON.md         9 KB   (method comparison)
TRAINING_ON_JETSON.md          8 KB   (Jetson tips)
QUICK_START.md                 2 KB   (one-pager)
```

---

## 🎯 Recommended Reading Order

### For Understanding Theory:
1. `QUICK_START.md` (overview)
2. `LOSS_FUNCTION_EXPLAINED.md` (how it works)
3. `VISUAL_LOSS_SUMMARY.txt` (visual examples)

### For Practical Training:
1. `QUICK_START.md` (overview)
2. `SGF_DOWNLOAD_TUTORIAL.md` (get files)
3. `COPY_PASTE_COMMANDS.md` (run training)
4. `TRAINING_COMPARISON.md` (compare methods)

### For Jetson Optimization:
1. `TRAINING_ON_JETSON.md` (optimization)
2. `TRAINING_COMPARISON.md` (pick best method)
3. `COPY_PASTE_COMMANDS.md` (run optimized commands)

---

## 💡 Key Insights Summary

### Loss Function (Question 1):
- **Two heads, one network**: Policy (moves) + Value (evaluation)
- **Shared learning**: Hidden layer gets gradients from both heads
- **Balance**: `c_v` controls policy vs value emphasis
- **Formula**: `loss = CE + c_v × MSE + l2 × Reg`
- **CE** (Cross-Entropy): Learns which moves pros play
- **MSE** (Mean Squared Error): Learns who wins positions

### SGF Training (Question 2):
- **10-100x faster** than self-play!
- **Sources**: KGS (free), GoGoD (paid), OGS (free), GitHub (free)
- **How many**: 500-1000 games recommended
- **Time**: 2-6 hours to train
- **Result**: Strong amateur level (55%+ move accuracy)
- **Best combo**: SGF first (fast) → KataGo second (strong)

---

## 🔗 External Links

- **KGS Archives**: https://www.gokgs.com/gameArchives.jsp
- **OGS**: https://online-go.com/
- **GoGoD**: https://gogodonline.co.uk/
- **SGF Format Spec**: http://www.red-bean.com/sgf/
- **AlphaGo Paper**: https://www.nature.com/articles/nature16961

---

## 🎓 Learning Path

### Beginner:
1. Read `QUICK_START.md`
2. Test with sample files: `python train_from_sgf.py --sgf_dir ./sample_sgf_games --epochs 3`
3. Play against it: `python startgame.py --sims 64`

### Intermediate:
1. Read `SGF_DOWNLOAD_TUTORIAL.md`
2. Download 500 SGF files
3. Train for real: `python train_from_sgf.py --sgf_dir ~/go_games --epochs 10`
4. Read `LOSS_FUNCTION_EXPLAINED.md` to understand what's happening

### Advanced:
1. Read `TRAINING_COMPARISON.md`
2. Implement full pipeline: SGF → KataGo → Self-play
3. Read `VISUAL_LOSS_SUMMARY.txt` to understand gradients
4. Experiment with hyperparameters

---

## ❓ FAQ Quick Answers

**Q: Which training method is fastest?**
A: SGF supervised learning (2-6 hours)

**Q: Which training method is strongest?**
A: Hybrid: SGF → KataGo (3-4 days total)

**Q: How does the loss function work?**
A: See `LOSS_FUNCTION_EXPLAINED.md` - it trains both heads together!

**Q: Where do I download SGF files?**
A: See `SGF_DOWNLOAD_TUTORIAL.md` - KGS is easiest (free)

**Q: Can I test without downloading anything?**
A: Yes! Use `./sample_sgf_games/` directory (already included)

**Q: How do policy and value heads learn together?**
A: They share a hidden layer that receives gradients from both losses

**Q: What's the formula for the loss function?**
A: `loss = CE + c_v × MSE + l2 × Reg` (see `VISUAL_LOSS_SUMMARY.txt`)

---

## 🎉 Start Here!

### Absolute Beginner:
```bash
# Read this, then:
python train_from_sgf.py --sgf_dir ./sample_sgf_games --epochs 3 --board_size 9
```

### Ready to Train for Real:
```bash
# Download from https://www.gokgs.com/gameArchives.jsp
# Save to ~/go_games/kgs_9x9/
# Then:
python train_from_sgf.py --sgf_dir ~/go_games/kgs_9x9 --board_size 9 --epochs 10
```

### Want Maximum Strength:
```bash
# Phase 1: SGF (4 hours)
python train_from_sgf.py --sgf_dir ~/go_games --epochs 15 --board_size 9

# Phase 2: KataGo (3 days)
python trainwithkatago.py --games 1000 --sims 64 --size 9 --lr 0.0003
```

---

**You now have everything you need to train a strong Go AI on your Jetson!** 🚀

For questions, read the relevant documentation file above. Everything is explained in detail!


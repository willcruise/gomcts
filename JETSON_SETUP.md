# Running SGF Training on Jetson

## 🚀 Quick Setup Guide

### Step 1: Clone the Repository on Jetson

```bash
# SSH into your Jetson
ssh your_jetson_user@jetson_ip_address

# Clone the repo
cd ~
git clone https://github.com/willcruise/gomcts.git
cd gomcts
```

---

### Step 2: Install Dependencies

```bash
# Make sure you have Python 3.8+
python3 --version

# Install requirements
pip3 install -r requirements.txt

# For Jetson, PyTorch should already be installed
# If not, follow: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

---

### Step 3: Transfer Your SGF Files

You have 4,835 SGF files on your Mac. Transfer them to Jetson:

**Option A: Using scp (from your Mac):**
```bash
# From your Mac terminal:
cd /Users/williamcha/Downloads

# Transfer all SGF directories
scp -r Shusaku your_jetson_user@jetson_ip:/home/your_user/go_data/
scp -r Cho_Chikun your_jetson_user@jetson_ip:/home/your_user/go_data/
scp -r Go_Seigen your_jetson_user@jetson_ip:/home/your_user/go_data/
scp -r Dosaku your_jetson_user@jetson_ip:/home/your_user/go_data/
scp -r AlphaGo your_jetson_user@jetson_ip:/home/your_user/go_data/
scp -r ancient your_jetson_user@jetson_ip:/home/your_user/go_data/

# This will take 10-20 minutes depending on network speed
```

**Option B: Using rsync (faster, resumable):**
```bash
# From your Mac:
rsync -avz --progress /Users/williamcha/Downloads/Shusaku your_jetson_user@jetson_ip:/home/your_user/go_data/
rsync -avz --progress /Users/williamcha/Downloads/Cho_Chikun your_jetson_user@jetson_ip:/home/your_user/go_data/
# ... repeat for other directories
```

**Option C: USB Drive:**
```bash
# Copy to USB on Mac, then plug into Jetson
cp -r /Users/williamcha/Downloads/Shusaku /Volumes/YOUR_USB/
# ... etc

# On Jetson:
cp -r /media/YOUR_USB/Shusaku ~/go_data/
```

---

### Step 4: Enable Maximum Performance (Jetson)

```bash
# Maximize clock speeds for faster training
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

### Step 5: Start Training!

```bash
cd ~/gomcts

# For 19x19 training (using your data):
python3 train_from_sgf.py \
    --sgf_dir ~/go_data \
    --board_size 19 \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --device cuda

# Or use the simplified script:
./train_jetson.sh
```

**Expected output:**
```
Found 4835 SGF files
Processing game 100/4835... (8234 samples so far)
Processing game 500/4835... (41245 samples so far)
...
Loaded 4835 games with 312,456 total positions

Training set: 281,210 samples
Validation set: 31,246 samples
Training for 15 epochs with batch size 64

Epoch 1/15: Train Loss=3.234, Val Loss=3.456, Val Accuracy=28.3%, Time=3m15s
Epoch 2/15: Train Loss=2.145, Val Loss=2.389, Val Accuracy=37.1%, Time=3m12s
...
```

---

### Step 6: Monitor Training

**In another terminal:**
```bash
# Watch GPU usage
tegrastats

# Or watch the weights file updating
watch -n 10 'ls -lh ~/gomcts/weights.pt'

# Monitor temperature (important!)
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
```

**Keep temperature < 80°C!** If it gets too hot:
- Reduce batch_size to 32
- Add cooling (fan)
- Take breaks between epochs

---

### Step 7: Training Time Estimates

Based on your Jetson model:

**Jetson Nano (4GB):**
```
Batch size: 32
Time per epoch: ~8-10 minutes
Total (15 epochs): ~2-2.5 hours
```

**Jetson Xavier NX/AGX (8GB):**
```
Batch size: 64
Time per epoch: ~4-6 minutes
Total (15 epochs): ~1-1.5 hours
```

**Jetson Orin (16GB+):**
```
Batch size: 128
Time per epoch: ~2-3 minutes
Total (15 epochs): ~30-45 minutes
```

---

### Step 8: After Training

```bash
# Weights saved to:
~/gomcts/weights.pt

# Test your trained network:
python3 startgame.py --sims 64 --size 19

# Or continue with reinforcement learning:
python3 trainwithkatago.py \
    --games 500 \
    --sims 64 \
    --size 19 \
    --lr 0.0003
```

---

## 🔧 Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size
python3 train_from_sgf.py \
    --sgf_dir ~/go_data \
    --board_size 19 \
    --batch_size 32  # or even 16
```

### Training is Slow:
```bash
# Make sure CUDA is being used
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# Check you're on max performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Can't Find SGF Files:
```bash
# Check the path
ls ~/go_data/

# Should show:
# Shusaku/  Cho_Chikun/  Go_Seigen/  Dosaku/  AlphaGo/  ancient/

# If not, adjust --sgf_dir path
```

### Jetson Overheating:
```bash
# Monitor temp
cat /sys/devices/virtual/thermal/thermal_zone0/temp

# If > 80000 (80°C):
# - Add cooling fan
# - Reduce batch_size
# - Train in batches with breaks
```

---

## 📊 Expected Results

After 15 epochs on 4,835 games:

```
Final Stats:
  Train Loss: ~1.2
  Val Loss: ~1.5
  Val Accuracy: 55-60%

What this means:
  ✓ Network matches pro moves >55% of time
  ✓ Strong amateur level (5-10 kyu equivalent)
  ✓ Ready for MCTS play or reinforcement learning
```

---

## 🎯 Next Steps After Training

### Option 1: Just Play
```bash
python3 startgame.py --sims 100 --size 19
```

### Option 2: Continue Training with KataGo
```bash
python3 trainwithkatago.py \
    --games 1000 \
    --sims 64 \
    --size 19 \
    --lr 0.0003 \
    --auto_install_assets
```

### Option 3: Self-Play Refinement
```bash
python3 selftraining.py \
    --games 500 \
    --sims 64 \
    --size 19 \
    --lr 0.0001
```

---

## 📱 Remote Monitoring

To check progress from your Mac:

```bash
# SSH and check
ssh your_jetson_user@jetson_ip 'tail -20 ~/gomcts/training.log'

# Or use tmux on Jetson:
# On Jetson before starting training:
tmux new -s training
python3 train_from_sgf.py ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## ⚡ Quick Commands Reference

```bash
# Full training (recommended)
python3 train_from_sgf.py \
    --sgf_dir ~/go_data \
    --board_size 19 \
    --epochs 15 \
    --batch_size 64

# Memory-constrained
python3 train_from_sgf.py \
    --sgf_dir ~/go_data \
    --board_size 19 \
    --epochs 15 \
    --batch_size 32

# Test with sample files
python3 train_from_sgf.py \
    --sgf_dir ./sample_sgf_games \
    --board_size 9 \
    --epochs 3 \
    --batch_size 32

# Play against trained model
python3 startgame.py --sims 64 --size 19
```

---

## 🎉 You're All Set!

Your Jetson will now learn from 4,835 games by legendary Go players and become a strong amateur player! 🚀

Good luck with training! 🎲


# Step-by-Step: Download SGF Files and Train

## 📥 Complete Tutorial: From Download to Training

This guide walks you through **exactly** how to get SGF files and use them for training.

---

## Method 1: KGS Game Archives (FREE, Easiest) ⭐

### Step 1: Visit KGS Archives

1. Go to: https://www.gokgs.com/gameArchives.jsp
2. You'll see a calendar interface

### Step 2: Select Date Range

1. Click on dates to select games
2. Filter options:
   - **Rank**: Select "5d" or higher for quality
   - **Board size**: Select "9" for 9x9 games
   - **Type**: Select "Ranked" for serious games

### Step 3: Download Games

**Option A: Manual download (browser)**
```bash
# Create directory
mkdir -p ~/go_games/kgs_9x9
cd ~/go_games/kgs_9x9

# For each game on the website:
# 1. Click on the game link
# 2. Click "Download SGF"
# 3. Save to ~/go_games/kgs_9x9/
```

**Option B: Bulk download (advanced)**
```bash
# Install wget if not available
# brew install wget  # macOS
# sudo apt install wget  # Linux

# Download archives (example for a specific month)
wget -r -np -nd -A "*.sgf" \
  "https://www.gokgs.com/gameArchives.jsp?year=2024&month=1"
```

### Step 4: Verify Downloads

```bash
# Check how many SGF files you got
cd ~/go_games/kgs_9x9
ls -1 *.sgf | wc -l

# Look at a sample file
head -n 5 *.sgf | head -n 20
```

You should see something like:
```
(;GM[1]FF[4]CA[UTF-8]...
```

### Step 5: Train!

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.001
```

**Expected output:**
```
Found 523 SGF files
Processing game 100/523... (4253 samples so far)
Processing game 200/523... (8632 samples so far)
...
Loaded 523 games with 41,245 total positions

Training set: 37,121 samples
Validation set: 4,124 samples
Training for 10 epochs with batch size 128

Epoch 1/10: Train Loss=2.134, Val Loss=2.345, Val Accuracy=35.2%
Epoch 2/10: Train Loss=1.891, Val Loss=2.123, Val Accuracy=42.8%
...
Epoch 10/10: Train Loss=1.234, Val Loss=1.678, Val Accuracy=55.3%

Training complete!
Weights saved to: /Users/williamcha/Desktop/gomcts/weights.pt
```

---

## Method 2: OGS (Online-Go.com) (FREE)

### Step 1: Get API Access

1. Go to: https://online-go.com/
2. Create account (free)
3. Get your API token: https://online-go.com/developer

### Step 2: Download with API

```bash
# Create directory
mkdir -p ~/go_games/ogs_9x9
cd ~/go_games/ogs_9x9

# Install Python requests library if needed
pip install requests

# Create download script
cat > download_ogs.py << 'EOF'
import requests
import json
import time
import os

API_TOKEN = "YOUR_TOKEN_HERE"  # Get from online-go.com/developer
BOARD_SIZE = 9
MIN_RANK = 15  # 5 dan = rank 15 in OGS
OUTPUT_DIR = "."

headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Search for games
url = "https://online-go.com/api/v1/games/"
params = {
    "width": BOARD_SIZE,
    "height": BOARD_SIZE,
    "ranked": "true",
    "ordering": "-ended",
    "page_size": 100
}

page = 1
total_downloaded = 0

while page <= 10:  # Download up to 1000 games
    params["page"] = page
    print(f"Fetching page {page}...")
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        break
    
    data = response.json()
    games = data.get("results", [])
    
    if not games:
        break
    
    for game in games:
        game_id = game.get("id")
        
        # Download SGF
        sgf_url = f"https://online-go.com/api/v1/games/{game_id}/sgf"
        sgf_response = requests.get(sgf_url, headers=headers)
        
        if sgf_response.status_code == 200:
            filename = os.path.join(OUTPUT_DIR, f"ogs_{game_id}.sgf")
            with open(filename, 'w') as f:
                f.write(sgf_response.text)
            total_downloaded += 1
            print(f"Downloaded game {game_id} ({total_downloaded} total)")
        
        time.sleep(0.1)  # Rate limiting
    
    page += 1

print(f"\nDownloaded {total_downloaded} games to {OUTPUT_DIR}")
EOF

# Run the script
python download_ogs.py
```

### Step 3: Train!

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/ogs_9x9 \
    --board_size 9 \
    --epochs 10
```

---

## Method 3: GoGoD Database (PAID, Highest Quality)

### Step 1: Purchase

1. Go to: https://gogodonline.co.uk/
2. Purchase "GoGoD Database" (~$40 USD)
3. Download the database (large ZIP file)

### Step 2: Extract

```bash
# Create directory
mkdir -p ~/go_games/gogod
cd ~/go_games/gogod

# Extract (replace with your actual download path)
unzip ~/Downloads/GoGoD2024.zip

# Find 9x9 games specifically
find . -name "*.sgf" -exec grep -l "SZ\[9\]" {} \; > 9x9_files.txt

# Copy 9x9 games to separate folder
mkdir -p ~/go_games/gogod_9x9
while read file; do
    cp "$file" ~/go_games/gogod_9x9/
done < 9x9_files.txt
```

### Step 3: Train!

```bash
cd /Users/williamcha/Desktop/gomcts

python train_from_sgf.py \
    --sgf_dir ~/go_games/gogod_9x9 \
    --board_size 9 \
    --epochs 15 \
    --batch_size 128
```

---

## Method 4: GitHub Datasets (FREE, Variable Quality)

### Option A: Computer Go Dataset

```bash
# Clone repository
cd ~
git clone https://github.com/yenw/computer-go-dataset.git
cd computer-go-dataset

# Check contents
ls -lh

# Use the games
cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py \
    --sgf_dir ~/computer-go-dataset \
    --board_size 9 \
    --epochs 10
```

### Option B: Search GitHub

```bash
# Search GitHub for more datasets
# Visit: https://github.com/search?q=go+sgf+dataset

# Popular repositories:
# - https://github.com/featurecat/go-dataset
# - https://github.com/hughperkins/kgsgo-dataset-preprocessor
# - https://github.com/tensorflow/minigo (includes training data)
```

---

## Method 5: Generate from KataGo (No Download!)

If you already have KataGo running, generate your own high-quality games:

```bash
cd /Users/williamcha/Desktop/gomcts

# Generate 100 self-play games and save as SGF
python generate_katago_sgf.py \
    --games 100 \
    --output_dir ~/go_games/katago_generated

# Then train on them
python train_from_sgf.py \
    --sgf_dir ~/go_games/katago_generated \
    --board_size 9 \
    --epochs 10
```

I can create the `generate_katago_sgf.py` script if you want!

---

## 🔍 Verifying Your SGF Files

### Check file format:

```bash
# View first few lines of an SGF file
head -n 10 ~/go_games/kgs_9x9/*.sgf | head -n 30
```

Should look like:
```
(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]
RU[Japanese]SZ[9]KM[6.50]
PW[White Player]PB[Black Player]
;B[gc];W[cg];B[gg]...
```

### Check board size:

```bash
# Count 9x9 games
grep -l "SZ\[9\]" ~/go_games/*/*.sgf | wc -l

# Count 19x19 games
grep -l "SZ\[19\]" ~/go_games/*/*.sgf | wc -l
```

### Check game quality:

```bash
# Find games with rank information
grep -l "[BW]R\[" ~/go_games/*/*.sgf | head -5

# Look at a high-quality game
grep "[BW]R\[" ~/go_games/*/*.sgf | head -5
```

---

## 🎯 Recommended Workflow

### Day 1: Quick Start (500 games)

```bash
# Download 500 games from KGS (1-2 hours manual work)
# Visit https://www.gokgs.com/gameArchives.jsp
# Filter: 9x9, 5d+, recent dates
# Save to ~/go_games/kgs_9x9/

# Train (2-3 hours)
cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10 \
    --batch_size 128

# Test
python startgame.py --sims 64
```

### Week 1: Scale Up (1000+ games)

```bash
# Purchase GoGoD or download more from KGS/OGS
# Aim for 1000+ games

# Train with more epochs
python train_from_sgf.py \
    --sgf_dir ~/go_games/combined \
    --board_size 9 \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001
```

### Week 2: Refine with KataGo

```bash
# Continue training with reinforcement learning
python trainwithkatago.py \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.0003  # Lower LR since already trained
```

---

## 🐛 Troubleshooting

### Problem: "No SGF files found"

```bash
# Check directory path
ls -lh ~/go_games/kgs_9x9/

# Check file extensions
ls ~/go_games/kgs_9x9/*.sgf | head -5

# Make sure you're in the right directory
pwd
```

### Problem: "No valid samples extracted"

This means the SGF files are for different board sizes.

```bash
# Check board sizes in your SGF files
grep "SZ\[" ~/go_games/*/*.sgf | sort | uniq -c

# Filter for 9x9 only
mkdir -p ~/go_games/9x9_only
for file in ~/go_games/*/*.sgf; do
    if grep -q "SZ\[9\]" "$file"; then
        cp "$file" ~/go_games/9x9_only/
    fi
done

# Train on filtered files
python train_from_sgf.py \
    --sgf_dir ~/go_games/9x9_only \
    --board_size 9
```

### Problem: Files download as HTML instead of SGF

Some websites serve SGF files through redirects. Try:

```bash
# Use curl with redirect following
curl -L -o game.sgf "https://example.com/game.sgf?id=12345"
```

### Problem: Too few positions extracted

```bash
# Check game lengths
for file in ~/go_games/*/*.sgf; do
    moves=$(grep -o ";[BW]\[" "$file" | wc -l)
    echo "$file: $moves moves"
done | sort -t: -k2 -n

# Filter out very short games (< 20 moves)
mkdir -p ~/go_games/filtered
for file in ~/go_games/*/*.sgf; do
    moves=$(grep -o ";[BW]\[" "$file" | wc -l)
    if [ "$moves" -gt 20 ]; then
        cp "$file" ~/go_games/filtered/
    fi
done
```

---

## 📊 How Many Games Do You Need?

| Games | Positions | Training Time | Expected Strength |
|-------|-----------|---------------|-------------------|
| 100 | ~5,000 | 15 min | Weak amateur |
| 500 | ~25,000 | 1 hour | Decent amateur |
| 1,000 | ~50,000 | 2 hours | Strong amateur |
| 5,000 | ~250,000 | 8 hours | Very strong |
| 10,000+ | ~500,000+ | 15+ hours | Expert amateur |

**Recommendation**: Start with 500-1000 games, then scale up if needed.

---

## 🎓 Pro Tips

### 1. Combine Multiple Sources

```bash
# Mix KGS, OGS, and GoGoD games
mkdir -p ~/go_games/all_9x9
cp ~/go_games/kgs_9x9/*.sgf ~/go_games/all_9x9/
cp ~/go_games/ogs_9x9/*.sgf ~/go_games/all_9x9/
cp ~/go_games/gogod_9x9/*.sgf ~/go_games/all_9x9/

# Remove duplicates (by file size)
fdupes -dN ~/go_games/all_9x9/

# Train on combined dataset
python train_from_sgf.py \
    --sgf_dir ~/go_games/all_9x9 \
    --board_size 9 \
    --epochs 15
```

### 2. Filter by Player Strength

```bash
# Only keep games with strong players (5d+)
mkdir -p ~/go_games/5dan_plus
for file in ~/go_games/*/*.sgf; do
    if grep -qE "[BW]R\[[5-9]d\]" "$file"; then
        cp "$file" ~/go_games/5dan_plus/
    fi
done
```

### 3. Resume Training

```bash
# If training was interrupted, just run again
# The network loads existing weights.pt automatically
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 10 \
    --lr 0.0001  # Lower LR for fine-tuning
```

### 4. Data Augmentation

The training script could be extended to use 8-fold symmetry:

```python
# Each position can be rotated/flipped 8 ways
# This effectively multiplies your dataset by 8!
# (Not implemented yet, but possible enhancement)
```

---

## ✅ Complete Example: Start to Finish

```bash
# === Day 1: Setup and Download (1-2 hours) ===

# 1. Create directory
mkdir -p ~/go_games/kgs_9x9
cd ~/go_games/kgs_9x9

# 2. Visit https://www.gokgs.com/gameArchives.jsp
#    - Set filter: Board size = 9, Rank >= 5d
#    - Download ~500 games manually (click each, save SGF)

# 3. Verify downloads
ls -1 *.sgf | wc -l
# Should show ~500

# === Day 1: Train (2-3 hours) ===

# 4. Start training
cd /Users/williamcha/Desktop/gomcts
python train_from_sgf.py \
    --sgf_dir ~/go_games/kgs_9x9 \
    --board_size 9 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001

# === Day 1: Test ===

# 5. Play against your trained network
python startgame.py --sims 64

# 6. Check strength
# Your network should now play decent amateur-level Go!

# === Optional: Continue with KataGo (Week 1) ===

# 7. Improve further with reinforcement learning
python trainwithkatago.py \
    --games 1000 \
    --sims 64 \
    --size 9 \
    --lr 0.0003 \
    --auto_install_assets

# Final result: Strong amateur to low dan level!
```

---

## 📚 Additional Resources

- **KGS Archives**: https://www.gokgs.com/gameArchives.jsp
- **OGS API Docs**: https://ogs.docs.apiary.io/
- **GoGoD**: https://gogodonline.co.uk/
- **SGF Format**: http://www.red-bean.com/sgf/
- **Senseis Library** (Go resources): https://senseis.xmp.net/

---

## 🎉 Summary

**Easiest**: KGS manual download (500 games in 1-2 hours)  
**Best quality**: GoGoD purchase ($40, 1000s of games)  
**Most automated**: OGS API (requires programming)  
**No download**: Generate with KataGo (I can create script)

Then just run:
```bash
python train_from_sgf.py --sgf_dir YOUR_DIRECTORY --board_size 9 --epochs 10
```

That's it! 🚀


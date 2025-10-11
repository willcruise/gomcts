# How SGF Extraction Works - Detailed Walkthrough

## 📄 Input: SGF File (Text Format)

Let's use one of your Shusaku games as an example:

```
(;
PB[Yasuda Shusaku]
BR[4d]
PW[Ito Tokubee]
WR[5d]
RE[B+R]
JD[Tenpo 15-9-28]
DT[1844-11-08]

;B[qd];W[oc];B[ed];W[cd];B[df];W[ec];B[fc];W[dc];B[fd];W[cf]
;B[cg];W[bf];B[lc];W[dg];B[od];W[pd];B[pe];W[pc];B[nd];W[qc]
;B[re];W[nc];B[md];W[rd];B[qe];W[qk];B[dd];W[cc];B[fg];W[dh]
...
```

---

## 🔍 STEP-BY-STEP EXTRACTION

### Step 1: Read the File

```python
with open("Shusaku/101.sgf", 'r') as f:
    content = f.read()

# content is now a big string:
# "(;PB[Yasuda Shusaku]BR[4d]PW[Ito Tokubee]..."
```

---

### Step 2: Extract Metadata

#### A. Find Board Size

```python
# Look for "SZ[" in the string
if 'SZ[' in content:
    # If found: "...SZ[19]..."
    sz_start = content.index('SZ[') + 3  # Points to '1'
    sz_end = content.index(']', sz_start)  # Points after '9'
    sgf_size = int(content[sz_start:sz_end])  # = 19
    
    if sgf_size != 19:  # We want 19x19
        return None  # Skip this file
```

**Note:** Many old games don't have SZ tag (defaults to 19x19)

#### B. Extract Game Result

```python
# Look for "RE[" in the string
if 'RE[' in content:
    # Found: "...RE[B+R]..."
    re_start = content.index('RE[') + 3  # Points to 'B'
    re_end = content.index(']', re_start)
    result_str = content[re_start:re_end]  # "B+R"
    
    if 'B+' in result_str:
        result_value = 1.0   # Black wins
    elif 'W+' in result_str:
        result_value = -1.0  # White wins
    else:
        result_value = 0.0   # Draw
```

**Result for this game:** `result_value = 1.0` (Black/Shusaku won)

---

### Step 3: Initialize Empty Board

```python
board = Board(19, enforce_rules=True, forbid_suicide=True, ko_rule='simple')

# Board state:
# 19x19 grid, all zeros (empty)
# turn = 1 (Black to move)
```

---

### Step 4: Parse Moves One-by-One

The SGF contains moves like: `;B[qd];W[oc];B[ed]...`

#### Move 1: Black plays at Q16

**A. Find the move in the text**

```python
pos = 0  # Start at beginning of file
black_pos = content.find(';B[', pos)  # Found at position 95: ";B[qd]"
white_pos = content.find(';W[', pos)  # Found at position 102: ";W[oc]"

# Black move comes first (95 < 102)
next_pos = black_pos  # = 95
is_black = True
```

**B. Extract the coordinates**

```python
# Content at position 95: ";B[qd]..."
coord_start = next_pos + 3  # Points to 'q' in "qd"
coord_end = content.index(']', coord_start)  # Points after 'd'
coord_str = content[coord_start:coord_end]  # "qd"
```

**C. Convert SGF coordinates to board index**

SGF uses letter coordinates where 'a' = 0:
- 'q' is the 16th letter (q = 16)
- 'd' is the 4th letter (d = 3)

```python
col = ord('q') - ord('a')  # = 16 (column Q)
row = ord('d') - ord('a')  # = 3  (row 4 from top)

# For 19x19 board:
action = row * 19 + col  # = 3 * 19 + 16 = 73

# Visual: Position 73 on 19x19 board
#   a b c d e f g h i j k l m n o p q r s
# 1 · · · · · · · · · · · · · · · · · · ·
# 2 · · · · · · · · · · · · · · · · · · ·
# 3 · · · · · · · · · · · · · · · · · · ·
# 4 · · · ● · · · · · · · · · · · · ● · ·  ← Position 73 (Q4)
#                                   ↑
```

Wait, that's wrong! Let me recalculate - SGF coordinates are confusing!

Actually for 19x19:
- Bottom-left is 'a' = row 18 (in array coordinates)
- Top-left is 's' = row 0

So for "qd":
```python
col = ord('q') - ord('a')  # = 16 (column Q, 17th column from left)
row = ord('d') - ord('a')  # = 3  (4th row from bottom = 15 from top)

# In standard display (A-T from top):
# Row 'd' in SGF = 19 - 3 = 16 from top (row P)

action = row * 19 + col  # = 3 * 19 + 16 = 73
```

Actually, the code just uses row * size + col directly with SGF coordinates.

**D. Verify move is legal**

```python
legal_moves = board.legal_moves()  # Returns [0,1,2,...,360] (all squares + pass)

if action not in legal_moves:
    # This move is illegal, skip rest of game
    break
```

**E. Record training sample BEFORE playing**

```python
# Capture current board state
features = board.to_features().copy()
# Returns numpy array of shape (1436,) for 19x19:
#   - First 361 elements: Black stones (all 0s - empty board)
#   - Next 361 elements: White stones (all 0s - empty board)  
#   - Next 361 elements: Empty squares (all 1s - board empty)
#   - Last 361 elements: Turn indicator (all 1s - Black's turn)

# Outcome from Black's perspective (playing now)
outcome = result_value  # = 1.0 (Black won this game)

# Save the training sample
sample = (features, action, outcome)
samples.append(sample)
# samples = [(features_move1, 73, 1.0)]
```

**F. Play the move**

```python
board.play(action)  # Play at position 73

# Board now:
# - Grid[3, 16] = 1 (Black stone at Q4)
# - turn = -1 (White's turn now)
# - history = [73]
```

---

#### Move 2: White plays at P17

**A. Find next move**

```python
pos = 102  # Continue from after previous move
black_pos = content.find(';B[', pos)  # Found at position 109: ";B[ed]"
white_pos = content.find(';W[', pos)  # Found at position 102: ";W[oc]"

# White move comes first (102 < 109)
next_pos = white_pos  # = 102
is_black = False
```

**B. Extract coordinates**

```python
coord_str = "oc"
col = ord('o') - ord('a')  # = 14 (column O)
row = ord('c') - ord('a')  # = 2  (row C)
action = 2 * 19 + 14  # = 52
```

**C. Record training sample**

```python
features = board.to_features().copy()
# Now shows:
#   - First 361: Black stones (position 73 = 1, rest 0)
#   - Next 361: White stones (all 0 - White hasn't played yet)
#   - Next 361: Empty (position 73 = 0, rest 1)
#   - Last 361: Turn indicator (all -1s or 0s - White's turn)

# Outcome from White's perspective
outcome = -result_value  # = -1.0 (White lost this game!)

sample = (features, 52, -1.0)
samples.append(sample)
# samples = [
#   (features_move1, 73, 1.0),   # Black's move
#   (features_move2, 52, -1.0)   # White's move
# ]
```

**D. Play the move**

```python
board.play(52)
# Board now has Black stone at 73, White stone at 52
```

---

#### Move 3: Black plays at E16

```python
coord_str = "ed"
col = ord('e') - ord('a')  # = 4
row = ord('d') - ord('a')  # = 3
action = 3 * 19 + 4  # = 61

features = board.to_features().copy()
# Shows: Black at 73, White at 52, turn = Black

outcome = result_value  # = 1.0 (Black won)
samples.append((features, 61, 1.0))
board.play(61)
```

---

### Step 5: Continue Until End of Game

The parser keeps doing this for EVERY move:

```
Move 1: Black Q16  → sample(board_empty, 73, +1.0)
Move 2: White O17  → sample(board_1stone, 52, -1.0)
Move 3: Black E16  → sample(board_2stones, 61, +1.0)
Move 4: White C16  → sample(board_3stones, ..., -1.0)
...
Move 150: Black passes → sample(board_149stones, 361, +1.0)
Move 151: White passes → sample(board_149stones, 361, -1.0)

Result: 151 training samples from this one game!
```

---

## 📊 What Gets Extracted from One Game

**From Shusaku game 101.sgf:**

```python
samples = [
    # Move 1 (Black Q16)
    (
        array([0,0,0,...,1,1,1,...]),  # 1436 numbers: empty board
        73,                             # Action: position 73
        1.0                             # Outcome: Black won
    ),
    
    # Move 2 (White O17)
    (
        array([0,0,...,1,...,0,0,...,1,1,...]),  # Board with 1 black stone
        52,                                       # Action: position 52
        -1.0                                      # Outcome: White lost
    ),
    
    # ... 150 more samples ...
]

Total: 151 samples from one game
```

---

## 🔄 Repeat for All Games

```python
def load_sgf_dataset(sgf_dir: str, board_size: int = 19):
    all_samples = []
    
    # Find all SGF files
    sgf_files = glob.glob(os.path.join(sgf_dir, "**/*.sgf"), recursive=True)
    # For you: 4,835 files
    
    for sgf_file in sgf_files:
        samples = parse_sgf_file(sgf_file, board_size=19)
        
        if samples is not None:  # Valid game
            all_samples.extend(samples)
            # all_samples now has 151 more samples
    
    return all_samples

# Result:
# all_samples = [
#   (features_game1_move1, action, outcome),
#   (features_game1_move2, action, outcome),
#   ...
#   (features_game4835_move150, action, outcome)
# ]
# Total: ~300,000 samples
```

---

## 📦 What Each Sample Contains

### 1. Features (Input to Network)

For 19x19 board:
```python
features.shape = (1436,)  # 4 * 19 * 19

# Breaking it down:
features[0:361]     = Black stones (1 where Black, 0 elsewhere)
features[361:722]   = White stones (1 where White, 0 elsewhere)  
features[722:1083]  = Empty spaces (1 where empty, 0 elsewhere)
features[1083:1444] = Turn indicator (1 for Black, 0 for White)
```

**Example for move 3:**
```
Board state:
  a b c d e f g h i j k l m n o p q r s
1 · · · · · · · · · · · · · · · · · · ·
2 · · · · · · · · · · · · · · · · · · ·
3 · · · · · · · · · · · · · · ○ · · · ·  ← White at O3
4 · · · · · · · · · · · · · · · · ● · ·  ← Black at Q4
5 · · · · · · · · · · · · · · · · · · ·

Features:
  [0,0,0,...,1(at 73),...,0,0,    # Black stones
   0,0,0,...,1(at 52),...,0,0,    # White stones
   1,1,1,...,0(at 73),0(at 52),...1,1,  # Empty
   1,1,1,...,1,1,1,...]           # Turn (Black)
```

### 2. Action (Target Move)

```python
action = 61  # Pro played at position 61 (E16)
```

This is what we want the network to predict!

### 3. Outcome (Target Value)

```python
outcome = 1.0   # This player (Black) won the game
# or
outcome = -1.0  # This player (White) lost the game
```

This is what we want the value head to predict!

---

## 🎯 Key Insights

### 1. Sequential Processing
```
Game moves: M1 → M2 → M3 → ... → M150
Board state: Empty → 1 stone → 2 stones → ... → Full board
Samples: S1, S2, S3, ..., S150
```

Each sample shows the board BEFORE the pro's move.

### 2. Alternating Perspectives
```
Sample 1: Black's view, outcome = +1.0 (won)
Sample 2: White's view, outcome = -1.0 (lost)
Sample 3: Black's view, outcome = +1.0 (won)
Sample 4: White's view, outcome = -1.0 (lost)
```

The network learns from both sides!

### 3. Rich Dataset
```
From 1 game (150 moves):
  - 150 board positions
  - 150 expert moves
  - All labeled with final outcome
  
From 4,835 games:
  - ~300,000 positions
  - Early game, middle game, endgame
  - Different styles (Shusaku, Go Seigen, etc.)
```

### 4. Coordinate Conversion

SGF uses letters ('aa' to 'ss' for 19x19):
```
'a' = 0, 'b' = 1, ..., 's' = 18

"qd" → (row=3, col=16) → index = 3*19 + 16 = 73
```

### 5. Feature Representation

The board gets converted to 4 "planes":
```
Plane 1: Your stones
Plane 2: Opponent stones
Plane 3: Empty spaces
Plane 4: Whose turn

This is how CNNs see images - multiple channels!
```

---

## ✅ Summary

**Extraction Process:**

1. ✅ Read SGF file (text)
2. ✅ Parse metadata (who won)
3. ✅ Initialize empty board
4. ✅ For each move:
   - Capture board state → features
   - Record pro's move → action
   - Label with outcome → value
   - Play the move
   - Repeat
5. ✅ Return list of (features, action, outcome) tuples

**Result:** Your 4,835 games become 300,000 labeled training examples!

Each example teaches the network:
- "In this position..." (features)
- "...the pro played here..." (action)
- "...and this player won" (outcome)

The network learns by seeing thousands of these examples! 🎉


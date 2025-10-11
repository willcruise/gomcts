# Loss Function Explained: How Both Heads Are Trained

## 🧠 Network Architecture Recap

Your neural network has **two heads** (outputs) sharing a common hidden layer:

```
Input (board state)
    ↓
[Hidden Layer] ← Shared representation
    ↓     ↓
    ↓     ↓
[Policy] [Value]
   ↓       ↓
Moves   Position
(82)    evaluation
        (-1 to +1)
```

**Both heads learn together** from the same shared features!

---

## 📐 The Complete Loss Function

Looking at `policyneural.py` line 332:

```python
loss = ce + c_v * mse + l2 * reg
```

Let me break down each component:

---

## 1️⃣ Policy Head Loss (Cross-Entropy)

### Line 323-324:
```python
logp = F.log_softmax(logits, dim=1)
ce = -(t_pi * logp).sum(dim=1).mean()
```

### What it does:
Measures how well your network's move predictions match the target moves.

### Mathematical explanation:

**Cross-Entropy Loss:**
```
CE = -Σ(target[i] * log(prediction[i]))
```

Where:
- `target[i]` = target probability for action i (usually 1.0 for the pro's move, 0.0 for others)
- `prediction[i]` = network's predicted probability for action i (after softmax)

### Example:

Board position: Pro plays at E5 (action index 40)

```
Target distribution:
[0, 0, 0, ..., 1.0, ..., 0, 0]  ← 1.0 at index 40
              ↑
            E5 (correct move)

Network prediction (before training):
[0.01, 0.02, 0.03, ..., 0.05, ..., 0.01, 0.02]  ← 0.05 at index 40
                       ↑
                    E5 (only 5% confidence!)

Cross-Entropy = -log(0.05) = 2.996  (high loss = bad!)

After training:
[0.001, 0.002, ..., 0.65, ..., 0.001, 0.002]  ← 0.65 at index 40
                    ↑
                 E5 (65% confidence!)

Cross-Entropy = -log(0.65) = 0.431  (low loss = good!)
```

### Why cross-entropy?
- Heavily penalizes confident wrong predictions
- Encourages high probability on correct moves
- Works naturally with softmax probabilities

### Gradient flow:
```python
∂CE/∂logits = (prediction - target) / batch_size
```

This means:
- If network predicts 0.05 but target is 1.0 → gradient pushes up
- If network predicts 0.8 but target is 0.0 → gradient pushes down

---

## 2️⃣ Value Head Loss (MSE)

### Line 325:
```python
mse = F.mse_loss(value, t_v)
```

### What it does:
Measures how well your network's position evaluation matches the actual game outcome.

### Mathematical explanation:

**Mean Squared Error:**
```
MSE = mean((prediction - target)²)
```

Where:
- `prediction` = network's value output (between -1 and +1)
- `target` = game outcome from current player's perspective
  - +1.0 if current player won
  - -1.0 if current player lost
  - 0.0 if draw

### Example:

**Position from a game Black eventually won:**

```
Position A (Black to move):
  Network prediction: v = +0.2  (thinks Black is slightly ahead)
  Actual outcome: Black won → target = +1.0
  
  MSE = (0.2 - 1.0)² = 0.64  (moderate error)
  Gradient: ∂MSE/∂v = 2(0.2 - 1.0) = -1.6
  → Network learns to be MORE optimistic for this position

Position B (White to move, same game):
  Network prediction: v = +0.1  (thinks White is slightly ahead)
  Actual outcome: Black won → target = -1.0 (White lost!)
  
  MSE = (0.1 - (-1.0))² = 1.21  (large error!)
  Gradient: ∂MSE/∂v = 2(0.1 - (-1.0)) = 2.2
  → Network learns to be MORE pessimistic for this position
```

### Why MSE?
- Simple and differentiable
- Quadratic penalty (larger errors penalized more)
- Works well for regression tasks

### Alternative: Huber Loss
Some implementations use Huber loss (less sensitive to outliers):
```python
# Not in your code, but an option:
huber = F.smooth_l1_loss(value, t_v)
```

---

## 3️⃣ Regularization Term (L2)

### Lines 327-331:
```python
reg = (
    self.fc1.weight.pow(2).sum()
    + self.policy_head.weight.pow(2).sum()
    + self.value_head.weight.pow(2).sum()
)
```

### What it does:
Prevents overfitting by penalizing large weights.

### Mathematical explanation:

**L2 Regularization:**
```
Reg = Σ(weight²) for all weights
```

This encourages weights to stay small.

### Why L2?
- **Prevents overfitting**: Network can't just memorize training positions
- **Improves generalization**: Forces network to learn general patterns
- **Numerical stability**: Keeps weights from exploding

### Example:

Without L2:
```
Weights might grow very large:
W_policy[10,5] = 127.3  → overfits to specific positions
W_policy[20,8] = -89.4
```

With L2 (default l2=1e-4):
```
Weights stay moderate:
W_policy[10,5] = 2.3  → learns general patterns
W_policy[20,8] = -1.7
```

### Trade-off:
```python
# Line 332
loss = ce + c_v * mse + l2 * reg
#           ↑           ↑
#         default 1.0   default 1e-4
```

- Large L2 → strong regularization → may underfit
- Small L2 → weak regularization → may overfit

---

## 4️⃣ Combining Everything: The Full Loss

### Line 332:
```python
loss = ce + c_v * mse + l2 * reg
```

### The balance:

```
Total Loss = [Policy Loss] + c_v × [Value Loss] + l2 × [Regularization]
             ↑                ↑                    ↑
          Usually ~1.0     Usually ~0.5        Usually ~0.0001
```

### Why `c_v` (value weight)?

The `c_v` parameter balances policy vs value training:

```python
# Default: c_v = 1.0 (equal weight)
loss = ce + 1.0 * mse + l2 * reg

# More emphasis on policy (moves):
loss = ce + 0.5 * mse + l2 * reg

# More emphasis on value (position eval):
loss = ce + 2.0 * mse + l2 * reg
```

**Why is this needed?**
- Cross-entropy and MSE have different scales
- Policy loss typically ranges 0-5
- Value loss typically ranges 0-1
- `c_v` balances their influence

---

## 🔄 How Gradients Flow to Both Heads

### Forward pass:
```python
# Line 319-321
h = F.relu(self.fc1(x))        # Shared hidden layer
logits = self.policy_head(h)    # Policy output
value = torch.tanh(self.value_head(h))  # Value output
```

### Backward pass (Line 334):
```python
loss.backward()  # PyTorch computes all gradients automatically
```

This computes:
```
∂loss/∂policy_head weights  ← from CE term
∂loss/∂value_head weights   ← from MSE term
∂loss/∂fc1 weights          ← from BOTH CE and MSE terms!
```

### Key insight:
**The shared hidden layer (fc1) receives gradients from both heads!**

```
        Input
          ↓
    [Hidden Layer]
          ↓
    gradients from both
          ↓     ↓
      [Policy] [Value]
          ↓     ↓
       CE loss  MSE loss
```

The hidden layer learns features useful for **both** predicting moves AND evaluating positions!

---

## 📊 Training Example

Let's trace through one training step:

### Setup:
```python
Board position: Black to move
Pro's move: E5 (action 40)
Game outcome: Black won
```

### Forward pass:
```python
features = board.to_features()  # [324] for 9x9
h = relu(fc1(features))         # [256] hidden
logits = policy_head(h)         # [82] move scores
value = tanh(value_head(h))     # [1] position eval

# Before training:
logits[40] = 0.5  (E5 has score 0.5)
After softmax: prediction[40] = 0.05  (5% confidence)
value = 0.1  (thinks position is slightly good for Black)
```

### Loss computation:
```python
# Policy loss
CE = -log(0.05) = 2.996

# Value loss (Black to move, Black won)
MSE = (0.1 - 1.0)² = 0.81

# Regularization
reg = Σ(W²) = 150.0  (sum of all squared weights)

# Total loss
loss = 2.996 + 1.0 * 0.81 + 1e-4 * 150.0
     = 2.996 + 0.81 + 0.015
     = 3.821
```

### Backward pass:
```python
# Compute gradients
loss.backward()

# Policy head gradients (simplified):
∂loss/∂logits[40] = (0.05 - 1.0) = -0.95  (push up E5)
∂loss/∂logits[30] = (0.02 - 0.0) = +0.02  (push down other moves)

# Value head gradients:
∂loss/∂value = 2(0.1 - 1.0) = -1.8  (push value up)

# Shared layer gradients (receives from both!):
∂loss/∂fc1 = ∂CE/∂fc1 + c_v * ∂MSE/∂fc1
```

### Weight update (Line 350-358):
```python
# Gradient descent
policy_head.weight -= lr * ∂loss/∂policy_head
value_head.weight -= lr * ∂loss/∂value_head
fc1.weight -= lr * ∂loss/∂fc1  # Updated from both heads!
```

### After training:
```python
# Forward pass again:
logits[40] = 1.2  (E5 now has higher score!)
After softmax: prediction[40] = 0.15  (15% confidence - improved!)
value = 0.3  (now more optimistic about Black's position)

# Loss is lower:
CE = -log(0.15) = 1.897  (was 2.996)
MSE = (0.3 - 1.0)² = 0.49  (was 0.81)
loss = 1.897 + 0.49 + 0.015 = 2.402  (was 3.821) ✓
```

---

## 🎯 Key Takeaways

1. **Policy Head (Cross-Entropy)**:
   - Learns which moves are good
   - Trained on expert moves (supervised) or MCTS visit counts (RL)
   - Target is a probability distribution over moves

2. **Value Head (MSE)**:
   - Learns to evaluate positions
   - Trained on game outcomes
   - Target is +1 (win), -1 (loss), or 0 (draw)

3. **Shared Layer**:
   - Learns features useful for BOTH heads
   - Gets gradients from both losses
   - This is why multi-task learning works!

4. **Loss balancing**:
   - `c_v` controls policy vs value emphasis
   - `l2` controls regularization strength
   - Defaults (c_v=1.0, l2=1e-4) work well for most cases

5. **Training dynamics**:
   - Policy head typically learns faster (more direct signal)
   - Value head takes longer (only one outcome per game)
   - Both improve together due to shared representation

---

## 🔧 Hyperparameter Effects

### Value Weight (`c_v`):

```python
c_v = 0.5   # Policy-focused: learns good moves faster, value less accurate
c_v = 1.0   # Balanced: default, works well
c_v = 2.0   # Value-focused: position evaluation more accurate, moves less precise
```

### L2 Regularization (`l2`):

```python
l2 = 1e-5   # Light regularization: may overfit on small datasets
l2 = 1e-4   # Default: good balance
l2 = 1e-3   # Strong regularization: prevents overfitting but may underfit
```

### Learning Rate (`lr`):

```python
lr = 1e-4   # Conservative: slow but stable
lr = 1e-3   # Default: good for most cases
lr = 1e-2   # Aggressive: fast but may be unstable
```

---

## 📈 Monitoring Training

Watch these metrics:

```
Epoch 1: Train Loss=3.821, Policy_Loss=2.996, Value_Loss=0.810
Epoch 5: Train Loss=2.145, Policy_Loss=1.432, Value_Loss=0.698
Epoch 10: Train Loss=1.523, Policy_Loss=0.876, Value_Loss=0.632
```

**Good training:**
- Both losses decrease steadily
- Validation loss tracks training loss
- Policy accuracy increases (for supervised learning)

**Problems:**
- Loss increases → learning rate too high
- Val loss >> train loss → overfitting (increase L2)
- Loss plateaus early → learning rate too low or need more data

---

## 🔬 Advanced: Different Loss Functions

### Policy alternatives:

```python
# Current: Cross-Entropy (classification)
ce = -(target * log(prediction)).sum()

# Alternative 1: Focal Loss (focuses on hard examples)
focal = -alpha * (1-p)^gamma * log(p)

# Alternative 2: KL Divergence (used in some RL methods)
kl = (target * log(target/prediction)).sum()
```

### Value alternatives:

```python
# Current: MSE (L2 loss)
mse = (prediction - target)²

# Alternative 1: MAE (L1 loss, more robust)
mae = |prediction - target|

# Alternative 2: Huber Loss (combines L1 and L2)
huber = L1 if |error| < δ else L2
```

Your current setup (CE + MSE) is standard and works well!

---

## 💡 Why This Design Works

1. **Multi-task learning**: Shared representations help both tasks
2. **Complementary signals**: Policy learns "what to do", value learns "how good it is"
3. **Efficient**: One network does two jobs
4. **Proven**: This is exactly AlphaGo's architecture!

The genius is that good features for predicting moves (policy) are also good for evaluating positions (value)!


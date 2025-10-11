#!/usr/bin/env python3
"""
Train policy/value network from professional Go game records (SGF files).

This is supervised learning / imitation learning - much more sample-efficient
than pure self-play for limited compute environments like Jetson.

Approach:
1. Load SGF files containing pro games
2. For each position, extract:
   - Board state -> network input
   - Pro's move -> target policy (one-hot)
   - Game outcome -> target value
3. Train network to predict pro moves and game outcomes

This is similar to how the original AlphaGo was bootstrapped before 
reinforcement learning.
"""

import argparse
import glob
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from board import Board
from policyneural import MLPPolicyValue


# Minimal SGF parser for Go games
def parse_sgf_file(filepath: str, board_size: int = 9) -> Optional[List[Tuple[np.ndarray, int, float]]]:
    """
    Parse a single SGF file and extract training samples.
    
    Returns: List of (features, action, outcome) tuples, or None if invalid
    - features: board state features (4*N*N array)
    - action: move index (0 to N*N for moves, N*N for pass)
    - outcome: game result from current player's perspective (+1 win, -1 loss, 0 draw)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None
    
    # Extract board size from SGF if specified
    if 'SZ[' in content:
        try:
            sz_start = content.index('SZ[') + 3
            sz_end = content.index(']', sz_start)
            sgf_size = int(content[sz_start:sz_end])
            if sgf_size != board_size:
                return None  # Skip games with different board sizes
        except Exception:
            pass
    
    # Extract game result
    result_value = 0.0
    if 'RE[' in content:
        try:
            re_start = content.index('RE[') + 3
            re_end = content.index(']', re_start)
            result_str = content[re_start:re_end].upper()
            if 'B+' in result_str:
                result_value = 1.0  # Black wins
            elif 'W+' in result_str:
                result_value = -1.0  # White wins
            else:
                result_value = 0.0  # Draw or unknown
        except Exception:
            pass
    
    # Extract moves: ;B[xy] for black, ;W[xy] for white
    board = Board(board_size, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
    samples: List[Tuple[np.ndarray, int, float]] = []
    
    # Parse moves sequentially
    pos = 0
    move_count = 0
    max_moves = board_size * board_size * 2  # Safety limit
    
    while pos < len(content) and move_count < max_moves:
        # Find next move
        black_pos = content.find(';B[', pos)
        white_pos = content.find(';W[', pos)
        
        # Determine which comes first
        if black_pos == -1 and white_pos == -1:
            break
        elif black_pos == -1:
            next_pos = white_pos
            is_black = False
        elif white_pos == -1:
            next_pos = black_pos
            is_black = True
        else:
            next_pos = min(black_pos, white_pos)
            is_black = (next_pos == black_pos)
        
        # Check if this move is for the correct player
        current_player_is_black = (board.turn == 1)
        if is_black != current_player_is_black:
            # Skip move if it's not the expected player's turn
            pos = next_pos + 1
            continue
        
        # Extract move coordinates
        try:
            coord_start = next_pos + 3
            coord_end = content.index(']', coord_start)
            coord_str = content[coord_start:coord_end].strip()
            
            if len(coord_str) == 0 or coord_str == 'tt':
                # Pass move
                action = board.pass_index
            elif len(coord_str) == 2:
                # Normal move: convert SGF coordinates (aa=top-left) to board index
                col = ord(coord_str[0]) - ord('a')
                row = ord(coord_str[1]) - ord('a')
                
                # Validate coordinates
                if 0 <= row < board_size and 0 <= col < board_size:
                    action = row * board_size + col
                else:
                    # Invalid coordinates, skip
                    pos = coord_end + 1
                    continue
            else:
                # Malformed coordinate
                pos = coord_end + 1
                continue
            
            # Check if move is legal
            legal_moves = board.legal_moves()
            if action not in legal_moves:
                # Illegal move in SGF (can happen with bad files), skip rest of game
                break
            
            # Record training sample BEFORE playing the move
            features = board.to_features().copy()
            # Outcome from current player's perspective
            outcome = result_value if is_black else -result_value
            samples.append((features, int(action), float(outcome)))
            
            # Play the move
            board.play(action)
            move_count += 1
            pos = coord_end + 1
            
        except Exception:
            # Error parsing this move, skip to next
            pos = next_pos + 1
            continue
    
    # Return samples if we got at least a few moves
    return samples if len(samples) >= 10 else None


def load_sgf_dataset(sgf_dir: str, board_size: int = 9, max_games: Optional[int] = None) -> List[Tuple[np.ndarray, int, float]]:
    """
    Load all SGF files from a directory and extract training samples.
    
    Args:
        sgf_dir: Directory containing .sgf files
        board_size: Board size to filter for
        max_games: Maximum number of games to load (None = all)
    
    Returns:
        List of (features, action, outcome) tuples
    """
    sgf_files = glob.glob(os.path.join(sgf_dir, "**/*.sgf"), recursive=True)
    
    if not sgf_files:
        raise FileNotFoundError(f"No SGF files found in {sgf_dir}")
    
    print(f"Found {len(sgf_files)} SGF files")
    
    if max_games is not None:
        random.shuffle(sgf_files)
        sgf_files = sgf_files[:int(max_games)]
    
    all_samples: List[Tuple[np.ndarray, int, float]] = []
    games_loaded = 0
    
    for i, sgf_file in enumerate(sgf_files):
        if (i + 1) % 100 == 0:
            print(f"Processing game {i+1}/{len(sgf_files)}... ({len(all_samples)} samples so far)")
        
        samples = parse_sgf_file(sgf_file, board_size=board_size)
        if samples is not None:
            all_samples.extend(samples)
            games_loaded += 1
    
    print(f"\nLoaded {games_loaded} games with {len(all_samples)} total positions")
    return all_samples


def train_from_samples(net: MLPPolicyValue,
                       samples: List[Tuple[np.ndarray, int, float]],
                       epochs: int = 10,
                       batch_size: int = 128,
                       lr: float = 1e-3,
                       value_weight: float = 1.0,
                       l2: float = 1e-4,
                       val_split: float = 0.1,
                       checkpoint_every: int = 1,
                       board_size: int = 9) -> None:
    """
    Train the network on extracted samples using supervised learning.
    
    Args:
        net: Neural network to train
        samples: List of (features, action, outcome) tuples
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        value_weight: Weight for value loss vs policy loss
        l2: L2 regularization strength
        val_split: Fraction of data for validation
        checkpoint_every: Save weights every N epochs
        board_size: Board size (for action space)
    """
    if not samples:
        raise ValueError("No training samples provided")
    
    # Shuffle and split into train/val
    random.shuffle(samples)
    n_val = int(len(samples) * val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    print(f"\nTraining set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Learning rate: {lr}, Value weight: {value_weight}, L2: {l2}\n")
    
    num_actions = board_size * board_size + 1
    
    for epoch in range(epochs):
        # Shuffle training data each epoch
        random.shuffle(train_samples)
        
        # Training phase
        train_loss = 0.0
        train_batches = 0
        
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i:i+batch_size]
            
            # Prepare batch data
            features_batch = np.stack([s[0] for s in batch], axis=0)
            actions_batch = np.array([s[1] for s in batch], dtype=np.int64)
            outcomes_batch = np.array([s[2] for s in batch], dtype=np.float32).reshape(-1, 1)
            
            # Convert actions to one-hot policy targets
            target_pi = np.zeros((len(batch), num_actions), dtype=np.float32)
            for j, action in enumerate(actions_batch):
                target_pi[j, action] = 1.0
            
            # Forward pass
            _, _, cache = net.forward(features_batch)
            
            # Backward pass
            loss, grads = net.backward(cache, target_pi, outcomes_batch, l2=l2, c_v=value_weight)
            
            # Update weights
            net.step(grads, lr=lr, save_every=0)  # Don't save every step
            
            train_loss += loss
            train_batches += 1
        
        avg_train_loss = train_loss / max(1, train_batches)
        
        # Validation phase
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i+batch_size]
            
            features_batch = np.stack([s[0] for s in batch], axis=0)
            actions_batch = np.array([s[1] for s in batch], dtype=np.int64)
            outcomes_batch = np.array([s[2] for s in batch], dtype=np.float32).reshape(-1, 1)
            
            target_pi = np.zeros((len(batch), num_actions), dtype=np.float32)
            for j, action in enumerate(actions_batch):
                target_pi[j, action] = 1.0
            
            logits, values, cache = net.forward(features_batch)
            loss, _ = net.backward(cache, target_pi, outcomes_batch, l2=l2, c_v=value_weight)
            
            # Check top-1 accuracy
            predicted_actions = np.argmax(logits, axis=1)
            val_correct += np.sum(predicted_actions == actions_batch)
            val_total += len(batch)
            
            val_loss += loss
            val_batches += 1
        
        avg_val_loss = val_loss / max(1, val_batches)
        val_accuracy = 100.0 * val_correct / max(1, val_total)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Accuracy={val_accuracy:.1f}%")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            net._save()
            print(f"  → Checkpoint saved to weights.pt")
    
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Go neural network from professional game records (SGF files)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sgf_dir", type=str, required=True,
                        help="Directory containing SGF files")
    parser.add_argument("--board_size", type=int, default=9,
                        help="Board size to filter for")
    parser.add_argument("--max_games", type=int, default=None,
                        help="Maximum number of games to load (None = all)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--value_weight", type=float, default=1.0,
                        help="Weight for value loss vs policy loss")
    parser.add_argument("--l2", type=float, default=1e-4,
                        help="L2 regularization strength")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize network
    net = MLPPolicyValue(device=device)
    
    # Load SGF dataset
    print(f"\nLoading SGF files from: {args.sgf_dir}")
    samples = load_sgf_dataset(
        sgf_dir=args.sgf_dir,
        board_size=args.board_size,
        max_games=args.max_games
    )
    
    if not samples:
        raise RuntimeError("No valid samples extracted from SGF files")
    
    # Train
    train_from_samples(
        net=net,
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_weight=args.value_weight,
        l2=args.l2,
        val_split=args.val_split,
        checkpoint_every=args.checkpoint_every,
        board_size=args.board_size
    )
    
    print(f"\nWeights saved to: {net._weights_path_pt()}")
    print("You can now use these weights for MCTS-based play or continue with reinforcement learning!")


if __name__ == "__main__":
    main()


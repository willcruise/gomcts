#!/usr/bin/env python3
"""
Parallel SGF training with multi-worker support for maximum CPU/GPU utilization.

Improvements over train_from_sgf.py:
1. Parallel SGF file parsing (multi-process)
2. PyTorch DataLoader with workers (async batch loading)
3. Prefetching for GPU to stay busy
"""

import argparse
import glob
import os
import random
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from board import Board
from policyneural import MLPPolicyValue


def parse_sgf_file(filepath: str, board_size: int = 9) -> Optional[List[Tuple[np.ndarray, int, float]]]:
    """Parse a single SGF file - same as original."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None
    
    # Extract board size
    if 'SZ[' in content:
        try:
            sz_start = content.index('SZ[') + 3
            sz_end = content.index(']', sz_start)
            sgf_size = int(content[sz_start:sz_end])
            if sgf_size != board_size:
                return None
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
                result_value = 1.0
            elif 'W+' in result_str:
                result_value = -1.0
        except Exception:
            pass
    
    # Parse moves
    board = Board(board_size, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
    samples: List[Tuple[np.ndarray, int, float]] = []
    
    pos = 0
    move_count = 0
    max_moves = board_size * board_size * 2
    
    while pos < len(content) and move_count < max_moves:
        black_pos = content.find(';B[', pos)
        white_pos = content.find(';W[', pos)
        
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
        
        current_player_is_black = (board.turn == 1)
        if is_black != current_player_is_black:
            pos = next_pos + 1
            continue
        
        try:
            coord_start = next_pos + 3
            coord_end = content.index(']', coord_start)
            coord_str = content[coord_start:coord_end].strip()
            
            if len(coord_str) == 0 or coord_str == 'tt':
                action = board.pass_index
            elif len(coord_str) == 2:
                col = ord(coord_str[0]) - ord('a')
                row = ord(coord_str[1]) - ord('a')
                
                if 0 <= row < board_size and 0 <= col < board_size:
                    action = row * board_size + col
                else:
                    pos = coord_end + 1
                    continue
            else:
                pos = coord_end + 1
                continue
            
            legal_moves = board.legal_moves()
            if action not in legal_moves:
                break
            
            features = board.to_features().copy()
            outcome = result_value if is_black else -result_value
            samples.append((features, int(action), float(outcome)))
            
            board.play(action)
            move_count += 1
            pos = coord_end + 1
            
        except Exception:
            pos = next_pos + 1
            continue
    
    return samples if len(samples) >= 10 else None


def parse_sgf_worker(args):
    """Worker function for parallel SGF parsing."""
    filepath, board_size = args
    return parse_sgf_file(filepath, board_size)


def load_sgf_dataset_parallel(sgf_dir: str, 
                              board_size: int = 9,
                              max_games: Optional[int] = None,
                              num_workers: int = 4) -> List[Tuple[np.ndarray, int, float]]:
    """
    Load SGF files in parallel using multiple CPU workers.
    
    Args:
        sgf_dir: Directory containing SGF files
        board_size: Board size to filter for
        max_games: Max games to load
        num_workers: Number of parallel workers for parsing
    """
    sgf_files = glob.glob(os.path.join(sgf_dir, "**/*.sgf"), recursive=True)
    
    if not sgf_files:
        raise FileNotFoundError(f"No SGF files found in {sgf_dir}")
    
    print(f"Found {len(sgf_files)} SGF files")
    
    if max_games is not None:
        random.shuffle(sgf_files)
        sgf_files = sgf_files[:int(max_games)]
    
    # Parallel parsing with progress
    all_samples: List[Tuple[np.ndarray, int, float]] = []
    games_loaded = 0
    
    print(f"Parsing with {num_workers} workers...")
    
    # Create worker pool
    with mp.Pool(num_workers) as pool:
        # Prepare arguments
        args_list = [(f, board_size) for f in sgf_files]
        
        # Parse in parallel with progress
        for i, samples in enumerate(pool.imap_unordered(parse_sgf_worker, args_list)):
            if samples is not None:
                all_samples.extend(samples)
                games_loaded += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(sgf_files)} games... ({len(all_samples)} samples)")
    
    print(f"\nLoaded {games_loaded} games with {len(all_samples)} total positions")
    return all_samples


class SGFDataset(Dataset):
    """PyTorch Dataset for SGF training samples."""
    
    def __init__(self, samples: List[Tuple[np.ndarray, int, float]], num_actions: int):
        self.samples = samples
        self.num_actions = num_actions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, action, outcome = self.samples[idx]
        
        # Convert to tensors
        features_t = torch.from_numpy(features).float()
        
        # One-hot policy target
        target_policy = torch.zeros(self.num_actions, dtype=torch.float32)
        target_policy[action] = 1.0
        
        # Value target
        target_value = torch.tensor([outcome], dtype=torch.float32)
        
        return features_t, target_policy, target_value


def train_from_samples_parallel(net: MLPPolicyValue,
                                samples: List[Tuple[np.ndarray, int, float]],
                                epochs: int = 10,
                                batch_size: int = 128,
                                lr: float = 1e-3,
                                value_weight: float = 1.0,
                                l2: float = 1e-4,
                                val_split: float = 0.1,
                                checkpoint_every: int = 1,
                                board_size: int = 9,
                                num_workers: int = 4,
                                prefetch_factor: int = 2) -> None:
    """
    Train with PyTorch DataLoader for efficient multi-worker data loading.
    
    Args:
        num_workers: Number of DataLoader workers for async batch loading
        prefetch_factor: Batches to prefetch per worker
    """
    if not samples:
        raise ValueError("No training samples provided")
    
    num_actions = board_size * board_size + 1
    
    # Split train/val
    random.shuffle(samples)
    n_val = int(len(samples) * val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    print(f"\nTraining set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"DataLoader workers: {num_workers}, prefetch: {prefetch_factor}")
    print(f"Learning rate: {lr}, Value weight: {value_weight}, L2: {l2}\n")
    
    # Create datasets
    train_dataset = SGFDataset(train_samples, num_actions)
    val_dataset = SGFDataset(val_samples, num_actions)
    
    # Create DataLoaders with multi-worker support
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True if net._device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,  # Fewer workers for validation
        prefetch_factor=prefetch_factor,
        pin_memory=True if net._device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        net.fc1.train()
        net.policy_head.train()
        net.value_head.train()
        
        train_loss = 0.0
        train_batches = 0
        
        for features_batch, target_pi_batch, target_v_batch in train_loader:
            # Move to device
            features_batch = features_batch.to(net._device)
            target_pi_batch = target_pi_batch.to(net._device)
            target_v_batch = target_v_batch.to(net._device)
            
            # Convert to numpy for compatibility with current API
            X = features_batch.cpu().numpy()
            target_pi = target_pi_batch.cpu().numpy()
            target_v = target_v_batch.cpu().numpy()
            
            # Forward + backward
            _, _, cache = net.forward(X)
            loss, grads = net.backward(cache, target_pi, target_v, l2=l2, c_v=value_weight)
            net.step(grads, lr=lr, save_every=0)
            
            train_loss += loss
            train_batches += 1
        
        avg_train_loss = train_loss / max(1, train_batches)
        
        # Validation phase
        net.fc1.eval()
        net.policy_head.eval()
        net.value_head.eval()
        
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features_batch, target_pi_batch, target_v_batch in val_loader:
                X = features_batch.cpu().numpy()
                target_pi = target_pi_batch.cpu().numpy()
                target_v = target_v_batch.cpu().numpy()
                
                logits, values, cache = net.forward(X)
                loss, _ = net.backward(cache, target_pi, target_v, l2=l2, c_v=value_weight)
                
                # Accuracy
                predicted_actions = np.argmax(logits, axis=1)
                actual_actions = np.argmax(target_pi, axis=1)
                val_correct += np.sum(predicted_actions == actual_actions)
                val_total += len(X)
                
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
            print(f"  → Checkpoint saved")
    
    print("\nTraining complete!")


def main() -> None:
    # Force spawn for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(
        description="Parallel SGF training with multi-worker support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sgf_dir", type=str, required=True)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    
    # Parallelization options
    parser.add_argument("--parse_workers", type=int, default=4,
                       help="Number of workers for parallel SGF parsing")
    parser.add_argument("--dataloader_workers", type=int, default=4,
                       help="Number of DataLoader workers for batch loading")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Batches to prefetch per DataLoader worker")
    
    args = parser.parse_args()
    
    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Parse workers: {args.parse_workers}")
    print(f"DataLoader workers: {args.dataloader_workers}")
    
    # Initialize network
    net = MLPPolicyValue(device=device)
    
    # Load SGF files in parallel
    print(f"\nLoading SGF files from: {args.sgf_dir}")
    samples = load_sgf_dataset_parallel(
        sgf_dir=args.sgf_dir,
        board_size=args.board_size,
        max_games=args.max_games,
        num_workers=args.parse_workers
    )
    
    if not samples:
        raise RuntimeError("No valid samples extracted")
    
    # Train with parallel data loading
    train_from_samples_parallel(
        net=net,
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_weight=args.value_weight,
        l2=args.l2,
        val_split=args.val_split,
        checkpoint_every=args.checkpoint_every,
        board_size=args.board_size,
        num_workers=args.dataloader_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    print(f"\nWeights saved to: {net._weights_path_pt()}")


if __name__ == "__main__":
    main()


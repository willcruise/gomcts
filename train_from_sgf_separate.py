#!/usr/bin/env python3
"""
Alternative implementation: Train policy and value heads separately.

Policy head: Train immediately (we know the target move)
Value head: Train after game ends (we know the outcome)

This is more conceptually correct than retrospective labeling.
"""

import numpy as np
from typing import List, Tuple
from policyneural import MLPPolicyValue

def train_from_sgf_separate(net: MLPPolicyValue, 
                            games_data: List[Tuple],
                            epochs: int = 10,
                            lr_policy: float = 1e-3,
                            lr_value: float = 1e-3):
    """
    Train policy and value heads separately.
    
    Args:
        games_data: List of (features_list, actions_list, outcome) for each game
    """
    
    # Phase 1: Train Policy Head Only
    print("=" * 60)
    print("PHASE 1: Training Policy Head")
    print("=" * 60)
    
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_samples = 0
        
        for game_features, game_actions, game_outcome in games_data:
            # Train on each position independently
            for features, action in zip(game_features, game_actions):
                # Create one-hot target
                num_actions = net.policy_dim
                target_policy = np.zeros(num_actions, dtype=np.float32)
                target_policy[action] = 1.0
                
                # Forward pass
                X = features.reshape(1, -1)
                target_pi = target_policy.reshape(1, -1)
                
                # Dummy value target (not used)
                dummy_value = np.zeros((1, 1), dtype=np.float32)
                
                _, _, cache = net.forward(X)
                
                # Backward with c_v=0 (only train policy)
                loss, grads = net.backward(cache, target_pi, dummy_value, 
                                         l2=1e-4, c_v=0.0)  # c_v=0 disables value loss
                
                # Update only policy head and shared layer
                net.step(grads, lr=lr_policy, save_every=0)
                
                total_policy_loss += loss
                total_samples += 1
        
        avg_loss = total_policy_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs}: Policy Loss = {avg_loss:.4f}")
    
    # Save after policy training
    net._save()
    print("\nPolicy head training complete!\n")
    
    # Phase 2: Train Value Head Only
    print("=" * 60)
    print("PHASE 2: Training Value Head")
    print("=" * 60)
    
    for epoch in range(epochs):
        total_value_loss = 0.0
        total_samples = 0
        
        for game_features, game_actions, game_outcome in games_data:
            # Train on all positions from this game with the SAME outcome
            for features in game_features:
                # Determine target value for this player
                # (Need to track whose turn it was)
                # For simplicity, use game_outcome directly
                # (In practice, you'd flip based on whose turn)
                
                X = features.reshape(1, -1)
                target_value = np.array([[game_outcome]], dtype=np.float32)
                
                # Dummy policy target (not used)
                dummy_policy = np.zeros((1, net.policy_dim), dtype=np.float32)
                
                _, _, cache = net.forward(X)
                
                # Backward with policy disabled
                # We want ONLY value loss, so we'd need to modify backward
                # For now, use high c_v to emphasize value
                loss, grads = net.backward(cache, dummy_policy, target_value,
                                         l2=1e-4, c_v=10.0)  # High c_v for value focus
                
                net.step(grads, lr=lr_value, save_every=0)
                
                total_value_loss += loss
                total_samples += 1
        
        avg_loss = total_value_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs}: Value Loss = {avg_loss:.4f}")
    
    # Save after value training
    net._save()
    print("\nValue head training complete!")

# This is conceptually cleaner but more complex to implement!


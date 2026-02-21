#!/bin/bash
# Training script for Jetson Orin Nano with Docker

set -euo pipefail

# Default configuration
CONFIG="${1:-configs/train_multiworker.yaml}"

echo "========================================="
echo "  Go MCTS Self-Play Training"
echo "========================================="
echo ""
echo "Configuration: $CONFIG"
echo ""

# Ensure we're in the right directory
REPO_DIR="${REPO_DIR:-$HOME/gomcts}"
cd "$REPO_DIR"

# Run training in Docker
sudo docker run -it --rm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.05-py3-igpu \
  python3 -m gomcts.training.selftraining --config "/workspace/$CONFIG"

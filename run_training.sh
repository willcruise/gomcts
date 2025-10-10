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
cd ~/gomcts/gomcts

# Run training in Docker
sudo docker run -it --rm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/gomcts/gomcts:/workspace \
  nvcr.io/nvidia/pytorch:25.05-py3-igpu \
  python3 /workspace/selftraining.py --config /workspace/$CONFIG

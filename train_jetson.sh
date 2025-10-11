#!/bin/bash
# Quick-start training script for Jetson with limited compute
# This trains your neural network against KataGo for sample efficiency

set -e  # Exit on error

echo "================================================"
echo "  Training Go Neural Network on Jetson"
echo "  Strategy: Learn from KataGo (sample efficient)"
echo "================================================"
echo ""

# Default settings
GAMES=${1:-500}
SIMS=${2:-64}
SIZE=${3:-9}

echo "Configuration:"
echo "  - Games: $GAMES"
echo "  - MCTS Simulations: $SIMS"
echo "  - Board Size: ${SIZE}x${SIZE}"
echo ""

# Check if we should maximize Jetson performance
if command -v nvpmodel &> /dev/null; then
    echo "Detected Jetson - enabling max performance mode..."
    sudo nvpmodel -m 0 2>/dev/null || echo "(Skipping nvpmodel - may need sudo)"
    sudo jetson_clocks 2>/dev/null || echo "(Skipping jetson_clocks - may need sudo)"
    echo ""
fi

# Start training
echo "Starting training..."
echo "This will take approximately $(echo "scale=1; $GAMES * 3 / 60" | bc) hours"
echo ""

python trainwithkatago.py \
    --games $GAMES \
    --sims $SIMS \
    --size $SIZE \
    --lr 0.001 \
    --checkpoint_every 10 \
    --auto_install_assets \
    --device cuda \
    --require_gpu false

echo ""
echo "================================================"
echo "  Training Complete!"
echo "  Weights saved to: weights.pt"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Test your agent: python startgame.py --sims $SIMS"
echo "  2. Continue training: ./train_jetson.sh 500"
echo "  3. Monitor with: tegrastats (on Jetson)"


#!/bin/bash
# Backup weights to timestamped file on Jetson
# Run this after training completes

BACKUP_DIR=~/gomcts_weights_backups
mkdir -p $BACKUP_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE=~/gomcts/artifacts/weights.pt

if [ -f "$SOURCE" ]; then
    cp "$SOURCE" "$BACKUP_DIR/weights_${TIMESTAMP}.pt"
    echo "✅ Backed up to: $BACKUP_DIR/weights_${TIMESTAMP}.pt"
    
    # Keep only last 10 backups
    ls -t $BACKUP_DIR/weights_*.pt | tail -n +11 | xargs -r rm
    echo "📦 Kept last 10 backups:"
    ls -lh $BACKUP_DIR/
else
    echo "❌ Error: $SOURCE not found!"
fi




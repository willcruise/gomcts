Sample SGF Files for Testing
=============================

These are 3 sample 9x9 Go game records for testing the supervised learning training script.

To test the training script with these samples:

    python train_from_sgf.py \
        --sgf_dir ./sample_sgf_games \
        --board_size 9 \
        --epochs 3 \
        --batch_size 32

This is just for testing! For real training, you need many more games (1000+).
See SGF_TRAINING_GUIDE.md for where to download real game datasets.


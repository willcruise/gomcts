# Training Configuration Files

This directory contains YAML configuration files for different training scenarios.

## Available Configurations

### `train_single.yaml`
- **Purpose:** Simple single-worker training for testing
- **Total games:** 100
- **Workers:** 1 (no parallelism)
- **Use when:** Testing changes, debugging, or learning the system

### `train_multiworker.yaml` ⭐ **Recommended**
- **Purpose:** Full parallel training using all CPU cores
- **Total games:** 1002 (6 workers × 167 each)
- **Workers:** 6
- **Use when:** Standard training runs on Jetson Orin Nano

### `train_gpu_fast.yaml`
- **Purpose:** High-performance training with evaluation gates
- **Total games:** 504 (6 workers × 84 each)
- **Workers:** 6 (matches CPU cores)
- **Batch size:** 512 (larger for more GPU utilization)
- **Use when:** Overnight/long training with quality control

## Usage

### Basic usage with config file:
```bash
python3 selftraining.py --config configs/train_multiworker.yaml
```

### Override specific values:
```bash
# Use config but change learning rate
python3 selftraining.py --config configs/train_multiworker.yaml --lr 0.005

# Use config but train for fewer games
python3 selftraining.py --config configs/train_multiworker.yaml --games 500
```

### Docker usage:
```bash
sudo docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/gomcts/gomcts:/workspace \
  nvcr.io/nvidia/pytorch:25.05-py3-igpu \
  python3 /workspace/selftraining.py \
  --config /workspace/configs/train_multiworker.yaml
```

## Configuration Parameters

All YAML files support these parameters:

### Core Training
- `games`: Total number of self-play games
- `sims`: MCTS simulations per move
- `size`: Board size (9, 13, or 19)
- `device`: Computing device ("cuda" or "cpu")
- `require_gpu`: Fail if GPU not available

### Learning
- `lr`: Learning rate (default: 0.003)
- `value_weight`: Value loss weight (default: 1.0)
- `l2`: L2 regularization (default: 0.0001)

### Parallelism
- `workers`: Number of parallel worker processes
- `worker_games`: Games per worker (workers × worker_games = total games)

### Performance
- `mcts_batch_size`: Batch size for GPU inference (64-512)
- `mcts_flush_ms`: Flush timeout in ms (1.0-2.0)
- `use_cuda_graphs`: Enable CUDA graph optimization

### Checkpointing
- `checkpoint_every`: Save weights every N games
- `log_every`: Print progress every N games

### Evaluation (optional)
- `eval_every`: Run evaluation every N games (0 disables)
- `eval_games`: Number of evaluation games
- `eval_threshold`: Win rate threshold to accept updates (0.0-1.0)
- `eval_swap_colors`: Swap colors during evaluation

## Creating Custom Configurations

Copy an existing config and modify:

```bash
cp configs/train_multiworker.yaml configs/my_custom.yaml
# Edit my_custom.yaml
python3 selftraining.py --config configs/my_custom.yaml
```

## Notes

- Command-line arguments always override config file values
- **Worker count should match CPU core count** (6 cores on Jetson Orin Nano = 6 workers max)
  - Using more workers than cores causes context switching overhead and slows down training
  - Using fewer workers wastes CPU capacity
- Larger `mcts_batch_size` uses more GPU but may not always be faster
  - For MCTS, CPU is usually the bottleneck, not GPU
  - Batch sizes of 128-512 are typically optimal
- Enable `eval_every` for long training runs to ensure quality improvements

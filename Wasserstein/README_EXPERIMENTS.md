# Ternary DRO Mode Comparison Experiments

This directory contains scripts to run and monitor ternary concept training experiments with three DRO modes on two datasets.

## Experiments Running

6 experiments are currently running in parallel:

| Dataset | DRO Mode | Description |
|---------|----------|-------------|
| CEBaB | `post_hoc` | Post-hoc DRO (λ_dro=0, β=0) |
| CEBaB | `fixed_eps` | Fixed epsilon (ε=0.1) |
| CEBaB | `joint` | Joint training (default) |
| GoEmotions | `post_hoc` | Post-hoc DRO (λ_dro=0, β=0) |
| GoEmotions | `fixed_eps` | Fixed epsilon (ε=0.1) |
| GoEmotions | `joint` | Joint training (default) |

## Configuration

- **Epochs**: 50
- **Batch Size**: 16
- **Encoder**: distilbert-base-uncased (frozen)
- **Concepts**: 3 classes (negative/unknown/positive)
- **Aleatoric Head**: Enabled

## Directory Structure

```
experiments/ternary_dro_comparison/
├── cebab_post_hoc/
│   ├── checkpoints/         # Model checkpoints every 10 epochs
│   ├── eval_outputs/        # Evaluation outputs at end
│   └── training.log         # Full training log
├── cebab_fixed_eps/
├── cebab_joint/
├── goemotions_post_hoc/
├── goemotions_fixed_eps/
└── goemotions_joint/
```

## Monitoring

### Check status
```bash
./check_experiments.sh
```

### Follow logs in real-time
```bash
tail -f experiments/ternary_dro_comparison/*/training.log
```

### Follow specific experiment
```bash
tail -f experiments/ternary_dro_comparison/cebab_joint/training.log
```

## Running Individual Experiments

```bash
# Single experiment
./run_single_experiment.sh <dataset> <dro_mode> [epochs]

# Examples:
./run_single_experiment.sh cebab joint 50
./run_single_experiment.sh goemotions post_hoc 50
./run_single_experiment.sh cebab fixed_eps 50
```

## Running All Experiments

```bash
./run_all_dro_modes.sh
```

This will run all 6 experiments sequentially.

## DRO Modes Explained

### 1. post_hoc
- No DRO during training (λ_dro=0, β=0)
- Credal set applied post-hoc to trained model
- Tests the benefit of DRO vs standard training

### 2. fixed_eps
- Fixed epsilon value (ε=0.1)
- Constant ambiguity set size during training
- Tests the effect of fixed uncertainty quantification

### 3. joint
- Full DRO with learned epsilon
- Jointly optimizes task loss and robust loss
- Default mode with adaptive uncertainty

## Expected Output

Each experiment produces:
1. **Checkpoints**: Model saved every 10 epochs + best model
2. **Plots**: `training_results_50epochs.png` (6 subplots)
3. **Eval outputs**: Per-sample epsilon, aleatoric uncertainty, predictions
4. **Metrics**: Test accuracy, epsilon statistics, loss curves

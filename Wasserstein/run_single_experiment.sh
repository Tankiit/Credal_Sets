#!/bin/bash
# Run a single ternary DRO experiment
# Usage: ./run_single_experiment.sh <dataset> <dro_mode>

DATASET=${1:-cebab}
DRO_MODE=${2:-joint}
EPOCHS=${3:-50}

echo "========================================"
echo "Running: $DATASET with DRO mode: $DRO_MODE"
echo "========================================"

cd "$(dirname "$0")"

python train_ternary_50epochs.py \
    --dataset "$DATASET" \
    --dro_mode "$DRO_MODE" \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --checkpoint_dir "experiments/ternary_dro_comparison/${DATASET}_${DRO_MODE}/checkpoints" \
    --eval_outdir "experiments/ternary_dro_comparison/${DATASET}_${DRO_MODE}/eval_outputs" \
    --save_eval

echo "========================================"
echo "Completed: $DATASET - $DRO_MODE"
echo "========================================"

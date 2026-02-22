#!/bin/bash
# Run all 6 experiments: CEBaB + GoEmotions × 3 DRO modes
# Each experiment runs for 50 epochs

set -e

cd "$(dirname "$0")"

# Configuration
EPOCHS=50
BATCH_SIZE=16
BASE_DIR="experiments/ternary_dro_comparison"

echo "========================================"
echo "TERNARY DRO MODE COMPARISON"
echo "========================================"
echo "Datasets: cebab, goemotions"
echo "DRO Modes: post_hoc, fixed_eps, joint"
echo "Epochs: $EPOCHS"
echo "Total experiments: 6"
echo ""

# Create results summary file
SUMMARY_FILE="$BASE_DIR/summary.txt"
mkdir -p "$BASE_DIR"
echo "DRO Mode Comparison Results" > "$SUMMARY_FILE"
echo "=============================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local dro_mode=$2
    local exp_dir="$BASE_DIR/${dataset}_${dro_mode}"

    mkdir -p "$exp_dir"

    echo "========================================"
    echo "Running: $dataset with DRO mode: $dro_mode"
    echo "Output dir: $exp_dir"
    echo "========================================"
    echo ""

    local log_file="$exp_dir/training.log"

    local cmd="python train_ternary_50epochs.py \
        --dataset $dataset \
        --dro_mode $dro_mode \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --checkpoint_dir $exp_dir/checkpoints \
        --eval_outdir $exp_dir/eval_outputs \
        --save_eval"

    if [ "$dro_mode" = "fixed_eps" ]; then
        cmd="$cmd --fixed_eps 0.1"
    fi

    # Run and capture output
    start_time=$(date +%s)
    eval $cmd 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Record result
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $dataset - $dro_mode (duration: ${duration}s)" >> "$SUMMARY_FILE"
        echo "  ✓ $dataset - $dro_mode completed (${duration}s)"
    else
        echo "FAILED: $dataset - $dro_mode (duration: ${duration}s)" >> "$SUMMARY_FILE"
        echo "  ✗ $dataset - $dro_mode failed (${duration}s)"
    fi

    return $exit_code
}

# Run all experiments
experiments=(
    "cebab post_hoc"
    "cebab fixed_eps"
    "cebab joint"
    "goemotions post_hoc"
    "goemotions fixed_eps"
    "goemotions joint"
)

failed=0
for exp in "${experiments[@]}"; do
    read -r dataset dro_mode <<< "$exp"
    if ! run_experiment "$dataset" "$dro_mode"; then
        failed=$((failed + 1))
    fi
    echo ""
done

# Final summary
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"
cat "$SUMMARY_FILE"
echo ""
total=${#experiments[@]}
success=$((total - failed))
echo "Completed: $success/$total experiments"

if [ $failed -eq 0 ]; then
    echo "🎉 All experiments completed successfully!"
    exit 0
else
    echo "⚠️  $failed experiment(s) failed."
    exit 1
fi

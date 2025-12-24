#!/bin/bash

# =============================================================================
# CREDENCE Head Ablation Study for HateXplain
# =============================================================================
# Runs experiments with different numbers of ensemble heads to analyze
# the effect of ensemble size on uncertainty estimation and performance.
#
# Usage: ./run_head_ablation.sh [ENCODER]
#
# Examples:
#   ./run_head_ablation.sh roberta-base
#   ./run_head_ablation.sh modernbert
#   ./run_head_ablation.sh microsoft/deberta-v3-base
# =============================================================================

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "CREDENCE Head Ablation Study for HateXplain"
    echo ""
    echo "Usage: ./run_head_ablation.sh [ENCODER]"
    echo ""
    echo "This script runs experiments with different numbers of ensemble heads"
    echo "to analyze the effect of ensemble size on uncertainty estimation."
    echo ""
    echo "Head counts tested: 3, 5, 7, 10, 15"
    echo ""
    echo "Available encoders (short names):"
    echo "  roberta, roberta-base          - RoBERTa base (2019)"
    echo "  deberta, deberta-v3-base       - DeBERTa-v3 base (2021)"
    echo "  distilbert, distilbert-base    - DistilBERT (2019)"
    echo "  modernbert, modernbert-base    - ModernBERT base (2024 - SOTA)"
    echo ""
    echo "You can also use full model names from HuggingFace."
    echo ""
    echo "Examples:"
    echo "  ./run_head_ablation.sh roberta-base"
    echo "  ./run_head_ablation.sh modernbert"
    exit 0
fi

# Default encoder
ENCODER="${1:-roberta-base}"

# Parse encoder name
case "$ENCODER" in
    "roberta"|"roberta-base")
        ENCODER_FULL="roberta-base"
        ;;
    "deberta"|"deberta-v3"|"deberta-v3-base")
        ENCODER_FULL="microsoft/deberta-v3-base"
        ;;
    "distilbert"|"distilbert-base")
        ENCODER_FULL="distilbert-base-uncased"
        ;;
    "modernbert"|"modernbert-base")
        ENCODER_FULL="answerdotai/ModernBERT-base"
        ;;
    "modernbert-large")
        ENCODER_FULL="answerdotai/ModernBERT-large"
        ;;
    *)
        ENCODER_FULL="$ENCODER"
        ;;
esac

# Head counts to test
HEAD_COUNTS=(3 5 7 10 15)

# Dataset
DATASET="hatexplain"

# Experiment settings
EPOCHS=30
BATCH_SIZE=16
LR="1e-4"
MAX_LENGTH=128
SEED=42

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ENCODER_SHORT=$(echo "$ENCODER_FULL" | sed 's/.*\///' | sed 's/-base//')
OUTPUT_BASE="./results/head_ablation_${ENCODER_SHORT}_${TIMESTAMP}"

mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/ablation_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

echo "Head Ablation Study: HateXplain"
echo "Encoder: $ENCODER_FULL"
echo "Head counts: ${HEAD_COUNTS[*]}"
echo "Output: $OUTPUT_BASE"
echo ""

log "Starting head ablation study"
log "Encoder: $ENCODER_FULL"
log "Dataset: $DATASET"
log "Head counts: ${HEAD_COUNTS[*]}"

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    log "GPU: Available ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'))"
else
    log "GPU: Not available, using CPU"
fi

RESULTS_SUMMARY=()

for N_HEADS in "${HEAD_COUNTS[@]}"; do
    echo ""
    echo "="*70
    echo "Running with $N_HEADS heads"
    echo "="*70
    
    OUTPUT_DIR="$OUTPUT_BASE/n_heads_${N_HEADS}"
    
    log "Running $DATASET with $N_HEADS heads (epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR)"
    
    START_TIME=$(date +%s)
    
    python credence.py \
        --encoder_name "$ENCODER_FULL" \
        --dataset "$DATASET" \
        --label_type "default" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --n_heads "$N_HEADS" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        2>&1 | tee "$OUTPUT_DIR.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "$N_HEADS heads: ERROR - Failed with exit code $EXIT_CODE"
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log "$N_HEADS heads completed in ${DURATION}s"
    
    RESULTS_FILE="$OUTPUT_DIR/${DATASET}_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        ACC=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['accuracy']*100:.1f}\")" 2>/dev/null || echo "N/A")
        RHO_EPI=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['disagree_error_corr']:.3f}\")" 2>/dev/null || echo "N/A")
        RHO_ALE=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['ambig_unknown_corr']:.3f}\")" 2>/dev/null || echo "N/A")
        MEAN_DISAGREE=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['mean_disagreement']:.4f}\")" 2>/dev/null || echo "N/A")
        
        RESULTS_SUMMARY+=("$N_HEADS heads: Acc=$ACC%, ρ_epi=$RHO_EPI, ρ_ale=$RHO_ALE, mean_disagree=$MEAN_DISAGREE (${DURATION}s)")
        log "$N_HEADS heads: Acc=$ACC%, ρ_epi=$RHO_EPI, ρ_ale=$RHO_ALE, mean_disagree=$MEAN_DISAGREE"
    elif [ $EXIT_CODE -ne 0 ]; then
        RESULTS_SUMMARY+=("$N_HEADS heads: Failed (exit code $EXIT_CODE)")
        log "$N_HEADS heads: ERROR - Failed to run (exit code $EXIT_CODE)"
    else
        RESULTS_SUMMARY+=("$N_HEADS heads: Results file not found")
        log "$N_HEADS heads: WARNING - Results file not found"
    fi
done

echo ""
echo "="*70
echo "Head Ablation Summary"
echo "="*70
echo "Encoder: $ENCODER_FULL"
echo "Dataset: $DATASET"
echo ""
for RESULT in "${RESULTS_SUMMARY[@]}"; do
    echo "  $RESULT"
done
echo ""
echo "Results saved to: $OUTPUT_BASE"

log "All ablation experiments completed"

# Generate summary JSON
python << EOF
import json
import os

output_base = "$OUTPUT_BASE"
encoder = "$ENCODER_FULL"
dataset = "$DATASET"
head_counts = [${HEAD_COUNTS[*]}]

summary = {
    "encoder": encoder,
    "dataset": dataset,
    "timestamp": "$TIMESTAMP",
    "head_counts": {},
}

for n_heads in head_counts:
    results_file = f"{output_base}/n_heads_{n_heads}/{dataset}_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        
        test = data.get("test_results", {})
        summary["head_counts"][n_heads] = {
            "accuracy": test.get("accuracy", 0),
            "rho_epi": test.get("disagree_error_corr", 0),
            "p_epi": test.get("disagree_error_pval", 1),
            "rho_ale": test.get("ambig_unknown_corr", 0),
            "p_ale": test.get("ambig_unknown_pval", 1),
            "disagree_ratio": test.get("disagree_ratio", 0),
            "mean_disagreement": test.get("mean_disagreement", 0),
            "mean_ambiguity": test.get("mean_ambiguity", 0),
            "mean_credal_width": test.get("mean_credal_width", 0),
        }

summary_path = f"{output_base}/ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_path}")
print("\n" + "="*70)
print(f"{'Heads':<8} {'Acc':>8} {'ρ_epi':>8} {'ρ_ale':>8} {'Disagree':>10} {'Ambiguity':>10}")
print("-"*70)
for n_heads in sorted(summary["head_counts"].keys()):
    metrics = summary["head_counts"][n_heads]
    print(f"{n_heads:<8} {metrics['accuracy']*100:>7.1f}% {metrics['rho_epi']:>8.3f} {metrics['rho_ale']:>8.3f} {metrics['mean_disagreement']:>10.4f} {metrics['mean_ambiguity']:>10.4f}")
print("="*70)
EOF


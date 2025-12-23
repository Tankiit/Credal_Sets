#!/bin/bash

set -e

ENCODER="${1:-roberta-base}"

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
    *)
        ENCODER_FULL="$ENCODER"
        ;;
esac

DATASETS=(
    "cebab"
    "hatexplain"
    "goemotions"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ENCODER_SHORT=$(echo "$ENCODER_FULL" | sed 's/.*\///' | sed 's/-base//')
OUTPUT_BASE="./results/${ENCODER_SHORT}_${TIMESTAMP}"

EPOCHS_DEFAULT=40
LR="1e-4"
BATCH_SIZE=16
N_HEADS=5
MAX_LENGTH=128
SEED=42

get_epochs() {
    case "$1" in
        "cebab")          echo 40 ;;
        "hatexplain")     echo 30 ;;
        "goemotions")     echo 25 ;;
        "civil_comments") echo 15 ;;
        "sst5")           echo 30 ;;
        *)                echo $EPOCHS_DEFAULT ;;
    esac
}

get_batch_size() {
    case "$1" in
        "goemotions")     echo 32 ;;
        "civil_comments") echo 32 ;;
        "sst5")           echo 32 ;;
        *)                echo $BATCH_SIZE ;;
    esac
}

get_label_type() {
    case "$1" in
        "cebab") echo "ternary" ;;
        *)       echo "ternary" ;;
    esac
}

mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/run_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

echo "Encoder:    $ENCODER_FULL"
echo "Datasets:   ${DATASETS[*]}"
echo "Output:     $OUTPUT_BASE"
echo ""

log "Starting CREDENCE multi-dataset run"
log "Encoder: $ENCODER_FULL"
log "Datasets: ${DATASETS[*]}"

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    log "GPU: Available ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'))"
else
    log "GPU: Not available, using CPU"
fi

RESULTS_SUMMARY=()

for DATASET in "${DATASETS[@]}"; do
    echo "Dataset: $DATASET"
    
    EPOCHS=$(get_epochs "$DATASET")
    BS=$(get_batch_size "$DATASET")
    LABEL_TYPE=$(get_label_type "$DATASET")
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET"
    
    log "Running $DATASET (epochs=$EPOCHS, batch_size=$BS)"
    
    START_TIME=$(date +%s)
    
    python credence.py \
        --encoder_name "$ENCODER_FULL" \
        --dataset "$DATASET" \
        --label_type "$LABEL_TYPE" \
        --epochs "$EPOCHS" \
        --batch_size "$BS" \
        --lr "$LR" \
        --n_heads "$N_HEADS" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        2>&1 | tee "$OUTPUT_DIR.log"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log "$DATASET completed in ${DURATION}s"
    
    RESULTS_FILE="$OUTPUT_DIR/${DATASET}_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        ACC=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['accuracy']*100:.1f}\")" 2>/dev/null || echo "N/A")
        RHO_EPI=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['disagree_error_corr']:.3f}\")" 2>/dev/null || echo "N/A")
        RHO_ALE=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(f\"{d['test_results']['ambig_unknown_corr']:.3f}\")" 2>/dev/null || echo "N/A")
        
        RESULTS_SUMMARY+=("$DATASET: Acc=$ACC%, ρ_epi=$RHO_EPI, ρ_ale=$RHO_ALE (${DURATION}s)")
        log "$DATASET: Acc=$ACC%, ρ_epi=$RHO_EPI, ρ_ale=$RHO_ALE"
    else
        RESULTS_SUMMARY+=("$DATASET: Results file not found")
        log "$DATASET: WARNING - Results file not found"
    fi
done

echo ""
echo "Encoder: $ENCODER_FULL"
echo ""
for RESULT in "${RESULTS_SUMMARY[@]}"; do
    echo "  $RESULT"
done
echo ""
echo "Results saved to: $OUTPUT_BASE"

log "All experiments completed"

python << EOF
import json
import os

output_base = "$OUTPUT_BASE"
encoder = "$ENCODER_FULL"
datasets = "${DATASETS[*]}".split()

summary = {
    "encoder": encoder,
    "timestamp": "$TIMESTAMP",
    "datasets": {},
}

for ds in datasets:
    results_file = f"{output_base}/{ds}/{ds}_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        
        test = data.get("test_results", {})
        summary["datasets"][ds] = {
            "accuracy": test.get("accuracy", 0),
            "rho_epi": test.get("disagree_error_corr", 0),
            "p_epi": test.get("disagree_error_pval", 1),
            "rho_ale": test.get("ambig_unknown_corr", 0),
            "p_ale": test.get("ambig_unknown_pval", 1),
            "disagree_ratio": test.get("disagree_ratio", 0),
            "mean_credal_width": test.get("mean_credal_width", 0),
        }

summary_path = f"{output_base}/summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_path}")
print("\n" + "="*70)
print(f"{'Dataset':<15} {'Acc':>8} {'ρ_epi':>8} {'ρ_ale':>8} {'Width':>8}")
print("-"*70)
for ds, metrics in summary["datasets"].items():
    print(f"{ds:<15} {metrics['accuracy']*100:>7.1f}% {metrics['rho_epi']:>8.3f} {metrics['rho_ale']:>8.3f} {metrics['mean_credal_width']:>8.3f}")
print("="*70)
EOF


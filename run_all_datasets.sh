#!/bin/bash

# Activate virtual environment
source ../../../../../torch-multimodal/bin/activate

# =============================================================================
# CREDENCE Multi-Dataset Runner
# =============================================================================
# Usage: ./run_all_datasets.sh [ENCODER]
#
# Available encoders (short names):
#   Encoder Models:
#   - roberta, roberta-base          (RoBERTa base, 2019)
#   - roberta-large                   (RoBERTa large, 2019)
#   - deberta, deberta-v3, deberta-v3-base  (DeBERTa-v3 base, 2021)
#   - deberta-v3-large                (DeBERTa-v3 large, 2021)
#   - distilbert, distilbert-base     (DistilBERT, 2019)
#   - modernbert, modernbert-base     (ModernBERT base, 2024 - SOTA)
#   - modernbert-large                (ModernBERT large, 2024 - SOTA)
#
#   LLM Models (auto-detected, uses frozen feature extraction by default):
#   - phi-3, phi-3-mini, phi-3.5-mini
#   - mistral, mistral-7b, mistral-instruct
#   - llama-3.1-8b, llama-3.2-1b, llama-3.2-3b
#   - qwen-0.5b, qwen-1.5b, qwen-3b, qwen-7b
#   - gemma-2b, gemma-9b
#   
#   Note: LLMs use frozen feature extraction by default (fast, memory efficient).
#   To use LoRA fine-tuning instead, add --use_lora flag (slower, more memory).
#
# You can also use full model names:
#   - answerdotai/ModernBERT-base
#   - microsoft/deberta-v3-base
#   - meta-llama/Llama-3.2-3B-Instruct
#   - etc.
#
# Examples:
#   ./run_all_datasets.sh roberta-base
#   ./run_all_datasets.sh modernbert
#   ./run_all_datasets.sh answerdotai/ModernBERT-base
# =============================================================================

# Don't exit on error - we want to continue processing other datasets
# set -e

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "CREDENCE Multi-Dataset Runner"
    echo ""
    echo "Usage: ./run_all_datasets.sh [ENCODER]"
    echo ""
    echo "Available encoders (short names):"
    echo ""
    echo "Encoder Models (frozen encoder, train heads):"
    echo "  roberta, roberta-base          - RoBERTa base (2019)"
    echo "  roberta-large                  - RoBERTa large (2019)"
    echo "  deberta, deberta-v3-base      - DeBERTa-v3 base (2021)"
    echo "  deberta-v3-large               - DeBERTa-v3 large (2021)"
    echo "  distilbert, distilbert-base    - DistilBERT (2019)"
    echo "  modernbert, modernbert-base    - ModernBERT base (2024 - SOTA)"
    echo "  modernbert-large               - ModernBERT large (2024 - SOTA)"
    echo ""
    echo "LLM Models (frozen feature extraction, auto-detected):"
    echo "  phi-3, phi-3-mini              - Phi-3 Mini (Microsoft)"
    echo "  phi-3.5, phi-3.5-mini         - Phi-3.5 Mini (Microsoft)"
    echo "  mistral, mistral-7b            - Mistral 7B"
    echo "  mistral-instruct               - Mistral 7B Instruct"
    echo "  llama-3.1-8b                  - Llama 3.1 8B"
    echo "  llama-3.1-instruct            - Llama 3.1 8B Instruct"
    echo "  llama-3.2-1b                  - Llama 3.2 1B"
    echo "  llama-3.2-3b                  - Llama 3.2 3B"
    echo "  llama-3.2-3b-instruct         - Llama 3.2 3B Instruct"
    echo "  qwen-0.5b, qwen2.5-0.5b       - Qwen 2.5 0.5B"
    echo "  qwen-1.5b, qwen2.5-1.5b       - Qwen 2.5 1.5B"
    echo "  qwen-3b, qwen2.5-3b           - Qwen 2.5 3B"
    echo "  qwen-7b, qwen2.5-7b           - Qwen 2.5 7B"
    echo "  gemma-2b, gemma-2-2b          - Gemma 2 2B"
    echo "  gemma-9b, gemma-2-9b          - Gemma 2 9B"
    echo ""
    echo "  Note: LLMs use frozen feature extraction by default (fast, memory efficient)."
    echo "        To use LoRA fine-tuning instead, add --use_lora flag (slower, more memory)."
    echo ""
    echo "You can also use full model names from HuggingFace:"
    echo "  answerdotai/ModernBERT-base"
    echo "  microsoft/deberta-v3-base"
    echo "  etc."
    echo ""
    echo "Examples:"
    echo "  ./run_all_datasets.sh roberta-base"
    echo "  ./run_all_datasets.sh modernbert"
    echo "  ./run_all_datasets.sh answerdotai/ModernBERT-base"
    exit 0
fi

ENCODER="${1:-roberta-base}"

# Check for profiler flag (optional arguments after encoder name)
ENABLE_PROFILER=false
PROFILER_WARMUP=1
PROFILER_ACTIVE=3
PROFILER_REPEAT=1
PROFILER_OUTPUT_DIR=""

# Parse optional profiler flags (skip encoder argument)
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable_profiler)
            ENABLE_PROFILER=true
            shift
            ;;
        --profiler_warmup)
            PROFILER_WARMUP="$2"
            shift 2
            ;;
        --profiler_active)
            PROFILER_ACTIVE="$2"
            shift 2
            ;;
        --profiler_repeat)
            PROFILER_REPEAT="$2"
            shift 2
            ;;
        --profiler_output_dir)
            PROFILER_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            # Unknown argument, ignore or warn
            echo "Warning: Unknown argument: $1 (ignoring)"
            shift
            ;;
    esac
done

case "$ENCODER" in
    "roberta"|"roberta-base")
        ENCODER_FULL="roberta-base"
        ;;
    "roberta-large")
        ENCODER_FULL="roberta-large"
        ;;
    "deberta"|"deberta-v3"|"deberta-v3-base")
        ENCODER_FULL="microsoft/deberta-v3-base"
        ;;
    "deberta-v3-large")
        ENCODER_FULL="microsoft/deberta-v3-large"
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
    # LLM Models (use frozen feature extraction by default)
    "phi-3"|"phi-3-mini")
        ENCODER_FULL="microsoft/phi-3-mini-4k-instruct"
        IS_LLM=true
        ;;
    "phi-3.5"|"phi-3.5-mini")
        ENCODER_FULL="microsoft/Phi-3.5-mini-instruct"
        IS_LLM=true
        ;;
    "mistral"|"mistral-7b")
        ENCODER_FULL="mistralai/Mistral-7B-v0.1"
        IS_LLM=true
        ;;
    "mistral-instruct")
        ENCODER_FULL="mistralai/Mistral-7B-Instruct-v0.3"
        IS_LLM=true
        ;;
    "llama-3.1"|"llama-3.1-8b")
        ENCODER_FULL="meta-llama/Llama-3.1-8B"
        IS_LLM=true
        ;;
    "llama-3.1-instruct")
        ENCODER_FULL="meta-llama/Llama-3.1-8B-Instruct"
        IS_LLM=true
        ;;
    "llama-3.2-1b")
        ENCODER_FULL="meta-llama/Llama-3.2-1B"
        IS_LLM=true
        ;;
    "llama-3.2-3b")
        ENCODER_FULL="meta-llama/Llama-3.2-3B"
        IS_LLM=true
        ;;
    "llama-3.2-3b-instruct")
        ENCODER_FULL="meta-llama/Llama-3.2-3B-Instruct"
        IS_LLM=true
        ;;
    "qwen-0.5b"|"qwen2.5-0.5b")
        ENCODER_FULL="Qwen/Qwen2.5-0.5B"
        IS_LLM=true
        ;;
    "qwen-1.5b"|"qwen2.5-1.5b")
        ENCODER_FULL="Qwen/Qwen2.5-1.5B"
        IS_LLM=true
        ;;
    "qwen-3b"|"qwen2.5-3b")
        ENCODER_FULL="Qwen/Qwen2.5-3B"
        IS_LLM=true
        ;;
    "qwen-7b"|"qwen2.5-7b")
        ENCODER_FULL="Qwen/Qwen2.5-7B"
        IS_LLM=true
        ;;
    "gemma-2b"|"gemma-2-2b")
        ENCODER_FULL="google/gemma-2-2b"
        IS_LLM=true
        ;;
    "gemma-9b"|"gemma-2-9b")
        ENCODER_FULL="google/gemma-2-9b"
        IS_LLM=true
        ;;
    *)
        # If not a known alias, check if it's an LLM by model name pattern
        # Use as-is (allows full model names)
        ENCODER_FULL="$ENCODER"
        # Auto-detect LLM models by checking if name contains LLM indicators
        if [[ "$ENCODER" == *"phi"* ]] || \
           [[ "$ENCODER" == *"mistral"* ]] || \
           [[ "$ENCODER" == *"llama"* ]] || \
           [[ "$ENCODER" == *"qwen"* ]] || \
           [[ "$ENCODER" == *"gemma"* ]] || \
           [[ "$ENCODER" == *"meta-llama"* ]] || \
           [[ "$ENCODER" == *"microsoft/phi"* ]] || \
           [[ "$ENCODER" == *"mistralai"* ]] || \
           [[ "$ENCODER" == *"Qwen"* ]] || \
           [[ "$ENCODER" == *"google/gemma"* ]]; then
            IS_LLM=true
        fi
        ;;
esac

# Initialize IS_LLM if not set
IS_LLM=${IS_LLM:-false}

DATASETS=(
    "cebab"
    "hatexplain"
    "goemotions"
    "civil_comments"
    "tid8"
    "chaosnli"
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
        "tid8")           echo 10 ;;
        "chaosnli")       echo 10 ;;
        *)                echo $EPOCHS_DEFAULT ;;
    esac
}

get_batch_size() {
    case "$1" in
        "goemotions")     echo 32 ;;
        "civil_comments") echo 32 ;;
        "tid8")           echo 32 ;;
        "chaosnli")       echo 32 ;;
        *)                echo $BATCH_SIZE ;;
    esac
}

get_label_type() {
    case "$1" in
        "cebab") echo "ternary" ;;
        *)       echo "default" ;;
    esac
}

get_lr() {
    case "$1" in
        "goemotions")     echo "5e-5" ;;
        "chaosnli")       echo "2e-5" ;;
        "tid8")           echo "2e-5" ;;
        *)                echo "$LR" ;;
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
if [ "$IS_LLM" = true ]; then
    log "LLM model detected - using frozen feature extraction (default)"
fi
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
    DATASET_LR=$(get_lr "$DATASET")
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET"
    
    log "Running $DATASET (epochs=$EPOCHS, batch_size=$BS, lr=$DATASET_LR)"
    
    START_TIME=$(date +%s)
    
    # Build command - LLMs use frozen feature extraction by default
    # (freeze_llm=True is the default in ExperimentConfig)
    CMD="python credence.py \
        --encoder_name \"$ENCODER_FULL\" \
        --dataset \"$DATASET\" \
        --label_type \"$LABEL_TYPE\" \
        --epochs \"$EPOCHS\" \
        --batch_size \"$BS\" \
        --lr \"$DATASET_LR\" \
        --n_heads \"$N_HEADS\" \
        --max_length \"$MAX_LENGTH\" \
        --output_dir \"$OUTPUT_DIR\" \
        --seed \"$SEED\""
    
    # Note: LLMs use frozen feature extraction by default (freeze_llm=True)
    # To use LoRA fine-tuning instead, add: --use_lora (and ensure --freeze_llm is not set)
    
    # Add profiler flags if enabled
    if [ "$ENABLE_PROFILER" = true ]; then
        CMD="$CMD --enable_profiler"
        CMD="$CMD --profiler_warmup \"$PROFILER_WARMUP\""
        CMD="$CMD --profiler_active \"$PROFILER_ACTIVE\""
        CMD="$CMD --profiler_repeat \"$PROFILER_REPEAT\""
        if [ -n "$PROFILER_OUTPUT_DIR" ]; then
            CMD="$CMD --profiler_output_dir \"$PROFILER_OUTPUT_DIR\""
        fi
    fi
    
    eval $CMD 2>&1 | tee "$OUTPUT_DIR.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "$DATASET: ERROR - Failed with exit code $EXIT_CODE"
    fi
    
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
    elif [ $EXIT_CODE -ne 0 ]; then
        RESULTS_SUMMARY+=("$DATASET: Failed (exit code $EXIT_CODE)")
        log "$DATASET: ERROR - Failed to run (exit code $EXIT_CODE)"
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


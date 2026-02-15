#!/bin/bash

# =============================================================================
# Variational Credal CBM - Multi-Dataset Runner
# =============================================================================
# Usage: ./run_all_datasets.sh [ENCODER] [DATASET] [OPTIONS]
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
#
# Available datasets:
#   - amiqa  (Multi-aspect sentiment analysis)
#   - cebab  (Restaurant review dataset)
#
# You can also use full model names:
#   - answerdotai/ModernBERT-base
#   - microsoft/deberta-v3-base
#   - meta-llama/Llama-3.2-3B-Instruct
#   - etc.
#
# Examples:
#   # Run with RoBERTa on all datasets
#   ./run_all_datasets.sh roberta
#
#   # Run with ModernBERT on specific dataset
#   ./run_all_datasets.sh modernbert amiqa
#
#   # Run with full model name
#   ./run_all_datasets.sh answerdotai/ModernBERT-base cebab
#
#   # Run with experiments
#   ./run_all_datasets.sh deberta cebab --run-intervention --run-ablation
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENCODER=${1:-"distilbert-base-uncased"}
DATASET=${2:-"all"}
EXTRA_ARGS="${@:3}"  # Any additional arguments

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

# Function to run experiment
run_experiment() {
    local encoder=$1
    local dataset=$2

    print_header "Running: Encoder=${encoder}, Dataset=${dataset}"

    # Create experiment name
    encoder_safe=$(echo "$encoder" | tr '/' '-')

    # Set save directory
    save_dir="./experiments/${dataset}/${encoder_safe}"

    # Run training
    print_info "Starting training..."
    python main.py \
        --dataset "$dataset" \
        --encoder "$encoder" \
        --save-dir "$save_dir" \
        --epochs 10 \
        --batch-size 16 \
        --lr 2e-5 \
        $EXTRA_ARGS

    if [ $? -eq 0 ]; then
        print_success "Experiment completed: ${dataset} with ${encoder}"
        print_info "Results saved to: ${save_dir}"
    else
        print_error "Experiment failed: ${dataset} with ${encoder}"
        return 1
    fi

    echo ""
}

# Main execution
print_header "Variational Credal CBM - Multi-Dataset Experiments"

print_info "Encoder: ${ENCODER}"
print_info "Dataset: ${DATASET}"
print_info "Extra args: ${EXTRA_ARGS}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found in current directory"
    exit 1
fi

# Run experiments
if [ "$DATASET" == "all" ]; then
    print_info "Running experiments on all datasets..."
    echo ""

    for dataset in "amiqa" "cebab"; do
        run_experiment "$ENCODER" "$dataset"
        if [ $? -ne 0 ]; then
            print_warning "Skipping remaining experiments due to error"
            exit 1
        fi
    done
else
    run_experiment "$ENCODER" "$DATASET"
fi

print_header "All Experiments Complete!"
print_success "Results are saved in ./experiments/ directory"

echo ""
print_info "To view results, check:"
echo "  - ./experiments/<dataset>/<encoder>/config.json"
echo "  - ./experiments/<dataset>/<encoder>/best_model.pt"
echo "  - ./experiments/<dataset>/<encoder>/final_model.pt"
echo ""

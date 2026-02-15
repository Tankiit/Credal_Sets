#!/bin/bash

# Quick Encoder Benchmark Example
# ================================
# This script demonstrates how to run a quick encoder comparison
#
# Usage:
#   bash run_quick_benchmark.sh

echo "=========================================="
echo "Quick Encoder Benchmark on HateXplain"
echo "=========================================="
echo ""
echo "This will benchmark 3 encoders with 1 epoch each:"
echo "  1. distilbert (baseline)"
echo "  2. roberta (classic)"
echo "  3. modernbert (SOTA)"
echo ""
echo "Estimated time: ~10 minutes"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Run the benchmark
python run_encoder_benchmark.py \
    --dataset hatexplain \
    --encoders distilbert roberta modernbert \
    --num-epochs 1 \
    --results-dir ./results

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/hatexplain_encoder_benchmark_latest.json"
echo "  - results/hatexplain_encoder_benchmark_latest.csv"
echo ""
echo "View results with:"
echo "  cat results/hatexplain_encoder_benchmark_latest.csv"
echo ""

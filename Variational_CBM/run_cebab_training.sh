#!/bin/bash

# Memory-optimized training script for CEBaB
# This script runs the training with proper memory management

echo "=================================="
echo "CEBaB Training Script"
echo "=================================="

# Set PyTorch memory fraction
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with memory optimization
python main_train_cebab.py 2>&1 | tee training_log.txt

echo ""
echo "Training complete! Check training_log.txt for details."

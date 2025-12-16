.PHONY: help train test clean install

# Default values
DATASET ?= cebab
BATCH_SIZE ?= 16
MAX_EPOCHS ?= 10
OUTPUT_DIR ?= ./outputs
ACCELERATOR ?= auto
DEVICES ?= 1
PRECISION ?= 16-mixed

# Model config defaults
ENCODER_NAME ?= distilbert-base-uncased
VARIATIONAL_FAMILY ?= mean_field
NUM_MC_SAMPLES ?= 10
LEARNING_RATE ?= 2e-5
KL_WEIGHT ?= 1e-4
CONCEPT_WEIGHT ?= 0.5
ALEATORIC_WEIGHT ?= 0.1

# Scheduler defaults
USE_LR_SCHEDULER ?= 1
LR_SCHEDULER_FACTOR ?= 0.5
LR_SCHEDULER_PATIENCE ?= 3
LR_SCHEDULER_MIN_LR ?= 1e-7
LR_SCHEDULER_MODE ?= max
EARLY_STOPPING_PATIENCE ?= 3

help:
	@echo "Variational Credal CBM - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make train                    # Train with default settings"
	@echo "  make train DATASET=sst2       # Train on SST-2 dataset"
	@echo "  make train BATCH_SIZE=32      # Train with batch size 32"
	@echo "  make test                     # Test with best checkpoint"
	@echo ""
	@echo "Dataset options:"
	@echo "  DATASET=cebab|hatexplain|goemotions|civil_comments|sst2|ag_news"
	@echo ""
	@echo "Training options:"
	@echo "  BATCH_SIZE=<int>              # Batch size (default: 16)"
	@echo "  MAX_EPOCHS=<int>              # Max epochs (default: 10)"
	@echo "  OUTPUT_DIR=<path>             # Output directory (default: ./outputs)"
	@echo "  ACCELERATOR=auto|gpu|cpu|mps  # Accelerator (default: auto)"
	@echo "  DEVICES=<int>                 # Number of devices (default: 1)"
	@echo "  PRECISION=32|16-mixed|bf16-mixed  # Precision (default: 16-mixed)"
	@echo ""
	@echo "Model config options:"
	@echo "  ENCODER_NAME=<str>            # Encoder name (default: distilbert-base-uncased)"
	@echo "  VARIATIONAL_FAMILY=mean_field|low_rank  # Variational family (default: mean_field)"
	@echo "  NUM_MC_SAMPLES=<int>           # MC samples (default: 10)"
	@echo "  LEARNING_RATE=<float>         # Learning rate (default: 2e-5)"
	@echo "  KL_WEIGHT=<float>             # KL weight (default: 1e-4)"
	@echo "  CONCEPT_WEIGHT=<float>        # Concept weight (default: 0.5)"
	@echo "  ALEATORIC_WEIGHT=<float>      # Aleatoric weight (default: 0.1)"
	@echo ""
	@echo "Scheduler options:"
	@echo "  USE_LR_SCHEDULER=0|1           # Use ReduceLROnPlateau (default: 1)"
	@echo "  LR_SCHEDULER_FACTOR=<float>   # LR reduction factor (default: 0.5)"
	@echo "  LR_SCHEDULER_PATIENCE=<int>   # Epochs to wait (default: 3)"
	@echo "  LR_SCHEDULER_MIN_LR=<float>   # Minimum LR (default: 1e-7)"
	@echo "  LR_SCHEDULER_MODE=max|min     # Monitor mode (default: max)"
	@echo "  EARLY_STOPPING_PATIENCE=<int>  # Early stopping patience (default: 3, -1 to disable)"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATASET=cebab BATCH_SIZE=32 MAX_EPOCHS=20"
	@echo "  make train DATASET=sst2 ENCODER_NAME=bert-base-uncased"
	@echo "  make train DATASET=ag_news VARIATIONAL_FAMILY=low_rank NUM_MC_SAMPLES=20"
	@echo "  make test OUTPUT_DIR=./outputs"

train:
	@echo "Training Variational Credal CBM..."
	@echo "Dataset: $(DATASET)"
	@echo "Batch size: $(BATCH_SIZE)"
	@echo "Max epochs: $(MAX_EPOCHS)"
	@echo ""
	@if [ "$(USE_LR_SCHEDULER)" = "1" ]; then \
		python main.py \
			--dataset $(DATASET) \
			--batch_size $(BATCH_SIZE) \
			--max_epochs $(MAX_EPOCHS) \
			--output_dir $(OUTPUT_DIR) \
			--accelerator $(ACCELERATOR) \
			--devices $(DEVICES) \
			--precision $(PRECISION) \
			--encoder_name $(ENCODER_NAME) \
			--variational_family $(VARIATIONAL_FAMILY) \
			--num_mc_samples $(NUM_MC_SAMPLES) \
			--learning_rate $(LEARNING_RATE) \
			--kl_weight $(KL_WEIGHT) \
			--concept_weight $(CONCEPT_WEIGHT) \
			--aleatoric_weight $(ALEATORIC_WEIGHT) \
			--tokenizer_name $(ENCODER_NAME) \
			--use_lr_scheduler \
			--lr_scheduler_factor $(LR_SCHEDULER_FACTOR) \
			--lr_scheduler_patience $(LR_SCHEDULER_PATIENCE) \
			--lr_scheduler_min_lr $(LR_SCHEDULER_MIN_LR) \
			--lr_scheduler_mode $(LR_SCHEDULER_MODE) \
			--early_stopping_patience $(EARLY_STOPPING_PATIENCE); \
	else \
		python main.py \
			--dataset $(DATASET) \
			--batch_size $(BATCH_SIZE) \
			--max_epochs $(MAX_EPOCHS) \
			--output_dir $(OUTPUT_DIR) \
			--accelerator $(ACCELERATOR) \
			--devices $(DEVICES) \
			--precision $(PRECISION) \
			--encoder_name $(ENCODER_NAME) \
			--variational_family $(VARIATIONAL_FAMILY) \
			--num_mc_samples $(NUM_MC_SAMPLES) \
			--learning_rate $(LEARNING_RATE) \
			--kl_weight $(KL_WEIGHT) \
			--concept_weight $(CONCEPT_WEIGHT) \
			--aleatoric_weight $(ALEATORIC_WEIGHT) \
			--tokenizer_name $(ENCODER_NAME) \
			--no_lr_scheduler \
			--lr_scheduler_factor $(LR_SCHEDULER_FACTOR) \
			--lr_scheduler_patience $(LR_SCHEDULER_PATIENCE) \
			--lr_scheduler_min_lr $(LR_SCHEDULER_MIN_LR) \
			--lr_scheduler_mode $(LR_SCHEDULER_MODE) \
			--early_stopping_patience $(EARLY_STOPPING_PATIENCE); \
	fi

test:
	@echo "Testing Variational Credal CBM..."
	@echo "Output dir: $(OUTPUT_DIR)"
	@echo ""
	python main.py \
		--test_only \
		--output_dir $(OUTPUT_DIR) \
		--dataset $(DATASET)

clean:
	@echo "Cleaning output directory..."
	rm -rf $(OUTPUT_DIR)
	@echo "Done!"

install:
	@echo "Installing dependencies..."
	pip install torch torchvision torchaudio
	pip install pytorch-lightning
	pip install transformers
	pip install datasets
	pip install torchmetrics
	pip install scipy
	@echo "Done!"

# Quick training examples
train-cebab:
	make train DATASET=cebab

train-sst2:
	make train DATASET=sst2

train-ag-news:
	make train DATASET=ag_news

# Advanced training examples
train-bert:
	make train ENCODER_NAME=bert-base-uncased

train-lowrank:
	make train VARIATIONAL_FAMILY=low_rank NUM_MC_SAMPLES=20

train-large-batch:
	make train BATCH_SIZE=32 MAX_EPOCHS=20


# Using Custom Output Directories for Experiments

## Overview

You can now organize your cached features by experiment using the `--output_dir` and `--experiment_name` options. This helps keep different experimental setups separate.

## Basic Usage

### Default Behavior
```bash
# Saves to ./latent_cache/
python dro_expts.py --dataset cebab --rank 128
```

Creates:
```
latent_cache/
├── cebab_low_rank_rank128_svd_train_latents.npy
├── cebab_low_rank_rank128_svd_train_labels.npy
├── cebab_low_rank_rank128_svd_val_latents.npy
├── cebab_low_rank_rank128_svd_val_labels.npy
├── cebab_low_rank_rank128_svd_test_latents.npy
├── cebab_low_rank_rank128_svd_test_labels.npy
└── cebab_low_rank_rank128_svd_metadata.json
```

### Custom Output Directory
```bash
# Save to a specific directory
python dro_expts.py --dataset cebab --rank 128 --output_dir ./my_cache
```

Creates:
```
my_cache/
├── cebab_low_rank_rank128_svd_*.npy
└── cebab_low_rank_rank128_svd_metadata.json
```

### Using Experiment Names (Recommended!)
```bash
# Create experiment subdirectory
python dro_expts.py --dataset cebab --rank 128 --experiment_name dro_baseline
```

Creates:
```
latent_cache/
└── dro_baseline/                    # ← Experiment subdirectory
    ├── cebab_low_rank_rank128_svd_train_latents.npy
    ├── cebab_low_rank_rank128_svd_train_labels.npy
    ├── cebab_low_rank_rank128_svd_val_latents.npy
    ├── cebab_low_rank_rank128_svd_val_labels.npy
    ├── cebab_low_rank_rank128_svd_test_latents.npy
    ├── cebab_low_rank_rank128_svd_test_labels.npy
    └── cebab_low_rank_rank128_svd_metadata.json
```

## Practical Examples

### Example 1: Organize by Experiment Type

```bash
# Strong DRO experiment
python dro_expts.py --dataset cebab --rank 128 \
    --experiment_name dro_strong

# Weak DRO experiment
python dro_expts.py --dataset cebab --rank 128 \
    --experiment_name dro_weak

# No width penalty experiment
python dro_expts.py --dataset cebab --rank 128 \
    --experiment_name dro_no_width
```

Directory structure:
```
latent_cache/
├── dro_strong/
│   └── cebab_low_rank_rank128_svd_*.npy
├── dro_weak/
│   └── cebab_low_rank_rank128_svd_*.npy
└── dro_no_width/
    └── cebab_low_rank_rank128_svd_*.npy
```

### Example 2: Organize by Rank Comparison

```bash
# Compare different ranks
python dro_expts.py --dataset cebab --rank 32 --experiment_name rank32
python dro_expts.py --dataset cebab --rank 64 --experiment_name rank64
python dro_expts.py --dataset cebab --rank 128 --experiment_name rank128
python dro_expts.py --dataset cebab --rank 256 --experiment_name rank256
```

Directory structure:
```
latent_cache/
├── rank32/
├── rank64/
├── rank128/
└── rank256/
```

### Example 3: Organize by Extraction Method

```bash
# Compare extraction methods
python dro_expts.py --dataset cebab --experiment_name cls_baseline
python dro_expts.py --dataset cebab --rank 128 --experiment_name svd128
python dro_expts.py --dataset cebab --rank 128 --pca --experiment_name pca128
python dro_expts.py --dataset cebab --method multi_layer --experiment_name multilayer
```

### Example 4: Completely Custom Organization

```bash
# Custom base directory + experiment name
python dro_expts.py --dataset cebab --rank 128 \
    --output_dir ./experiments/2025-02-05 \
    --experiment_name dro_baseline

python dro_expts.py --dataset cebab --rank 128 \
    --output_dir ./experiments/2025-02-05 \
    --experiment_name dro_strong
```

Directory structure:
```
experiments/2025-02-05/
├── dro_baseline/
│   └── cebab_low_rank_rank128_svd_*.npy
└── dro_strong/
    └── cebab_low_rank_rank128_svd_*.npy
```

## Loading from Custom Directories

When loading cached features, specify the same `output_dir`:

```python
from dro_expts import create_cached_dataloaders

# Load from custom experiment directory
train_loader, val_loader, test_loader, metadata = create_cached_dataloaders(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
    output_dir="./latent_cache/dro_strong",  # ← Match experiment name
    batch_size=16,
)

# Or use completely custom path
train_loader, val_loader, test_loader, metadata = create_cached_dataloaders(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
    output_dir="./experiments/2025-02-05/dro_baseline",
    batch_size=16,
)
```

## Benefits of Organized Directories

### 1. **Easy Comparison**
```bash
# Extract all variants
for strength in weak strong; do
    python dro_expts.py --dataset cebab --rank 128 \
        --experiment_name dro_${strength}
done

# Train on each variant
python train.py --output_dir ./latent_cache/dro_weak
python train.py --output_dir ./latent_cache/dro_strong
```

### 2. **Clean Experiments**
```bash
# Remove old experiment
rm -rf latent_cache/old_experiment/

# Keep only current experiments
ls latent_cache/
```

### 3. **Shareable Experiments**
```bash
# Share with team
tar -czf dro_baseline.tar.gz latent_cache/dro_baseline/
scp dro_baseline.tar.gz teammate:/path/to/project/

# Teammate extracts and uses instantly!
tar -xzf dro_baseline.tar.gz
python train.py --output_dir latent_cache/dro_baseline
```

### 4. **Version Control Friendly**
```
experiments/
├── v1_baseline/
├── v2_strong_dro/
├── v3_with_concepts/
└── v4_final/
```

## Tips

### Tip 1: Descriptive Experiment Names
```bash
# Good: Descriptive
--experiment_name dro_strong_lambda0.5_steps20
--experiment_name rank_ablation_study
--experiment_name final_model_2025-02-05

# Bad: Vague
--experiment_name exp1
--experiment_name test
--experiment_name temp
```

### Tip 2: Include Key Parameters in Name
```bash
# Format: {experiment}_{key_param}_{value}
--experiment_name dro_rank128
--experiment_name dro_lr1e4
--experiment_name dro_steps20
```

### Tip 3: Date-Based Organization
```bash
# Organize by date
OUTPUT_DIR=./experiments/$(date +%Y-%m-%d)
python dro_expts.py --dataset cebab --rank 128 \
    --output_dir $OUTPUT_DIR \
    --experiment_name dro_baseline
```

Creates:
```
experiments/2025-02-05/dro_baseline/
```

### Tip 4: Quick Experiment Switching
```bash
# Set environment variable
export EXP_DIR="./latent_cache/dro_baseline"

# Use in scripts
python train.py --output_dir $EXP_DIR
python evaluate.py --output_dir $EXP_DIR
```

## Summary

- ✅ **`--output_dir`**: Specify base cache directory
- ✅ **`--experiment_name`**: Create subdirectory for organization
- ✅ **Easy to organize**: By experiment, rank, method, date, etc.
- ✅ **Clean management**: Delete/move entire experiment folders
- ✅ **Shareable**: Easy to pack and share with team

**Recommended workflow:**
```bash
python dro_expts.py --dataset cebab --rank 128 \
    --experiment_name dro_baseline_v1
```

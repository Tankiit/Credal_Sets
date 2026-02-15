# DRO Experiments - Latent Extraction Guide

## Overview

`dro_expts.py` extracts and caches latent embeddings from datasets for Distributionally Robust Optimization (DRO) experiments. It supports multiple extraction methods and datasets with automatic caching.

## Quick Start

### Basic Usage

```bash
# Extract CEBaB with default settings (SVD-128)
python dro_expts.py --dataset cebab

# Extract with lower rank
python dro_expts.py --dataset sst2 --rank 32

# Extract all datasets
python dro_expts.py --dataset all
```

### Advanced Usage

```bash
# Use PCA instead of SVD
python dro_expts.py --dataset cebab --pca --rank 64

# Multi-layer aggregation
python dro_expts.py --dataset cebab --method multi_layer --layers 0 3 5 -1

# Token-level extraction
python dro_expts.py --dataset sst2 --method token_level
```

## Features

### 1. **Multiple Extraction Methods**

| Method | Description | Output Dim | Use Case |
|--------|-------------|------------|----------|
| **cls** | CLS token only | 768 | Baseline |
| **low_rank** | SVD/PCA compression | 32-256 | Fast training |
| **multi_layer** | Layer aggregation | 3072 | Rich features |
| **token_level** | All tokens | 98304 | Analysis |

### 2. **Automatic Caching**

```bash
# First run: extracts and caches
python dro_expts.py --dataset cebab --rank 64

# Subsequent runs: loads from cache (instant!)
# (Automatically detects existing cache)
```

Cache files are saved to `./latent_cache/`:
```
latent_cache/
├── cebab_low_rank_rank64_svd_train_latents.npy
├── cebab_low_rank_rank64_svd_train_labels.npy
├── cebab_low_rank_rank64_svd_val_latents.npy
├── cebab_low_rank_rank64_svd_val_labels.npy
├── cebab_low_rank_rank64_svd_test_latents.npy
├── cebab_low_rank_rank64_svd_test_labels.npy
└── cebab_low_rank_rank64_svd_metadata.json
```

### 3. **Dataset-Specific Defaults**

Each dataset has optimized default settings:

```python
DEFAULT_EXTRACTION_CONFIGS = {
    "cebab": {"rank": 128, "svd": True},
    "hatexplain": {"rank": 128, "svd": True},
    "sst2": {"rank": 64, "svd": True},
    "imdb": {"rank": 64, "svd": True},
    # ... etc
}
```

### 4. **Batch Processing**

Extract multiple datasets in one command:

```bash
# Extract specific datasets
python dro_expts.py --dataset cebab hatexplain sst2

# Extract ALL datasets
python dro_expts.py --dataset all
```

## Command-Line Arguments

### Required

- `--dataset`: Dataset name (or "all")

### Optional

#### Extraction Method
- `--method`: `cls`, `low_rank`, `multi_layer`, `token_level`
- `--rank`: Target rank for low-rank (default: dataset-specific)
- `--svd`: Use SVD (default)
- `--pca`: Use PCA instead of SVD
- `--layer`: Layer to extract (`first`, `last`, `mean`)
- `--aggregate`: Aggregation method (`cls`, `mean`, `max`)

#### Multi-Layer Options
- `--layers`: Layers to extract (e.g., `0 3 5 -1`)
- `--aggregation`: `concat`, `mean`, `sum`

#### Encoder
- `--encoder`: Model name (default: `distilbert-base-uncased`)
- `--device`: Device (default: auto-detect)

#### Output
- `--output_dir`: Cache directory (default: `./latent_cache`)
- `--batch_size`: Batch size for extraction (default: 32)

## Examples

### Example 1: CEBaB with Custom Rank

```bash
python dro_expts.py \
    --dataset cebab \
    --rank 64 \
    --batch_size 16
```

Output:
```
cebab_low_rank_rank64_svd_train_latents.npy      # 9848 x 64
cebab_low_rank_rank64_svd_train_labels.npy      # 9848
cebab_low_rank_rank64_svd_val_latents.npy       # 1673 x 64
cebab_low_rank_rank64_svd_val_labels.npy        # 1673
cebab_low_rank_rank64_svd_test_latents.npy      # 1689 x 64
cebab_low_rank_rank64_svd_test_labels.npy       # 1689
```

### Example 2: SST-2 with PCA

```bash
python dro_expts.py \
    --dataset sst2 \
    --pca \
    --rank 32 \
    --aggregate cls
```

### Example 3: Multi-Layer for HateXplain

```bash
python dro_expts.py \
    --dataset hatexplain \
    --method multi_layer \
    --layers 0 3 5 -1 \
    --aggregation concat
```

Output dimension: `batch_size x 3072` (4 layers × 768 dims)

## Using Cached Features

### Loading Cached Features

```python
from dro_expts import create_cached_dataloaders, load_latents

# Method 1: Create PyTorch DataLoaders directly
train_loader, val_loader, test_loader, metadata = create_cached_dataloaders(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
    output_dir="./latent_cache",
    batch_size=32,
)

# Use in training loop
for batch in train_loader:
    features = batch['features']  # [batch, 128] - pre-extracted!
    labels = batch['labels']
    # No encoder forward pass needed!
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
```

### Manual Loading

```python
from dro_expts import load_latents

# Load cached latents
results, metadata = load_latents(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
    output_dir="./latent_cache",
)

train_latents = results['train_latents']  # [N, 128]
train_labels = results['train_labels']    # [N]
```

## Performance

### Extraction Time (CPU, SST-2)

| Method | Rank | Time | Memory |
|--------|------|------|--------|
| **cls** | 768 | ~5 min | 100% |
| **low_rank** | 64 | ~6 min | 8% |
| **low_rank** | 128 | ~7 min | 17% |
| **multi_layer** | 3072 | ~15 min | 400% |

### Training Speedup (Using Cached Features)

- **Without cache**: Encoder forward pass every batch
- **With cache**: No encoder, 10-50x faster!

## Troubleshooting

### Issue: Out of Memory

```bash
# Solution 1: Reduce batch size
python dro_expts.py --dataset cebab --batch_size 8

# Solution 2: Reduce rank
python dro_expts.py --dataset cebab --rank 32

# Solution 3: Use CPU
python dro_expts.py --dataset cebab --device cpu
```

### Issue: Slow Extraction

```bash
# Extract once, reuse many times
python dro_expts.py --dataset cebab --rank 64

# Cached features load instantly in subsequent runs
```

### Issue: Wrong Dimensions

```bash
# Check metadata
cat latent_cache/cebab_low_rank_rank128_svd_metadata.json

# Ensure rank matches what you expect
python dro_expts.py --dataset cebab --rank 64  # Not 128!
```

## Integration with Training

### Option 1: Online Extraction (Slow)

```python
# Extract during training (slow)
for batch in dataloader:
    hidden_states = encoder(batch['input_ids'], batch['attention_mask'])
    outputs = model(hidden_states)  # [batch, 768]
    loss.backward()
```

### Option 2: Offline Extraction (Fast, Recommended)

```bash
# 1. Extract once
python dro_expts.py --dataset cebab --rank 128

# 2. Train on cached features (10-50x faster!)
```

```python
# Python training code
train_loader, val_loader, test_loader, metadata = create_cached_dataloaders(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
)

for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(batch['features'])  # [batch, 128]
        loss.backward()
        optimizer.step()
```

## Summary of Files

| File | Description |
|------|-------------|
| `dro_expts.py` | Main extraction script |
| `latent_cache/` | Cached embeddings |
| `extraction_summary.json` | Summary of all extractions |

## Next Steps

1. **Extract features** for your datasets:
   ```bash
   python dro_expts.py --dataset cebab --rank 128
   ```

2. **Train CREDENCE** on cached features (10-50x faster!)

3. **Experiment** with different ranks and methods

4. **Analyze** latent space quality

---

**For more details on low-rank extraction, see `LOW_RANK_SUMMARY.md` or `ENCODER_README.md`.**

# File Structure Overview

```
Wasserstein/
├── Core Implementation
│   ├── dataloader.py              # Multi-dataset loader (11 datasets)
│   ├── encoder.py                 # Frozen encoder + low-rank extraction
│   ├── config.py                  # Complete configuration system
│   └── dro_expts.py               # DRO experiments + extraction
│
├── Documentation
│   ├── ENCODER_README.md          # Encoder documentation
│   ├── LOW_RANK_SUMMARY.md        # Low-rank features overview
│   ├── QUICK_START.md             # Quick reference
│   ├── DRO_EXPTS_README.md        # DRO experiments guide
│   ├── COMPLETE_SUMMARY.md        # This file
│   └── FILE_STRUCTURE.md          # This file
│
├── Utilities
│   ├── test_low_rank_extraction.py    # 5 usage examples
│   └── analyze_latents.py             # Analysis & visualization
│
├── Cache (Generated)
│   └── latent_cache/
│       ├── sst2_low_rank_rank64_svd_*.npy    # SST-2 features
│       ├── cebab_low_rank_rank128_svd_*.npy  # CEBaB features (TODO)
│       ├── hatexplain_low_rank_*.npy          # HateXplain (TODO)
│       └── extraction_summary.json            # Summary of all extractions
│
└── Example Output
    └── example_config.json         # Example configuration
```

## File Sizes

```
Core:
  dataloader.py          600+ lines  (Multi-dataset support)
  encoder.py             607 lines   (Low-rank extraction)
  config.py              440 lines   (Configuration system)
  dro_expts.py           745 lines   (DRO extraction pipeline)

Documentation:
  ENCODER_README.md      8.8 KB      (Complete guide)
  LOW_RANK_SUMMARY.md    7.5 KB      (Feature overview)
  QUICK_START.md         3.4 KB      (Quick reference)
  DRO_EXPTS_README.md    6.2 KB      (DRO guide)
  COMPLETE_SUMMARY.md    12 KB       (Master summary)
  FILE_STRUCTURE.md      2 KB        (This file)

Utilities:
  test_low_rank_extraction.py  10 KB  (Examples)
  analyze_latents.py           12 KB  (Analysis tools)
```

## Quick Start Commands

```bash
# 1. Extract features
python dro_expts.py --dataset cebab --rank 128

# 2. Test encoder
python encoder.py

# 3. Test config
python config.py

# 4. Analyze latents
python analyze_latents.py --dataset sst2

# 5. View examples
python test_low_rank_extraction.py
```

## Key Features by File

### dataloader.py
- ✅ 11 datasets supported
- ✅ Concept labels (ternary: 0/1/2)
- ✅ Multi-annotator handling
- ✅ Verified against HuggingFace

### encoder.py
- ✅ Low-rank (SVD, PCA)
- ✅ Multi-layer aggregation
- ✅ Token-level extraction
- ✅ Cached feature loading

### config.py
- ✅ Architecture, PGD, Loss configs
- ✅ Preset configurations
- ✅ Validation & pretty-printing
- ✅ JSON save/load

### dro_expts.py
- ✅ Extract all splits at once
- ✅ Automatic caching
- ✅ Batch processing
- ✅ PyTorch DataLoader wrapper

## Dataset Support

| Dataset | Status | Concepts | Classes |
|---------|--------|----------|---------|
| CEBaB | ✅ Ready | 4 | 3 |
| HateXplain | ✅ Ready | 2 | 3 |
| Civil Comments | ✅ Ready | 6 | 2 |
| GoEmotions | ✅ Ready | 28 | 28 |
| SST-2 | ✅ Tested | 0 | 2 |
| SST-5 | ✅ Ready | 0 | 5 |
| IMDB | ✅ Ready | 0 | 2 |
| Yelp | ✅ Ready | 0 | 5 |
| ChaosNLI | ✅ Ready | 0 | 3 |
| TID-8 | ✅ Ready | 0 | 3 |

## Configuration Templates

```python
# CEBaB (default)
config = get_cebab_config()
# → 4 concepts, 3 classes, 768-dim

# Low-rank variant
config = get_low_rank_config("cebab", rank=128)
# → Compressed to 128-dim

# Multi-layer variant
config = get_multilayer_config("cebab")
# → 3072-dim (4 layers × 768)

# Strong DRO
config = get_dro_strong_config()
# → λ_rob=0.5, T=20

# Weak DRO
config = get_dro_weak_config()
# → λ_rob=0.01, T=5
```

## Extraction Methods

| Method | Command | Output Dim | Use Case |
|--------|---------|------------|----------|
| CLS | `--method cls` | 768 | Baseline |
| SVD-64 | `--rank 64` | 64 | Fast training |
| SVD-128 | `--rank 128` | 128 | Balanced |
| PCA-128 | `--pca --rank 128` | 128 | Denoised |
| Multi-layer | `--method multi_layer` | 3072 | Rich features |
| Token-level | `--method token_level` | 98304 | Analysis |

## Typical Workflow

```bash
# 1. Extract (one-time, ~10 min)
python dro_expts.py --dataset cebab --rank 128

# 2. Train (10-50x faster!)
python train.py --use_cached --rank 128

# 3. Analyze
python analyze_latents.py --dataset cebab
```

## Success Metrics

✅ **SST-2 extracted**: 67,349 samples × 64 dims
✅ **Compression**: 12x (64 vs 768 dims)
✅ **Memory**: 16.44 MB (vs 198 MB)
✅ **Time**: 11 minutes (CPU)
✅ **Ready to use**: Load and train immediately
```


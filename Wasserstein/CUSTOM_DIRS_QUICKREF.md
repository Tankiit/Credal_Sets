# Quick Reference: Custom Output Directories

## New Feature! 🎉

You can now organize cached features by experiment using `--experiment_name`

## Usage

### Default (saves to ./latent_cache/)
```bash
python dro_expts.py --dataset cebab --rank 128
```

### With Experiment Name (Recommended!)
```bash
python dro_expts.py --dataset cebab --rank 128 --experiment_name dro_baseline
```

Creates:
```
latent_cache/
└── dro_baseline/
    ├── cebab_low_rank_rank128_svd_*.npy
    └── cebab_low_rank_rank128_svd_metadata.json
```

## Examples

### Organize by Experiment
```bash
python dro_expts.py --dataset cebab --rank 128 --experiment_name dro_weak
python dro_expts.py --dataset cebab --rank 128 --experiment_name dro_strong
python dro_expts.py --dataset cebab --rank 128 --experiment_name dro_no_width
```

### Organize by Rank
```bash
python dro_expts.py --dataset cebab --rank 32 --experiment_name rank32
python dro_expts.py --dataset cebab --rank 64 --experiment_name rank64
python dro_expts.py --dataset cebab --rank 128 --experiment_name rank128
```

### Organize by Date
```bash
python dro_expts.py --dataset cebab --rank 128 \
    --output_dir ./experiments/$(date +%Y-%m-%d) \
    --experiment_name dro_baseline
```

## Loading from Custom Directory

```python
from dro_expts import create_cached_dataloaders

# Specify the experiment directory
train_loader, val_loader, test_loader, metadata = create_cached_dataloaders(
    dataset_name="cebab",
    extraction_config={"method": "low_rank", "rank": 128, "svd": True},
    output_dir="./latent_cache/dro_baseline",  # ← Your experiment dir
    batch_size=16,
)
```

## Benefits

✅ **Organized** - Keep experiments separate
✅ **Clean** - Delete old experiments easily
✅ **Shareable** - Pack and share with team
✅ **Trackable** - Know what's in each directory

## Command-Line Options

```
--output_dir       Base directory (default: ./latent_cache)
--experiment_name   Experiment subdirectory (optional)
```

---

**See CUSTOM_DIRS_GUIDE.md for detailed examples!**

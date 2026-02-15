# Quick Reference: Results Files

## Training Outputs (in `checkpoints/hybrid_credal_{dataset}/`)

| File | Format | Contents | When to Use |
|------|--------|----------|-------------|
| `epoch_XXX_metrics.json` | JSON | Per-epoch detailed metrics | Detailed epoch analysis |
| `best_model.pt` | PyTorch | Best model checkpoint | Loading best model |
| `checkpoint_epoch_N.pt` | PyTorch | Periodic checkpoints | Resuming training |
| `training_history.json` | JSON | Complete training log | Full reproducibility |
| `training_history.csv` | CSV | Epoch-by-epoch summary | Quick plotting/analysis |
| `final_results.json` | JSON | Test results + summary | Final performance |

## Key Metrics in Each File

### Per-Epoch Files (`epoch_XXX_metrics.json`)
```json
{
  "epoch": 1,
  "epoch_time": 142.8,                    // ⏱️ Seconds
  "train": {
    "loss": 1.2345,                       // 📉 Training loss
    "accuracy": 0.4567                    // 📈 Training accuracy
  },
  "val": {
    "loss": 1.2789,                       // 📉 Validation loss
    "accuracy": 0.4456,                   // 📈 Validation accuracy
    "concept_accs": {                     // 🎯 Per-concept
      "food": 0.6234,
      "service": 0.5891,
      ...
    },
    "mean_sigma_epi": 0.5234,             // 🔵 Epistemic width
    "mean_eu": -0.8923,                   // 🔵 Epistemic uncertainty
    "mean_au": 0.3456,                    // 🟠 Aleatoric uncertainty
    "rho_eu_au": -0.048,                  // ✅ Orthogonality (< 0.3)
    "rho_eu_error": 0.101,                // ✅ Calibration (> 0.2)
    "rho_ale_entropy": 0.358              // ✅ Aleatoric quality (> 0.3)
  },
  "timestamp": "2026-01-23T15:30:45"      // 🕐 When saved
}
```

### Training History (`training_history.json`)
```json
{
  "metadata": {                           // 📋 Run info
    "dataset": "CEBaB",
    "encoder": "distilbert-base-uncased",
    "device": "cuda",
    "command_line_args": {...},           // 🎛️ Exact args used
    "dataset_metadata": {                 // 📊 Dataset info
      "train_size": 8000,
      "val_size": 1000,
      "test_size": 1000,
      "num_classes": 5,
      "num_concepts": 4
    }
  },
  "data_statistics": {                    // 📦 Data loader stats
    "train_size": 8000,
    "val_size": 1000,
    "train_batches": 1000,
    "val_batches": 125,
    "batch_size": 8
  },
  "training_config": {                    // ⚙️ Hyperparameters
    "num_epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_steps": 100
  },
  "training_time": {                      // ⏱️ Timing info
    "total_seconds": 1428.5,
    "total_minutes": 23.8,
    "avg_time_per_epoch": 142.8
  },
  "best_val_accuracy": 0.5123,            // 🏆 Best performance
  "best_epoch": 8,
  "history": [...]                        // 📜 All epochs
}
```

### Final Results (`final_results.json`)
```json
{
  "test_metrics": {                       // 🧪 Test set performance
    "accuracy": 0.5012,
    "loss": 1.2987,
    "concept_accs": {...},
    "mean_eu": -0.8923,
    "mean_au": 0.3456,
    "rho_eu_au": -0.048,
    ...
  },
  "training_summary": {                   // 📊 Training overview
    "best_val_accuracy": 0.5123,
    "best_epoch": 8,
    "total_training_time": 23.8
  },
  "run_metadata": {...},                  // 📋 Run info
  "data_statistics": {...},               // 📦 Data info
  "training_config": {...}                // ⚙️ Hyperparameters
}
```

## Quick Loading Examples

### Load and Plot:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (fastest)
df = pd.read_csv('checkpoints/hybrid_credal_cebab/training_history.csv')

# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(df['epoch'], df['train_acc'], label='Train')
axes[0].plot(df['epoch'], df['val_acc'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Uncertainty
axes[1].plot(df['epoch'], df['mean_eu'], label='EU')
axes[1].plot(df['epoch'], df['mean_au'], label='AU')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Uncertainty')
axes[1].legend()

plt.tight_layout()
plt.savefig('curves.png')
```

### Check Uncertainty Quality:
```python
import pandas as pd

df = pd.read_csv('checkpoints/hybrid_credal_cebab/training_history.csv')

# Check final correlations (last epoch)
final = df.iloc[-1]
print(f"Orthogonality (ρ_EU_AU): {final['rho_eu_au']:.3f} (target: < 0.3)")
print(f"Calibration (ρ_EU_Error): {final['rho_eu_error']:.3f} (target: > 0.2)")
print(f"Aleatoric (ρ_AU_Entropy): {final['rho_au_entropy']:.3f} (target: > 0.3)")
```

### Load Best Model:
```python
import torch

checkpoint = torch.load('checkpoints/hybrid_credal_cebab/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Check which epoch was best
print(f"Best model from epoch {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['metrics']['accuracy']:.4f}")
```

### Compare Runs:
```python
import pandas as pd
import matplotlib.pyplot as plt

encoders = ['distilbert', 'roberta', 'modernbert']

for enc in encoders:
    path = f'checkpoints/hybrid_credal_cebab_{enc}/training_history.csv'
    df = pd.read_csv(path)
    plt.plot(df['epoch'], df['val_acc'], label=enc)

plt.xlabel('Epoch')
plt.ylabel('Val Accuracy')
plt.legend()
plt.savefig('comparison.png')
```

## Metric Interpretation

### Accuracy Metrics:
| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| `accuracy` | Dataset-dependent | Higher = better |
| `concept_accs[concept]` | 0.5-1.0 | Per-concept performance |

### Uncertainty Metrics:
| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| `mean_eu` | Dataset-dependent | Average epistemic uncertainty |
| `mean_au` | Dataset-dependent | Average aleatoric uncertainty |
| `rho_eu_au` | **< 0.3** | Orthogonality (lower = better) |
| `rho_eu_error` | **> 0.2** | Calibration (higher = better) |
| `rho_au_entropy` | **> 0.3** | Aleatoric quality (higher = better) |

### Quality Checks:
✅ **Good Model**: ρ_EU_AU < 0.3, ρ_EU_Error > 0.2, ρ_AU_Entropy > 0.3
⚠️ **Check**: If any correlation is outside target range

## File Locations

```
checkpoints/
├── hybrid_credal_cebab/                    # CEBaB results
│   ├── epoch_001_metrics.json
│   ├── ...
│   ├── best_model.pt
│   ├── training_history.json
│   ├── training_history.csv
│   └── final_results.json
├── hybrid_credal_hatexplain/               # HateXplain results
│   └── ...
└── hybrid_credal_goemotions/               # GoEmotions results
    └── ...
```

## Common Tasks

### Task 1: "How did training progress?"
```bash
# View CSV
cat checkpoints/hybrid_credal_cebab/training_history.csv | column -t -s,

# Or plot with Python
python -c "import pandas as pd; df = pd.read_csv('checkpoints/hybrid_credal_cebab/training_history.csv'); print(df[['epoch', 'train_acc', 'val_acc']])"
```

### Task 2: "What was the best performance?"
```bash
# Check final results
python -c "import json; r = json.load(open('checkpoints/hybrid_credal_cebab/final_results.json')); print(f\"Best val: {r['training_summary']['best_val_accuracy']:.4f} (epoch {r['training_summary']['best_epoch']})\")"
```

### Task 3: "Which concept performed worst?"
```bash
# Load final results
python -c "import json; r = json.load(open('checkpoints/hybrid_credal_cebab/final_results.json')); accs = r['test_metrics']['concept_accs']; print(f\"Worst: {min(accs, key=accs.get)} ({accs[min(accs, key=accs.get)]:.4f})\")"
```

### Task 4: "How long did training take?"
```bash
# Check training history
python -c "import json; r = json.load(open('checkpoints/hybrid_credal_cebab/training_history.json')); print(f\"Total: {r['training_time']['total_minutes']:.1f} minutes ({r['training_time']['avg_time_per_epoch']:.1f}s per epoch)\")"
```

### Task 5: "Reproduce this run"
```bash
# Get exact command
python -c "import json; r = json.load(open('checkpoints/hybrid_credal_cebab/training_history.json')); args = r['metadata']['command_line_args']; cmd = 'python main_train_hybrid_multi_dataset.py ' + ' '.join([f'--{k} {v}' if not isinstance(v, bool) else f'--{k}' for k, v in args.items() if v]); print(cmd)"
```

## Summary

✅ **Every epoch saved** as `epoch_XXX_metrics.json`
✅ **Complete history** in `training_history.json`
✅ **Quick analysis** with `training_history.csv`
✅ **Best model** in `best_model.pt`
✅ **Final results** in `final_results.json`
✅ **All metadata** for reproducibility
✅ **Time tracking** per epoch and total
✅ **Uncertainty quality** metrics tracked

Everything saved, nothing missed! 🎯

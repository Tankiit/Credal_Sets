# Quick Start: Ternary Concepts for CEBaB

## What Changed?
Added proper ternary concept support using CrossEntropyLoss instead of BCE with the {0, 0.5, 1} workaround.

## How to Use

### Option 1: Using example.ipynb (Already Configured)
```bash
jupyter notebook example.ipynb
# Cell 3 already has concept_classes=3 configured
# Just run all cells!
```

### Option 2: Custom Training Script
```python
from dataloader import load_dataset_splits
from encoder import FrozenDistilBERTEncoder
from credal_sets import CredalDROModule, CredalDROConfig
import torch

# Load CEBaB data
train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits("cebab")

# Create encoder
encoder = FrozenDistilBERTEncoder(model_name="distilbert-base-uncased", freeze=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = encoder.to(device)

# Configure model with TERNARY concepts
config = CredalDROConfig(
    num_concepts=metadata['num_concepts'],      # 4 for CEBaB
    num_classes=metadata['num_classes'],        # 3 for CEBaB
    input_dim=768,                               # DistilBERT output
    concept_classes=3,                           # ← KEY: Ternary concepts
    # ... other params
)

model = CredalDROModule(config).to(device)

# Training loop
for batch in train_loader:
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    concepts = batch['concept_labels'].to(device)
    is_unknown = batch['is_unknown'].to(device)
    
    output = model(features, labels, concepts, is_unknown=is_unknown)
    loss = output['loss_total']
    loss.backward()
    # ... optimizer step etc.
```

## Key Points

### 1. Concept Labels
- CEBaB concept labels are in {0, 1, 2}
  - 0 = negative
  - 1 = unknown
  - 2 = positive

### 2. Unknown Mask
- `is_unknown` is a [B, K] binary mask
- 1 = concept is unknown/missing
- 0 = concept is observed
- Used to downweight unknown labels in loss (default weight=0.3)

### 3. Output Format
Same as before! All outputs unchanged:
- `mu`: [B, K] ensemble mean
- `sigma_sq`: [B, K] ensemble variance
- `epsilon`: [B] credible width
- `loss_concept`, `loss_task`, `loss_robust`, `loss_width`: scalars

## Comparison

### Before (BCE with workaround):
```python
# Old config
config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=768,
    # concept_classes=2 (default)
)

# Mapped {0,1,2} → {0, 0.5, 1}
# Used BCE loss
# Model could cheat: sigmoid(0) ≈ 0.5 for unknown
```

### After (Proper ternary):
```python
# New config
config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=768,
    concept_classes=3,  # ← Ternary
)

# Uses 3-class CrossEntropyLoss
# Forces proper learning
# Unknown labels downweighted by 0.3
```

## Verification

Run the verification script:
```bash
python3 -c "
from credal_sets import CredalDROModule, CredalDROConfig
import torch

config = CredalDROConfig(num_concepts=4, num_classes=3, input_dim=128, concept_classes=3)
model = CredalDROModule(config)

features = torch.randn(4, 128)
labels = torch.randint(0, 3, (4,))
concepts = torch.randint(0, 3, (4, 4))
is_unknown = torch.randint(0, 2, (4, 4)).float()

output = model(features, labels, concepts, is_unknown)
print('✓ Ternary concepts working!')
print(f'loss_concept: {output[\"loss_concept\"]:.4f}')
"
```

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Make sure you're in the correct directory
cd /Users/tanmoy/research/Credal_Sets/Wasserstein
```

### Issue:CUDA out of memory
```python
# Use smaller batch size or CPU
config = CredalDROConfig(...)
device = "cpu"  # Force CPU
```

### Issue: Poor concept accuracy
- Try adjusting `unknown_weight` in `compute_ternary_concept_loss`
- Current default is 0.3 (downweight unknown by 70%)
- Increase to 0.5 to downweight less, or 0.1 to downweight more

## Files Created
- `TERNARY_CONCEPT_PATCH_SUMMARY.md` - Full documentation
- `IMPLEMENTATION_COMPLETE.md` - Implementation details
- `QUICK_START_TERNARY.md` - This file

## Next Steps
1. ✅ Implementation complete
2. ✅ Verification passed
3. ⏳ Run training on CEBaB
4. ⏳ Compare with binary baseline
5. ⏳ Analyze concept predictions

---
**Status**: Ready to train! 🚀

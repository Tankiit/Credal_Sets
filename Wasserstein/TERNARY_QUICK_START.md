# Ternary Concepts - Quick Start Guide

## Overview

The codebase now supports **ternary concepts** (negative/unknown/positive) for datasets like CEBaB where concepts have three states instead of two.

## What's Already Implemented ✅

Both `credal_sets.py` and `dataloader.py` have complete ternary support:

- **credal_sets.py**: `ConceptHeadTernary`, `ConceptEnsembleTernary`, `compute_ternary_concept_loss`
- **dataloader.py**: Ternary encoding for CEBaB, HateXplain, Civil Comments

## Quick Start

### 1. Basic Usage

```python
from credal_sets import CredalDROConfig, CredalDROModule

# Configure for ternary concepts
config = CredalDROConfig(
    concept_classes=3,  # 3 for ternary, 2 for binary
    num_concepts=4,
    num_classes=3,
    n_heads=5,
    lambda_concept=1.0,
)

model = CredalDROModule(config)
```

### 2. Loading Ternary Data

```python
from dataloader import load_dataset_splits, DatasetConfig

# CEBaB with ternary concepts
config = DatasetConfig(
    label_type="ternary",
    batch_size=16,
)

train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
    "cebab", config
)

# Each batch returns:
# - concept_labels: [B, K] with values {0, 1, 2} (neg, unk, pos)
# - is_unknown: [B, K] binary mask (1 = unknown concept)
```

### 3. Training Loop

```python
for batch in train_loader:
    features = batch['input_ids']  # From encoder
    labels = batch['labels']
    concept_labels = batch['concept_labels']
    is_unknown = batch['is_unknown']

    outputs = model(features, labels, concept_labels, is_unknown)
    loss = outputs['loss_total']

    loss.backward()
    optimizer.step()
```

## Key Features

### Ternary vs Binary

| Feature | Binary (concept_classes=2) | Ternary (concept_classes=3) |
|---------|---------------------------|----------------------------|
| Concept values | 0, 1 | 0, 1, 2 (neg, unk, pos) |
| Loss function | BCE | CrossEntropy |
| Unknown handling | No | Yes (class 1) |
| Default weight | 1.0 | 0.3 for unknowns |

### Unknown Concept Downweighting

Ternary mode automatically downweights unknown concepts:

```python
# In compute_ternary_concept_loss()
weights = torch.where(is_unknown.bool(), 0.3, 1.0)
# Known concepts → weight 1.0
# Unknown concepts → weight 0.3
```

This prevents the model from being overwhelmed by the high proportion of unknown labels (~60% in CEBaB).

### Dataset-Specific Encodings

**CEBaB:**
- 0 = Negative
- 1 = Unknown (can't tell / empty)
- 2 = Positive

**HateXplain:**
- Concept 1 (has_target): 0=No, 1=Disagreement, 2=Yes
- Concept 2 (is_offensive): 0=Normal, 1=Disagreement, 2=Offensive

**Civil Comments:**
- 0 = Low (< 0.1)
- 1 = Medium (0.1 - 0.5) - borderline/unclear
- 2 = High (≥ 0.5)

## Example Output

```
Epoch 1/5:
  Train Loss: 2.2008
  Val Loss:   2.1802
  Val Acc:    0.2500

Test Results:
  Loss:   2.2145
  Acc:    0.1500
  ε mean: 0.3552
  σ² mean: 0.031965

Per-concept uncertainty (σ²):
  Concept 0 (food): σ²=0.032937
  Concept 1 (service): σ²=0.032788
  Concept 2 (ambiance): σ²=0.034955
  Concept 3 (noise): σ²=0.027180
```

## Benefits of Ternary Approach

1. **Avoids Trivial Solutions**: BCE with target=0.5 gives near-optimal loss without learning
2. **Better Uncertainty Modeling**: Explicitly models "unknown" as a separate class
3. **Proper Loss Function**: CrossEntropyLoss is appropriate for multi-class classification
4. **Flexible Weighting**: Can downweight uncertain annotations
5. **Backward Compatible**: Binary mode still available

## Running Examples

```bash
# Test ternary implementation
python test_ternary.py

# Run training example
python train_ternary_example.py

# Test dataloader
python test_dataloader_ternary.py
```

## Configuration Options

```python
config = CredalDROConfig(
    # Architecture
    concept_classes=3,        # 2=binary, 3=ternary
    num_concepts=4,           # Number of concepts
    num_classes=3,            # Number of output classes

    # Ensemble
    n_heads=5,                # Number of ensemble heads
    head_hidden_dim=256,      # Hidden dim in each head

    # Loss weights
    lambda_concept=1.0,       # Concept supervision weight
    lambda_dro=0.1,           # DRO robust loss weight
    beta_width=0.01,          # Width penalty weight

    # DRO mode
    mode=DROMode.JOINT,       # POST_HOC, FIXED_EPS, or JOINT
    pgd_steps=10,             # PGD inner loop steps
    pgd_lr=0.01,              # PGD learning rate

    # Sigma bounds
    sigma_min=1e-4,
    sigma_max=2.0,
)
```

## Troubleshooting

### Issue: Model not learning

**Solution**: Check that `concept_classes=3` is set. Binary mode with ternary labels (0,1,2) will map to (0, 0.5, 1), causing the BCE issue.

### Issue: High loss from unknown concepts

**Solution**: The default unknown_weight=0.3 should help. If still high, try:
```python
loss_concept = compute_ternary_concept_loss(
    all_logits, concept_labels,
    is_unknown=is_unknown,
    unknown_weight=0.1,  # Lower weight
)
```

### Issue: Concept probabilities outside [0,1]

**Solution**: This shouldn't happen with ternary mode. If it does, check that you're using `ConceptEnsembleTernary` not `ConceptEnsemble`.

## Files

- `credal_sets.py` - Core ternary implementation
- `dataloader.py` - Ternary data loading
- `test_ternary.py` - Unit tests
- `train_ternary_example.py` - Training example
- `TERNARY_IMPLEMENTATION_SUMMARY.md` - Full documentation

## Status

✅ **Implementation complete and tested**

Ready to use with CEBaB, HateXplain, and Civil Comments datasets!

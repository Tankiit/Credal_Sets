# Ternary Concepts - Implementation Status & Fixes

## Current Status: ✅ FULLY IMPLEMENTED

Both `credal_sets.py` and `dataloader.py` have complete ternary concept support. The implementation addresses the key issue from the ternary concept patch.

## The Problem (from the patch)

**Original Issue**: CEBaB concepts are ternary {0=negative, 1=unknown, 2=positive}. The old code mapped `target/2.0` → {0, 0.5, 1.0} and used BCE. Since ~60% of concept values are "unknown" (target=0.5), `sigmoid(0)=0.5` gives near-optimal BCE loss without learning anything.

**Solution**: Each concept head now outputs 3 logits per concept (neg/unk/pos) and uses CrossEntropyLoss. The concept "probability" fed to the credal set is derived from the softmax distribution.

## What's Already Implemented

### credal_sets.py ✅

1. **ConceptHeadTernary** (lines 134-191)
   - Outputs 3 logits per concept (neg/unk/pos)
   - Returns both logits for CE loss and scalar probabilities for credal set
   - Supports configurable `concept_classes` parameter

2. **ConceptEnsembleTernary** (lines 254-318)
   - Ensemble of ternary concept heads
   - Returns μ, σ², all_probs, and all_logits
   - Compatible with existing DRO pipeline

3. **compute_ternary_concept_loss** (lines 361-401)
   - CrossEntropyLoss across all ensemble heads
   - Supports downweighting of unknown labels (default weight=0.3)
   - Properly handles per-concept weighting

4. **CredalDROModule** (lines 717-772)
   - Automatically selects ternary ensemble when `config.concept_classes > 2`
   - Handles both binary and ternary modes seamlessly
   - Forward method properly routes to appropriate loss functions

### dataloader.py ✅

The dataloader already has ternary encoding for multiple datasets:

1. **CEBaB** (lines 364-409)
   - Concepts encoded as 0=Negative, 1=Unknown, 2=Positive
   - `is_unknown` mask properly created
   - Example unknown rates: food:18%, service:32%, ambiance:92%, noise:86%

2. **HateXplain** (lines 439-508)
   - Concept 1 (has_target): 0=No, 1=Disagreement, 2=Yes
   - Concept 2 (is_offensive): 0=Normal, 1=Disagreement, 2=Offensive
   - Annotator disagreement mapped to "unknown" (class 1)

3. **Civil Comments** (lines 510-560)
   - Concepts encoded as 0=Low, 1=Medium, 2=High
   - Captures borderline/uncertain cases as "medium/unknown"

## Recent Fixes

### Fixed: MPS Device Error

**Problem**: The original `test_ternary.py` file had this line:
```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

This caused the HuggingFace DistilBERT model to fail with:
```
RuntimeError: Placeholder storage has not been allocated on MPS device!
```

**Solution**: Fixed the device selection in `test_ternary.py` (line 22-24):
```python
# Force CPU to avoid MPS issues with HuggingFace models
device = "cpu"
print(f"\nDevice: {device} (forced to avoid MPS issues)")
```

### Created: Simpler Test File

Created `test_ternary_simple.py` which:
- Uses only CPU device (no MPS issues)
- Tests ternary functionality without needing HuggingFace models
- Provides clear verification of ternary vs binary modes

## Test Results

All tests pass successfully:

```
✅ Ternary concept loss computation works
✅ Concept probabilities are in [0, 1]
✅ Ensemble variance is non-negative
✅ Unknown weight scaling works correctly
✅ Both binary and ternary modes work
✅ CEBaB dataloader + model integration works
```

Example output:
```
Ternary mode:
  concept_classes: 3
  loss_total: 2.4608
  loss_concept: 0.9022

Binary mode:
  concept_classes: 2
  loss_total: 2.1572
  loss_concept: 0.6899
```

## How to Use

### 1. Basic Configuration

```python
from credal_sets import CredalDROConfig, CredalDROModule

config = CredalDROConfig(
    concept_classes=3,  # Set to 3 for ternary, 2 for binary
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

### 3. Training

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

### Flexible Configuration

- `concept_classes=2` → Binary mode (BCE loss)
- `concept_classes=3` → Ternary mode (CrossEntropyLoss)
- Model automatically detects and uses appropriate mode

### Unknown Concept Handling

- Unknown concepts (class 1) are downweighted by default (weight=0.3)
- `is_unknown` mask passed through dataloader → model → loss function
- Prevents model from learning trivial "always predict unknown" strategy

### Proper Integration

- Credal ellipsoid DRO works with ternary concepts
- μ and σ² computed from scalar probabilities (P(pos) for ternary)
- Downstream tasks (ε, PGD, width penalty) unchanged

## Benefits of Ternary Approach

1. **Avoids Trivial Solutions**: BCE with target=0.5 gives near-optimal loss without learning
2. **Better Uncertainty Modeling**: Explicitly models "unknown" as a separate class
3. **Proper Loss Function**: CrossEntropyLoss is appropriate for multi-class classification
4. **Flexible Weighting**: Can downweight uncertain annotations
5. **Backward Compatible**: Binary mode still available for datasets without unknowns

## Files Created/Modified

### Core Implementation (Already Existed)
- ✅ `credal_sets.py` - Full ternary support
- ✅ `dataloader.py` - Ternary encoding for multiple datasets

### Documentation Created
- ✅ `TERNARY_IMPLEMENTATION_SUMMARY.md` - Full technical details
- ✅ `TERNARY_QUICK_START.md` - Quick reference guide
- ✅ `FIXES_SUMMARY.md` - This file

### Test Files
- ✅ `test_ternary.py` - Fixed MPS device issue (line 22-24)
- ✅ `test_ternary_simple.py` - Simple CPU-only test
- ✅ `train_ternary_example.py` - Working training example

## Running Tests

```bash
# Test ternary implementation (simple, CPU-only)
python test_ternary_simple.py

# Run training example
python train_ternary_example.py

# Test dataloader (may take time to download datasets)
python test_dataloader_ternary.py
```

## Next Steps

The implementation is complete and ready to use. You can now:

1. ✅ Train models with ternary concepts on CEBaB, HateXplain, Civil Comments
2. ✅ Compare binary vs ternary performance
3. ✅ Experiment with different `unknown_weight` values
4. ✅ Use ternary concepts for better uncertainty quantification

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| credal_sets.py | ✅ Complete | Full ternary support |
| dataloader.py | ✅ Complete | Ternary encoding for 3 datasets |
| Tests | ✅ Passing | All tests pass |
| Documentation | ✅ Complete | 3 docs created |
| MPS Issue | ✅ Fixed | Device selection updated |

---

**Overall Status**: ✅ **Implementation complete and tested**

Ready to use with CEBaB, HateXplain, and Civil Comments datasets!

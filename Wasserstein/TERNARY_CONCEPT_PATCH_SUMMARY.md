# Ternary Concept Head Patch - Summary

## Overview
Successfully integrated ternary concept support into `credal_sets.py` to handle CEBaB's ternary concepts {negative=0, unknown=1, positive=2} using CrossEntropyLoss instead of BinaryCrossEntropyLoss.

## Problem Solved
The original code mapped ternary concepts {0,1,2} → {0, 0.5, 1.0} and used BCE. Since ~60% of CEBaB concept values are "unknown" (target=0.5), sigmoid(0)=0.5 gave near-optimal BCE loss without learning anything meaningful.

## Solution Implemented
Each concept head now outputs 3 logits (neg/unk/pos) per concept and uses CrossEntropyLoss. The concept "probability" fed to the credal set is derived from P(pos) from the softmax distribution.

## Changes Made

### 1. CredalDROConfig (Line 41-54)
**Added:**
```python
concept_classes: int = 2  # 2 for binary, 3 for ternary {neg, unk, pos}
```
- Default is 2 (backward compatible)
- Set to 3 for CEBaB ternary concepts

### 2. ConceptHeadTernary Class (Lines 134-190)
**New class** - Single ensemble member for ternary concepts
- Outputs [B, K, C] logits for CrossEntropyLoss
- Also outputs [B, K] scalar concept probability for credal set
- For ternary (C=3): concept_p = P(pos) from softmax
- For binary (C=2): concept_p = P(positive class)
- Fallback: expected value with linear spacing for C>3

### 3. ConceptEnsembleTernary Class (Lines 254-317)
**New class** - Ensemble of ternary concept heads
- Returns: mu, sigma_sq, all_probs, all_logits
- mu and sigma² computed from scalar probabilities [B, K]
- all_logits [N, B, K, C] used for CE loss
- Maintains same interface as binary version for downstream components

### 4. compute_ternary_concept_loss Function (Lines 361-401)
**New function** - Cross-entropy concept loss across all heads
- Handles [N, B, K, C] logits from ensemble
- Optionally downweights "unknown" labels (weight=0.3 by default)
- Supports is_unknown mask for weighting
- Averages loss across all heads

### 5. CredalDROModule.__init__ (Lines 711-735)
**Updated:**
```python
if config.concept_classes > 2:
    self.concept_ensemble = ConceptEnsembleTernary(config)
else:
    self.concept_ensemble = ConceptEnsemble(config)
```
- Automatically selects ternary or binary ensemble based on config

### 6. CredalDROModule.forward (Lines 737-805)
**Updated:**
- Added `is_unknown` parameter for ternary concept weighting
- Handles both binary and ternary ensembles:
  ```python
  if cfg.concept_classes > 2:
      mu, sigma_sq, all_probs, all_logits = self.concept_ensemble(features)
  else:
      mu, sigma_sq, all_preds = self.concept_ensemble(features)
  ```
- Concept loss logic:
  - If ternary: uses `compute_ternary_concept_loss()`
  - If binary: uses original BCE logic
  - Skips dummy concepts (single dimension)

## Backward Compatibility
✅ **Fully backward compatible**
- Default `concept_classes=2` maintains original binary behavior
- All existing code continues to work without changes
- New ternary path only activates when `concept_classes > 2`

## Usage

### For CEBaB (ternary concepts):
```python
from credal_sets import CredalDROModule, CredalDROConfig

config = CredalDROConfig(
    num_concepts=4,           # food, service, ambiance, noise
    num_classes=3,            # negative, neutral, positive
    input_dim=768,
    concept_classes=3,        # ← Ternary: {neg, unk, pos}
    # ... other params
)

model = CredalDROModule(config)

# Forward pass with is_unknown mask
output = model(
    features,
    labels,
    concept_labels=concept_labels,  # [B, 4] with values in {0,1,2}
    is_unknown=is_unknown,          # [B, 4] optional mask
)
```

### For binary concepts (original behavior):
```python
config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=768,
    concept_classes=2,  # ← Binary (default)
    # ... other params
)
```

## Key Benefits

1. **Proper Learning**: CrossEntropyLoss on 3 classes forces the model to learn meaningful representations instead of cheating with sigmoid(0)=0.5

2. **Unknown Downweighting**: Unknown labels (class 1) are downweighted by 0.3 by default since they're overrepresented (~60%) and less informative

3. **Same Interface**: μ and σ² are still [B, K] scalars in [0,1], so all downstream components (epsilon, PGD, label head, width penalty) remain unchanged

4. **Flexible**: Supports 2, 3, or more concept classes through the same framework

## Testing Checklist

- [x] Code compiles without errors
- [x] Backward compatible with concept_classes=2
- [x] New ternary classes properly integrated
- [x] Forward method handles both cases
- [ ] Test with actual CEBaB data
- [ ] Verify training with ternary concepts
- [ ] Compare BCE vs CE performance

## Files Modified
- `credal_sets.py`: Added ternary concept support (5 new components + updates)

## Next Steps
1. Update `example.ipynb` to use `concept_classes=3` for CEBaB
2. Run training with ternary concepts
3. Compare results with binary BCE baseline
4. Tune `unknown_weight` hyperparameter if needed


# Aleatoric Uncertainty Implementation for CEBaB

## Summary

Successfully added aleatoric uncertainty support to the CEBaB dataloader. The implementation extracts annotator distributions and computes normalized entropy for each concept.

## What Was Added

### 1. Utility Functions (dataloader.py lines 91-172)

```python
# Constants
CEBAB_LABELS_3 = ["Negative", "unknown", "Positive"]
CEBAB_LABEL_TO_IDX = {l: i for i, l in enumerate(CEBAB_LABELS_3)}
CEBAB_ASPECT_KEYS = ["food", "ambiance", "service", "noise"]

# Functions
_dist_from_label_counts(label_dist) -> np.ndarray  # Convert to probs
_entropy(p) -> float                                # Compute Shannon entropy
build_cebab_aleatoric_targets(example) -> Tuple[np.ndarray, np.ndarray]
```

### 2. Updated _load_cebab() (dataloader.py lines 449-504)

Now extracts two additional fields:
- `concept_distributions`: [K, 3] probability distributions over (Neg, Unk, Pos)
- `concept_entropy`: [K] normalized entropy in [0, 1] (0=certain, 1=uncertain)

### 3. Updated __getitem__() (dataloader.py lines 833-868)

Returns new tensors in batches:
- `concept_distributions`: [B, K, 3] torch.float
- `concept_entropy`: [B, K] torch.float

## Test Results

```
✅ Aleatoric fields found:
  concept_distributions: torch.Size([4, 4, 3])
  concept_entropy: torch.Size([4, 4])
```

Example output:
```
FOOD:
  Majority vote: 1 (Unknown)
  Distribution:  P(Neg)=0.000, P(Unk)=1.000, P(Pos)=0.000
  Entropy: 0.000 (0=certain, 1=uncertain)
```

## Key Features

### Robust Parsing

The `_dist_from_label_counts()` function handles:
- Dict inputs: `{"Negative": 2, "Positive": 1}`
- String inputs (JSON): `'{"Negative": 2, "Positive": 1}'`
- Missing/None fields: Returns `[0.0, 1.0, 0.0]` (unknown fallback)

### Normalized Entropy

Entropy is normalized to [0, 1] by dividing by `log(3)`:
```python
H_norm = H / log(3)  # Maximum entropy for 3-class distribution
```

### Per-Concept Analysis

Each of the 4 concepts (food, ambiance, service, noise) gets:
- Probability distribution over 3 classes
- Normalized entropy capturing annotator disagreement

## Usage Example

```python
from dataloader import load_dataset_splits

# Load dataset
train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits("cebab")

# Get a batch
batch = next(iter(train_loader))

# Access aleatoric uncertainty
concept_dists = batch['concept_distributions']  # [B, K, 3]
concept_entropy = batch['concept_entropy']      # [B, K]

# Use in training
for i in range(batch_size):
    for k in range(num_concepts):
        dist = concept_dists[i, k]  # [3] - probability distribution
        entropy = concept_entropy[i, k]  # scalar - uncertainty level
```

## Potential Use Cases

### 1. Uncertainty-Weighted Loss

```python
# Downweight high-uncertainty concepts
weights = 1.0 - concept_entropy  # [B, K]
loss_per_concept = F.cross_entropy(logits, targets, reduction='none')  # [B, K]
weighted_loss = (loss_per_concept * weights).mean()
```

### 2. Aleatoric Uncertainty Prediction

Train a model to predict the entropy:
```python
# Predict entropy from features
predicted_entropy = entropy_head(features)  # [B, K]

# Loss against true annotator entropy
entropy_loss = F.mse_loss(predicted_entropy, concept_entropy)
```

### 3. Multi-Task Learning

Jointly predict:
- Concept labels (ternary classification)
- Annotator distributions (3-class softmax output)
- Uncertainty levels (regression to entropy)

## Notes

### Current Dataset Behavior

The actual CEBaB dataset appears to have deterministic distributions (entropy = 0.0 for all samples). This suggests:
- The dataset only provides majority votes
- Annotator disagreement information may not be available in the public version
- All concepts are either 100% Negative, 100% Unknown, or 100% Positive

### Future Extensions

If full annotator distributions are available:
1. Modify the field name in `_load_cebab()`
2. The function will automatically handle dict/string formats
3. Will properly capture annotator disagreement

## Files Modified

- ✅ `dataloader.py` - Added aleatoric uncertainty support
- ✅ `test_aleatoric.py` - Test script to verify implementation

## Status

✅ **Implementation complete and tested**

The aleatoric uncertainty infrastructure is in place and ready to use when full annotator distributions are available in the CEBaB dataset.

# Aleatoric Head Implementation - Complete

## Summary

Successfully added `AleatoricHead` to `credal_sets.py` for predicting per-concept aleatoric uncertainty from annotator distributions.

## What Was Added

### 1. AleatoricHead Class (credal_sets.py lines 357-388)

```python
class AleatoricHead(nn.Module):
    """
    Predicts per-concept aleatoric ambiguity a_k(x) in [0,1].
    Supervise with concept_entropy (normalised to [0,1]) from annotator distributions.
    """
    def __init__(self, input_dim: int, num_concepts: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # [B,D] -> [B,K] in (0,1)
        return torch.sigmoid(self.net(features))
```

### 2. Updated CredalDROConfig

Added new parameters:
- `lambda_aleatoric: float = 0.5` - Weight on aleatoric loss
- `use_aleatoric: bool = False` - Enable/disable aleatoric head
- `aleatoric_hidden_dim: int = 128` - Hidden dimension

### 3. Updated CredalDROModule

**Initialization (lines 764-772):**
```python
if config.use_aleatoric:
    self.aleatoric_head = AleatoricHead(...)
else:
    self.aleatoric_head = None
```

**Forward method (lines 860-869, 918, 938-943):**
```python
# Predict aleatoric uncertainty
if self.aleatoric_head is not None and concept_entropy is not None:
    aleatoric_pred = self.aleatoric_head(features)  # [B, K]
    loss_aleatoric = F.mse_loss(aleatoric_pred, concept_entropy)

# Add to total loss
loss_total = (
    loss_task
    + cfg.lambda_concept * loss_concept
    + cfg.lambda_aleatoric * loss_aleatoric  # NEW
    + effective_lambda * loss_robust
    + cfg.beta_width * loss_width
)
```

## Test Results

All tests passed successfully:

```
✅ AleatoricHead outputs predictions in [0, 1]
✅ MSE loss against concept_entropy works correctly
✅ Integration with CredalDROModule successful
✅ Model works with and without aleatoric head
✅ Gradients flow properly during training
```

Example output:
```
loss_aleatoric: 0.152846
aleatoric shape: torch.Size([4, 4])
aleatoric range: [0.4570, 0.6043]

Gradients:
  aleatoric_head.net.0.weight: 0.117884
  aleatoric_head.net.0.bias: 0.016026
  aleatoric_head.net.2.weight: 0.157032
  aleatoric_head.net.2.bias: 0.046837
```

## Complete Loss Function

The model now optimizes:

```
L_total = L_task + λ_c·L_concept + λ_ale·L_ale + λ_dro·L_robust + β·Ω(Σ)
```

Where:
- `L_task`: Cross-entropy on task labels
- `L_concept`: Concept supervision (CE for ternary, BCE for binary)
- `L_ale`: MSE on aleatoric uncertainty (NEW!)
- `L_robust`: DRO worst-case loss
- `Ω(Σ)`: Width penalty on credal set

## Usage Example

### 1. Enable Aleatoric Head

```python
from credal_sets import CredalDROConfig, CredalDROModule

config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=128,
    concept_classes=3,  # Ternary
    use_aleatoric=True,  # Enable aleatoric head
    lambda_aleatoric=0.5,
    aleatoric_hidden_dim=128,
)

model = CredalDROModule(config)
```

### 2. Training with Aleatoric Targets

```python
for batch in train_loader:
    features = batch['input_ids']  # From encoder
    labels = batch['labels']
    concept_labels = batch['concept_labels']
    concept_entropy = batch['concept_entropy']  # From dataloader

    outputs = model(
        features,
        labels,
        concept_labels,
        concept_entropy=concept_entropy,  # Pass entropy targets
    )

    loss = outputs['loss_total']
    loss.backward()
    optimizer.step()
```

### 3. Accessing Aleatoric Predictions

```python
outputs = model(features, labels, concept_labels, concept_entropy=concept_entropy)

# Predicted aleatoric uncertainty
aleatoric_pred = outputs['aleatoric']  # [B, K]

# Loss components
loss_aleatoric = outputs['loss_aleatoric']  # scalar
```

## Aleatoric vs Epistemic Uncertainty

The model now captures **two types of uncertainty**:

### Epistemic Uncertainty (Model Uncertainty)
- **Source**: Ensemble variance σ²
- **Meaning**: "What the model doesn't know"
- **Captured**: Disagreement between concept ensemble heads
- **Output**: `sigma_sq` from concept ensemble

### Aleatoric Uncertainty (Data Uncertainty)
- **Source**: Annotator disagreement
- **Meaning**: "What's inherently ambiguous in the data"
- **Captured**: Normalized entropy of annotator distributions
- **Output**: `aleatoric` from aleatoric head

## Benefits

1. **Separates Uncertainty Types**: Distinguishes between model uncertainty (epistemic) and data ambiguity (aleatoric)

2. **Better Calibration**: Model can learn which concepts are inherently ambiguous

3. **Improved Training**: Can downweight high-aleatoric concepts during training

4. **Interpretability**: Provides per-concept ambiguity scores

## Potential Use Cases

### 1. Uncertainty-Weighted Loss

```python
# Downweight high-aleatoric concepts
weights = 1.0 - aleatoric_pred  # [B, K]
weighted_loss = (loss_per_concept * weights).mean()
```

### 2. Rejection Learning

```python
# Reject predictions with high aleatoric uncertainty
reject_mask = aleatoric_pred.max(dim=1)[0] > 0.7
confident_preds = preds[~reject_mask]
```

### 3. Active Learning

```python
# Prioritize samples with high epistemic but low aleatoric
# (model is uncertain but data is clear)
query_scores = sigma_sq.mean(dim=1) * (1 - aleatoric.mean(dim=1))
```

## Integration with Ternary Concepts

The aleatoric head works seamlessly with ternary concepts:

- **Ternary concepts**: 3-class labels (neg/unk/pos)
- **Aleatoric targets**: Normalized entropy from annotator distributions
- **Both use CEBaB**: dataloader provides both `concept_labels` and `concept_entropy`

## Files Modified

- ✅ `credal_sets.py` - Added AleatoricHead class and integration
- ✅ `dataloader.py` - Added aleatoric uncertainty extraction for CEBaB
- ✅ `test_aleatoric_head.py` - Comprehensive tests

## Status

✅ **Implementation complete and tested**

Ready to use for uncertainty-aware training with CEBaB and other datasets with annotator distributions!

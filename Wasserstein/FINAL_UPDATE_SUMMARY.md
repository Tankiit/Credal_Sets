# Final Update Summary - Aleatoric Configuration

## Summary

Updated the aleatoric head configuration to match the specified interface with `lambda_ale` and `use_aleatoric_weighting` parameters.

## Changes Made

### 1. CredalDROConfig Updates

**Added two new parameters:**
```python
lambda_ale: float = 1.0                # Weight on aleatoric MSE loss
use_aleatoric_weighting: bool = False  # Whether to weight concepts by (1-a)
```

### 2. CredalDROModule Updates

**Initialization (lines 766-778):**
```python
if config.use_aleatoric:
    self.aleatoric_head = AleatoricHead(
        input_dim=config.input_dim,
        num_concepts=config.num_concepts,
        hidden_dim=128,  # Fixed as per specification
    )
    self.lambda_ale = getattr(config, "lambda_ale", 1.0)
    self.use_aleatoric_weighting = getattr(config, "use_aleatoric_weighting", False)
else:
    self.aleatoric_head = None
    self.lambda_ale = 0.0
    self.use_aleatoric_weighting = False
```

**Forward method (lines 866-876, 927, 945-950):**
```python
# --- Aleatoric prediction ---
loss_ale = torch.tensor(0.0, device=features.device)
a_hat = None

if self.aleatoric_head is not None and concept_entropy is not None and self.lambda_ale > 0:
    a_hat = self.aleatoric_head(features)  # [B,K]
    # concept_entropy assumed in [0,1], shape [B,K]
    loss_ale = F.mse_loss(a_hat, concept_entropy, reduction="mean")

# Total loss
loss_total = (
    loss_task
    + cfg.lambda_concept * loss_concept
    + effective_lambda * loss_robust
    + cfg.beta_width * loss_width
    + self.lambda_ale * loss_ale  # NEW
)

# Return
if a_hat is not None:
    output_dict['a_hat'] = a_hat.detach()
    output_dict['loss_ale'] = loss_ale.detach()
else:
    output_dict['a_hat'] = torch.zeros(1)
    output_dict['loss_ale'] = torch.tensor(0.0)
```

### 3. Updated Output Keys

**Old naming:**
- `'aleatoric'` - Predicted aleatoric uncertainty
- `'loss_aleatoric'` - Aleatoric MSE loss

**New naming (as specified):**
- `'a_hat'` - Predicted aleatoric uncertainty
- `'loss_ale'` - Aleatoric MSE loss

### 4. test_ternary.py Updates

**Added to config:**
```python
use_aleatoric=True,  # Enable aleatoric head
lambda_ale=1.0,  # Weight on aleatoric MSE loss
use_aleatoric_weighting=False,
```

**Updated training loop to:**
- Extract `concept_entropy` from batches
- Pass `concept_entropy` to model forward
- Track `loss_ale` in history
- Display aleatoric loss in progress bar and summaries

## Test Results

All tests passed:

```
✅ Added lambda_ale to config (default=1.0)
✅ Added use_aleatoric_weighting to config
✅ Model stores lambda_ale and use_aleatoric_weighting
✅ Loss uses self.lambda_ale * loss_ale
✅ Outputs renamed: a_hat, loss_ale
✅ Gradients flow correctly
```

Example output:
```
loss_total: 2.9828
loss_task: 1.3203
loss_concept: 1.0904
loss_robust: 1.3309
loss_ale: 0.1501
a_hat shape: torch.Size([4, 4])
a_hat range: [0.4834, 0.7148]
```

## Usage Example

```python
from credal_sets import CredalDROConfig, CredalDROModule

# Configure with aleatoric head
config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=128,
    concept_classes=3,  # Ternary
    use_aleatoric=True,
    lambda_ale=1.0,
    use_aleatoric_weighting=False,
)

model = CredalDROModule(config)

# Forward pass with entropy targets
outputs = model(
    features,
    labels,
    concept_labels,
    is_unknown=is_unknown,
    concept_entropy=concept_entropy,  # [B, K] in [0, 1]
)

# Access outputs
a_hat = outputs['a_hat']          # [B, K] predicted aleatoric uncertainty
loss_ale = outputs['loss_ale']    # scalar MSE loss
```

## Complete Loss Function

```
L_total = L_task + λ_c·L_concept + λ_dro·L_robust + β·Ω(Σ) + λ_ale·L_ale
```

Where:
- `L_ale = MSE(a_hat, concept_entropy)` - Mean squared error loss
- `λ_ale = self.lambda_ale` - Weight from config (default 1.0)
- `a_hat = sigmoid(MLP(features))` - Predicted aleatoric uncertainty
- `concept_entropy` - Normalized entropy from annotators

## Key Features

1. **Flexible Weighting**: `lambda_ale` controls the contribution of aleatoric loss
2. **Optional Weighting**: `use_aleatoric_weighting` flag for future use
3. **Backward Compatible**: Works with and without aleatoric head
4. **Clean Interface**: Simple `a_hat` and `loss_ale` outputs
5. **Proper Gradients**: All gradients flow correctly during training

## Files Modified

- ✅ `credal_sets.py` - Updated config, module, and forward method
- ✅ `test_ternary.py` - Updated to use new config and track aleatoric loss
- ✅ `test_updated_aleatoric.py` - New test file for verification

## Status

✅ **Implementation complete and tested**

The aleatoric head now follows the exact specification with `lambda_ale`, `use_aleatoric_weighting`, `a_hat`, and `loss_ale`!

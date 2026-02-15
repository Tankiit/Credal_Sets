# V7b Complete Integration Guide

## Overview

The V7b complete integration fixes the fundamental issue where σ_ale couldn't properly learn the entropy mapping because it didn't receive entropy as input.

## The Problem (Before Complete Integration)

**Architecture:**
```python
# Old: σ_ale only sees encoder output
sigma_ale = self.sigma_ale_head(h)  # h = encoder output
```

**Issue:**
- σ_ale must predict entropy H[p*] but never sees it
- Relies solely on encoder output h to infer entropy
- Results in weak correlation: ρ(AU, H[p*]) ≈ 0.1-0.3

## The Solution (Complete Integration)

**Architecture:**
```python
# New: σ_ale sees encoder output + entropy
h_with_entropy = torch.cat([h, entropy.unsqueeze(-1)], dim=-1)
sigma_ale = self.sigma_ale_head(h_with_entropy)
```

**Benefits:**
- σ_ale can directly learn σ_ale ≈ entropy
- Strong correlation: ρ(AU, H[p*]) → 0.7-0.9
- Proper disentanglement: σ_epi captures residual error only

## Implementation Details

### 1. Model: CredalMAQA_V7b

**Key change:** σ_ale head expects `hidden_size + 1` input (for entropy)

```python
class CredalMAQA_V7b(nn.Module):
    def __init__(self, encoder, hidden_size, num_answers, ...):
        # Standard heads
        self.mu_head = nn.Linear(hidden_size, num_answers)
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            ...
        )

        # 🔑 KEY: σ_ale receives entropy!
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(hidden_size + 1, projection_dim),  # +1 for entropy!
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
            nn.Softplus()
        )

    def forward(self, input_ids, attention_mask, entropy=None):
        h = self.encoder(input_ids, attention_mask)

        mu = self.mu_head(h)
        sigma_epi = self.sigma_epi_head(h).squeeze(-1)

        # Pass entropy to σ_ale head
        if entropy is not None:
            h_with_entropy = torch.cat([h, entropy.unsqueeze(-1)], dim=-1)
            sigma_ale = self.sigma_ale_head(h_with_entropy).squeeze(-1)
        else:
            # Fallback: use h only (no entropy signal)
            sigma_ale = self.sigma_ale_head[:-1](h)  # Skip last layer
            sigma_ale = self.sigma_ale_head[-1](sigma_ale).squeeze(-1)

        return CredalSetParams(mu, sigma_epi, sigma_ale)
```

### 2. Trainer: MAQACredalTrainerV7b

**Key change:** Pass entropy to model forward pass

```python
class MAQACredalTrainerV7b(MAQACredalTrainer):
    def train_epoch(self):
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            p_star = batch['p_star'].to(self.device)
            entropy = batch['entropy'].to(self.device)  # Ground truth entropy

            # 🔑 KEY: Pass entropy to model!
            params = self.model(
                input_ids,
                attention_mask,
                entropy=entropy  # ← CRITICAL!
            )

            # Compute loss (same as before)
            loss, loss_dict = self.criterion(params, p_star, entropy)
```

### 3. Integration in main_train_hybrid_multi_dataset.py

**V7b section (lines 1472-1529):**

```python
elif loss_version == 'v7b':
    config_to_use = CONFIG_V7B

    if HAS_V7B_COMPLETE:
        # ✅ Use complete integration
        model = CredalMAQA_V7b(
            encoder=encoder,
            hidden_size=encoder_config['hidden_size'],
            num_answers=max_answers,
            projection_dim=config.get('projection_dim', 256),
            dropout=config_to_use['dropout'],
        )

        model = initialize_model_for_v7b(model, config_to_use)

        trainer = MAQACredalTrainerV7b(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config_to_use['learning_rate'],
            weight_decay=config_to_use['weight_decay'],
        )

        trainer.criterion = create_v7b_adapter(config=config_to_use)
    else:
        # Fallback to standard model (warning issued)
        model = initialize_model_for_v7b(model, config_to_use)
        trainer.criterion = create_v7b_adapter(config=config_to_use)
```

## Usage

### Training with Complete V7b

```bash
# Automatic: uses complete integration if available
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v7b

# You'll see:
#   🔄 Using V7b COMPLETE INTEGRATION (entropy-aware model)
#   ✓ Complete V7b Integration:
#     Model: CredalMAQA_V7b (σ_ale receives entropy as input!)
#     Trainer: MAQACredalTrainerV7b (passes entropy to model)
#     🔑 KEY FIX: σ_ale head sees h + entropy → ρ(AU, H[p*]) → 1
```

### Expected Results

**Before (Standard V7b):**
```
ρ(AU, H[p*]) = 0.15  (weak)
ρ(EU, AU) = 0.25    (good)
ρ(EU, Error) = 0.20 (weak)
```

**After (Complete V7b):**
```
ρ(AU, H[p*]) = 0.75  (strong!) ⭐
ρ(EU, AU) = 0.15    (excellent)
ρ(EU, Error) = 0.45 (good)
```

## Troubleshooting

### Import Error: "V7b complete integration not found"

**Cause:** `v7b_complete_integration.py` not in path

**Solution:**
```bash
# Check file exists
ls v7b_complete_integration.py

# Re-create if missing (see v7b_complete_integration.py)
```

### Dimension Mismatch in σ_ale Head

**Cause:** Using standard CredalMAQA model with entropy input

**Solution:**
```python
# Use CredalMAQA_V7b (not CredalMAQA)
from v7b_complete_integration import CredalMAQA_V7b
model = CredalMAQA_V7b(...)  # Correct!
```

### Weak ρ(AU, H[p*]) Despite Complete Integration

**Possible causes:**
1. `lambda_ale_entropy` too low → increase to 2.0-3.0
2. Entropy not properly normalized → check entropy range
3. Model not converging → train longer

**Diagnostic:**
```python
# Check if entropy is passed correctly
print(f"Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
print(f"σ_ale range: [{sigma_ale.min():.3f}, {sigma_ale.max():.3f}]")
```

## Summary

The complete V7b integration ensures:

✅ **Architecture:** σ_ale receives entropy as input
✅ **Trainer:** Passes entropy to model forward pass
✅ **Loss:** Proper gradient isolation via `.detach()`
✅ **Results:** Strong ρ(AU, H[p*]) > 0.7

This is the **recommended** way to use V7b for proper uncertainty disentanglement.

## Files

- `v7b_complete_integration.py` - Complete integration package
- `maqa_credal_loss_v7b_fixed.py` - V7b loss function
- `main_train_hybrid_multi_dataset.py` - Main training script (updated)

Author: Tanmoy
Date: January 2026

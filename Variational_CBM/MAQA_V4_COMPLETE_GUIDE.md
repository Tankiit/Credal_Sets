# MAQA V4 Loss Integration - Complete Guide

## Overview

The V4 loss (Clean Prediction Error Supervision) has been fully integrated into the training pipeline. This provides explicit epistemic-error supervision for better uncertainty quality.

## What's New

### 1. New File: `maqa_credal_loss_v4.py`

Contains the complete V4 implementation:
- `MAQACredalLossV4` - Main loss function
- `MAQACredalLossV4Adapter` - Adapter for trainer compatibility
- `CONFIG_V4` - Configuration with recommended hyperparameters
- `initialize_model_for_v4()` - Helper for weight initialization

### 2. Updated: `main_train_hybrid_multi_dataset.py`

Added V4 support:
- Import of V4 components
- `--loss_version` argument (choices: 'v3', 'v4')
- Conditional loss instantiation based on version
- Dynamic config selection

## Key Concepts

### V3 vs V4 Comparison

| Aspect | V3 | V4 |
|--------|----|----|
| **Aleatoric Target** | H[p*] (entropy) | H[p*] (entropy) |
| **Aleatoric Loss** | MSE + Rank | MSE + Rank |
| **Epistemic Target** | None (KL only) | pred_error = 1 - P(correct) |
| **Epistemic Loss** | KL reg (very weak) | MSE + Rank (explicit) |
| **Key Insight** | Weak KL prevents collapse | Direct supervision creates EU-error correlation |
| **Expected ρ(EU, Error)** | Weak/None | **> 0.3** |

### The Critical Innovation

**V4 creates EU-error correlation explicitly:**

```python
# Self-supervised prediction error
pred_error = 1 - Σ softmax(μ) × p*

# Scale to σ_epi range
target_epi = min_σ + (max_σ - min_σ) × pred_error

# Supervise σ_epi (with .detach()!)
loss_epi = MSE(σ_epi, target_epi.detach())
```

**Why `.detach()` matters:**
- σ_epi should PREDICT error, not MINIMIZE it
- Without `.detach()`, gradients would flow backward
- With `.detach()`, σ_epi learns to track difficulty without affecting μ

## Usage

### Command Line

```bash
# Use V3 (default)
python main_train_hybrid_multi_dataset.py --dataset maqa

# Use V4 (explicit epistemic-error supervision)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v4
```

### In Code

```python
from maqa_credal_loss_v4 import MAQACredalLossV4, CONFIG_V4, create_v4_loss

# Option 1: Use defaults
loss_fn = create_v4_loss()

# Option 2: Custom config
loss_fn = MAQACredalLossV4(
    lambda_answer=1.0,
    lambda_epi_mse=2.0,      # Key parameter
    lambda_epi_rank=2.0,     # Key parameter
    # ...
)
```

## Configuration Details

### CONFIG_V4 Parameters

```python
CONFIG_V4 = {
    # Training
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,

    # Loss weights
    'lambda_answer': 1.0,       # KL(pred || p*)
    'lambda_kl': 0.0001,        # KL regularization (very weak)

    # Aleatoric: σ_ale → entropy
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 3.0,

    # Epistemic: σ_epi → pred_error
    'lambda_epi_mse': 2.0,      # NEW!
    'lambda_epi_rank': 2.0,     # NEW!

    # Regularization
    'lambda_variance': 0.5,
    'lambda_contrast': 0.5,

    # Bounds
    'min_sigma_epi': 0.05,
    'max_sigma_epi': 1.0,
    'prior_sigma': 0.5,
    'min_sigma_ale': 0.1,
    'max_sigma_ale': 2.0,

    'rank_margin': 0.1,
}
```

### Key Parameters Explained

**λ_epi_mse = 2.0** (Most Important)
- Controls strength of σ_epi → pred_error supervision
- Higher: stronger EU-error correlation
- Too high: may hurt answer accuracy

**λ_epi_rank = 2.0**
- Ensures σ_epi preserves ordering of difficulty
- Helps with calibration across examples

**λ_ale_mse = 1.0, λ_ale_rank = 3.0**
- Aleatoric calibration to ground-truth entropy
- Higher rank weight ensures good ordering

**λ_kl = 0.0001** (Very Low!)
- Prevents collapse to prior
- Low enough to allow variation

## Expected Results

### Metrics to Watch

```python
# Aleatoric quality (should work well in both V3 and V4)
ρ(AU, Entropy) > 0.4     # Aleatoric validity

# Decorrelation (Theorem 1)
ρ(EU, AU) < 0.1          # Should work in both

# Epistemic quality (V4 improvement!)
ρ(EU, Error) > 0.3       # NEW: Should be much better in V4
```

### Training Behavior

**V3:**
- σ_epi varies randomly around prior
- Weak correlation with errors
- May need many epochs to learn anything

**V4:**
- σ_epi explicitly tracks prediction difficulty
- Clear correlation with errors from early epochs
- Faster convergence of uncertainty quality

## Model Initialization

V4 includes smart initialization:

```python
from maqa_credal_loss_v4 import initialize_model_for_v4

model = initialize_model_for_v4(model)
```

**What it does:**
- Initializes σ_epi head bias → ~0.3
- Initializes σ_ale head bias → ~0.5
- Better starting point for training

## Loss Components (V4)

The total loss is:

```
L = λ_answer × L_answer
  + λ_kl × L_kl
  + λ_ale_mse × L_ale_mse
  + λ_ale_rank × L_ale_rank
  + λ_epi_mse × L_epi_mse      ← NEW!
  + λ_epi_rank × L_epi_rank    ← NEW!
  + λ_variance × L_variance
  + λ_contrast × L_contrast
```

Where:
- `L_answer = KL(pred || p*)`
- `L_kl = KL(q(μ,σ) || p_prior)`
- `L_ale_mse = MSE(σ_ale, H[p*])`
- `L_ale_rank = RankingLoss(σ_ale, H[p*])`
- `L_epi_mse = MSE(σ_epi, scaled_pred_error)` ← NEW!
- `L_epi_rank = RankingLoss(σ_epi, pred_error)` ← NEW!

## Troubleshooting

### Issue: ρ(EU, Error) still low

**Solutions:**
1. Increase `lambda_epi_mse` (try 3.0 or 4.0)
2. Check that pred_error is being computed correctly
3. Verify `.detach()` is in place
4. Ensure σ_epi has enough capacity (hidden_dim)

### Issue: Answer accuracy drops

**Solutions:**
1. Reduce `lambda_epi_mse` (try 1.0)
2. Increase `lambda_answer` (try 2.0)
3. Check learning rate (may be too high)

### Issue: σ_epi collapses

**Solutions:**
1. Verify `lambda_kl = 0.0001` (should be low)
2. Check bounds: `min_sigma_epi`, `max_sigma_epi`
3. Ensure variance regularization is active

## Theory Behind V4

### The Problem with V3

In V3, σ_epi is only regularized by KL:
```
L_kl = KL(q(μ,σ) || p_prior)
```

This says "stay close to prior" but NOT "be high for hard examples".

### The V4 Solution

We define a self-supervised proxy for difficulty:
```
pred_error = 1 - Σ softmax(μ)[a] × p*[a]
```

Then explicitly supervise σ_epi:
```
L_epi = MSE(σ_epi, scale(pred_error))
```

Now σ_epi has a DIRECT signal to track prediction difficulty!

### Why This Creates Correlation

**In ensembles:**
- Hard examples → members disagree → high variance → high EU
- Easy examples → members agree → low variance → low EU

**In V4 variational:**
- Hard examples → low confidence → high pred_error → high σ_epi
- Easy examples → high confidence → low pred_error → low σ_epi

We CREATE the signal that ensembles get implicitly!

## Citation

If you use V4, please cite:

```bibtex
@misc{maqa_v4_2026,
  title={Clean Prediction Error Supervision for Epistemic Uncertainty},
  author={Tanmoy},
  year={2026},
  note={Explicit supervision creates EU-error correlation in variational methods}
}
```

## Summary

✅ **V4 is fully integrated** and ready to use
✅ **Just add `--loss_version v4`** to enable
✅ **Expected improvement:** ρ(EU, Error) from ~0.0 to >0.3
✅ **No breaking changes:** V3 remains default

**Status:** Production Ready
**Date:** January 2026
**Author:** Tanmoy

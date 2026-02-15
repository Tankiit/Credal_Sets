# MAQA V6 - Hard Decorrelation

## Problem with V5

Your training history shows V5 failed:
```
Epoch 1: ρ(EU, AU) = -0.02  ✓ (good start)
Epoch 6: ρ(EU, AU) = 0.92   ✗ (catastrophic!)
```

**Root cause:** Soft decorrelation (λ=1.0) was overwhelmed by other losses. Both σ_epi and σ_ale grew together over time.

## V6 Solution: HARD Decorrelation

V6 uses **three** mechanisms to enforce decorrelation:

### 1. **Very Strong Decorrelation Loss**
```python
lambda_decorr = 20.0  # Was 1.0 in V5 (20x stronger!)
loss = |correlation|²  # Squared for stronger gradient at high correlation
```

### 2. **Orthogonalization**
```python
# Remove the component of σ_epi that's correlated with σ_ale
σ_epi_orth = σ_epi - proj(σ_epi onto σ_ale)
```

This **architecturally** enforces decorrelation, not just through loss penalty.

### 3. **Residual Error Supervision**
```python
# Instead of supervising on pred_error, supervise on residual
residual = pred_error - E[pred_error | entropy]
```

This removes the entropy-correlated part from the epistemic target.

## Usage

```bash
# V6 (RECOMMENDED)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v6

# V5 (soft decorrelation - may fail)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v5
```

## Key Differences

| Aspect | V5 | V6 |
|--------|----|----|
| **λ_decorr** | 2.0 | **20.0** (10x stronger) |
| **Orthogonalization** | ❌ No | ✅ **Yes** |
| **Residual Error** | ❌ No | ✅ **Yes** |
| **Decorrelation** | Soft penalty | **Hard constraint** |
| **Expected ρ(EU, AU)** | ? | **< 0.3** |

## How V6 Works

### Step 1: Orthogonalize σ_epi
```python
σ_epi_orth = σ_epi - proj(σ_epi onto σ_ale)
```

This **removes** the correlated component from σ_epi before using it for any loss computation.

### Step 2: Residual Error Target
```python
# What part of pred_error is explained by entropy?
expected_error = E[pred_error | entropy]

# Residual = unexplained part (truly epistemic)
residual = pred_error - expected_error

# Supervise σ_epi on residual (not total error)
target_epi = scale(residual)
```

### Step 3: Strong Decorrelation Penalty
```python
loss_decorr = 20.0 × |correlation(σ_epi, σ_ale)|²
```

The squared penalty gives much stronger gradients when correlation is high.

## Expected Results

Based on the hard constraints:

| Metric | V5 (observed) | V6 (expected) |
|--------|--------------|---------------|
| ρ(EU, AU) | **0.92** ❌ | **< 0.3** ✅ |
| ρ(AU, Entropy) | ? | > 0.3 |
| ρ(EU, Error) | ? | > 0.2 |

V6 should maintain ρ(EU, AU) < 0.3 throughout training, not just at initialization.

## Monitoring During Training

Watch these metrics:

```python
# Should stay LOW throughout training
actual_eu_au_corr: 0.02, 0.05, 0.08, 0.10, ... (not 0.92!)

# After orthogonalization (should be ~0)
corr_after_orthogonalization: ~0.0
```

If `actual_eu_au_corr` starts climbing, the orthogonalization is working but the raw σs are still correlating. The loss on the orthogonalized version keeps the training stable.

## Why This Should Work

**V5 failure mode:**
- Both σ_epi → pred_error and σ_ale → entropy
- pred_error and entropy are correlated (ρ ≈ 0.5-0.7)
- Both losses push σs in same direction
- λ_decorr=2.0 too weak to stop this

**V6 solution:**
- Orthogonalize σ_epi → removes correlated component **before** loss
- Supervise on residual error → target itself is decorrelated
- λ_decorr=20.0 → very strong even if orthogonalization fails
- Multiple layers of protection against correlation

## Configuration

```python
CONFIG_V6 = {
    # Aleatoric (unchanged)
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 2.0,

    # Epistemic (slightly reduced to compensate)
    'lambda_epi_mse': 1.0,   # Was 2.0
    'lambda_epi_rank': 1.0,  # Was 2.0

    # DECORRELATION (KEY!)
    'lambda_decorr': 20.0,  # Was 2.0 (10x stronger)

    # Other
    'lambda_answer': 1.0,
    'lambda_kl': 0.0001,
    'lambda_variance': 0.5,
}
```

## Tuning

**If ρ(EU, AU) is still too high:**
```python
'lambda_decorr': 50.0,  # Even stronger
```

**If EU-Error or AU-Entropy correlations drop too much:**
```python
'lambda_epi_mse': 2.0,   # Increase epistemic supervision
'lambda_ale_mse': 2.0,   # Increase aleatoric supervision
'lambda_decorr': 10.0,   # Reduce decorrelation strength
```

## Summary

V6 = V4 + **HARD** decorrelation

- V4: Explicit supervision (good)
- V5: V4 + soft decorrelation (failed)
- **V6: V4 + hard decorrelation (should work)**

**Recommended for all MAQA training.**

---

**Status:** Ready to test
**Expected improvement:** ρ(EU, AU) stays < 0.3 throughout training
**Author:** Tanmoy
**Date:** January 2026

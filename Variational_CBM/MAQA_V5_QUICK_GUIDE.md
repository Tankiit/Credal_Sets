# MAQA V5 - Quick Guide

## TL;DR

V5 fixes V4's correlation problem by adding an explicit decorrelation loss.

## The Problem with V4

In V4, both σ_epi and σ_ale track **correlated signals**:
- σ_ale → entropy (H[p*])
- σ_epi → pred_error (1 - P(correct))
- **Problem:** entropy and pred_error are correlated! (ρ ≈ 0.5-0.7)
- **Result:** σ_epi and σ_ale become correlated → ρ(EU, AU) can be high

This violates Theorem 1 (EU and AU should be decorrelated).

## V5 Solution

Keep V4's supervision, **add explicit decorrelation penalty**:

```python
loss_decorr = |correlation(σ_epi, σ_ale)|
```

This enforces all three objectives:
1. ✅ ρ(EU, Error) > 0.2 (epistemic validity)
2. ✅ ρ(AU, Entropy) > 0.3 (aleatoric validity)
3. ✅ ρ(EU, AU) < 0.3 (decorrelation)

## Usage

```bash
# V5 (RECOMMENDED)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v5

# V4 (may have high EU-AU correlation)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v4

# V3 (baseline)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v3
```

## Key Config

```python
CONFIG_V5 = {
    # Same as V4
    'lambda_ale_mse': 1.0,
    'lambda_ale_rank': 3.0,
    'lambda_epi_mse': 2.0,
    'lambda_epi_rank': 2.0,

    # NEW: Decorrelation weight
    'lambda_decorr': 2.0,  # High to enforce ρ(EU, AU) → 0
}
```

## Expected Results

| Metric | V3 | V4 | V5 |
|--------|----|----|----|
| ρ(EU, Error) | ~0.0 | > 0.3 | > 0.2 |
| ρ(AU, Entropy) | > 0.4 | > 0.4 | > 0.3 |
| ρ(EU, AU) | < 0.1 | **?** | < 0.3 ✅ |

V4 might get:
- Good EU-Error correlation ✅
- Good AU-Entropy correlation ✅
- **But high EU-AU correlation** ❌ (because signals are correlated)

V5 consistently gets all three ✅

## Why V5 Works

The decorrelation loss directly penalizes correlation:

```python
def _decorrelation_loss(self, sigma_epi, sigma_ale):
    # Standardize
    epi_std = (sigma_epi - sigma_epi.mean()) / sigma_epi.std()
    ale_std = (sigma_ale - sigma_ale.mean()) / sigma_ale.std()

    # Correlation
    corr = (epi_std * ale_std).mean()

    # Penalize absolute correlation
    return corr.abs()
```

The optimizer learns to:
- Make σ_epi track pred_error (for epi_mse loss)
- Make σ_ale track entropy (for ale_mse loss)
- **Keep them decorrelated** (for decorr loss)

It finds a balance where all three objectives are satisfied!

## Tuning

**If ρ(EU, AU) is still too high:**
```python
'lambda_decorr': 3.0,  # Increase (was 2.0)
```

**If EU-Error or AU-Entropy correlations drop:**
```python
'lambda_decorr': 1.0,  # Decrease (was 2.0)
'lambda_epi_mse': 3.0,  # Increase epi supervision
'lambda_ale_mse': 2.0,  # Increase ale supervision
```

## Which Version to Use?

**Use V5 if:**
- You want all three validity conditions ✅
- You care about Theorem 1 (decorrelation)
- You want robust uncertainty decomposition

**Use V4 if:**
- You only care about EU-Error correlation
- High EU-AU correlation is acceptable
- You want simpler loss

**Use V3 if:**
- You want a baseline
- You don't need epistemic validity

## Summary

V5 = V4 + Decorrelation

It's the same as V4, but with an explicit penalty for correlation between EU and AU.

**Recommended:** Always use V5 for MAQA training.

---

**Date:** January 2026
**Status:** Production Ready ✅
**Author:** Tanmoy

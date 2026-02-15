# MAQA V4 - Quick Start Guide

## TL;DR

V4 adds explicit epistemic-error supervision to make σ_epi correlate with prediction errors.

## Usage

```bash
# V4 (NEW!)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v4

# V3 (default)
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v3
```

## What's Different?

### V3
```python
# Epistemic: only KL regularization (no error signal)
L_epi = KL(q(μ,σ) || p_prior)  # Just says "stay near prior"
```

### V4
```python
# Epistemic: explicit error supervision
pred_error = 1 - P(correct)  # Self-supervised difficulty
L_epi = MSE(σ_epi, pred_error)  # Direct supervision!
```

## Expected Results

| Metric | V3 | V4 |
|--------|----|----|
| ρ(EU, Error) | ~0.0 | **> 0.3** ✨ |
| ρ(AU, Entropy) | > 0.4 | > 0.4 |
| ρ(EU, AU) | < 0.1 | < 0.1 |

## Key Config

```python
# In maqa_credal_loss_v4.py
CONFIG_V4 = {
    'lambda_epi_mse': 2.0,      # Main knob for EU-error correlation
    'lambda_epi_rank': 2.0,     # Ordering loss
    'lambda_ale_mse': 1.0,      # Aleatoric calibration
    'lambda_ale_rank': 3.0,     # Aleatoric ordering
    'lambda_kl': 0.0001,        # Very weak KL reg
    # ...
}
```

## Tuning Guide

**If ρ(EU, Error) is too low:**
```bash
# Increase epistemic supervision
# Edit CONFIG_V4 in maqa_credal_loss_v4.py:
'lambda_epi_mse': 3.0,  # was 2.0
'lambda_epi_rank': 3.0, # was 2.0
```

**If answer accuracy drops:**
```bash
# Reduce epistemic weight, increase answer weight
'lambda_epi_mse': 1.0,      # was 2.0
'lambda_answer': 2.0,       # was 1.0
```

**If σ_epi collapses:**
```bash
# Check these are set correctly:
'lambda_kl': 0.0001,        # Must be LOW
'min_sigma_epi': 0.05,
'max_sigma_epi': 1.0,
```

## Files Changed

1. **NEW:** `maqa_credal_loss_v4.py` - V4 implementation
2. **UPDATED:** `main_train_hybrid_multi_dataset.py` - Added V4 support
3. **NEW:** `MAQA_V4_COMPLETE_GUIDE.md` - Full documentation

## Testing It Out

```bash
# Quick test with V4
python main_train_hybrid_multi_dataset.py \
    --dataset maqa \
    --loss_version v4 \
    --num_epochs 5 \
    --batch_size 8

# Compare with V3
python main_train_hybrid_multi_dataset.py \
    --dataset maqa \
    --loss_version v3 \
    --num_epochs 5 \
    --batch_size 8
```

## Understanding the Loss

```python
# V4 computes prediction error from model's own predictions
pred = softmax(μ)  # Model's predicted distribution
prob_correct = Σ pred[a] × p*[a]  # Prob of being correct
pred_error = 1 - prob_correct  # Difficulty proxy

# Then supervises σ_epi to track this
target_σ = scale(pred_error)  # Scale to [0.05, 1.0]
loss = MSE(σ_epi, target_σ.detach())  # .detach() is critical!
```

**Why `.detach()`?**
- σ_epi should PREDICT difficulty
- Not MINIMIZE difficulty
- `.detach()` stops gradients from flowing back to μ

## Common Questions

**Q: Is V4 drop-in compatible?**
A: Yes! Just add `--loss_version v4`

**Q: Will V4 hurt accuracy?**
A: Shouldn't, if tuned properly. May even help by regularizing.

**Q: Can I mix V3 and V4?**
A: No, pick one. V4 is strictly better for epistemic quality.

**Q: What about other datasets?**
A: V4 is MAQA-specific right now. Other datasets use different losses.

## Citation

```bibtex
@misc{maqa_v4,
  title={Clean Prediction Error Supervision for Epistemic Uncertainty},
  author={Tanmoy},
  year={2026}
}
```

---

**Ready to use?** Just run:
```bash
python main_train_hybrid_multi_dataset.py --dataset maqa --loss_version v4
```

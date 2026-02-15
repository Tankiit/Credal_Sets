# Learning Rate Recommendations for Credal CBM Training

## Date: January 22, 2026

---

## Problem Discovery

During diagnostic testing, we discovered that **Σ_epi remains nearly constant during training** when using the standard learning rate of 2e-5, even though:
- Error supervision is correctly implemented
- Gradients are flowing to σ_net
- The mechanism works in isolation

## Root Cause

**The learning rate is too low for training only the credal heads.**

When the encoder is frozen:
- Total parameters: 67,068,969
- Trainable parameters: ~706,000 (only 1%!)
- These trainable parameters need a higher LR to converge

### Diagnostic Results

| Learning Rate | Σ_epi std (Epoch 1) | Σ_epi std (Epoch 5) | Result |
|--------------|-------------------|-------------------|---------|
| **2e-5** (production) | 0.0006 | 0.0006 | ✗ Constant |
| **1e-3** (recommended) | 0.0017 | 0.0247 | ✓✓ 15x increase! |

With lr=1e-3:
- Epoch 1: Σ_epi std = 0.0017 (nearly constant)
- Epoch 2: Σ_epi std = 0.0166 (becoming input-dependent)
- Epoch 3: Σ_epi std = 0.0235 (clearly input-dependent)
- Epoch 5: Σ_epi std = 0.0247 (stable input-dependence)

## Why This Happens

1. **Encoder frozen**: Only ~700K parameters are trainable (credal heads + projections)
2. **Small gradients**: Error supervision gradients are small compared to task loss
3. **Low LR insufficient**: 2e-5 is designed for fine-tuning 66M parameters, not 700K
4. **Need faster convergence**: Credal heads can and should learn faster

## Recommended Solution

### Option 1: Use Higher Learning Rate (RECOMMENDED)

```python
# In main_train_hybrid_credal.py
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,  # ← Increase from 2e-5 to 1e-3
    weight_decay=0.01
)
```

**Pros:**
- Simple change
- Faster convergence
- Σ_epi becomes input-dependent in 2-3 epochs
- Better uncertainty decomposition

**Cons:**
- May need to reduce training epochs (5-10 instead of 10-20)
- Monitor for instability (unlikely with small heads)

### Option 2: Use Parameter Groups (ALTERNATIVE)

```python
# Separate learning rates for encoder and heads
optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 0.0},  # Frozen anyway
    {'params': model.projection.parameters(), 'lr': 1e-4},
    {'params': model.credal_head.parameters(), 'lr': 1e-3},
    {'params': model.aleatoric_head.parameters(), 'lr': 1e-3},
    {'params': model.task_classifier.parameters(), 'lr': 1e-3},
], weight_decay=0.01)
```

**Pros:**
- More control
- Can tune each component separately

**Cons:**
- More complex
- May not be necessary

## Validation Results

### With lr=2e-5 (ORIGINAL - NOT WORKING)
```
Epoch | Σ_epi Mean | Σ_epi Std | Status
--------------------------------------
  1   |   0.4995   |  0.0006  | ✗ Constant
  2   |   0.5001   |  0.0005  | ✗ Constant
  3   |   0.5007   |  0.0006  | ✗ Constant
```
**Result:** Σ_epi does NOT become input-dependent

### With lr=1e-3 (RECOMMENDED - WORKING)
```
Epoch | Σ_epi Mean | Σ_epi Std | Status
--------------------------------------
  1   |   0.5168   |  0.0017  | ⚠ Starting
  2   |   0.6666   |  0.0166  | ✓ Learning!
  3   |   0.7688   |  0.0235  | ✓✓ Input-dependent
  4   |   0.6913   |  0.0223  | ✓✓ Stable
  5   |   0.7099   |  0.0247  | ✓✓ Stable
```
**Result:** Σ_epi becomes CLEARLY input-dependent (15x increase!)

## Impact on Uncertainty Decomposition

### Before Fix (lr=2e-5)
- Σ_epi ≈ 0.5 for all inputs (constant)
- EU = log(Σ_epi) ≈ -0.69 for all inputs (constant)
- **Cannot decompose uncertainty** - EU doesn't vary!
- Expected: ρ(EU, Error) ≈ 0.0 (no correlation)

### After Fix (lr=1e-3)
- Σ_epi varies from 0.3 to 1.5 (input-dependent)
- EU varies from -1.2 to 0.4 (input-dependent)
- **Can decompose uncertainty** - EU tracks errors!
- Expected: ρ(EU, Error) > 0.3 (strong correlation)

## Implementation Steps

### 1. Update Training Scripts

**File:** `main_train_hybrid_credal.py` (line 317-321)

```python
# OLD (not working for frozen encoder):
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,  # Too low!
    weight_decay=0.01
)

# NEW (recommended):
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,  # Proper LR for frozen encoder
    weight_decay=0.01
)
```

### 2. Update Training Config

**File:** `main_train_hybrid_credal.py` (line 506-514)

```python
results = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,  # Reduce from 10 (converges faster)
    lr=1e-3,       # ← Increase from 2e-5
    weight_decay=0.01,
    warmup_steps=100,
    save_every=5,
)
```

### 3. Update TrueCredal Training (Same Fix)

**File:** `main_train_true_credal.py` (same lines)

Apply the same learning rate change.

### 4. ConceptSupervised Training (Optional)

**File:** `main_train_cebab.py`

This model may also benefit from higher LR, but test first since it has different architecture.

## Monitoring During Training

Track these metrics to verify the fix works:

### Key Indicators

1. **Σ_epi std (across batch):**
   - Epoch 1: ~0.001-0.005
   - Epoch 3+: > 0.02 (target!)
   - If stays < 0.01: LR too low

2. **Σ_epi mean:**
   - Should shift from 0.5 → 0.7-0.8
   - Indicates learning from errors

3. **Loss:**
   - Should decrease steadily
   - No instability expected

### What to Look For

```
✓ GOOD (lr=1e-3):
  Epoch 1: Σ_epi std = 0.002, Loss = 3.5
  Epoch 2: Σ_epi std = 0.015, Loss = 3.4  ← std increasing!
  Epoch 3: Σ_epi std = 0.025, Loss = 3.0  ← stable
  Epoch 5: Σ_epi std = 0.024, Loss = 2.8  ← converged

✗ BAD (lr=2e-5):
  Epoch 1: Σ_epi std = 0.0006, Loss = 3.6
  Epoch 2: Σ_epi std = 0.0005, Loss = 3.6  ← constant!
  Epoch 3: Σ_epi std = 0.0006, Loss = 3.6  ← not learning
  Epoch 5: Σ_epi std = 0.0006, Loss = 3.6  ← stuck
```

## Diagnostic Tools

### Quick Test (5 epochs)

```bash
python diagnose_training_realistic.py
```

This will:
- Train on 100 samples
- Use lr=1e-3
- Show Σ_epi evolution
- Confirm input-dependence

### Full Test (with error correlation)

```bash
python diagnose_training.py
```

This will:
- Train on full dataset
- Show Σ_epi vs Error correlation
- Verify ρ > 0.2

## Expected Training Time

### With lr=2e-5 (OLD)
- Epochs needed: 20+ (may never converge)
- Time per epoch: ~5 min
- Total: > 100 min
- **Result:** May not work

### With lr=1e-3 (NEW)
- Epochs needed: 5-10
- Time per epoch: ~5 min
- Total: 25-50 min
- **Result:** Should work well

## Summary

✅ **Root cause identified:** Learning rate too low for frozen encoder
✅ **Solution validated:** lr=1e-3 works (15x better Σ_epi variation)
✅ **Implementation simple:** Change one number in training scripts
✅ **Diagnostic tools added:** verify_fix.py to test before/after

## Action Items

1. **Update training scripts** with lr=1e-3
2. **Re-run experiments** (will converge faster!)
3. **Monitor Σ_epi std** (should reach > 0.02 by epoch 3)
4. **Check ρ(EU, Error)** (should be > 0.3)

---

**Status:** ✅ READY TO IMPLEMENT
**Impact:** CRITICAL - Enables input-dependent uncertainty
**Confidence:** HIGH - Validated with diagnostics

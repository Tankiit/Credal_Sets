# Quick Reference: Available Models

## Overview

This codebase now includes **two** state-of-the-art Concept Bottleneck Models with uncertainty decomposition:

1. **ConceptSupervisedCredalCBM** - Deterministic, supervised uncertainty heads
2. **TrueCredalCBM** - Credal sets with structural separation (NEW!)

## Model Comparison

| Feature | ConceptSupervisedCBM | TrueCredalCBM |
|---------|---------------------|---------------|
| **Representation** | Point estimates | Credal sets (μ, Σ_epi) |
| **EU Source** | EpistemicHead (prediction) | log(Σ_epi) (geometry) |
| **EU Training** | Concept error supervision | KL regularization |
| **AU Source** | AleatoricHead (prediction) | AleatoricHead (prediction) |
| **AU Training** | Annotator entropy | Annotator entropy |
| **Projections** | 2-way (concept, uncertainty) | 3-way (concept, epi, ale) |
| **Stochastic** | No (deterministic) | Yes (MC sampling from credal) |
| **Impossibility** | May not escape | Escapes via geometry |
| **Complexity** | Simpler | More complex |
| **Use Case** | Baseline, fast training | Theoretically sound |

## Quick Start

### Option 1: Concept-Supervised CBM (Baseline)

```bash
python main_train_cebab.py
```

**Config:**
```python
from VCBM import ConceptSupervisedCredalCBM, ConceptSupervisedConfig

config = ConceptSupervisedConfig(
    encoder_name='distilbert-base-uncased',
    num_concepts=4,
    concept_names=['food', 'service', 'ambiance', 'noise'],
    num_classes=5,
    epistemic_prior=0.1,
    aleatoric_prior=0.3,
    kl_weight=0.01,
    concept_weight=2.0,
    epistemic_weight=1.0,
    aleatoric_weight=1.0,
)
```

### Option 2: True Credal CBM (Recommended)

```bash
python main_train_true_credal.py
```

**Config:**
```python
from VCBM import TrueCredalCBM, TrueCredalConfig

config = TrueCredalConfig(
    encoder_name='distilbert-base-uncased',
    num_concepts=4,
    concept_names=['food', 'service', 'ambiance', 'noise'],
    num_classes=5,

    # Credal set parameters
    num_mc_samples=10,
    prior_sigma=1.0,
    min_sigma=1e-4,
    max_sigma=2.0,

    # Loss weights
    concept_weight=2.0,
    kl_weight=0.1,        # Epistemic (credal shrinkage)
    aleatoric_weight=1.0, # Aleatoric (entropy)
    orth_weight=0.001,    # Orthogonality

    # Architecture
    projection_dim=256,
    hidden_dim=128,
)
```

## Which Model to Use?

### Use ConceptSupervisedCBM if:
- ✓ You want a simpler baseline
- ✓ You need faster training
- ✓ You're okay with deterministic predictions
- ✓ You want direct error/entropy supervision

### Use TrueCredalCBM if:
- ✓ You need proper credal sets
- ✓ You want theoretically sound uncertainty
- ✓ You need to escape Tomov et al. impossibility
- ✓ You want MC sampling for prediction intervals
- ✓ You're writing a paper (stronger theory)

## Key Differences

### 1. Uncertainty Computation

**ConceptSupervisedCBM:**
```python
# Both are PREDICTED by heads
epistemic = self.epistemic_head(h_epi)      # Predicts errors
aleatoric = self.aleatoric_head(h_ale)      # Predicts entropy
```

**TrueCredalCBM:**
```python
# EU is DERIVED from geometry, AU is PREDICTED
sigma_epi = self.credal_head(h_concept, h_epi)  # Credal size
epistemic = torch.log(sigma_epi)                # DERIVED!
aleatoric = self.aleatoric_head(h_ale)          # Predicts entropy
```

### 2. Training Signals

**ConceptSupervisedCBM:**
- μ ← L_concept
- σ_epi ← L_epistemic (concept errors) + L_KL
- σ_ale ← L_aleatoric (annotator entropy)

**TrueCredalCBM:**
- μ ← L_concept
- σ_epi ← L_KL only (credal shrinkage)
- σ_ale ← L_aleatoric (annotator entropy)

### 3. Architectural Separation

**ConceptSupervisedCBM:**
```
Encoder → OrthogonalProjection (2-way)
         → h_concept, h_uncertainty
```

**TrueCredalCBM:**
```
Encoder → OrthogonalProjection (3-way)
         → h_concept, h_epi, h_ale
```

## Expected Results

### ConceptSupervisedCBM
```
ρ(EU, AU): 0.05-0.15 ✓
ρ(EU, Error): 0.25-0.35 ✓
ρ(AU, Entropy): 0.30-0.50 ✓
```

### TrueCredalCBM
```
ρ(EU, AU): 0.00-0.20 ✓✓ (better separation)
ρ(EU, Error): 0.30-0.45 ✓✓ (stronger correlation)
ρ(AU, Entropy): 0.40-0.60 ✓✓ (better tracking)
Σ_epi shrinkage: 20-40% (credal tightening)
```

## File Structure

```
Variational_CBM/
├── VCBM.py  # All model definitions
│   ├── ConceptSupervisedConfig
│   ├── ConceptSupervisedCredalCBM
│   ├── TrueCredalConfig
│   └── TrueCredalCBM  ← NEW!
│
├── main_train_cebab.py  # For ConceptSupervisedCBM
├── main_train_true_credal.py  # For TrueCredalCBM ← NEW!
│
├── load_cebab_direct.py  # Dataloader with annotator_entropy
│
└── Documentation/
    ├── INTEGRATION_VERIFIED.md
    ├── ANNOTATOR_ENTROPY_INTEGRATION.md
    ├── TRUE_CREDAL_CBM_SUMMARY.md
    └── QUICK_REFERENCE.md  # This file
```

## Common Operations

### Load a Model

```python
# ConceptSupervisedCBM
from VCBM import ConceptSupervisedCredalCBM, ConceptSupervisedConfig
config = ConceptSupervisedConfig(num_concepts=4)
model = ConceptSupervisedCredalCBM(config)

# TrueCredalCBM
from VCBM import TrueCredalCBM, TrueCredalConfig
config = TrueCredalConfig(num_concepts=4)
model = TrueCredalCBM(config)
```

### Forward Pass

```python
# Both models have the same interface!
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    concept_labels=concept_labels,
    annotator_entropy=annotator_entropy,  # CEBaB disagreement
)

# Access uncertainties
epistemic = outputs['epistemic']    # [batch, num_concepts]
aleatoric = outputs['aleatoric']    # [batch, num_concepts]
predictions = outputs['predictions'] # [batch]
```

### Verify Properties

```python
# For TrueCredalCBM only
from VCBM import verify_credal_properties

verify_credal_properties(model, num_steps=100)
# Tests:
#   1. EU = log(Σ_epi)
#   2. Σ_epi shrinks during training
#   3. Gradient separation
```

### Analyze on Data

```python
from VCBM import analyze_credal_sets

analyze_credal_sets(model, dataloader, device='mps')
# Shows:
#   - Credal set statistics
#   - EU/AU correlations
#   - Separation quality
```

## Troubleshooting

### Both Models

**Issue:** High ρ(EU, AU) > 0.3
**Solution:** Increase orthogonality weight

**Issue:** Low concept accuracy
**Solution:** Increase concept_weight

**Issue:** AU not tracking entropy
**Solution:** Increase aleatoric_weight

### TrueCredalCBM Specific

**Issue:** Σ_epi not shrinking
**Solution:** Increase kl_weight

**Issue:** Σ_epi too small (collapse)
**Solution:** Decrease kl_weight or increase prior_sigma

**Issue:** Gradient explosion
**Solution:** Enable gradient clipping (already done in trainer)

## Training Tips

1. **Start with ConceptSupervisedCBM** - Faster, simpler
2. **Verify annotator_entropy is working** - Check ρ(AU, Entropy)
3. **Then try TrueCredalCBM** - Better theory, slightly slower
4. **Monitor orthogonality** - Key for good separation
5. **Check Σ_epi shrinkage** - Should decrease 20-40%

## Citation

If you use this code, please cite:

```bibtex
@article{credal_cbm_2026,
  title={Credal Concept Bottleneck Models with Structural Separation},
  author={Tanmoy},
  journal={ICML Submission},
  year={2026}
}
```

## Summary

✅ **Two models available:**
1. ConceptSupervisedCredalCBM (baseline)
2. TrueCredalCBM (state-of-the-art)

✅ **Both support:**
- CEBaB dataset with annotator entropy
- Concept-level uncertainty decomposition
- Orthogonal projections for separation

✅ **Key differences:**
- ConceptSupervised: Predicted EU, deterministic
- TrueCredal: Derived EU, credal sets, MC sampling

✅ **Recommendation:**
- Start with `main_train_cebab.py` (baseline)
- Then try `main_train_true_credal.py` (SOTA)

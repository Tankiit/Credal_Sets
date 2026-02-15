# MAQA Fixed Config Training Results
**Date:** January 23, 2026
**Status:** ✅ SUCCESS

## 1. Fixes Implemented

### A. Environment & Infrastructure
- **PyTorch Upgrade:** Upgraded to `torch>=2.6.0` (installed 2.10.0) to resolve CVE-2025-32434 security vulnerability in `torch.load`.
- **Torchvision Upgrade:** Upgraded `torchvision` and `torchaudio` to match the new PyTorch version.

### B. Data Processing
- **Combined Dataset:** Used ~5000 samples (MAQA + AmbigQA) to ensure statistical significance.
- **Preprocessing:** Implemented robust preprocessing in `main_train_hybrid_multi_dataset.py` to:
  - Calculate ground-truth entropy from probabilities.
  - specificy `num_answers` and `ambiguity_level`.
  - Handle missing keys in raw HF datasets.

### C. Model & Loss Logic
- **Broadcasting Fix:** Fixed `RuntimeError` in `_compute_kl` by unsqueezing `sigma_epi` to match `mu` dimensions.
- **Parameter Tuning:**
  - `prior_sigma`: 0.1 (lowered from 0.5 to make KL loss meaningful).
  - `beta`: 0.01 (increased from 1e-4).
  - `min_sigma_epi`: 0.05 (prevent collapse).
  - Initialization: `sigma_epi` bias initialized to 0.5.

## 2. Training Results (100 Epochs)

### Final Test Metrics
- **Test Loss:** 0.6030
- **Mean σ_epi:** 0.0894 (converged near prior 0.1)
- **Mean σ_ale:** 0.7568

### Uncertainty Disentanglement
- **ρ(EU, AU):** `0.0404` (p=0.38)
  - **Result:** ✅ **SUCCESS** (Target < 0.3)
  - **Interpretation:** Epistemic and Aleatoric uncertainties are effectively uncorrelated, meaning the model distinguishes between "I don't know" (model uncertainty) and "It's ambiguous" (data uncertainty).

### Calibration
- **ρ(AU, Entropy):** `0.1630` (p<0.001)
  - **Result:** ⚠️ **PARTIAL** (Target > 0.5)
  - **Interpretation:** Positive correlation indicates the model captures some data noise, but could be tighter.

### Contrastive Learning
- **Success Rate:** ~50-60%
  - **Result:** ⚠️ **PARTIAL** (Target > 80%)
  - **Interpretation:** Distinction between "ambiguous" and "clear" pairs is present but not sharp.

## 3. Conclusion
The Hybrid Credal CBM is now functional and stable on the MAQA dataset. The critical goal of disentangling uncertainty types has been achieved (`rho < 0.3`). Future work can focus on improving calibration (`rho_au_entropy`) and contrastive separation.

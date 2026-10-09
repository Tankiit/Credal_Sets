# 07. Companion Research Modules & Extensions

This document provides an overview of complementary research modules present in the repository outside of the main concept reparameterization framework.

---

## 📂 Overview of Additional Research Modules

```
Credal_Sets/
├── Optimization/               # Distributionally Robust Optimization (DRO) & Credal Sets
├── Random_Sets/                # Random Sets Uncertainty Quantification (RSUQ) & Belief Functions
├── Patch_Uncertainty/          # Possibilistic Attention & GradCAM for Vision Transformers (DINO)
└── Multi_Sets/                 # Multi-set Credal Extensions & Soft Concept Ensembles
```

---

## 1. `Optimization/` (DRO & Credal Optimization)
- **Files**: [`Optimization/main.py`](file:///Users/cril/tanmoy/research/Credal_Sets/Optimization/main.py), [`Optimization/models.py`](file:///Users/cril/tanmoy/research/Credal_Sets/Optimization/models.py)
- **Focus**: Distributionally Robust Optimization (DRO) over credal probability sets.
- **Key Functionality**: Computes worst-case expected loss over credal sets of probability distributions subject to divergence constraints (Wasserstein, KL divergence, or $f$-divergence bounds).

---

## 2. `Random_Sets/` (Random Set Uncertainty Quantification - RSUQ)
- **Files**: [`Random_Sets/rsuq/`](file:///Users/cril/tanmoy/research/Credal_Sets/Random_Sets/rsuq), `Random_Sets/d2d-belief/`
- **Focus**: Implements random sets theory, Dempster-Shafer belief functions, and uncertainty quantification under epistemic ambiguity.
- **Package**: `rsuq` (packaged in `rsuq_package.zip`).

---

## 3. `Patch_Uncertainty/` (Possibilistic Attention & Patch Uncertainty)
- **Files**: [`Patch_Uncertainty/attention_possibilistic_gradcam_hf.py`](file:///Users/cril/tanmoy/research/Credal_Sets/Patch_Uncertainty/attention_possibilistic_gradcam_hf.py), `credence_dino_sanity.ipynb`
- **Focus**: Computes possibilistic Grad-CAM attention maps for Vision Transformers (DINO/DINOv2) to measure spatial uncertainty in image patches.

---

## 4. `Multi_Sets/` (Multi-Set Credal Extensions)
- **Files**: `Multi_Sets/arr-sce/`
- **Focus**: Multi-set credal extensions and soft concept ensemble models for robust prediction under multi-annotator concept ambiguity.

# 06. Codebase Map & API Reference

This document maps all files, modules, classes, and function signatures in the `Credal_Sets` repository.

---

## 📁 Repository Directory Structure

```
Credal_Sets/
├── download.py                  # CLI entrypoint for downloading & standardizing datasets
├── train.py                     # CLI entrypoint for training CBM / CEM models
├── train_concept_models.py      # Auditable trainer for CBM, residual-CBM, and CEM with trace export
├── twin.py                      # CLI entrypoint for building & verifying twin models
├── sweep.py                     # CLI entrypoint for parameter sweep across twin models
├── twin_seeds.py                # CLI entrypoint for multi-seed twin sampling
├── gauge_vs_seed.py             # CLI entrypoint for gauge vs seed realism evaluation
├── probe_residual.py            # CLI entrypoint for residual probing & leakage audit
├── aggregate.py                 # CLI entrypoint for summarizing sweep results & leakage test
├── explain.py                   # CLI entrypoint for concept push attribution explanations
├── requirements.txt             # Core Python package dependencies
├── concept_datasets/            # Dataset downloading and standardization module
│   ├── __init__.py              # Dataset package init & registry
│   ├── _common.py               # Common JSONL schema & metadata handler
│   ├── cebab.py                 # CEBaB dataset downloader
│   ├── civil_comments.py        # Civil Comments dataset downloader
│   ├── goemotions.py            # GoEmotions dataset downloader
│   └── imdb_cad.py              # IMDB-CAD dataset downloader
├── concept_models/              # Core neural network & reparameterization library
│   ├── __init__.py              # Package init
│   ├── models.py                # ConceptModel base, CBM, CEM & factory functions
│   ├── reparam.py               # invisible_map, weight folding, make_twin
│   ├── features.py              # Text feature extraction & disk caching
│   └── training.py              # Joint loss function, train loop & evaluation
└── wiki/                        # LLM Wiki Vault documentation
```

---

## 🔌 API Module Breakdown

### 1. `concept_models/models.py`
- [`ConceptModel(nn.Module)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L51): Abstract base class for concept bottleneck architectures.
- [`CBM(ConceptModel)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L71): Concept Bottleneck Model implementation.
  - `calibrate_interventions(x, concepts)`: Sets `logit_on` and `logit_off` to training medians.
  - `readout_blocks()`: Returns list containing `(slice, R)`.
  - `fold_concept_layer(B, t)`: Folds reparameterization matrix into linear concept layer.
- [`CEM(ConceptModel)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L122): Concept Embedding Model implementation.
  - `embeddings(x)`: Computes active $c^+$ and inactive $c^-$ embeddings.
  - `readout_blocks()`: Returns $k$ per-concept readout blocks $[s^+ ; s^-]$.
  - `fold_concept_layer(B, t)`: Folds reparameterization matrices into generator weights `gen_w` and bias `gen_b`.
- [`build_model(config)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L166): Model instantiation factory.
- [`load_model(path, device="cpu")`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L174): Deserialization helper.

---

### 2. `concept_models/reparam.py`
- [`invisible_map(R, rotate=1.0, mix=1.0, shift=0.0, generator=None, tol=1e-8)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/reparam.py#L28): Solves for affine transformation $(B, t)$ preserving readout $R$.
- [`fold_head(model, B, t)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/reparam.py#L52): Updates downstream head weights $\tilde{W} = W B^{-1}$ and bias $\tilde{b} = b - W B^{-1} t$.
- [`make_twin(model, rotate=1.0, mix=1.0, shift=0.0, seed=0)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/reparam.py#L60): Creates deep copy of model and applies reparameterization.

---

### 3. `concept_models/features.py`
- [`get_features(dataset, split, encoder_name="sentence-transformers/all-mpnet-base-v2", cache_dir="cache")`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/features.py): Extracts dense embeddings using SentenceTransformer and caches them in PyTorch format.

---

### 4. `concept_models/training.py`
- [`train_epoch(model, dataloader, optimizer, concept_weight=5.0, device="cpu")`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/training.py): Runs one epoch of joint task Cross-Entropy + concept BCE training.
- [`evaluate(model, dataloader, device="cpu")`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/training.py): Evaluates task accuracy, macro F1, and mean concept ROC-AUC.
- [`evaluate_interventions(model, dataloader, device="cpu")`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/training.py): Evaluates task accuracy when all ground-truth concept labels are intervened.

---

### 5. `concept_datasets/_common.py`
- [`DatasetMeta`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_datasets/_common.py): Dataclass holding dataset metadata (`dataset`, `class_names`, `concept_names`, `sizes`).
- [`save_dataset(dataset_name, splits, meta)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_datasets/_common.py): Serializes JSONL splits and `meta.json`.
- [`load_split(dataset_name, split)`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_datasets/_common.py): Loads raw JSONL records.

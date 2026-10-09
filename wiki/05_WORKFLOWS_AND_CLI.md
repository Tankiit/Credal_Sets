# 05. CLI References & Execution Workflows

This document outlines command-line scripts, training recipes, evaluation parameters, and reference benchmarks.

---

## 1. CLI Reference Guide

### 📥 1. Dataset Downloading (`download.py`)
Downloads raw datasets from HuggingFace, standardizes schemas, and writes `data/<dataset>/{train,val,test}.jsonl + meta.json`.

```bash
# Download all 4 datasets
python download.py all

# Download specific dataset
python download.py cebab
python download.py goemotions
python download.py civil_comments
python download.py imdb_cad
```

---

### 🏋️ 2. Auditable Concept Model Training (`train_concept_models.py`)
Trains auditable CBM, residual-CBM, and CEM architectures on cached features with explicit trace export and data contract validation.

```bash
# Generate synthetic demo dataset
python train_concept_models.py demo --out demo_data

# Train CBM, residual-CBM, or CEM architecture
python train_concept_models.py train --data demo_data --arch cbm --out runs/cbm
python train_concept_models.py train --data demo_data --arch cem --out runs/cem

# Export batch trace embeddings for audit
python train_concept_models.py export --data demo_data --checkpoint runs/cem/best.pt --out traces/cem

# Run standalone self-test validation
python train_concept_models.py self-test
```

#### Key Arguments
- `--dataset`: `cebab`, `goemotions`, `civil_comments`, `imdb_cad`.
- `--model`: `cbm` or `cem`.
- `--residual_dim`: Residual dimension $r$ for CBM (default `0`).
- `--emb_dim`: Embedding dimension $m$ per concept for CEM (default `16`).
- `--head_input`: `probs` or `logits` (CBM only).
- `--head`: `linear` (default) or `mlp`.
- `--concept_weight`: Loss weight for concept BCE (default `5.0`).
- `--p_int`: RandInt intervention probability during CEM training (default `0.25`).
- `--encoder`: Pretrained Transformer model name (default `sentence-transformers/all-mpnet-base-v2`).

---

### ♊ 3. Twin Model Generation & Diagnostics (`twin.py`)
Constructs an invisible reparameterization twin of a trained model and evaluates invariance.

```bash
# Construct twin model for a run directory
python twin.py runs/cebab-cem-s0 --rotate 2.0 --mix 1.0 --shift 1.0 --seed 3

# Output saved to:
#   runs/<run_name>/twins/<tag>.pt
#   runs/<run_name>/twins/<tag>_report.json
```

---

### 📊 4. Full Tier-0 Parameter Sweep (`sweep.py`)
Executes a multi-parameter sweep over rotation, mixing, shift, and twin seeds across all trained models in `runs/`.

```bash
# Run full sweep across trained runs
python sweep.py --out results/tier0.jsonl

# Aggregate results into summary table and leakage test
python aggregate.py --jsonl results/tier0.jsonl
```

---

### 🌱 5. Twin Seeds Sampling Sweep (`twin_seeds.py`)
Evaluates twin reparameterizations across multiple twin random seeds ($n=20$) with fixed $\text{rot}=1.0, \text{shift}=0.0$.

```bash
python twin_seeds.py --out results/twin_seeds.jsonl --n_tseeds 20
```

---

### 📏 6. Gauge vs. Seed Realism Scale (`gauge_vs_seed.py`)
Compares metric alignment (CKA, SVCCA/CCA, kNN) under twin reparameterizations vs. seed-to-seed variance across independently trained models.

```bash
python gauge_vs_seed.py --run runs/cebab-cbm-r16-logits-s0
```

---

### 🔍 5. Model Explanations & Attribution (`explain.py`)
Decomposes task predictions into additive per-concept concept pushes.

```bash
# Interactively explain 5 random validation samples from a CEM run
python explain.py runs/cebab-cem-s0 --n 5 --top 6

# Explain specific sample indices on test split
python explain.py runs/cebab-cbm-r16-logits-s0 --split test --ids 0 1 2 3
```

---

## 2. Benchmark Reference Results (Test Split, Seed 0)

| Run Name | Task Acc | Concept AUC | Acc (All Intervened) | Invisible Dims | Max $\Delta \text{logit}$ (Twin) | Twin $z$ Rel. Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cebab-cbm` | 0.530 | 0.896 | 0.516 | 0 | 0 | 0 |
| `cebab-cbm-r16` | 0.658 | 0.897 | 0.660 | 16 | $8 \times 10^{-6}$ | 0.58 |
| `cebab-cem` | 0.649 | 0.897 | 0.680 | 112 | $9 \times 10^{-6}$ | 1.81 |
| `goemotions-cbm` | 0.630 | 0.880 | 0.915 | 0 | 0 | 0 |
| `goemotions-cem` | 0.623 | 0.874 | 0.976 | 392 | $1 \times 10^{-5}$ | 2.39 |
| `civil_comments-cbm` | 0.785 | 0.850 | 0.932 | 0 | 0 | 0 |
| `civil_comments-cem` | 0.785 | 0.851 | 0.933 | 84 | $4 \times 10^{-6}$ | 2.76 |
| `imdb_cad-cbm` | 0.892 | 0.865 | 0.914 | 0 | 0 | 0 |
| `imdb_cad-cem` | 0.874 | 0.855 | 0.926 | 224 | $6 \times 10^{-6}$ | 2.42 |

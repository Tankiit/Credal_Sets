# Readout-invisible reparameterization of concept models

A concept model compares its concept layer `z` to the concept labels only through a readout
`R`. Anything `R` cannot see is not pinned down by training. This repo:

1. downloads 4 text datasets with concept annotations into one common format,
2. trains **CBM** and **CEM** models on them, plus three models built from the
   [PyC](https://pytorch-concepts.readthedocs.io/en/latest/guides/using_low_level.html)
   low-level API (`pyc-cbm`, `pyc-cem`, `pyc-hyper`),
3. builds the **twin** of a trained model: `z` moved along directions `R` ignores, head
   compensated. The task logits and concept probabilities stay identical (up to float
   precision) while the internals change. The twin is then ready for any diagnostic.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # pick the build for your GPU
pip install -r requirements.txt
```

## Quick start

```bash
python download.py all                         # -> data/<dataset>/{train,val,test}.jsonl + meta.json
python train.py --dataset cebab --model cem    # -> runs/cebab-cem-lr0.001-p8-s0/{model.pt,metrics.json}
python twin.py runs/cebab-cem-lr0.001-p8-s0    # -> runs/cebab-cem-lr0.001-p8-s0/twins/<tag>.pt + report.json
```

The first `train.py` call on a dataset embeds the texts with a frozen encoder and caches
the features in `cache/`. That takes a few minutes; every later run takes seconds.

To see how a model reasons on individual examples:

```bash
python explain.py runs/cebab-cbm-s0            # -> per-concept breakdown of 10 validation examples
```

For the PyC models, the concept-level read-out (CaCE per concept, concept -> answer weights,
NCC, intervention curves by policy), all computed with PyC tools:

```bash
python train.py --dataset cebab --model pyc-cem
python semantics.py runs/cebab-pyc-cem-lr0.001-p8-s0   # -> runs/<run>/semantics-test.json
```

How to read the outputs: [runs/README.md](runs/README.md) (scores and twin reports) and
[explanations/README.md](explanations/README.md) (per-example explanations).

## Layout

```
download.py                 CLI: download + standardize datasets
train.py                    CLI: train a CBM / CEM / PyC model
twin.py                     CLI: build + verify the twin of a trained model
explain.py                  CLI: per-concept breakdown of a model's answers
semantics.py                CLI: concept-level read-out of a PyC model, with PyC tools
runs/                       trained models, scores, twins (guide: runs/README.md)
explanations/               explain.py outputs (guide: explanations/README.md)
concept_datasets/           one file per dataset, each with download()
    cebab.py  goemotions.py  civil_comments.py  imdb_cad.py
    _common.py              common on-disk format (documented there)
concept_models/
    models.py               CBM, CEM (concept layer z, readout R, head)
    pyc_models.py           pyc-cbm, pyc-cem, pyc-hyper (PyC low-level layers)
    reparam.py              the construction (invisible_map, make_twin)
    features.py             frozen-encoder features + cache
    training.py             joint training, evaluation
```

## Datasets

Every example is `{"text", "label", "concepts": [0/1,...], "info"}`. All concepts are binary,
so every model uses the same readout (one sigmoid per concept). Ternary aspects
(Positive/Negative/unknown) become `<aspect>_pos` and `<aspect>_neg`, and "unknown" is both 0.
`meta.json` has the class names, concept names, concept groups, split sizes and concept
prevalence.

| name | source | task (classes) | concepts | train / val / test |
|---|---|---|---|---|
| `cebab` | HF `CEBaB/CEBaB` | star rating (5) | food, service, ambiance, noise × pos/neg (8) | 9848 / 1673 / 1689 |
| `goemotions` | HF `go_emotions` simplified | sentiment group: pos/neg/ambiguous/neutral (4) | 28 emotions | 42588 / 5303 / 5349 |
| `civil_comments` | HF `google/civil_comments` | toxic ≥ 0.5 (2) | 6 toxicity sub-types, score ≥ 0.2 (soft scores kept) | 40000 / 5000 / 10000, class-balanced subsample |
| `imdb_cad` | IMDB-C (Tan et al. 2024) + CAD (Kaushik et al. 2020) | sentiment (2) | 8 movie aspects × pos/neg (16) | 2000 / 500 / 500, + `cad_pairs` 4880 |

Dataset notes:
- **CEBaB**: `info` keeps `original_id` / `edit_goal` / `edit_type`, so you can rebuild the
  counterfactual edit pairs.
- **GoEmotions**: the label uses the official sentiment grouping. Comments that have both
  positive and negative emotions are dropped. The label is a function of the concepts and
  is never itself a concept.
- **Civil Comments**: the concept threshold is 0.2 because at 0.5 `severe_toxicity` is
  never on. Use `--soft_concepts` to train on the raw annotator fractions.
- **IMDB CAD**: the concept labels come from IMDB-C. Despite its name, only about 6% of
  IMDB-C reviews are CAD reviews. The real CAD original/counterfactual pairs are the extra
  split `cad_pairs`, which has no concept labels (`info.pair_id`, `info.is_original`).

## Models

Both run on frozen text features (`sentence-transformers/all-mpnet-base-v2`, mean-pooled;
change it with `--encoder`). Both have the same structure:

```
x --trunk--> h --concept layer--> z --head--> task logits
                                  └──readout R──> concept logits ──sigmoid──> concept probs
```

| `--model` | concept layer `z` | readout `R` | invisible dims |
|---|---|---|---|
| `cbm` | concept logits (k) | identity | **0**: control, nothing can move |
| `cbm --residual_dim r` | [concept logits ; residual (r)] | `[I_k, 0]` | r |
| `cem --emb_dim m` | concat of `z_i = p_i c_i+ + (1-p_i) c_i-` | shared score `s([c+; c-])` | k·(m−2) |

Model options:
- **CBM head input:** `--head_input probs` (the default, as in Koh et al.) or `logits`.
  Only with `logits` is the head linear in all of `z`, which lets the twin also mix concept
  logits into the residual.
- **Training:** joint, `task CE + concept_weight · concept BCE`, with `concept_weight = 5`
  by default. CEM uses random interventions during training (`--p_int 0.25`).
- **Learning rate:** starts at `--lr` (1e-3). When the validation loss has not improved for
  `--patience` (8) epochs, the best weights are restored and the learning rate is multiplied
  by `--lr_factor` (0.3); training stops when the next rate would be below `--min_lr` (1e-5),
  or after `--epochs` (300). `--lr_factor 0` stops at the first plateau, which is how the
  runs without `-lr…-p…` in their name were trained (patience 8, at most 60 / 100 epochs).

### PyC models (`concept_models/pyc_models.py`)

Built from PyC low-level layers (`pytorch-concepts==1.0.0a5`), on the same trunk and with
the same training and evaluation. Concepts are named with PyC `Annotations`
(`model.annotate(out["concept_probs"])["food_pos"]`).

| `--model` | concept encoder | task predictor |
|---|---|---|
| `pyc-cbm` | `LinearEmbeddingToConcept` | `LinearConceptToConcept` on concept probs (same function class as `cbm`: a cross-check) |
| `pyc-cem` | `LinearEmbeddingEncoder` (one embedding per concept) → shared `LinearEmbeddingToConcept` | `MixConceptEmbeddingToConcept`: `c±` from a Linear + LeakyReLU of the embedding, mixed by `p` |
| `pyc-hyper` | `LinearEmbeddingToConcept` | `HyperlinearConceptEmbeddingToConcept`: per-text concept weights `W(x)` from class embeddings, `logits = W(x) p + b` |

`pyc-cem` differs from `cem`: the concept is read from one embedding rather than from
`[c+; c-]`, and `c±` come out of a nonlinearity. The PyC models have no `readout_blocks`, so
`twin.py` does not apply to them.
- **Head:** `--head linear` (default) or `mlp`.

## The construction (`concept_models/reparam.py`)

For each readout block `R`, split `z` into what `R` sees, `a`, and its null space, `b`:

```
a' = a                        (untouched: concept probabilities identical)
b' = M a + G b + t            (G = rotation, M = "mix", t = "shift")
```

This gives `z' = B z + t` with `R B = R`. `B` is folded into the layer that produces `z`,
and `B^{-1}` into the head's first layer. The twin therefore has **the same architecture**
and different weights.

```bash
python twin.py runs/cebab-cbm-r16-logits-s0 --rotate 2 --mix 1 --shift 1 --seed 3
```

`report.json` shows what stayed the same (`max_abs_d_logits`, `max_abs_d_concept_probs`,
about 1e-6) and what moved (`rel_change_z`, `mean_cos_z`). It also includes a first
diagnostic: logits under concept interventions. Interventions notice the twin only when
`mix ≠ 0`, i.e. on a CBM with a residual and `head_input=logits`. For a CEM they are blind,
because the same `B` acts on `c+` and `c-`.

From Python:

```python
from concept_models.models import load_model
from concept_models.reparam import make_twin
from concept_models.features import get_features

model = load_model("runs/cebab-cem-s0/model.pt")
twin, info = make_twin(model, rotate=1.0, mix=1.0, shift=0.0, seed=0)
X = get_features("cebab", "test")["X"]
out, out_twin = model(X), twin(X)       # dicts: z, concept_logits, logits (+ c_plus, c_minus for CEM)
```

## Reference results (test split, seed 0)

| run | task acc | concept AUC | acc, all concepts intervened | invisible dims | max Δlogit (twin) | twin z rel. change |
|---|---|---|---|---|---|---|
| cebab-cbm | 0.530 | 0.896 | 0.516 | 0 | 0 | 0 |
| cebab-cbm-r16 | 0.658 | 0.897 | 0.660 | 16 | 8e-6 | 0.58 |
| cebab-cem | 0.649 | 0.897 | 0.680 | 112 | 9e-6 | 1.81 |
| goemotions-cbm | 0.630 | 0.880 | 0.915 | 0 | 0 | 0 |
| goemotions-cem | 0.623 | 0.874 | 0.976 | 392 | 1e-5 | 2.39 |
| civil_comments-cbm | 0.785 | 0.850 | 0.932 | 0 | 0 | 0 |
| civil_comments-cem | 0.785 | 0.851 | 0.933 | 84 | 4e-6 | 2.76 |
| imdb_cad-cbm | 0.892 | 0.865 | 0.914 | 0 | 0 | 0 |
| imdb_cad-cem | 0.874 | 0.855 | 0.926 | 224 | 6e-6 | 2.42 |

On CEBaB the vanilla CBM reaches 0.53, which is also what logistic regression gets from
the *true* concepts. Its bottleneck is therefore tight, and the residual adds about 13
points of accuracy that bypass the concepts.

## Design choices to know about

- **Frozen encoder.** The construction only touches the concept layer and the head.
  End-to-end fine-tuning is not implemented.
- **Linear generators in the CEM.** The CEM's `c±` are a linear map of a shared trunk; the
  paper uses a per-concept Linear + LeakyReLU. This keeps the fold exact.
- **CBM intervention values.** With `head_input=logits`, an intervention writes the median
  training logit of true positives / negatives. Koh et al.'s 5th/95th percentile breaks on
  rare concepts.
- **Downloads.** `concept_datasets/__init__.py` sets `HF_HUB_DISABLE_XET=1`, because the
  Hub's Xet backend fails on some networks.

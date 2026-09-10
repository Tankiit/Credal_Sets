# Frozen DINOv3 vs. Human Uncertainty — dataset-agnostic pipeline

Extract frozen DINOv3 embeddings for **any** image dataset, then compare
the embedding-space geometry against human annotation uncertainty to find
where the model's structure and human perception disagree ("blind spots").

Two ready-made examples are included — **CIFAR-10 + CIFAR-10H** (single
soft label per image) and **CUB-200-2011** (multiple binary attributes per
image, with real probability *intervals*, so genuine
aleatoric/epistemic decomposition is possible) — plus a generic builder for
your own dataset.

Run this on your own machine with a GPU and internet access — it needs to
download CIFAR-10/CUB images and the (gated) DINOv3 checkpoint from
Hugging Face, neither of which is reachable from a sandboxed environment.

## Architecture: everything speaks "manifest"

The three pipeline stages don't know anything about CIFAR or CUB
specifically — they only know a common file contract:

- **`manifest.csv`** — columns `id, image_path, label`. One row per image.
  `id` is the join key used everywhere (not row position), `label` is an
  optional 0-indexed ground-truth class integer.
- **`classes.txt`** (optional) — one class name per line, line *i* = class *i*.
- One of two annotation formats:
  - **`soft_labels.{csv,npy}`** — single-label point probabilities (e.g.
    CIFAR-10H). `.csv`: `id, prob_0..prob_{K-1}`, joined by id.  `.npy`:
    `(N, K)`, positional (row *i* = manifest row *i*).
  - **`attribute_credal.csv`** — multi-label probability *intervals* (e.g.
    CUB attributes): `id, lower_0..lower_{A-1}, upper_0..upper_{A-1}`.

Full details (and why id-keyed joins are safer than positional ones) are in
`common.py`'s module docstring — read that first if anything is unclear.

Bring your own dataset by producing these files yourself; nothing else
needs to change. `00_build_manifest_imagefolder.py` covers the common case
of one-subfolder-per-class.

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

DINOv3 checkpoints are gated on Hugging Face: accept the license at
`https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m` and run
`huggingface-cli login` (or set `HF_TOKEN`) before step 2.

If you hit a `CAS Client Error` / `File reconstruction error` from
`huggingface_hub` during download, that's their newer "Xet" storage backend
being flaky on some networks — set `export HF_HUB_DISABLE_XET=1` before
running, and if a previous attempt left a partial download, clear it first:
`rm -rf ~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m`.

## 1. Build the manifest (dataset-specific, choose one)

**CIFAR-10 + CIFAR-10H** (single-label, point-probability example):
```bash
python download_data_cifar.py --cifar10h-dir data/cifar-10h/data --out-dir dataset_cifar10h
```
Skips re-downloading `cifar10h-probs.npy` if you already have it there.
Produces `dataset_cifar10h/{manifest.csv, classes.txt, soft_labels.csv, images/*.png}`.

**CUB-200-2011** (multi-label, credal-interval example): download and
extract `CUB_200_2011.tgz` from `https://www.vision.caltech.edu/datasets/cub_200_2011/`
yourself first, then:
```bash
python download_data_cubs.py --cub-dir /path/to/CUB_200_2011 --out-dir dataset_cub --idm-s 1.0
```
Produces `dataset_cub/{manifest.csv, classes.txt, attributes.txt, attribute_credal.csv}`.
See the script's module docstring for exactly how worker votes + certainty
levels get turned into `[lower, upper]` intervals (Imprecise Dirichlet
Model) — `--idm-s` controls how much residual epistemic uncertainty a
single vote leaves (bigger = wider intervals for the same evidence).

## 2. Extract frozen DINOv3 features (same command for every dataset)

```bash
python feature_extraction.py --manifest dataset_cifar10h/manifest.csv --images-base-dir dataset_cifar10h \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m --batch-size 128 --out-dir features_cifar10h
```

```bash
python feature_extraction.py --manifest dataset_cub/manifest.csv --images-base-dir dataset_cub \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m --batch-size 128 --out-dir features_cub
```

- Frozen throughout: `eval()`, every parameter `requires_grad_(False)`,
  forward pass under `torch.inference_mode()`. No fine-tuning.
- Images are resized to DINOv3's expected input size via its own
  `AutoImageProcessor`, whatever their native resolution — fine for
  CIFAR's 32×32 upsampled or CUB's larger native photos alike.
- Saves `embeddings.npy` (CLS token, the global image embedding),
  `labels.npy`, `ids.npy` — all in manifest row order — plus `meta.json`.
  `--save-patch-tokens` additionally saves the full patch/register token
  grid if you want dense (not just global) features later.

ViT-L over 10k images is a few minutes on a modern GPU; use
`facebook/dinov3-vits16-pretrain-lvd1689m` for a quick smoke test first.

## 3. Analyze — two modes

### `--mode single-label` (CIFAR-10H-style: one soft-label vector per image)

```bash
python blindspot_analysis.py --mode single-label \
  --features-dir features_cifar10h --soft-labels dataset_cifar10h/soft_labels.csv \
  --classes dataset_cifar10h/classes.txt --manifest dataset_cifar10h/manifest.csv \
  --images-base-dir dataset_cifar10h/ \
  --out-dir results_cifar10h
```

- **Human side**: Shannon entropy of the soft label — a *total*
  uncertainty measure. A single point probability can't be decomposed into
  aleatoric/epistemic parts (that needs a set of distributions, not one),
  so this is deliberately not called AU or EU.
- **Model side**: k-NN label purity in DINOv3 embedding space (cosine
  distance) — fraction of an image's k nearest neighbors sharing its
  ground-truth label. A local, aleatoric-flavored read of the geometry, no
  classifier trained.
- Cross-tabulated (rank-based median split — robust to the many ties a
  discrete/skewed metric like this produces) into 4 quadrants; the two
  interesting ones are `model_blind_spot` (humans confident, model geometry
  locally confused) and `human_blind_spot` (model geometry clean, humans
  disagreed).

### `--mode multi-label` (CUB-style: many concepts per image, real intervals)

```bash
python blindspot_analysis.py --mode multi-label \
  --features-dir features_cub --credal-csv dataset_cub/attribute_credal.csv \
  --attributes dataset_cub/attributes.txt --classes dataset_cub/classes.txt \
  --manifest dataset_cub/manifest.csv --images-base-dir dataset_cub/ \
  --out-dir results_cub
```



- **Human side**: per attribute, closed-form binary credal-set metrics
  (Hüllermeier et al. 2022) from the `[lower, upper]` interval:
  `AU = min(lower, 1-upper)`, `EU = upper-lower`, `TU = min(1-lower, upper)`
  — averaged across attributes into one `human_AU` / `human_EU` / `human_TU`
  per image. `EU` here is genuine epistemic uncertainty (interval width),
  not disagreement relabeled.
- **Model side**: `model_knn_distance` — mean cosine distance to an image's
  k nearest embedding-space neighbors, density-based and
  epistemic-flavored (Deep-Deterministic-Uncertainty style: sparse regions
  are where a frozen deterministic model has the least basis for any
  prediction, independent of whether neighbors agree on a label).
  `model_knn_purity` (same as single-label mode, on the manifest's
  single-label ground truth, e.g. species) is reported alongside for
  aleatoric-flavored context.
- **Main comparison**: `human_EU` vs. `model_knn_distance` — two different
  epistemic-uncertainty proxies, one from annotation intervals, one from
  embedding density. Also reports `human_AU` vs. `1 - model_knn_purity` for
  context. Cross-tabulated into the same 4-quadrant blind-spot structure.
  Each blind-spot row also lists the top-EU attribute names for that image
  (`--top-attrs`, default 3) so you can see *what* was ambiguous, not just
  *that* it was.

### Outputs (either mode), in `--out-dir`

- `per_image_metrics.csv` — every metric, per image
- `correlation.txt` — Spearman correlation between the human and model signal
- `quadrant_scatter.png` — scatter plot, quadrant-colored
- `blind_spots_*.csv` — top-N most extreme images per blind-spot quadrant
  (ranked by combined rank-distance from the quadrant boundary)
- `blind_spot_images/` — PNGs of those images (skip with `--no-images`)
- `summary.json` — machine-readable summary of the above

Tune `--k` (neighbors, default 10) and `--top-n` (images per report,
default 30).

## Files

```
common.py                         shared manifest/credal-set/kNN utilities
download_data_cifar.py            CIFAR-10 + CIFAR-10H -> manifest.csv + soft_labels.csv
download_data_cubs.py             CUB-200-2011 -> manifest.csv + attribute_credal.csv
00_build_manifest_imagefolder.py  generic <class>/<image> folder -> manifest.csv
feature_extraction.py             frozen DINOv3 embeddings, any manifest.csv
blindspot_analysis.py             single-label / multi-label blind-spot analysis
requirements.txt
```

## Why these specific uncertainty metrics

A single averaged probability vector (CIFAR-10H's format) only supports
*total* uncertainty (Shannon entropy) — decomposing into aleatoric vs.
epistemic parts requires a *set* of plausible distributions, not a point.
CUB's per-worker certainty levels (not visible / guessing / probably /
definitely) provide exactly the extra structure needed to build that set,
so the multi-label mode's `AU`/`EU`/`TU` are real, not just relabeled
entropy. The formulas used here are the interval/credal-set decomposition
(Abellán & Moral 2000; Hüllermeier & Waegeman 2021; Hüllermeier et al.
2022) rather than the theoretically purer but combinatorially expensive
Generalized Hartley measure, which is infeasible once the number of
concepts grows past ~5–6 (CUB has 312 attributes).

On the model side, a single frozen deterministic embedding model
fundamentally cannot produce a credal set either (no ensembling, no
dropout — one forward pass, one distribution). `model_knn_distance`
(density-based) is the standard substitute for a model-side epistemic
proxy without training or ensembling anything (in the spirit of Deep
Deterministic Uncertainty / Mahalanobis-OOD methods); `model_knn_purity`
(label-based) is the aleatoric-flavored counterpart already used in
single-label mode.
# Supervised concept experiments

Experiment entry points/configs are separated into
[`experiments/synthetic`](../experiments/synthetic/README.md) and
[`experiments/real`](../experiments/real/README.md). This package holds shared
scientific objects and training mechanics. See the
[ICLR execution plan](../experiments/README.md) for study order and current status.
The old `concept_audit.experiment` command is retained as a compatibility dispatcher:
explicit feature-cache flags route to real-data runs; otherwise it routes to synthetic.
Use the new family-specific entry points when supplying JSON configs.

The scientific core is native PyTorch: fixed supervision readout `R`, task head
`h`, admissible transformations, and diagnostic audit runners. PyC implements an
optional encoder and donor replacement utility. Probly and Tinker are not imported
or installed by this package.

## Run the first experiment

From the repository root:

```bash
pip install -e .
python -m experiments.synthetic.run --out results/synthetic/native.json
python -m experiments.synthetic.run --readout identity --out results/synthetic/identity.json
python -m experiments.synthetic.run --readout grouped --out results/synthetic/grouped.json
python -m experiments.synthetic.run --backend cem --out results/synthetic/cem.json
```

These commands train on **synthetic features**, without downloading a backbone or
dataset. Each writes a JSON audit and a `.pt` state-dict checkpoint. The synthetic
data test the software, not the paper's scientific claims. Identity R admits only
the identity A, so the default coordinate readout includes unsupervised directions
to exercise nontrivial equivalence.

## Hugging Face datasets and frozen features

To start with the existing CIFAR-10 copy on this machine:

```python
from concept_audit.data import load_dataset

dataset = load_dataset("cifar10", split="test",
                       data_dir="/Users/cril/tanmoy/research/data")
```

```bash
python feature_extraction.py --dataset cifar10 --data-dir /Users/cril/tanmoy/research/data --split test --out-dir features/real/cifar10/dinov2/test
```

`data_dir` reads the existing `cifar-10-batches-py` files with downloads disabled,
then exposes them as an in-memory HF dataset. The shared row interface is unchanged.
Local files are read only; feature outputs stay in this project. Use `train` for
the training split, or `test[:8]` for a small extraction check. Local CIFAR-100 Python
batches are also supported. HF `revision`/`config` options apply only to Hub loading.
The visual backbone may still need downloading if its weights are not cached.

Hub image datasets go through Hugging Face `datasets.load_dataset`, behind
one wrapper shared with the existing extraction script:

```python
from concept_audit.data import load_dataset
# Also available as: from dataloaders import load_dataset

train = load_dataset("cifar10", split="train", cache_dir="data/huggingface")
test = load_dataset("cifar100", split="test", cache_dir="data/huggingface")
birds = load_dataset("cub", split="train", cache_dir="data/huggingface")
image, concepts, task_label = birds[0]  # RGB PIL, vector or None, integer
```

| Alias | HF repository | Image / task label |
| --- | --- | --- |
| `cifar10` | [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) | `img` / `label` |
| `cifar100` | [uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100) | `img` / `fine_label` |
| `cub` | [bentrevett/caltech-ucsd-birds-200-2011](https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011) | `image` / `label` |

HF handles downloads and caching. Pass `revision` (a commit hash for reproducible
runs), `config`, and a split such as `train[:100]` when needed. Other HF repository
IDs work with explicit `image_column`, `label_column`, and `concept_column`.
Images retain their full frame; the wrapper applies no bounding-box cropping.

```bash
python feature_extraction.py --dataset cifar10 --split test --out-dir features/real/cifar10/test
python feature_extraction.py --dataset cifar100 --split train --out-dir features/real/cifar100/train
python feature_extraction.py --dataset cub --split test --out-dir features/real/cub/test
```

These default HF mirrors supply classification labels, **not concept vectors**.
CUB's selected mirror has images, labels, and bounding boxes; its original 312
attribute annotations are not included. The wrapper returns `concepts=None`
unless annotations are explicitly provided. Image-only extraction writes
`embeddings.npy`, `labels.npy`, and `meta.json`; these caches alone cannot train a
supervised concept model. CIFAR-100 coarse class IDs are not silently treated as
binary concept vectors.

For concept-audit training, supply a HF vector column or an external `(N,K)`
NumPy annotation matrix in exactly the selected HF split's row order:

```python
from concept_audit.backbones import build_backbone
from concept_audit.data.extraction import extract_cache

train = load_dataset("cub", split="train", concepts_file="annotations/cub_train.npy",
                     require_concepts=True, cache_dir="data/huggingface")
backbone = build_backbone("hf", "facebook/dinov2-base", "cpu")
extract_cache(train, backbone, "features/real/cub/train")
```

```bash
python feature_extraction.py --dataset cub --split train --concepts-file annotations/cub_train.npy --require-concepts --out-dir features/real/cub/train
python feature_extraction.py --dataset cub --split test --concepts-file annotations/cub_test.npy --require-concepts --out-dir features/real/cub/test
python -m experiments.real.run --features-dir features/real/cub/train --eval-features-dir features/real/cub/test --out results/real/cub_native.json
```

Alternatively use `--concept-column attributes` with a HF repository containing
per-image binary/soft attribute vectors. Concept files must match the **requested
slice**, not the unsliced full dataset. Shape checks cannot establish semantic
alignment: the caller must join annotations correctly before supplying them.
Targets must be finite and in [0,1]; missing labels are not supported yet.

Extraction preserves HF order and records repository, split, requested revision,
HF fingerprint, class names, and annotation-file checksum in `meta.json`.
Use disjoint training/evaluation splits with the same concept and class ordering.
The backbone and all audit code remain independent of HF dataset column names.
`python -m concept_audit.data.extraction` exposes the same CLI as
`feature_extraction.py`. CUB-S and Shapes3D can be supplied through compatible HF
repositories with explicit column mappings; no aliases are defined for them yet.

## Model and readout contract

`ConceptModel` exposes `encode(z)`, `concept_readout(c)`,
`predict_from_concepts(c)`, and `intervene(c, concept_id, value)`.
Representations are flat `(B,D)` tensors; explicit `blocks` map concepts to latent
coordinates. `substitute_donor(c, j, donor)` replaces a declared block from a full
donor representation. Readout rows and blocks are separate objects.

`NativeCBM` is a continuous/logit bottleneck with a linear encoder and task head.
`NativeCEM` mixes positive/negative feature-dependent embedding blocks, supervised
through a native block projection. It is a CEM-like experimental variant, **not**
a claimed reproduction of standard CEM training. Its readout is always a block
projection, irrespective of the CLI `--readout` setting.

Readouts are fixed registered buffers and remain visible in checkpoints:
`IdentityReadout`, `CoordinateReadout`, `GroupReadout` (group means),
`BlockProjectionReadout`, and `LinearReadout`. The training runner uses
`BCEWithLogitsLoss(R(c), concepts) + CrossEntropyLoss(h(c), labels)`.
Scores are logits; sigmoid is not inserted between R and the equivalence check.

`intervene` sets one **score**, preserving other readout scores by a minimum-norm
latent update. Supplying 0 or 1 means a score of 0 or 1, not a guaranteed binary
concept probability. Impossible independent changes in rank-deficient R raise an
error. Donor substitution avoids choosing arbitrary finite logits for hard labels.

## Equivalence and diagnostics

With row-batched representations the transformation is `c @ A.T`, and the
compensated task head is `W @ inv(A)` with unchanged bias. A is generated as
`matrix_exp(N @ B)`, where columns of N span `ker(R)`. Singular, non-admissible,
nonfinite, and severely ill-conditioned transformations are rejected.

The audit verifies both readout and logit errors. Native and PyC experiments call
exactly the same runner. Exact compensation explicitly requires a linear head;
the four-method interface alone cannot expose its weights. `ReparameterizedModel`
owns a frozen snapshot and transports score interventions and full-donor
substitutions through A. Its `substitute` block value is in the source basis;
prefer `substitute_donor` when comparing equivalent representations. Applying
native minimum-norm or coordinate replacement directly in the new basis is a
different intervention, not automatically an equivalent causal operation.

The registry evaluates one value per concept for:

| Diagnostic | Definition |
| --- | --- |
| Own alignment | Maximum absolute correlation of block coordinates with its concept target |
| Cross-concept association | Maximum absolute correlation with other concept targets |
| Correlation share | Own alignment divided by the sum across concept targets |
| Task probe | Held-out accuracy of a block-only ridge least-squares classifier |
| Head sensitivity | Frobenius norm of the block's task-head weights |

These are explicit baseline diagnostics; association is not proof of leakage or
causality. Constant variables return zero correlation. Probes are fit exclusively
on training rows and refit for each condition, with fixed ridge strength 1. Their
regularization and coordinate correlation can themselves depend on parameterization.

Every registry diagnostic is run under four controls:

1. Exact equivalence: preserve R and logits, compare diagnostic values.
2. Structural: zero a concept block's outgoing task-head weights, preserve c/R.
3. Informational: permute block values within each split with head fixed.
4. Consequence: donor substitution plus held-out `Acc(original)-Acc(substituted)`.

The latter three are operational baselines, not guarantees of isolated causal
effects. Unconditional permutation can disrupt concept correlations and move
representations off their joint data distribution. Consequence drops can be
negative. Reports include seeds, A, R, blocks, split sizes, and diagnostic values.
Real-data claims need repeated seeds and a dataset-specific intervention protocol.

## Optional PyC

```bash
pip install -e '.[concepts]'
python -m experiments.synthetic.run --backend pyc --out results/synthetic/pyc.json
python -m unittest discover -s tests -v
```

The extra pins `pytorch-concepts==1.0.0a5`, imported as `torch_concepts`, following
the [official distribution](https://pypi.org/project/pytorch-concepts/1.0.0a5/).
The adapter uses its `LinearEmbeddingToConcept` encoder and `DoIntervention`
replacement strategy; native code owns R, h, block selection, and every audit.
See the [upstream low-level API](https://pytorch-concepts.readthedocs.io/en/latest/guides/using_low_level.html).
Tests compare native and actual PyC outputs, gradients, interventions, and all four
audits with matched weights. The PyC-specific test skips when PyC is absent.

No `conformal` extra is declared yet: Probly is reserved for a separate calibrated
uncertainty track. Tinker/VLM work and richer PyC CEM backends are also separate
extensions rather than dependencies of this first implementation.

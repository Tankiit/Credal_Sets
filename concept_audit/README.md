# Supervised concept experiments

The scientific core is native PyTorch: fixed supervision readout `R`, task head
`h`, admissible transformations, and diagnostic audit runners. PyC implements an
optional encoder and donor replacement utility. Probly and Tinker are not imported
or installed by this package.

## Run the first experiment

From the repository root:

```bash
pip install -e .
python -m concept_audit.experiment --out results/native.json
python -m concept_audit.experiment --readout identity --out results/identity.json
python -m concept_audit.experiment --readout grouped --out results/grouped.json
python -m concept_audit.experiment --backend cem --out results/cem.json
```

These commands train on **synthetic features**, without downloading a backbone or
dataset. Each writes a JSON audit and a `.pt` state-dict checkpoint. The synthetic
data test the software, not the paper's scientific claims. Identity R admits only
the identity A, so the default coordinate readout includes unsupervised directions
to exercise nontrivial equivalence.

## Frozen image features

The existing `models.backbones` implementation is re-exported without modification:

```python
from concept_audit.backbones import build_backbone
from concept_audit.data.extraction import extract_cache

backbone = build_backbone("hf", "facebook/dinov2-base", "cpu")
# Each native dataset returns (PIL_image, binary_concept_vector, integer_task_label).
extract_cache(train_dataset, backbone, "features/cub/train")
extract_cache(test_dataset, backbone, "features/cub/test")
```

Here `train_dataset` and `test_dataset` are user-supplied datasets, not included CUB
loaders. For timm, use `build_backbone("timm", "resnet50", "cpu")`.
Every split contains aligned `embeddings.npy` (N,D), `concepts.npy` (N,K), and
`labels.npy` (N,). Binary concept targets may be soft values in [0,1]. Missing
concept annotations are not supported by this first training runner.

```bash
python -m concept_audit.experiment \
  --features-dir features/cub/train \
  --eval-features-dir features/cub/test \
  --out results/cub_native.json
```

The caller must supply genuinely disjoint splits with identical concept/class
ordering. The existing CIFAR-10H cache has no concept annotations and is not by
itself a supervised concept dataset. Dataset-specific CUB, CUB-S, and Shapes3D
loaders remain future additions; no attributes are fabricated from class labels.

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
python -m concept_audit.experiment --backend pyc --out results/pyc.json
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

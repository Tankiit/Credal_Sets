# ICLR experiment sequence

Experiment code and configs are separated by data family. Models, readouts,
diagnostics, training mechanics, and audits remain shared in `concept_audit/`.

```text
experiments/
  synthetic/
    data.py                 # controlled synthetic generator
    run.py                  # synthetic-only training and baseline audits
    configs/                # full, partial, grouped supervision
    README.md
  real/
    extract.py              # image datasets -> frozen feature caches
    run.py                  # annotated real-data caches -> training and audits
    configs/                # real-data experiment configurations
    README.md
concept_audit/
  training.py               # common training/audit mechanics, no dataset selection
results/
  synthetic/                # synthetic reports and checkpoints
  real/                     # real-data reports and checkpoints
features/
  real/                     # image-feature caches, separated by dataset/backbone/split
```

Run modules from the repository root. Config paths are relative to the current
working directory; CLI arguments override JSON values. Explicit `--out` paths
remain supported. Existing result files are not relocated or relabeled.

## Progress and TensorBoard

Both families use tqdm for training epochs, training batches, and audit stages.
Image extraction also displays batch progress. TensorBoard logging is enabled for
every training run, with a unique run directory beneath the output directory's
`tensorboard/` folder. Thus the defaults keep synthetic and real logs separate.
Override the log root with `--log-dir`; the resolved run directory is recorded in
the JSON report as `tensorboard_log_dir`.

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/partial.json
tensorboard --logdir results
```

Open the URL printed by TensorBoard. Scalars include sample-weighted per-epoch
training/evaluation total, concept, and task losses; task accuracy; readout rank
and nullity; equivalence errors; and per-concept diagnostic/consequence results.
Training metrics summarize minibatches during optimization; evaluation metrics
use the fixed held-out split after each epoch. Evaluation does not select a
checkpoint or change optimization. Audit values are logged at the final epoch.
The run's configuration and data provenance are logged as text. Event writers
are flushed and closed even if training or an audit raises an error.

## Reviewer-driven execution order

| Order | Experiment | Objection it addresses | Planned evidence |
| --- | --- | --- | --- |
| 1 | Synthetic exact equivalence | Is the algebraic mechanism real? | rank(R), nullity, RA=R, readout/logit errors under full, partial, grouped/projected supervision |
| 2 | One-checkpoint diagnostic audit | Do reported metrics actually depend on parameterization? | D(c) vs D(Ac), yes/no changes and maximum relative changes across admissible A |
| 3 | Synthetic structural vs informational controls | What phenomena do the metrics respond to? | Behavioral fingerprint (equivalence, structural, informational sensitivity) |
| 4 | Synthetic substitution consequence | Does a diagnostic predict intervention failure? | Per-concept diagnostics vs task-behavior changes under a declared substitution protocol |
| 5 | Shapes3D | Is this specific to synthetic feature vectors? | Essential equivalence, source-separation, and consequence battery on known image factors |
| 6 | CUB + frozen DINOv2 | Does the mechanism matter on a real CBM benchmark? | Diagnostic instability and consequence prediction with fixed images/features/splits/encoder family |
| 7 | CUB readout variants | Is the supervision operator responsible? | Per-coordinate, partial, grouped, class-level, and feasible block/CEM supervision comparisons |
| 8 | CUB-S | Does group/soft human supervision leave the predicted ambiguity? | Targeted within-group underdetermination test |
| 9 | Backbone robustness | Is the result peculiar to DINOv2? | Compact DINOv2/ResNet-50/ConvNeXt-Tiny instability and consequence comparison |
| 10 | Tinker/VLM | Does this extend beyond explicit CBMs? | Optional dose-response sweep: 0, .1, .25, .5, 1, 2 |

**Decision checkpoint after step 4:** if diagnostics do not separate under the
controlled interventions, stop and rethink the scientific story before expanding
to real-data benchmarks. This is a scientific review gate, not an automatic rule
that declares success from a passing software test.

## Current implementation versus remaining study

The synthetic runner currently trains one model and applies the four existing
baseline audits to that checkpoint. Reports now include rank(R) and unconstrained
dimension in addition to exact-equivalence errors. Full/partial/grouped configs
are executable sanity checks; they are not a completed paper experiment matrix.

The existing structural baseline zeros outgoing head weights; the informational
baseline permutes blocks. Whitening/orthogonality/normalization controls, leakage
injection, nuisance rewards, masking, restricted encoders, and adversarial removal
still need dedicated controlled conditions. Neither baseline should be relabeled
as an isolated structural-only or information-only manipulation without validation.

The one-checkpoint table still needs repeated admissible transforms, a declared
near-zero denominator policy for relative changes, and a change tolerance.
Completeness and W_j^mom need explicit mathematical definitions before they are
added to the registry. Existing cross-concept correlation is an association proxy,
not an established leakage estimator. Consequence prediction needs a specified
association/evaluation protocol across concepts, checkpoints, and seeds.

Shapes3D, the CUB supervision study, CUB-S, backbone robustness, and Tinker remain
planned stages. CUB's example config requires genuine annotated caches. Local
CIFAR-10 is an extraction/integration starting point, not a replacement for
Shapes3D/CUB concept-ground-truth experiments. Probly remains outside this track.

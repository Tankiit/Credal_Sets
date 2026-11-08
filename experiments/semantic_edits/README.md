# Visual semantic edits — CVPR pilot

Runnable CPU experiment scaffold. This is a **2-D plumbing pilot**, not CLEVR,
not a pretrained vision experiment, and not evidence of a novel method.
The original `synthetic_tabular.py` is not modified.

## Run

Python 3.10+:

```bash
python -m pip install numpy Pillow
python -m unittest discover -s tests -p 'test_semantic_edits.py' -v
OPENBLAS_NUM_THREADS=1 python -m experiments.semantic_edits.run --out results/semantic_edits/pilot --seed 0
```

For independent model seeds, keep `--data-seed` fixed and change `--seed` and
`--out`. Existing output directories are rejected. The lightweight ridge CBM
is deterministic; model seeds affect residual projections and the MLP.

## Files

- `experiments/semantic_edits/data.py`: renderer, paired colour edits, scene manifests,
  fixed spatial pixel features, held-out red-square combinations.
- `experiments/semantic_edits/models.py`: ridge concept predictor, optional random projected
  pixel residual, frozen attribute reader, coordinate/linear/MLP operators.
- `experiments/semantic_edits/metrics.py`: target success, unaffected accuracy, joint success,
  edited-image agreement, and preservation conditional on initially correct answers.
- `experiments/semantic_edits/run.py`: training, validation checkpoint selection, evaluation,
  artifact export, exact rendered-image overlap filtering across splits.
- `tests/test_semantic_edits.py`: six protocol tests.

## Protocol

Two objects occupy left/right slots with independently varied position, colour
and shape. Object identity is slot identity. The only supported action is
`set_colour(object, target_colour)`; unchanged-colour requests are omitted.
Each source scene and all its edits belong to one split. Red squares are excluded
from both endpoints of training and validation pairs. Test results are stratified
by whether either endpoint contains a red square. This is combination holdout,
not evidence of generalization to unseen individual colours or shapes.

The encoder uses resized RGB pixels. A ridge predictor learns 10 concept scores
(two objects, each with three colour and two shape scores). The optional residual
contains 16 fixed random projections of centered pixel features. These are
simplified linear CBM prototypes, not trained neural CBM/CEM architectures.
The frozen downstream reader answers four categorical attribute questions.

Learned operators both use exactly the training edit pairs and representation
regression targets. The linear operator is affine in representation and action;
the MLP is a one-hidden-layer residual map, trained with Adam and selected by
validation representation MSE. Coordinate replacement sets the requested colour
block to a one-hot target and retains all other entries. Identity and the actual
edited-image representation are controls. The latter is a perception reference,
not a guaranteed oracle; ground-truth attributes supply correctness targets.

All test metrics use the frozen reader. Bootstrap intervals resample source scenes
and estimate uncertainty conditional on the fitted model, not across training seeds.
Point metrics average per scene, then across scenes, so scenes with more retained
pairs do not dominate. Conditional preservation is pooled over initially correct
unaffected questions and has no bootstrap interval in this version.

## Outputs

Each output directory contains:

- `config.json`: parameters, versions, source hashes.
- `scenes.json`, `pairs.jsonl`: sampled scenes and actual retained split/pair records.
- `predictions.jsonl`: IDs, seeds, strata, predictions, truth and per-pair metrics.
- `metrics.json`: per-model/method/stratum aggregates and scene bootstrap intervals.
- `*_checkpoint.npz`: fitted arrays, including encoder, head and edit operators.
- `*_test_arrays.npz`: test representations, actions, targets and edited representations.
- `pairs.png`: original/edited image pairs.

The pre-integration smoke validation used 80/20/40 source scenes and 40 optimization steps.
All six tests passed and the full pipeline completed. Coordinate replacement
achieved perfect joint attribute accuracy in that smoke run; learned edits did
not outperform it. This validates the controls, not the proposed research claim.

## Next implementation stage

1. Replace `features()` with cached frozen pretrained image features, keyed by
   image hash, encoder identity and preprocessing version. Keep paired-image
   splitting upstream of feature caching.
2. Add PyTorch neural CBM and residual CBM training behind the representation
   boundary. Do not interpret this random-projection residual as a learned bypass.
3. Replace slot identity with explicit object/region IDs for richer scenes.
4. Add affected and unaffected relational task questions; attribute readout alone
   is too weak for the proposed CVPR claim.
5. Add composed-edit evaluation and one verified natural-image pair adapter.
6. Run multiple independent training seeds, matched-budget baselines and
   paired scene-bootstrap differences before drawing comparative conclusions.

Do not escalate model complexity just to make coordinate replacement fail.
First establish a meaningful visual task where the proposed correction is needed.
The ICML identifiability study is intentionally outside this pilot.

## Native-model experiment (Shapes3D, MPI3D)

The toy pilot above is a software control. The research experiment uses the
student's frozen checkpoints (`concept_audit`) and cached DINOv3 features.

### Commands (repository root, `.venv`; see `requirements.txt` + `pytorch-concepts==1.0.0a5`)

```bash
python -m experiments.semantic_edits.inventory --out results/semantic_edits/inventory
python -m experiments.semantic_edits.build_pairs --dataset shapes3d --out results/semantic_edits/pairs/shapes3d
python -m experiments.semantic_edits.build_pairs --dataset mpi3d    --out results/semantic_edits/pairs/mpi3d
python -m experiments.semantic_edits.train_native --pairs-dir results/semantic_edits/pairs/shapes3d \
    --checkpoint ignore/results_experiments/cbm_results_seed0/shapes3d_dataset/audit.pt \
    --out results/semantic_edits/runs/shapes3d/cbm_seed0          # repeat per checkpoint
python -m experiments.semantic_edits.evaluate_native --run-dir results/semantic_edits/runs/shapes3d/cbm_seed0
python -m experiments.semantic_edits.summarize_native
```

Runs: Shapes3D CBM/CEM seeds 0 and 12; MPI3D CBM/CEM seed 12 (only seed available); 3 operator seeds each.

### Files

- `inventory.py`: checkpoint/cache inventory; every checkpoint must reproduce its reported accuracy.
- `build_pairs.py`: raw-factor single-factor pairs, context-grouped splits, held-out combinations,
  automated validity/disjointness assertions (IDs and image-byte hashes), montage.
- `native_adapter.py`: frozen checkpoint loading, hash-bound latent cache, readout/logit access, decoding.
- `torch_operators.py`: affine and MLP edit operators (identity-initialised), held-out factor readers.
- `train_native.py`: forward-pass reproduction, readout-unit checks, reader fitting, calibration,
  operator fitting with validation selection. Never loads test pairs.
- `evaluate_native.py`: one-shot frozen test evaluation, per-pair CSV, readouts/logits, context bootstrap.
- `summarize_native.py`: result tables, figures, decision-gate evidence.
- Tests: `tests/test_semantic_edit_pairs.py`, `tests/test_semantic_edit_adapter.py`.

### Protocol

- **Edit**: Shapes3D `floor` hue (10 levels); MPI3D `background` colour (3 levels). Both are supervised
  concepts and are not part of the task label (`shape*10+object`, `shape*4+color`), so the label must stay fixed.
- **Splits**: context = all raw factors except the edited one; contexts split 60/20/20 before pairs are drawn.
- **Unseen combinations**: some (edited value, `shape`) conjunctions never appear at either endpoint of an
  operator-train/val pair; every value and every shape still appears in training.
- **Base-model overlap**: every image was in the base model's train or eval cache, so results are scoped to
  *operator* generalization. `base_model_heldout_endpoints` marks pairs with both images in the eval cache only.
- **Methods**: identity; native `intervene` with logit(target) (`native_logit`) and with per-level median scores
  from training images (`native_calibrated`); affine; MLP; references using the target image: donor-block
  substitution and the real edited-image state.
- **Judging**: held-out MLP readers trained on original images from operator-train contexts read *every* raw
  factor (also unsupervised ones). The model's own readout is reported separately because native interventions
  set it by construction. `joint` = edited factor reads as requested AND all other factors read correctly AND
  task prediction correct. `cond. retention` = preservation among pairs whose source was read correctly.

### Results (unseen combinations; see `results/semantic_edits/summary/results.md`)

Joint success, mean over 3 operator seeds; real edited image = practical ceiling under these readers.

| run | native (cal.) | affine | MLP | real edited image* |
|---|---|---|---|---|
| Shapes3D CBM s0 | 0.103 | 0.208 | 0.237 | 0.437 |
| Shapes3D CBM s12 | 0.050 | 0.199 | 0.253 | 0.444 |
| Shapes3D CEM s0 | 0.100 | 0.388 | 0.434 | 0.737 |
| Shapes3D CEM s12 | 0.088 | 0.376 | 0.435 | 0.731 |
| MPI3D CBM s12 | 0.072 | 0.056 | 0.040 | 0.088 |
| MPI3D CEM s12 | 0.130 | 0.137 | 0.125 | 0.234 |

### Decision gate (outcome: **Revise** on Shapes3D, **Stop** on MPI3D)

1. *Native failure is not a unit error*: on Shapes3D, calibrated native intervention sets the model's own
   readout to the requested level for 100% of pairs, yet independent readers see the requested floor in only
   ~19-32%. The latent does encode floor (readers: 84-95% on real images); the model's own concept readout is the
   weak part (calibrated decoding 33-42%). Native intervention moves a poorly aligned readout axis.
2. *MLP vs affine on unseen combinations*: on Shapes3D the MLP wins joint success in all 4 models x 3 seeds
   (+0.03 to +0.06; every context-bootstrap 95% CI excludes 0). On MPI3D it loses in every run and seed.
3. *Survives preservation?* Partly. The Shapes3D gain comes from target success; on CBM the MLP preserves
   other factors worse than affine (cond. retention -0.05 to -0.09); on CEM it is neutral. Every learned edit
   blurs wall/orientation relative to the real image. The MLP's advantage shrinks from seen to unseen
   combinations (CEM s0 joint 0.585 seen -> 0.434 unseen) while affine is stable.
4. *Across model seeds*: Shapes3D findings agree across seeds 0 and 12 for both architectures.

**Reading**: MPI3D background is already solved by native intervention (100% target success; colour, shape,
size and height preserved at 98-100%); no learned-edit claim there. On Shapes3D, learned paired operators
clearly beat native intervention and reach ~55-60% of the real-image ceiling, but the nonlinear gain over affine
is small, partly paid for in preservation, and smaller on unseen combinations. Per the plan this is **Revise**,
not Continue: do not claim a nonlinear-edit advantage. The robust finding is the native/real-image gap.

Caveats: MPI3D `dof1`/`dof2` (40 levels) are read poorly even on real images (28-55%), which dominates its
absolute preservation numbers; no tolerance was predeclared, so none is applied. No extra readout/preservation
loss ablation was run (operators use paired latent reconstruction only). Composed edits, a second
architecture beyond CBM/CEM and natural images (CUB) are post-gate work and not done.

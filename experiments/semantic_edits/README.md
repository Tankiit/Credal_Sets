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

## Integration with uncertainty-images

Run from the repository root. This namespace is included by the existing
`experiments*` package discovery; no new dependency or CLI registration is needed.
The pilot deliberately does not change the student's training or audit paths.

The next production adapter should reuse:
- `concept_audit.models.base.ConceptModel.encode(z)` for latent states;
- `concept_readout(c)` and `predict_from_concepts(c)` as frozen readers;
- `intervene(c, concept_id, value)` for the existing minimum-norm **score-space**
  baseline (not a probability edit unless the readout actually uses probabilities);
- `substitute_donor` only as a paired-donor reference, since it needs the edited
  example at evaluation time;
- existing image extraction and feature caches for both endpoints.

This adapter is not implemented here. The NumPy prototype does not load student
checkpoints. Its colour one-hot replacement is specific to the toy schema and
must not be applied to CEM coordinates as though they were concept probabilities.

For Shapes3D, the existing exporter assigns `shape_idx * 10 + object_idx` as the
task label and supervises floor hue, wall hue, scale and orientation. Preserve
all raw factors and stable IDs before constructing single-factor pairs; the
current four concept columns alone cannot verify that other factors stay fixed.
Group paired examples before splitting. Do not use the existing row-random cache
splitter on a collection of source/edited endpoints. Changing floor hue should
preserve that current task label; testing a changed task answer requires an
explicit new target specification.

No results/checkpoints or downloaded image datasets are committed with this module.

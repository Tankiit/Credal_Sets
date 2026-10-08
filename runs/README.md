# How to read the results in `runs/`

Each folder in `runs/` is one trained model. This guide explains the folder names, the
scores in `metrics.json`, and the twin checks in `twins/*.report.json`.

## Folder names

`<dataset>-<model>[-r16][-logits]-s<seed>`, for example `cebab-cbm-r16-logits-s0`.

| Part | Meaning |
|---|---|
| `cebab`, `goemotions`, `civil_comments`, `imdb_cad` | the dataset |
| `cbm` | Concept Bottleneck Model: the answer is computed from the concepts |
| `cem` | Concept Embedding Model: each concept is a vector of 16 numbers instead of one |
| `-r16` | the CBM also has 16 extra "free" numbers (the residual) that are never compared to any label |
| `-logits` | the CBM's final layer reads the raw concept scores instead of probabilities (needed for the twin's `mix` option) |
| `-s0` | random seed 0 |

There are 4 models per dataset, from the most constrained to the most free:

| Model | Hidden room for a twin |
|---|---|
| `cbm` | none: this is the control, no twin is possible |
| `cbm-r16` | the 16 free numbers |
| `cbm-r16-logits` | the 16 free numbers, and concept information can also be mixed into them |
| `cem` | 14 hidden directions per concept |

## Inside each folder

```
runs/cebab-cem-s0/
├── model.pt                 the trained model (not in git)
├── metrics.json             its scores
└── twins/
    ├── <tag>.pt             the twin model (not in git)
    └── <tag>.report.json    the check that the twin behaves like the original
```

## `metrics.json`: how good is the model?

The file has four parts:

- `args`: the options used to train the model
- `config`: the model's size (number of concepts, classes, layers)
- `metrics`: the scores (below)
- `history`: the validation loss after each training pass ("epoch"). Lower is better. Training
  stops automatically when it stops improving.

`metrics` has scores for the `val` split (used to decide when to stop training) and the
`test` split (never seen during training). **Report the `test` numbers.**

| Field | What it measures | How to read it |
|---|---|---|
| `task_acc` | % of final answers that are correct | 0.65 = 65% correct |
| `task_macro_f1` | accuracy averaged over the classes, so rare classes count as much as common ones | lower than `task_acc` means the model does worse on some classes |
| `concept_acc` | % of concepts predicted correctly (yes/no) | can look high just because most concepts are "no" most of the time, so prefer the AUC |
| `concept_mean_auc` | how well the model ranks texts with a concept above texts without it, averaged over concepts | 0.5 = random guessing, 1.0 = perfect. About 0.85–0.90 here. |
| `task_acc_full_intervention` | accuracy when **every** predicted concept is replaced by the true human label | see below |

**How to use `task_acc_full_intervention`:** compare it with `task_acc`.

- **Large increase** (e.g. GoEmotions CBM: 0.63 → 0.92): the model really decides through
  its concepts, so correcting the concepts fixes its answers.
- **No increase** (e.g. GoEmotions CBM-r16: 0.62 → 0.63): the model mostly ignores its
  concepts and decides through the hidden residual. Its concept "explanation" is not what
  drives the answer.

## `twins/<tag>.report.json`: does the twin behave exactly like the original?

The **twin** is a copy of the model whose hidden numbers have been rearranged where the
concept readout can't see them, with its final layer adjusted to undo the change. It should
give **the same concepts and the same answers** while being **different inside**.

The file name says how the twin was made: `rot1.0-mix1.0-shift0.0-seed0`.

| Part | Meaning |
|---|---|
| `rot` | how strongly the hidden numbers are rotated |
| `mix` | how much concept information is copied into the hidden numbers (only possible for `cem` and `-logits` models, so it is 0 for the others) |
| `shift` | a constant added to the hidden numbers |
| `seed` | random seed for the change |

The fields, in four groups:

**1. Size of the change**

| Field | Meaning |
|---|---|
| `invisible_dims` | how many hidden directions could be changed. **0 for `cbm`**: nothing can move. |
| `concept_layer_dim` | total size of the model's inner layer `z` |
| `cond_B` | how "stretched" the change is. 1 = pure rotation. Larger values are fine as long as the checks below stay tiny. |

**2. Must be (almost) zero: the twin looks identical from outside**

| Field | Meaning | Good value |
|---|---|---|
| `max_abs_d_logits` | largest difference in the final-answer scores, over all test examples | about 1e-5 or smaller (computer rounding) |
| `max_abs_d_concept_probs` | largest difference in any concept probability | about 1e-6 or smaller |

If either of these is large, the twin is broken.

**3. Should be large: the twin is different inside**

| Field | Meaning | How to read it |
|---|---|---|
| `rel_change_z` | how much the inner layer `z` changed, relative to its size | 0 = no change, 1 = changed by as much as its own size. 0 for `cbm`. |
| `mean_cos_z` | average similarity between the original's and the twin's `z` for the same text | 1 = same direction, 0 = unrelated. Lower means more different. |

**4. First diagnostic: can interventions tell the twin apart?**

Half of the concepts (chosen at random) are replaced by the true human labels in both
models, and their answers are compared.

| Field | Meaning |
|---|---|
| `intervention_max_abs_d_logits` | largest difference in answer scores after the correction. Tiny = interventions can't see the twin. |
| `intervention_pred_agreement` | % of texts where both models give the same answer after the correction. 1.0 = can't tell them apart. |
| `intervention_acc_original` / `intervention_acc_twin` | accuracy of each model after the correction |

What we found: for the **CEM** twins, agreement is 1.0, so interventions are fooled. For the
**`cbm-r16-logits`** twins made with `mix=1`, agreement drops (0.78 on CEBaB), so
interventions catch the twin. The mixing hid a copy of the concepts in the residual, and
that copy is not corrected.

## Rebuilding these files

```bash
python train.py --dataset cebab --model cem      # -> runs/cebab-cem-s0/metrics.json
python twin.py runs/cebab-cem-s0                 # -> runs/cebab-cem-s0/twins/*.report.json
```

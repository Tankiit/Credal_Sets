# How to read the explanations

Each `.txt` file shows how one trained model (from `runs/`) handled 10 random examples from
the validation split. For each example, the file shows which concepts the model detected and
how much each concept pushed it towards its answer.

The file name is the model's name, for example `cebab-cbm-s0.txt` is the plain CBM on
CEBaB. See [runs/README.md](../runs/README.md) for what the names mean.

There are two generations of files, on the **same 10 examples** per dataset, so they can be
compared side by side:

- `<dataset>-<model>-s0.txt`: the first runs (training stopped at the first plateau).
- `<dataset>-<model>-lr0.001-p8-s0.txt`: the runs where the learning rate is lowered at each
  plateau, including the three PyC models (`pyc-cbm`, `pyc-cem`, `pyc-hyper`).

## One example, explained

```
Example #829
  Text: Always dependably good service, but the food is mediocre.
  True answer: 3_stars   Model's answer: 3_stars (43% sure)   ✓ correct
  All answers: 1_star 7%  2_stars 22%  3_stars 43%  4_stars 9%  5_stars 19%

  concept               human label  model thinks                      push towards '3_stars'
  food_pos              no           ████████············   39%        +0.10
  food_neg              yes          ██████████████████··   89%        +0.47
  service_pos           yes          ██████████████████··   90%        +0.21
  ...
  (constant, same for every text)                                       +0.10
  TOTAL lead of '3_stars' over the average answer                       +0.99
```

| Line / column | Meaning |
|---|---|
| `Example #829` | the example's line number in `data/<dataset>/val.jsonl` (counting from 0) |
| `True answer` | what the human annotators said |
| `Model's answer (43% sure)` | the model's choice and how confident it is |
| `All answers` | the model's probability for every possible answer (they add up to 100%) |
| `human label` | whether the human said this concept is present |
| `model thinks` | the model's probability that the concept is present (the bar shows the same number) |
| `← disagrees with human` | the model's yes/no (above or below 50%) differs from the human label |
| `push` | how much this concept moved the model **towards** the answer it chose. Positive = towards it, negative = away from it, near 0 = no effect. |
| `(residual: 16 unlabelled numbers)` | only in `-r16` models: the push from the 16 hidden numbers |
| `(other concepts)` | only when `--top` is used: the pushes of the concepts not listed, added together |
| `(constant, ...)` | the model's built-in preference for this answer, the same for every text |
| `TOTAL` | the sum of all the pushes and the constant: how far this answer's score is above the average answer's score. The highest total wins. |

The pushes add up **exactly** to the total, because the model's final layer is a plain
weighted sum. Nothing is approximated.

**Reading this example:** the model saw "food bad" (+0.47) and "service good" (+0.21), and
concluded 3 stars. That is a sensible reason, and it matches the humans.

## What to look for

**Where a mistake comes from.** When the answer is wrong, check:

- **A concept was misread** (`← disagrees with human` on a concept with a large push). The
  error comes from concept detection. Example: "the food is not always great" read as
  `food_pos` 88%.
- **The concepts are right, but the answer is wrong anyway.** The step from concepts to
  answer is too coarse. Example: every negative concept is detected, but the model says
  2 stars instead of 1, because yes/no concepts can't express *how* bad something was.

**Whether the concepts really drive the answer.** In the `-r16` files, compare the
residual's push with the concepts' pushes. If the residual line carries almost all of the
total (e.g. +3.86 out of +3.90), the model is deciding through its hidden numbers, and its
concept explanation is mostly decoration.

**CEM files read differently.** In a CEM, each concept is a vector, not a single number, so
a concept can push even when it is *absent*. For example, "food is NOT bad" can push towards
5 stars. The numbers are correct but less intuitive than in a CBM.

**PyC files.** `pyc-cbm` reads like a CBM (and gives the same numbers as `cbm`).
`pyc-cem` reads like a CEM: a concept can push even when it is absent. In `pyc-hyper`, the
weight of each concept is **recomputed for every text**, so the pushes of one example
cannot be carried over to another, and absent concepts often push towards the chosen
answer too (a 7% concept times a large weight). For what a concept does on average over
all texts, see `semantics-test.json` in the run folder (`cace`, `weights`).

**Datasets with many concepts.** For GoEmotions (28) and IMDB (16), only the 8 concepts with
the largest push are listed. The rest are summed into `(other concepts)`.

## Rebuilding or exploring

```bash
python explain.py runs/cebab-cbm-s0                          # 10 random validation examples
python explain.py runs/cebab-cbm-s0 --n 20 --seed 1          # 20 other examples
python explain.py runs/cebab-cbm-s0 --ids 829 1047           # specific examples
python explain.py runs/goemotions-cem-s0 --top 6             # only the 6 strongest concepts
python explain.py runs/cebab-cbm-s0 > explanations/cebab-cbm-s0.txt   # save to a file
python explain.py runs/cebab-pyc-hyper-lr0.001-p8-s0                  # works for PyC models too
```

The same `--seed` gives the same examples for every model on a dataset, so the files can be
compared side by side.

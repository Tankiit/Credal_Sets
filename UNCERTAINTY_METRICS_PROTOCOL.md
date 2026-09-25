# Uncertainty and Annotator-Disagreement Metrics

This file fixes the evaluation protocol used for the ICML/NeurIPS credal
experiments. Metrics must be reported per seed and as the arithmetic mean
across seeds. A proxy must never be presented as genuine annotator
disagreement.

## Model uncertainty

For concept models, retain the complete `[N, K]` arrays:

- `sigma_epi`: epistemic scale
- `EU = log(sigma_epi + 1e-10)`: epistemic uncertainty
- `AU = sigma_ale`: aleatoric uncertainty

Report per-concept values, pooled values obtained by flattening `(i, k)`, and
sample-level values obtained by averaging over `k`.

For MAQA/AmbigQA, also report the model-native `EU = sigma_epi` alongside
`log(sigma_epi + 1e-10)`.

## Disagreement measures

Given a vote or answer distribution `p` over `C` outcomes, calculate:

- normalized entropy: `-sum(p log p) / log(C)`
- variation ratio: `1 - max(p)`
- pairwise disagreement (Gini): `1 - sum(p**2)`
- margin uncertainty: `1 - (p_(1) - p_(2))`
- any-disagreement indicator: `1[max(p) < 1]`
- effective outcomes: `exp(-sum(p log p))`

## Dataset targets

### CEBaB

Genuine worker distributions:

- `food_aspect_label_distribution`
- `service_aspect_label_distribution`
- `ambiance_aspect_label_distribution`
- `noise_aspect_label_distribution`
- `review_label_distribution`

Report every disagreement measure per aspect, pooled across `(i, k)`,
sample-mean across aspects, and separately for the review distribution.

### HateXplain

The three values in `annotators.label` are genuine task-label votes. Report
all disagreement measures above. Also report `1 - consensus_score` and an
any-disagreement indicator. Target-list disagreement is a separate auxiliary
quantity and must not be called task-label annotator entropy.

### GoEmotions

The Hugging Face `simplified` configuration does not expose identifiable
annotator vote counts. Multiple emotion labels are a multi-label target, not
annotator disagreement. Report label count, multi-label indicator, and
label-set entropy only as ambiguity proxies.

### MAQA/AmbigQA

Use the annotated answer distribution `p_star`. Report all distributional
measures above, answer count, and the provided/derived ambiguity level.
These are answer-distribution ambiguity measures; call them annotator
disagreement only when `p_star` is demonstrably constructed from worker vote
counts.

## Associations

For every valid target `Z`, report:

- Spearman `rho(AU, Z)` and Pearson `r(AU, Z)`
- Spearman `rho(EU, AU)` and Pearson `r(EU, AU)`
- Spearman/Pearson association between EU and task error
- per-concept and pooled association between EU and concept error
- AUROC of EU for detecting task errors
- AUROC of AU for detecting any disagreement/high ambiguity

Undefined statistics caused by a constant array must be stored as `null`, not
silently replaced by zero.

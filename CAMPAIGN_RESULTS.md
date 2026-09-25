# Retraining campaign results

Generated 2026-09-25 17:51 UTC by `scripts/campaign_report.py` from `outputs/icml_2026_reeval/`. Mean ± sample SD over seeds; `(n=k)` means only k seeds have a defined value. `—` means undefined, including correlations with a collapsed (constant) ambiguity head. CEBaB ρ(AU,H) and concept accuracy exclude unannotated aspects (`CEBAB_MASK_AUDIT.md`).

## Retrained, corrected CEBaB labels (3 seeds)

| Configuration | Seeds | Acc % | Concept acc % | ρ(EU,AU) | ρ(EU,Err) | ρ(AU,H) | AUROC EU | AUROC MaxProb |
|---|---|---|---|---|---|---|---|---|
| CEBaB DistilBERT | 42,123,2024 | 77.8 ± 0.1 | 78.1 ± 0.7 | -0.14 ± 0.03 | 0.32 ± 0.02 | 0.22 ± 0.01 | 0.72 ± 0.02 | 0.78 ± 0.01 |
| CEBaB RoBERTa | 42,123,2024 | 78.2 ± 0.4 | 79.4 ± 0.8 | 0.00 ± 0.02 | 0.32 ± 0.02 | 0.24 ± 0.01 | 0.72 ± 0.01 | 0.79 ± 0.01 |

## Ablations, DistilBERT, corrected labels (3 seeds)

| Configuration | Seeds | Acc % | Concept acc % | ρ(EU,AU) | ρ(EU,Err) | ρ(AU,H) | AUROC EU | AUROC MaxProb |
|---|---|---|---|---|---|---|---|---|
| Default (λ_a = 2, λ_d = 0) | 42,123,2024 | 77.8 ± 0.1 | 78.1 ± 0.7 | -0.14 ± 0.03 | 0.32 ± 0.02 | 0.22 ± 0.01 | 0.72 ± 0.02 | 0.78 ± 0.01 |
| No ambiguity supervision (λ_a = 0) | 42,123,2024 | 77.4 ± 0.8 | 77.7 ± 0.9 | — | 0.32 ± 0.01 | — | 0.72 ± 0.01 | 0.79 ± 0.00 |
| Decorrelation penalty (λ_d = 5) | 42,123,2024 | 77.2 ± 0.4 | 78.2 ± 0.9 | — | 0.30 (n=1) | 0.22 ± 0.01 (n=2) | 0.71 (n=1) | 0.79 ± 0.00 |

## Other datasets, seed 42 added (3 seeds)

| Configuration | Seeds | Acc % | Concept acc % | ρ(EU,AU) | ρ(EU,Err) | ρ(AU,H) | AUROC EU | AUROC MaxProb |
|---|---|---|---|---|---|---|---|---|
| HateXplain | 42,123,2024 | 57.6 ± 0.4 | — | — | 0.19 ± 0.02 | — | 0.61 ± 0.01 | 0.68 ± 0.01 |
| GoEmotions | 42,123,2024 | 44.6 ± 0.3 | — | — | 0.04 ± 0.04 | — | 0.52 ± 0.02 | 0.75 ± 0.00 |
| MAQA* (λ_decorr = 5) | 42,123,2024 | 48.9 ± 1.7 | — | 0.01 ± 0.04 | 0.01 ± 0.02 | 0.21 ± 0.03 | 0.50 ± 0.01 | 0.63 ± 0.01 |
| AmbigQA* only (λ_decorr = 5) | 42,123,2024 | 52.0 ± 0.6 | — | 0.05 ± 0.09 | -0.07 ± 0.04 | 0.11 ± 0.07 | 0.46 ± 0.02 | 0.65 ± 0.01 |

## Previous checkpoints, original labels, masked evaluation (2 seeds)

| Configuration | Seeds | Acc % | Concept acc % | ρ(EU,AU) | ρ(EU,Err) | ρ(AU,H) | AUROC EU | AUROC MaxProb |
|---|---|---|---|---|---|---|---|---|
| CEBaB DistilBERT | 123,2024 | 76.1 ± 0.5 | 73.6 ± 0.3 | -0.09 ± 0.04 | 0.21 ± 0.06 | 0.23 ± 0.04 | 0.64 ± 0.04 | 0.76 ± 0.01 |
| CEBaB RoBERTa | 123,2024 | 77.6 ± 0.8 | 74.5 ± 1.1 | -0.03 ± 0.03 | 0.22 ± 0.02 | 0.30 ± 0.03 | 0.66 ± 0.01 | 0.80 ± 0.00 |

## Baselines (error-detection AUROC with bootstrap 95% CI)

| Group | Ensemble acc % | Ens. MaxProb | Ens. entropy | Ens. mutual info | Mean CBM EU | MC-dropout MI (per seed) |
|---|---|---|---|---|---|---|
| `cebab_3class_distilbert_seed{}_100ep_fixed` seeds [42, 123, 2024] | 78.8 | 0.775 [0.751, 0.798] | 0.764 [0.740, 0.786] | 0.722 [0.693, 0.749] | 0.720 [0.694, 0.745] | 42: 0.733, 123: 0.730, 2024: 0.685 |
| `cebab_3class_distilbert_seed{}_100ep_fixed_decorr5` seeds [42, 123, 2024] | 79.1 | 0.768 [0.743, 0.790] | 0.758 [0.733, 0.781] | 0.705 [0.677, 0.732] | 0.686 [0.658, 0.712] | — |
| `cebab_3class_distilbert_seed{}_100ep_fixed_noale` seeds [42, 123, 2024] | 78.4 | 0.792 [0.769, 0.814] | 0.778 [0.754, 0.801] | 0.710 [0.682, 0.737] | 0.717 [0.690, 0.743] | — |
| `cebab_3class_roberta_base_seed{}_100ep_fixed` seeds [42, 123, 2024] | 79.2 | 0.792 [0.769, 0.814] | 0.775 [0.750, 0.797] | 0.655 [0.623, 0.683] | 0.722 [0.697, 0.747] | 42: 0.689, 123: 0.644, 2024: 0.717 |

## Status

- Missing runs (not trained or not yet collected): none
- Runs with a collapsed ambiguity head: `cebab_3class_distilbert_seed42_100ep_fixed_noale`, `cebab_3class_distilbert_seed123_100ep_fixed_noale`, `cebab_3class_distilbert_seed2024_100ep_fixed_noale`, `cebab_3class_distilbert_seed123_100ep_fixed_decorr5`, `hatexplain_distilbert_seed42_100ep`, `hatexplain_seed123_100ep`, `hatexplain_seed2024_100ep`, `goemotions_distilbert_seed42_100ep`, `goemotions_seed123_100ep`, `goemotions_seed2024_100ep`
- Runs with a collapsed epistemic head (EU metrics undefined): `cebab_3class_distilbert_seed42_100ep_fixed_decorr5`, `cebab_3class_distilbert_seed2024_100ep_fixed_decorr5`

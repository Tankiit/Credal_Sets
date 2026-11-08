# Concept-Audit Analysis Report

## Comparison across runs

| Run | Train | Eval | #Concepts | Rank | Unconstrained dim | Max logit error | Condition # | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|
| cub_dataset | 9430 | 2358 | 624 | 624 (full rank) | 624 | 1.96e-10 | 3.3152 | 0.8715 |
| cubs_dataset | 115 | 29 | 624 | 624 (full rank) | 624 | 5.46e-12 | 3.3152 | 0.2414 |
| shapes3d_dataset | 384000 | 96000 | 8 | 8 (full rank) | 8 | 9.09e-13 | 2.9886 | 0.9753 |

## cub_dataset

- Source: `ignore/results_experiments/cbm_results_seed12/cub_dataset/audit.json`
- backend=`native`, epochs=`100`, seed=`12`
- train_samples=`9430`, eval_samples=`2358`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`624` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 1.964508555829525e-10 | 3.315159408076245 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0018 mean=0.0681 max=0.2943 |
| cross_concept_abs_correlation | min=0.1213 mean=0.2036 max=0.3966 |
| own_correlation_share | min=0.0001 mean=0.0021 max=0.0091 |
| block_task_probe_accuracy | min=0.0127 mean=0.0223 max=0.0360 |
| block_head_frobenius_norm | min=17.7350 mean=23.6682 max=30.1239 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0018 mean=0.0687 max=0.2943 |
| cross_concept_abs_correlation | min=0.1166 mean=0.2034 max=0.3966 |
| own_correlation_share | min=0.0001 mean=0.0021 max=0.0089 |
| block_task_probe_accuracy | min=0.0119 mean=0.0222 max=0.0365 |
| block_head_frobenius_norm | min=19.6290 mean=25.7466 max=32.2965 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0018 mean=0.0681 max=0.2943 |
| cross_concept_abs_correlation | min=0.1213 mean=0.2036 max=0.3966 |
| own_correlation_share | min=0.0001 mean=0.0021 max=0.0091 |
| block_task_probe_accuracy | min=0.0127 mean=0.0223 max=0.0360 |
| block_head_frobenius_norm | min=0.0000 mean=23.6347 max=30.1239 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0018 mean=0.0680 max=0.2943 |
| cross_concept_abs_correlation | min=0.0725 mean=0.2033 max=0.3966 |
| own_correlation_share | min=0.0001 mean=0.0021 max=0.0091 |
| block_task_probe_accuracy | min=0.0038 mean=0.0222 max=0.0360 |
| block_head_frobenius_norm | min=17.7350 mean=23.6682 max=30.1239 |

### Consequence (substitution) test

- baseline_accuracy=`0.8715012669563293`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=-0.0017 mean=0.0005 max=0.0025

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 187 | 0.0025 |
| 199 | 0.0025 |
| 277 | 0.0025 |
| 516 | 0.0025 |
| 61 | 0.0021 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 359 | -0.0017 |
| 325 | -0.0013 |
| 388 | -0.0013 |
| 460 | -0.0013 |
| 463 | -0.0013 |

tensorboard_log_dir: `ignore/results_experiments/cbm_results_seed12/cub_dataset/tensorboard/real-native-coordinates-seed12-20260925T080019-9373b573`

---

## cubs_dataset

- Source: `ignore/results_experiments/cbm_results_seed12/cubs_dataset/audit.json`
- backend=`native`, epochs=`100`, seed=`12`
- train_samples=`115`, eval_samples=`29`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`624` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 5.4569682106375694e-12 | 3.315159408076245 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2235 max=0.6825 |
| cross_concept_abs_correlation | min=0.4621 mean=0.6192 max=0.8013 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0058 |
| block_task_probe_accuracy | min=0.0000 mean=0.0688 max=0.2069 |
| block_head_frobenius_norm | min=2.2573 mean=2.7637 max=3.2642 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2270 max=0.6825 |
| cross_concept_abs_correlation | min=0.4348 mean=0.6204 max=0.8189 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0054 |
| block_task_probe_accuracy | min=0.0000 mean=0.0687 max=0.2069 |
| block_head_frobenius_norm | min=2.2147 mean=2.9512 max=3.9434 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2235 max=0.6825 |
| cross_concept_abs_correlation | min=0.4621 mean=0.6192 max=0.8013 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0058 |
| block_task_probe_accuracy | min=0.0000 mean=0.0688 max=0.2069 |
| block_head_frobenius_norm | min=0.0000 mean=2.7588 max=3.2642 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2237 max=0.6825 |
| cross_concept_abs_correlation | min=0.4621 mean=0.6192 max=0.8013 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0058 |
| block_task_probe_accuracy | min=0.0000 mean=0.0687 max=0.2069 |
| block_head_frobenius_norm | min=2.2573 mean=2.7637 max=3.2642 |

### Consequence (substitution) test

- baseline_accuracy=`0.24137930572032928`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=-0.0345 mean=-0.0121 max=0.0000

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 0 | 0.0000 |
| 2 | 0.0000 |
| 3 | 0.0000 |
| 4 | 0.0000 |
| 6 | 0.0000 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 1 | -0.0345 |
| 5 | -0.0345 |
| 8 | -0.0345 |
| 9 | -0.0345 |
| 10 | -0.0345 |

tensorboard_log_dir: `ignore/results_experiments/cbm_results_seed12/cubs_dataset/tensorboard/real-native-coordinates-seed12-20260925T080016-9e6cf72d`

---

## shapes3d_dataset

- Source: `ignore/results_experiments/cbm_results_seed12/shapes3d_dataset/audit.json`
- backend=`native`, epochs=`100`, seed=`12`
- train_samples=`384000`, eval_samples=`96000`
- num_concepts=`8`, readout_rank=`8`, unconstrained_dim=`8` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 9.094947017729282e-13 | 2.9885752724179127 |

### equivalence.baseline (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| cross_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| own_correlation_share | min=0.4138 mean=0.4429 max=0.4692 |
| block_task_probe_accuracy | min=0.0512 mean=0.0645 max=0.0749 |
| block_head_frobenius_norm | min=16.1420 mean=21.7007 max=25.4787 |

### equivalence.transformed (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| cross_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| own_correlation_share | min=0.4072 mean=0.4438 max=0.4677 |
| block_task_probe_accuracy | min=0.0505 mean=0.0637 max=0.0747 |
| block_head_frobenius_norm | min=18.6034 mean=25.2505 max=30.5542 |

### structural (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| cross_concept_abs_correlation | min=0.7761 mean=0.8764 max=0.9412 |
| own_correlation_share | min=0.4138 mean=0.4429 max=0.4692 |
| block_task_probe_accuracy | min=0.0512 mean=0.0645 max=0.0749 |
| block_head_frobenius_norm | min=0.0000 mean=19.1368 max=25.4787 |

### informational (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0017 mean=0.7648 max=0.9412 |
| cross_concept_abs_correlation | min=0.0083 mean=0.7657 max=0.9412 |
| own_correlation_share | min=0.0416 mean=0.3922 max=0.4692 |
| block_task_probe_accuracy | min=0.0254 mean=0.0590 max=0.0749 |
| block_head_frobenius_norm | min=16.1420 mean=21.7007 max=25.4787 |

### Consequence (substitution) test

- baseline_accuracy=`0.9752500057220459`, replacement=`within_split_random_donor`
- accuracy_drop across 8 concepts: min=0.1550 mean=0.2453 max=0.3949

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 6 | 0.3949 |
| 0 | 0.3000 |
| 4 | 0.2918 |
| 7 | 0.2632 |
| 3 | 0.2158 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 1 | 0.1550 |
| 2 | 0.1635 |
| 5 | 0.1784 |
| 3 | 0.2158 |
| 7 | 0.2632 |

tensorboard_log_dir: `ignore/results_experiments/cbm_results_seed12/shapes3d_dataset/tensorboard/real-native-coordinates-seed12-20260925T080015-f6e097a2`

---

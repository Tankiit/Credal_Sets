# Concept-Audit Analysis Report

## Comparison across runs

| Run | Train | Eval | #Concepts | Rank | Unconstrained dim | Max logit error | Condition # | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|
| cub_dataset_results | 9430 | 2358 | 624 | 624 (full rank) | 624 | 4.37e-11 | 3.3077 | 0.8694 |
| cubs_dataset_results | 115 | 29 | 624 | 624 (full rank) | 624 | 1.14e-12 | 3.3077 | 0.2069 |
| shapes3d_dataset_results | 384000 | 96000 | 8 | 8 (full rank) | 8 | 7.39e-13 | 2.9424 | 0.9844 |

## cub_dataset_results

- Source: `ignore/results_experiments/cub_dataset_results/audit.json`
- backend=`native`, epochs=`100`, seed=`0`
- train_samples=`9430`, eval_samples=`2358`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`624` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 4.3655745685100555e-11 | 3.307747675534512 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0014 mean=0.0656 max=0.3126 |
| cross_concept_abs_correlation | min=0.1195 mean=0.2050 max=0.3816 |
| own_correlation_share | min=0.0000 mean=0.0020 max=0.0087 |
| block_task_probe_accuracy | min=0.0127 mean=0.0221 max=0.0339 |
| block_head_frobenius_norm | min=18.8347 mean=24.0519 max=30.9922 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0025 mean=0.0659 max=0.3126 |
| cross_concept_abs_correlation | min=0.1182 mean=0.2056 max=0.3957 |
| own_correlation_share | min=0.0001 mean=0.0020 max=0.0086 |
| block_task_probe_accuracy | min=0.0140 mean=0.0220 max=0.0348 |
| block_head_frobenius_norm | min=19.9972 mean=26.1869 max=33.1540 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0014 mean=0.0656 max=0.3126 |
| cross_concept_abs_correlation | min=0.1195 mean=0.2050 max=0.3816 |
| own_correlation_share | min=0.0000 mean=0.0020 max=0.0087 |
| block_task_probe_accuracy | min=0.0127 mean=0.0221 max=0.0339 |
| block_head_frobenius_norm | min=0.0000 mean=24.0131 max=30.9922 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0014 mean=0.0656 max=0.3126 |
| cross_concept_abs_correlation | min=0.0621 mean=0.2048 max=0.3816 |
| own_correlation_share | min=0.0000 mean=0.0020 max=0.0087 |
| block_task_probe_accuracy | min=0.0051 mean=0.0220 max=0.0339 |
| block_head_frobenius_norm | min=18.8347 mean=24.0519 max=30.9922 |

### Consequence (substitution) test

- baseline_accuracy=`0.8693808317184448`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=-0.0025 mean=-0.0004 max=0.0013

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 314 | 0.0013 |
| 563 | 0.0013 |
| 59 | 0.0008 |
| 62 | 0.0008 |
| 68 | 0.0008 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 493 | -0.0025 |
| 574 | -0.0025 |
| 70 | -0.0021 |
| 158 | -0.0021 |
| 269 | -0.0021 |

tensorboard_log_dir: `results/real/tensorboard/real-native-coordinates-seed0-20260917T153822-ff1b2428`

---

## cubs_dataset_results

- Source: `ignore/results_experiments/cubs_dataset_results/audit.json`
- backend=`native`, epochs=`100`, seed=`0`
- train_samples=`115`, eval_samples=`29`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`624` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 1.1368683772161603e-12 | 3.307747675534512 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2263 max=0.6125 |
| cross_concept_abs_correlation | min=0.4629 mean=0.6235 max=0.8694 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0049 |
| block_task_probe_accuracy | min=0.0000 mean=0.0585 max=0.1724 |
| block_head_frobenius_norm | min=2.2502 mean=2.6159 max=2.9250 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2265 max=0.6010 |
| cross_concept_abs_correlation | min=0.4396 mean=0.6216 max=0.8694 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0047 |
| block_task_probe_accuracy | min=0.0000 mean=0.0566 max=0.1724 |
| block_head_frobenius_norm | min=1.9942 mean=2.7722 max=3.8663 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2263 max=0.6125 |
| cross_concept_abs_correlation | min=0.4629 mean=0.6235 max=0.8694 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0049 |
| block_task_probe_accuracy | min=0.0000 mean=0.0585 max=0.1724 |
| block_head_frobenius_norm | min=0.0000 mean=2.6118 max=2.9250 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2268 max=0.6125 |
| cross_concept_abs_correlation | min=0.4629 mean=0.6235 max=0.8694 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0049 |
| block_task_probe_accuracy | min=0.0000 mean=0.0584 max=0.1724 |
| block_head_frobenius_norm | min=2.2502 mean=2.6159 max=2.9250 |

### Consequence (substitution) test

- baseline_accuracy=`0.2068965584039688`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=0.0000 mean=0.0000 max=0.0000

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 0 | 0.0000 |
| 1 | 0.0000 |
| 2 | 0.0000 |
| 3 | 0.0000 |
| 4 | 0.0000 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 0 | 0.0000 |
| 1 | 0.0000 |
| 2 | 0.0000 |
| 3 | 0.0000 |
| 4 | 0.0000 |

tensorboard_log_dir: `results_experiments/cubs_dataset/tensorboard/real-native-coordinates-seed0-20260924T131039-062cb18c`

---

## shapes3d_dataset_results

- Source: `ignore/results_experiments/shapes3d_dataset_results/audit.json`
- backend=`native`, epochs=`100`, seed=`0`
- train_samples=`384000`, eval_samples=`96000`
- num_concepts=`8`, readout_rank=`8`, unconstrained_dim=`8` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 7.389644451905042e-13 | 2.9424204753664016 |

### equivalence.baseline (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| cross_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| own_correlation_share | min=0.4278 mean=0.4450 max=0.4569 |
| block_task_probe_accuracy | min=0.0533 mean=0.0608 max=0.0677 |
| block_head_frobenius_norm | min=18.2215 mean=21.9192 max=33.6082 |

### equivalence.transformed (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| cross_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| own_correlation_share | min=0.4254 mean=0.4465 max=0.4609 |
| block_task_probe_accuracy | min=0.0541 mean=0.0596 max=0.0678 |
| block_head_frobenius_norm | min=18.2526 mean=28.4771 max=49.1555 |

### structural (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| cross_concept_abs_correlation | min=0.8322 mean=0.9048 max=0.9502 |
| own_correlation_share | min=0.4278 mean=0.4450 max=0.4569 |
| block_task_probe_accuracy | min=0.0533 mean=0.0608 max=0.0677 |
| block_head_frobenius_norm | min=0.0000 mean=19.5879 max=33.6082 |

### informational (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0018 mean=0.7951 max=0.9502 |
| cross_concept_abs_correlation | min=0.0048 mean=0.7955 max=0.9502 |
| own_correlation_share | min=0.0840 mean=0.4018 max=0.4569 |
| block_task_probe_accuracy | min=0.0237 mean=0.0557 max=0.0677 |
| block_head_frobenius_norm | min=18.2215 mean=21.9192 max=33.6082 |

### Consequence (substitution) test

- baseline_accuracy=`0.9844270944595337`, replacement=`within_split_random_donor`
- accuracy_drop across 8 concepts: min=0.0992 mean=0.2345 max=0.3533

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 7 | 0.3533 |
| 3 | 0.2823 |
| 4 | 0.2685 |
| 5 | 0.2629 |
| 2 | 0.2477 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 0 | 0.0992 |
| 1 | 0.1381 |
| 6 | 0.2242 |
| 2 | 0.2477 |
| 5 | 0.2629 |

tensorboard_log_dir: `ignore/results_experiments/shapes3d_dataset/tensorboard/real-native-coordinates-seed0-20260924T134914-0a1c2fc8`

---

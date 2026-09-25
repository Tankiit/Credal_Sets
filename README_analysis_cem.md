# Concept-Audit Analysis Report

## Comparison across runs

| Run | Train | Eval | #Concepts | Rank | Unconstrained dim | Max logit error | Condition # | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|
| cub_dataset | 9430 | 2358 | 624 | 624 (RANK DEFICIT) | 1872 | 6.84e-10 | 3.9893 | 0.8830 |
| cubs_dataset | 115 | 29 | 624 | 624 (full rank) | 624 | 3.41e-12 | 3.2289 | 0.3103 |
| shapes3d_dataset | 384000 | 96000 | 8 | 8 (full rank) | 8 | 7.39e-13 | 2.9424 | 0.9844 |

## cub_dataset

- Source: `ignore/results_experiments/cem_results/cub_dataset/audit.json`
- backend=`cem`, epochs=`100`, seed=`0`
- train_samples=`9430`, eval_samples=`2358`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`1872` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 6.83940015733242e-10 | 3.9892863517874915 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0063 mean=0.0682 max=0.2561 |
| cross_concept_abs_correlation | min=0.1216 mean=0.1952 max=0.5663 |
| own_correlation_share | min=0.0002 mean=0.0019 max=0.0075 |
| block_task_probe_accuracy | min=0.0293 mean=0.0578 max=0.0941 |
| block_head_frobenius_norm | min=23.3524 mean=33.4481 max=44.1868 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0063 mean=0.0685 max=0.2554 |
| cross_concept_abs_correlation | min=0.1225 mean=0.1937 max=0.5663 |
| own_correlation_share | min=0.0002 mean=0.0019 max=0.0070 |
| block_task_probe_accuracy | min=0.0293 mean=0.0576 max=0.0891 |
| block_head_frobenius_norm | min=28.5753 mean=37.5347 max=48.1315 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0063 mean=0.0682 max=0.2561 |
| cross_concept_abs_correlation | min=0.1216 mean=0.1952 max=0.5663 |
| own_correlation_share | min=0.0002 mean=0.0019 max=0.0075 |
| block_task_probe_accuracy | min=0.0293 mean=0.0578 max=0.0941 |
| block_head_frobenius_norm | min=0.0000 mean=33.3992 max=44.1868 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0063 mean=0.0681 max=0.2561 |
| cross_concept_abs_correlation | min=0.0744 mean=0.1950 max=0.5663 |
| own_correlation_share | min=0.0002 mean=0.0019 max=0.0075 |
| block_task_probe_accuracy | min=0.0055 mean=0.0577 max=0.0941 |
| block_head_frobenius_norm | min=23.3524 mean=33.4481 max=44.1868 |

### Consequence (substitution) test

- baseline_accuracy=`0.8829516768455505`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=-0.0021 mean=0.0001 max=0.0025

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 68 | 0.0025 |
| 388 | 0.0025 |
| 90 | 0.0017 |
| 118 | 0.0017 |
| 151 | 0.0017 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 612 | -0.0021 |
| 262 | -0.0013 |
| 267 | -0.0013 |
| 278 | -0.0013 |
| 336 | -0.0013 |

tensorboard_log_dir: `ignore/results_experiments/cem_results/cub_dataset/tensorboard/real-cem-coordinates-seed0-20260924T144812-2abb521f`

---

## cubs_dataset

- Source: `ignore/results_experiments/cem_results/cubs_dataset/audit.json`
- backend=`native`, epochs=`100`, seed=`0`
- train_samples=`115`, eval_samples=`29`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`624` ✅ full rank

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 3.410605131648481e-12 | 3.228857863059529 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2266 max=0.6436 |
| cross_concept_abs_correlation | min=0.4681 mean=0.6274 max=0.8155 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0053 |
| block_task_probe_accuracy | min=0.0000 mean=0.0559 max=0.2069 |
| block_head_frobenius_norm | min=2.3609 mean=2.6737 max=3.0177 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2251 max=0.6436 |
| cross_concept_abs_correlation | min=0.4463 mean=0.6292 max=0.8759 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0055 |
| block_task_probe_accuracy | min=0.0000 mean=0.0555 max=0.1724 |
| block_head_frobenius_norm | min=1.9351 mean=2.8295 max=4.0787 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2266 max=0.6436 |
| cross_concept_abs_correlation | min=0.4681 mean=0.6274 max=0.8155 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0053 |
| block_task_probe_accuracy | min=0.0000 mean=0.0559 max=0.2069 |
| block_head_frobenius_norm | min=0.0000 mean=2.6695 max=3.0177 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2264 max=0.6436 |
| cross_concept_abs_correlation | min=0.4681 mean=0.6273 max=0.8155 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0053 |
| block_task_probe_accuracy | min=0.0000 mean=0.0557 max=0.2069 |
| block_head_frobenius_norm | min=2.3609 mean=2.6737 max=3.0177 |

### Consequence (substitution) test

- baseline_accuracy=`0.3103448152542114`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=0.0000 mean=0.0098 max=0.0690

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 38 | 0.0690 |
| 220 | 0.0690 |
| 234 | 0.0690 |
| 299 | 0.0690 |
| 516 | 0.0690 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 0 | 0.0000 |
| 1 | 0.0000 |
| 3 | 0.0000 |
| 4 | 0.0000 |
| 6 | 0.0000 |

tensorboard_log_dir: `ignore/results_experiments/cem_results/cubs_dataset/tensorboard/real-native-coordinates-seed0-20260924T144821-2168a2c2`

---

## shapes3d_dataset

- Source: `ignore/results_experiments/cem_results/shapes3d_dataset/audit.json`
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

tensorboard_log_dir: `ignore/results_experiments/cem_results/shapes3d_dataset/tensorboard/real-native-coordinates-seed0-20260924T144826-61dd031c`

---

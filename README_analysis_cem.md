# Concept-Audit Analysis Report

## Comparison across runs

| Run | Train | Eval | #Concepts | Rank | Unconstrained dim | Max logit error | Condition # | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|
| cub_dataset | 9430 | 2358 | 624 | 624 (RANK DEFICIT) | 1872 | 6.84e-10 | 3.9893 | 0.8830 |
| cubs_dataset | 115 | 29 | 624 | 624 (RANK DEFICIT) | 1872 | 2.55e-11 | 3.9893 | 0.2414 |
| shapes3d_dataset | 384000 | 96000 | 8 | 8 (RANK DEFICIT) | 24 | 4.55e-12 | 2.9200 | 0.9933 |

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
- backend=`cem`, epochs=`100`, seed=`0`
- train_samples=`115`, eval_samples=`29`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`1872` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 2.546585164964199e-11 | 3.9892863517874915 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2773 max=0.6849 |
| cross_concept_abs_correlation | min=0.5271 mean=0.6625 max=0.8782 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0038 |
| block_task_probe_accuracy | min=0.0000 mean=0.0921 max=0.3103 |
| block_head_frobenius_norm | min=3.1989 mean=3.7048 max=4.2707 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2769 max=0.6679 |
| cross_concept_abs_correlation | min=0.5182 mean=0.6635 max=0.8885 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0037 |
| block_task_probe_accuracy | min=0.0000 mean=0.0889 max=0.2414 |
| block_head_frobenius_norm | min=3.1467 mean=4.1094 max=5.7740 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2773 max=0.6849 |
| cross_concept_abs_correlation | min=0.5271 mean=0.6625 max=0.8782 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0038 |
| block_task_probe_accuracy | min=0.0000 mean=0.0921 max=0.3103 |
| block_head_frobenius_norm | min=0.0000 mean=3.6988 max=4.2707 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2773 max=0.6849 |
| cross_concept_abs_correlation | min=0.5271 mean=0.6625 max=0.8782 |
| own_correlation_share | min=0.0000 mean=0.0016 max=0.0038 |
| block_task_probe_accuracy | min=0.0000 mean=0.0921 max=0.3103 |
| block_head_frobenius_norm | min=3.1989 mean=3.7048 max=4.2707 |

### Consequence (substitution) test

- baseline_accuracy=`0.24137930572032928`, replacement=`within_split_random_donor`
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

tensorboard_log_dir: `ignore/results_experiments/cem_results/cubs_dataset/tensorboard/real-cem-coordinates-seed0-20260925T071340-7dbe1568`

---

## shapes3d_dataset

- Source: `ignore/results_experiments/cem_results/shapes3d_dataset/audit.json`
- backend=`cem`, epochs=`100`, seed=`0`
- train_samples=`384000`, eval_samples=`96000`
- num_concepts=`8`, readout_rank=`8`, unconstrained_dim=`24` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 1.554312234475219e-14 | 4.547473508864641e-12 | 2.9199856832518103 |

### equivalence.baseline (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| cross_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| own_correlation_share | min=0.3471 mean=0.3834 max=0.4095 |
| block_task_probe_accuracy | min=0.1195 mean=0.1562 max=0.1935 |
| block_head_frobenius_norm | min=24.8046 mean=41.7987 max=64.2845 |

### equivalence.transformed (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| cross_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| own_correlation_share | min=0.3705 mean=0.4018 max=0.4277 |
| block_task_probe_accuracy | min=0.1503 mean=0.1848 max=0.2333 |
| block_head_frobenius_norm | min=32.9113 mean=46.2866 max=69.1552 |

### structural (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| cross_concept_abs_correlation | min=0.7659 mean=0.8931 max=0.9422 |
| own_correlation_share | min=0.3471 mean=0.3834 max=0.4095 |
| block_task_probe_accuracy | min=0.1195 mean=0.1562 max=0.1935 |
| block_head_frobenius_norm | min=0.0000 mean=33.7632 max=53.4368 |

### informational (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0038 mean=0.7810 max=0.9422 |
| cross_concept_abs_correlation | min=0.0057 mean=0.7813 max=0.9422 |
| own_correlation_share | min=0.1095 mean=0.3459 max=0.4017 |
| block_task_probe_accuracy | min=0.0246 mean=0.1362 max=0.1935 |
| block_head_frobenius_norm | min=24.8046 mean=41.7987 max=64.2845 |

### Consequence (substitution) test

- baseline_accuracy=`0.9933333396911621`, replacement=`within_split_random_donor`
- accuracy_drop across 8 concepts: min=0.0142 mean=0.2059 max=0.4790

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 7 | 0.4790 |
| 0 | 0.4021 |
| 2 | 0.2042 |
| 5 | 0.1777 |
| 1 | 0.1730 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 4 | 0.0142 |
| 3 | 0.0460 |
| 6 | 0.1507 |
| 1 | 0.1730 |
| 5 | 0.1777 |

tensorboard_log_dir: `ignore/results_experiments/cem_results/shapes3d_dataset/tensorboard/real-cem-coordinates-seed0-20260925T071345-1398babd`

---

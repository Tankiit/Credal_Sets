# Concept-Audit Analysis Report

## Comparison across runs

| Run | Train | Eval | #Concepts | Rank | Unconstrained dim | Max logit error | Condition # | Baseline accuracy |
|---|---|---|---|---|---|---|---|---|
| cub_dataset | 9430 | 2358 | 624 | 624 (RANK DEFICIT) | 1872 | 5.38e-10 | 3.9581 | 0.8745 |
| cubs_dataset | 115 | 29 | 624 | 624 (RANK DEFICIT) | 1872 | 1.91e-11 | 3.9581 | 0.2069 |
| shapes3d_dataset | 384000 | 96000 | 8 | 8 (RANK DEFICIT) | 24 | 4.55e-12 | 3.2992 | 0.9846 |

## cub_dataset

- Source: `ignore/results_experiments/cem_results_seed12/cub_dataset/audit.json`
- backend=`cem`, epochs=`100`, seed=`12`
- train_samples=`9430`, eval_samples=`2358`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`1872` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 5.384208634495735e-10 | 3.95812934954447 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0111 mean=0.0655 max=0.2582 |
| cross_concept_abs_correlation | min=0.1201 mean=0.1929 max=0.3731 |
| own_correlation_share | min=0.0003 mean=0.0018 max=0.0063 |
| block_task_probe_accuracy | min=0.0344 mean=0.0583 max=0.0848 |
| block_head_frobenius_norm | min=24.4657 mean=33.6420 max=44.1107 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0103 mean=0.0653 max=0.2550 |
| cross_concept_abs_correlation | min=0.1134 mean=0.1923 max=0.3659 |
| own_correlation_share | min=0.0003 mean=0.0018 max=0.0065 |
| block_task_probe_accuracy | min=0.0331 mean=0.0585 max=0.0912 |
| block_head_frobenius_norm | min=28.5530 mean=37.7251 max=47.5535 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0111 mean=0.0655 max=0.2582 |
| cross_concept_abs_correlation | min=0.1201 mean=0.1929 max=0.3731 |
| own_correlation_share | min=0.0003 mean=0.0018 max=0.0063 |
| block_task_probe_accuracy | min=0.0344 mean=0.0583 max=0.0848 |
| block_head_frobenius_norm | min=0.0000 mean=33.5914 max=44.1107 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0111 mean=0.0655 max=0.2582 |
| cross_concept_abs_correlation | min=0.0807 mean=0.1928 max=0.3731 |
| own_correlation_share | min=0.0003 mean=0.0018 max=0.0063 |
| block_task_probe_accuracy | min=0.0042 mean=0.0582 max=0.0848 |
| block_head_frobenius_norm | min=24.4657 mean=33.6420 max=44.1107 |

### Consequence (substitution) test

- baseline_accuracy=`0.8744698762893677`, replacement=`within_split_random_donor`
- accuracy_drop across 624 concepts: min=-0.0025 mean=-0.0004 max=0.0013

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 157 | 0.0013 |
| 191 | 0.0013 |
| 197 | 0.0013 |
| 212 | 0.0013 |
| 352 | 0.0013 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 176 | -0.0025 |
| 17 | -0.0021 |
| 53 | -0.0021 |
| 30 | -0.0017 |
| 43 | -0.0017 |

tensorboard_log_dir: `ignore/results_experiments/cem_results_seed12/cub_dataset/tensorboard/real-cem-coordinates-seed12-20260925T075908-93c2fb4d`

---

## cubs_dataset

- Source: `ignore/results_experiments/cem_results_seed12/cubs_dataset/audit.json`
- backend=`cem`, epochs=`100`, seed=`12`
- train_samples=`115`, eval_samples=`29`
- num_concepts=`624`, readout_rank=`624`, unconstrained_dim=`1872` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 0.0 | 1.9099388737231493e-11 | 3.95812934954447 |

### equivalence.baseline (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2949 max=0.6911 |
| cross_concept_abs_correlation | min=0.5236 mean=0.6606 max=0.8519 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0037 |
| block_task_probe_accuracy | min=0.0000 mean=0.1080 max=0.2759 |
| block_head_frobenius_norm | min=3.4769 mean=3.8134 max=4.1141 |

### equivalence.transformed (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2956 max=0.6819 |
| cross_concept_abs_correlation | min=0.5346 mean=0.6580 max=0.8519 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0040 |
| block_task_probe_accuracy | min=0.0000 mean=0.1072 max=0.2759 |
| block_head_frobenius_norm | min=3.2655 mean=4.2345 max=5.4776 |

### structural (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2949 max=0.6911 |
| cross_concept_abs_correlation | min=0.5236 mean=0.6606 max=0.8519 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0037 |
| block_task_probe_accuracy | min=0.0000 mean=0.1080 max=0.2759 |
| block_head_frobenius_norm | min=0.0000 mean=3.8075 max=4.1141 |

### informational (aggregated over 624 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0000 mean=0.2948 max=0.6911 |
| cross_concept_abs_correlation | min=0.5236 mean=0.6606 max=0.8519 |
| own_correlation_share | min=0.0000 mean=0.0017 max=0.0037 |
| block_task_probe_accuracy | min=0.0000 mean=0.1079 max=0.2759 |
| block_head_frobenius_norm | min=3.4769 mean=3.8134 max=4.1141 |

### Consequence (substitution) test

- baseline_accuracy=`0.20689654350280762`, replacement=`within_split_random_donor`
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

tensorboard_log_dir: `ignore/results_experiments/cem_results_seed12/cubs_dataset/tensorboard/real-cem-coordinates-seed12-20260925T075913-1a33365d`

---

## shapes3d_dataset

- Source: `ignore/results_experiments/cem_results_seed12/shapes3d_dataset/audit.json`
- backend=`cem`, epochs=`100`, seed=`12`
- train_samples=`384000`, eval_samples=`96000`
- num_concepts=`8`, readout_rank=`8`, unconstrained_dim=`24` ⚠️ RANK DEFICIT

### Equivalence check

| max_readout_error | max_logit_error | condition_number |
|---|---|---|
| 1.509903313490213e-14 | 4.547473508864641e-12 | 3.2992332914070195 |

### equivalence.baseline (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| cross_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| own_correlation_share | min=0.3391 mean=0.3776 max=0.4165 |
| block_task_probe_accuracy | min=0.1016 mean=0.1574 max=0.1982 |
| block_head_frobenius_norm | min=19.1200 mean=38.5779 max=62.4083 |

### equivalence.transformed (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| cross_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| own_correlation_share | min=0.3513 mean=0.3885 max=0.4361 |
| block_task_probe_accuracy | min=0.1449 mean=0.1818 max=0.2537 |
| block_head_frobenius_norm | min=30.8408 mean=47.0155 max=76.9999 |

### structural (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| cross_concept_abs_correlation | min=0.7535 mean=0.8561 max=0.9264 |
| own_correlation_share | min=0.3391 mean=0.3776 max=0.4165 |
| block_task_probe_accuracy | min=0.1016 mean=0.1574 max=0.1982 |
| block_head_frobenius_norm | min=0.0000 mean=34.9203 max=62.4083 |

### informational (aggregated over 8 concepts)

| field | min / mean / max |
|---|---|
| own_concept_abs_correlation | min=0.0038 mean=0.7521 max=0.9264 |
| cross_concept_abs_correlation | min=0.0060 mean=0.7523 max=0.9264 |
| own_correlation_share | min=0.0995 mean=0.3424 max=0.4165 |
| block_task_probe_accuracy | min=0.0252 mean=0.1453 max=0.1982 |
| block_head_frobenius_norm | min=19.1200 mean=38.5779 max=62.4083 |

### Consequence (substitution) test

- baseline_accuracy=`0.9845728874206543`, replacement=`within_split_random_donor`
- accuracy_drop across 8 concepts: min=0.0020 mean=0.1946 max=0.4216

**Most impactful concepts (largest accuracy drop when substituted):**

| concept_id | accuracy_drop |
|---|---|
| 2 | 0.4216 |
| 3 | 0.3055 |
| 5 | 0.2771 |
| 4 | 0.2356 |
| 7 | 0.1710 |

**Least impactful / negative-drop concepts:**

| concept_id | accuracy_drop |
|---|---|
| 1 | 0.0020 |
| 6 | 0.0602 |
| 0 | 0.0840 |
| 7 | 0.1710 |
| 4 | 0.2356 |

tensorboard_log_dir: `ignore/results_experiments/cem_results_seed12/shapes3d_dataset/tensorboard/real-cem-coordinates-seed12-20260925T075903-45ccda9a`

---

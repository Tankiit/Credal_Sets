# Modal CBM/CEM sweep

24 runs: 4 datasets × 2 architectures × 3 seeds; 50 epochs/run.

## Test aggregation

| dataset | architecture | n | test_accuracy_mean | test_accuracy_std | test_task_loss_mean | test_task_loss_std | best_epoch_mean | best_epoch_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cebab | cbm | 3 | 0.6238 | 0.0166 | 0.8965 | 0.0185 | 49.3333 | 1.1547 |
| cebab | cem | 3 | 0.6560 | 0.0033 | 0.7994 | 0.0084 | 3.6667 | 0.5774 |
| civil_comments | cbm | 3 | 0.7778 | 0.0011 | 0.4680 | 0.0006 | 41.0000 | 3.6056 |
| civil_comments | cem | 3 | 0.7785 | 0.0028 | 0.4601 | 0.0021 | 2.3333 | 0.5774 |
| goemotions | cbm | 3 | 0.6300 | 0.0030 | 0.9051 | 0.0024 | 22.6667 | 1.1547 |
| goemotions | cem | 3 | 0.6121 | 0.0011 | 0.9230 | 0.0044 | 2.0000 | 0.0000 |
| imdb_cad | cbm | 3 | 0.8700 | 0.0092 | 0.3160 | 0.0017 | 50.0000 | 0.0000 |
| imdb_cad | cem | 3 | 0.8860 | 0.0053 | 0.3239 | 0.0078 | 2.0000 | 0.0000 |

`summary.csv` contains all aggregated metrics; `per_run.csv` contains every seed-level result.

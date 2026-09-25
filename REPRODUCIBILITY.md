# Living reproducibility workflow

Reproducibility is part of launching an experiment, not a cleanup step. Every
reported number must have a run manifest, an immutable output directory, and a
seed-specific result file.

## Launch every experiment through the wrapper

Local example:

```bash
python scripts/repro_run.py \
  --name cebab_3class_100ep \
  --seed 123 \
  --track configs/cebab.yaml \
  -- python train.py --dataset cebab --epochs 100 --seed 123
```

Modal example:

```bash
python scripts/repro_run.py \
  --name icml_table6_100ep \
  --seed 123 \
  --track modal_icml_2026_multiseed.py \
  -- modal run --detach modal_icml_2026_multiseed.py \
  --datasets cebab,hatexplain,goemotions,maqa \
  --cebab-three-class --seeds 123 \
  --epochs-override 100 --run-tag 100ep
```

The wrapper creates:

```text
repro_runs/<experiment>/<UTC timestamp>_seed<seed>/
├── manifest.json
└── requirements.freeze.txt
```

The manifest records the exact command, seed, start/end time, exit status, Git
commit and branch, dirty-worktree state, diff checksum, Python/platform details,
and checksums of explicitly tracked files and Python entrypoints.

## Rules for paper experiments

1. Use a new output directory for every dataset, seed, epoch budget, label
   mapping, and ablation. Never reuse or overwrite an earlier run directory.
2. Pin repository commits and package versions. A branch name alone is mutable.
3. Record the seed in both the manifest and saved test metrics.
4. Seed Python, NumPy, PyTorch, and CUDA. Disable cuDNN benchmarking when
   deterministic execution is required.
5. Record dataset source, revision, split names, filtering, and label mapping.
   A dataset name alone is insufficient.
6. Save final per-example targets, predictions, epistemic uncertainty, and
   aleatoric uncertainty. Aggregate JSON alone cannot recover AUROC later.
7. Preserve `test_metrics.json`, training history, best checkpoint, run
   metadata, and the per-example prediction file together.
8. Aggregate only runs with matching commit, configuration, dataset revision,
   label mapping, encoder, epoch budget, and metric implementation.
9. Treat an undefined correlation or a constant uncertainty vector as a failed
   diagnostic, not as a meaningful zero.
10. Before updating a paper table, verify the manifest and copy its run IDs into
    the table-generation input.

## ICML 2026 replication convention

- Seeds: `123`, `2024`
- Encoder: `distilbert-base-uncased`
- Loss: `v7b`
- Epochs for Table 6 replication: `100`
- CEBaB task: 1–2 stars = negative, 3 = neutral, 4–5 = positive; discard
  `no majority`; use `train_inclusive`, `validation`, and `test`
- MAQA task: combined MAQA-Star and AmbigQA-Star
- Modal result volume: `icml-2026-credal-results`
- Result suffix: `_100ep`

The Modal launcher is pinned to ICML branch commit
`532fd05f67238f4459c371e2dbb14bca3bdecd59`. MAQA support uses locally injected
files, whose checksums must be captured with `--track` when they change.

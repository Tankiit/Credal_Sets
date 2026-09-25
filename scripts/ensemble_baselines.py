"""Deep-ensemble and MC-dropout error-detection baselines from saved test arrays.

    python scripts/ensemble_baselines.py --root outputs/icml_2026_reeval \
        --group cebab_3class_distilbert_seed{}_100ep_fixed --seeds 42,123,2024

The seed runs of one configuration form a deep ensemble: task probabilities are
averaged, and each score is used to rank the ensemble's own errors. MC-dropout
scores (written by reeval_icml_modal.py with REEVAL_MC>0) are scored per seed
against that seed's errors. Writes <root>/baselines_<group>.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 2000


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


def auroc_ci(score, err, seed=0):
    rng = np.random.default_rng(seed)
    n = len(err)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if err[i].min() != err[i].max():
            vals.append(roc_auc_score(err[i], score[i]))
    return [float(roc_auc_score(err, score)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--group", required=True, help="run-id pattern with {} for the seed")
    p.add_argument("--seeds", default="42,123,2024")
    args = p.parse_args()
    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    arrays = [dict(np.load(root / args.group.format(s) / "test_arrays.npz")) for s in seeds]
    y = arrays[0]["y_true"]
    assert all(np.array_equal(a["y_true"], y) for a in arrays), "members were evaluated on different test orders"

    out = {"group": args.group, "seeds": seeds, "n": int(len(y))}
    probs = np.stack([a["probs"] for a in arrays])          # [M, N, J]
    mean = probs.mean(0)
    err = (mean.argmax(-1) != y).astype(float)
    eu = np.stack([a["eu"].mean(-1) for a in arrays]).mean(0)
    out["ensemble"] = {
        "accuracy": float(1 - err.mean()),
        "auroc_maxprob": auroc_ci(1 - mean.max(-1), err),
        "auroc_pred_entropy": auroc_ci(entropy(mean), err),
        "auroc_mutual_info": auroc_ci(entropy(mean) - entropy(probs).mean(0), err),
        "auroc_cbm_eu_mean": auroc_ci(eu, err),
    }
    per_seed = {}
    for s, a in zip(seeds, arrays):
        if "mc_probs" not in a:
            continue
        e = (a["mc_probs"].argmax(-1) != y).astype(float)
        per_seed[s] = {
            "accuracy": float(1 - e.mean()),
            "auroc_mc_maxprob": auroc_ci(1 - a["mc_probs"].max(-1), e, s),
            "auroc_mc_entropy": auroc_ci(a["mc_entropy"], e, s),
            "auroc_mc_mutual_info": auroc_ci(a["mc_mutual_info"], e, s),
        }
    if per_seed:
        out["mc_dropout"] = per_seed
    name = args.group.replace("{}", "X")
    (root / f"baselines_{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

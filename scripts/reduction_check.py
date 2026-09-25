"""Recompute Table 1 EU metrics under alternative epistemic-score reductions.

Usage: python reduction_check.py <reeval_root>
CBM arrays store per-concept log sigma_epi [N, C]; MAQA stores sigma_epi [N].
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

root = Path(sys.argv[1])


def variants(eu, kind):
    if kind == "maqa":
        return {"sigma (reported)": eu, "log sigma": np.log(eu)}
    sig = np.exp(eu)
    return {
        "mean log sigma (reported)": eu.mean(-1),
        "sum log sigma = 1/2 logdet": eu.sum(-1),
        "mean sigma": sig.mean(-1),
        "sum sigma^2 (trace)": (sig ** 2).sum(-1),
    }


rows = []
for run in sorted(p for p in root.iterdir() if (p / "test_arrays.npz").exists()):
    a = np.load(run / "test_arrays.npz")
    kind = "maqa" if run.name.startswith("maqa") else "cbm"
    err = (a["y_true"] != a["y_pred"]).astype(float)
    au = a["au_no_H_input"] if kind == "maqa" else a["au"].mean(-1)
    for name, u in variants(a["eu"], kind).items():
        rows.append((run.name, name,
                     spearmanr(u, au)[0], spearmanr(u, err)[0], roc_auc_score(err, u)))

print(f"{'run':42s} {'U_epi variant':30s} {'rho(EU,AU)':>10s} {'rho(EU,err)':>11s} {'AUROC':>7s}")
last = None
for r in rows:
    if last and r[0] != last:
        print()
    print(f"{r[0]:42s} {r[1]:30s} {r[2]:10.3f} {r[3]:11.3f} {r[4]:7.3f}")
    last = r[0]

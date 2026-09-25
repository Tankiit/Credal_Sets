"""Stratified AUROC, quadrant counts, and paired-bootstrap CIs from saved test arrays.

    python scripts/extra_diagnostics.py --root outputs/icml_2026_reeval

CEBaB uses H averaged over annotated aspects only (cebab_masked_rescore.json
documents the mask). Scores whose per-example values are constant are reported
as undefined rather than as numbers. Writes <root>/extra_diagnostics.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
COLLAPSED_SD = 1e-3


def auroc(score, err):
    if err.min() == err.max() or np.std(score) < 1e-12:
        return math.nan
    return float(roc_auc_score(err, score))


def paired_boot(err, a, b, seed):
    """95% CIs for AUROC(a), AUROC(b), and AUROC(b) - AUROC(a) on shared resamples."""
    rng = np.random.default_rng(seed)
    n = len(err)
    ra, rb, d = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        x, y = auroc(a[i], err[i]), auroc(b[i], err[i])
        if math.isfinite(x) and math.isfinite(y):
            ra.append(x), rb.append(y), d.append(y - x)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    d = np.array(d)
    return {"auroc_eu": [auroc(a, err), ci(ra)], "auroc_maxprob": [auroc(b, err), ci(rb)],
            "delta_maxprob_minus_eu": [auroc(b, err) - auroc(a, err), ci(d)],
            "p_delta_le_0": float((d <= 0).mean())}


def load_run(run_dir, cebab_mask):
    a = dict(np.load(run_dir / "test_arrays.npz"))
    kind = json.loads((run_dir / "run_manifest.json").read_text())["kind"]
    err = (a["y_true"] != a["y_pred"]).astype(float)
    eu = a["eu"].mean(1) if a["eu"].ndim > 1 else a["eu"]
    if kind == "maqa":
        au, conf = a["au_no_H_input"], 1 - a["maxprob"] if "maxprob" in a else None
    else:
        au = a["au"].mean(1) if a["au"].ndim > 1 else a["au"]
        conf = 1 - a["probs"].max(1)
    # Collapsed if the per-example AU score barely moves (HateXplain s2024 spans 0.9986-0.99995).
    au_const = bool(np.std(au) < COLLAPSED_SD)
    H = a.get("H")
    if H is not None and kind == "cebab3":
        H = np.where(cebab_mask, H, 0).sum(1) / np.maximum(cebab_mask.sum(1), 1)
    elif H is not None and H.ndim > 1:
        H = H.mean(1)
    return kind, err, eu, au, conf, H, au_const


def strata(H, err, eu, conf, kind):
    if kind == "cebab3":  # paper rule: H = 0, then split the rest at the 2/3 quantile
        q = np.quantile(H, 2 / 3)
        bins = {"low (H=0)": H == 0, "med": (H > 0) & (H < q), f"high (H>={q:.3f})": (H > 0) & (H >= q)}
    elif len(np.unique(H)) <= 3:  # binary/ternary target: one stratum per value
        bins = {f"H={v:.3f}": H == v for v in np.unique(H)}
    else:
        c = np.quantile(H, [1 / 3, 2 / 3])
        bins = {f"low (H<{c[0]:.3f})": H < c[0], "med": (H >= c[0]) & (H < c[1]), f"high (H>={c[1]:.3f})": H >= c[1]}
    out = {}
    for name, m in bins.items():
        out[name] = {"n": int(m.sum()), "errors": int(err[m].sum()),
                     "H_range": [float(H[m].min()), float(H[m].max())],
                     "auroc_eu": auroc(eu[m], err[m]),
                     "auroc_maxprob": auroc(conf[m], err[m]) if conf is not None else None}
    return out


def quadrants(err, eu, au, H, au_const):
    if au_const:
        return {"undefined": "AU head is constant; the median split on AU is meaningless"}
    hi_eu, hi_au = eu > np.median(eu), au > np.median(au)
    out = {}
    for name, m in (("trust", ~hi_eu & ~hi_au), ("data", hi_eu & ~hi_au),
                    ("review", ~hi_eu & hi_au), ("abstain", hi_eu & hi_au)):
        out[name] = {"n": int(m.sum()), "accuracy": float(1 - err[m].mean()) if m.any() else math.nan,
                     "mean_H": float(H[m].mean()) if H is not None and m.any() else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    root = Path(p.parse_args().root)
    spec = importlib.util.spec_from_file_location("rescore", Path(__file__).with_name("rescore_cebab_masked.py"))
    rescore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rescore)
    cebab_mask = rescore.raw_masks()[1]

    report = {}
    for run_dir in sorted(d for d in root.iterdir() if (d / "test_arrays.npz").exists()):
        seed = json.loads((run_dir / "run_manifest.json").read_text())["seed"]
        kind, err, eu, au, conf, H, au_const = load_run(run_dir, cebab_mask)
        r = {"kind": kind, "n": len(err), "au_constant": au_const,
             "quadrants": quadrants(err, eu, au, H, au_const)}
        if conf is not None:
            r["paired_bootstrap"] = paired_boot(err, eu, conf, seed)
        if H is not None:
            r["strata"] = strata(H, err, eu, conf, kind)
        report[run_dir.name] = r
    (root / "extra_diagnostics.json").write_text(json.dumps(report, indent=2))

    for run, r in report.items():
        print(f"\n== {run} (n={r['n']})")
        pb = r.get("paired_bootstrap")
        if pb:
            f = lambda v: f"{v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]"
            print(f"  AUROC EU {f(pb['auroc_eu'])} | MaxProb {f(pb['auroc_maxprob'])} | "
                  f"Δ {f(pb['delta_maxprob_minus_eu'])}")
        for name, s in r.get("strata", {}).items():
            mp = f"{s['auroc_maxprob']:.3f}" if s["auroc_maxprob"] is not None else "n/a"
            print(f"  stratum {name:22s} n={s['n']:4d} err={s['errors']:4d} EU={s['auroc_eu']:.3f} MaxProb={mp}")
        q = r["quadrants"]
        print("  quadrants:", q.get("undefined") or
              {k: (v["n"], round(v["accuracy"], 3), None if v["mean_H"] is None else round(v["mean_H"], 3))
               for k, v in q.items()})


if __name__ == "__main__":
    main()

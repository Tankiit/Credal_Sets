"""Compute headline metrics with confidence intervals from re-evaluated test arrays.

Reads <root>/<run_id>/{test_arrays.npz,run_manifest.json} written by
reeval_icml_modal.py and writes per-run metrics plus per-configuration
seed summaries (mean and sample SD across seeds, with the seed list).

CIs: Fisher z for Spearman rho (normal approximation, n-3 df) and a paired
percentile bootstrap over test examples (2000 resamples) for every metric.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

N_BOOT = 2000


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def auroc(score, err):
    if err.min() == err.max() or np.std(score) == 0:
        return math.nan
    return float(roc_auc_score(err, score))


def fisher_ci(r, n):
    if not math.isfinite(r):
        return [math.nan, math.nan]
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)]


def per_example(arrays, kind):
    y, yhat = arrays["y_true"], arrays["y_pred"]
    err = (y != yhat).astype(float)
    eu = arrays["eu"]
    eu = eu.mean(-1) if eu.ndim > 1 else eu
    if kind == "maqa":
        au = arrays["au_no_H_input"]
        au_leaky = arrays["au_with_H_input"]
    else:
        au = arrays["au"].mean(-1) if arrays["au"].ndim > 1 else arrays["au"]
        au_leaky = None
    H = arrays.get("H")
    if H is not None and H.ndim > 1:
        H = H.mean(-1)
    conf = 1 - arrays["probs"].max(-1) if "probs" in arrays else None
    return err, eu, au, H, conf, au_leaky


def metric_fns(err, eu, au, H, conf):
    fns = {
        "accuracy": lambda i: 1 - err[i].mean(),
        "rho_eu_au": lambda i: spearman(eu[i], au[i]),
        "rho_eu_err": lambda i: spearman(eu[i], err[i]),
        "auroc_eu_err": lambda i: auroc(eu[i], err[i]),
    }
    if H is not None:
        fns["rho_au_H"] = lambda i: spearman(au[i], H[i])
    if conf is not None:
        fns["auroc_maxprob_err"] = lambda i: auroc(conf[i], err[i])
    return fns


def aurc(score, err):
    """Area under the risk-coverage curve when abstaining on the highest scores first."""
    order = np.argsort(score, kind="stable")
    risks = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    return float(risks.mean())


def diagnostics(err, eu, au, H, conf):
    """Descriptive analyses on the test set; thresholds are test tertiles/medians."""
    out = {"aurc": {"eu": aurc(eu, err), "au": aurc(au, err)}}
    if conf is not None:
        out["aurc"]["maxprob"] = aurc(conf, err)
    out["aurc"]["oracle"] = aurc(err + 1e-9 * np.random.default_rng(0).random(len(err)), err)
    if H is not None and np.std(H) > 0:
        cuts = np.quantile(H, [1 / 3, 2 / 3])
        bins = np.digitize(H, cuts)
        strat = {}
        for b, name in enumerate(("low", "med", "high")):
            m = bins == b
            strat[name] = {
                "n": int(m.sum()),
                "errors": int(err[m].sum()),
                "H_range": [float(H[m].min()), float(H[m].max())] if m.any() else None,
                "auroc_eu": auroc(eu[m], err[m]) if m.any() else math.nan,
                "auroc_maxprob": auroc(conf[m], err[m]) if conf is not None and m.any() else math.nan,
            }
        out["ambiguity_strata"] = {"cuts": cuts.tolist(), "bins": strat}
    hi_eu, hi_au = eu > np.median(eu), au > np.median(au)
    quad = {}
    for name, m in (("trust", ~hi_eu & ~hi_au), ("data", hi_eu & ~hi_au),
                    ("review", ~hi_eu & hi_au), ("abstain", hi_eu & hi_au)):
        quad[name] = {
            "n": int(m.sum()),
            "accuracy": float(1 - err[m].mean()) if m.any() else math.nan,
            "mean_H": float(H[m].mean()) if H is not None and m.any() else None,
        }
    out["quadrants"] = quad
    return out


def summarize_run(run_dir: Path):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    arrays = dict(np.load(run_dir / "test_arrays.npz"))
    err, eu, au, H, conf, au_leaky = per_example(arrays, manifest["kind"])
    n = len(err)
    fns = metric_fns(err, eu, au, H, conf)
    full = np.arange(n)
    rng = np.random.default_rng(manifest["seed"])
    boot = defaultdict(list)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for name, fn in fns.items():
            boot[name].append(fn(idx))
    result = {"run_id": manifest["run_id"], "n": n, "manifest": manifest, "au_constant": bool(np.std(au) < 1e-4)}
    for name, fn in fns.items():
        value = fn(full)
        samples = np.asarray([b for b in boot[name] if math.isfinite(b)])
        entry = {"value": value}
        if samples.size:
            entry["boot95"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        if name.startswith("rho"):
            entry["fisher95"] = fisher_ci(value, n)
        result[name] = entry
    if au_leaky is not None:
        result["rho_au_H_leaky_input"] = {"value": spearman(au_leaky, H)}
    result["diagnostics"] = diagnostics(err, eu, au, H, conf)
    return result


def config_key(run_id: str) -> str:
    return run_id.replace("_seed123", "").replace("_seed2024", "").replace("_seed42", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    runs = [summarize_run(d) for d in sorted(root.iterdir()) if (d / "test_arrays.npz").exists()]
    (root / "per_run_metrics.json").write_text(json.dumps(runs, indent=2))

    groups = defaultdict(list)
    for r in runs:
        groups[config_key(r["run_id"])].append(r)
    summary = {}
    for key, members in groups.items():
        entry = {"seeds": [m["manifest"]["seed"] for m in members], "n_test": [m["n"] for m in members]}
        for metric in ("accuracy", "rho_eu_au", "rho_eu_err", "rho_au_H", "auroc_eu_err", "auroc_maxprob_err"):
            vals = [m[metric]["value"] for m in members if metric in m]
            if not vals:
                continue
            finite = [v for v in vals if math.isfinite(v)]
            entry[metric] = {
                "per_seed": vals,
                "mean": float(np.mean(finite)) if finite else math.nan,
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
            }
        entry["au_constant"] = [m["au_constant"] for m in members]
        summary[key] = entry
    (root / "seed_summary.json").write_text(json.dumps(summary, indent=2))

    for key, e in summary.items():
        cells = []
        for metric in ("accuracy", "rho_eu_au", "rho_eu_err", "rho_au_H", "auroc_eu_err", "auroc_maxprob_err"):
            if metric in e:
                cells.append(f"{metric}={e[metric]['mean']:.3f}±{e[metric]['sd']:.3f} {['%.3f' % v for v in e[metric]['per_seed']]}")
        print(key, e["seeds"], e["n_test"], "AUconst" if any(e["au_constant"]) else "", "\n   " + "\n   ".join(cells))


if __name__ == "__main__":
    main()

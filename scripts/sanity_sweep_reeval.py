"""Flag degenerate or invalid values in every re-evaluated test array.

    python scripts/sanity_sweep_reeval.py --root outputs/icml_2026_reeval

Checks: non-finite values, constant score columns, probabilities that do not
sum to one, out-of-range labels/entropies, ties that make rank metrics
unreliable, and entropy targets that take very few distinct values.
Writes <root>/sanity_sweep.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONST_SD = 1e-4


def check_run(run_dir):
    a = dict(np.load(run_dir / "test_arrays.npz"))
    issues, facts = [], {}
    for k, v in a.items():
        if v.dtype.kind == "f" and not np.isfinite(v).all():
            issues.append(f"{k}: {int((~np.isfinite(v)).sum())} non-finite values")
    n = len(a["y_true"])
    for k in a:
        if len(a[k]) != n:
            issues.append(f"{k}: length {len(a[k])} != n {n}")
    if "probs" in a:
        s = a["probs"].sum(-1)
        if np.abs(s - 1).max() > 1e-3:
            issues.append(f"probs rows sum to [{s.min():.4f}, {s.max():.4f}]")
        if a["y_true"].max() >= a["probs"].shape[1] or a["y_true"].min() < 0:
            issues.append("y_true outside class range")
        facts["pred_classes_used"] = int(len(np.unique(a["y_pred"])))
        facts["n_classes"] = int(a["probs"].shape[1])
    for key in ("eu", "au", "au_no_H_input", "au_with_H_input"):
        if key not in a:
            continue
        v = a[key].reshape(n, -1)
        sd = v.std(0)
        facts[f"{key}_sd_per_col"] = [float(x) for x in sd]
        facts[f"{key}_range"] = [float(v.min()), float(v.max())]
        const = np.where(sd < CONST_SD)[0]
        if len(const):
            issues.append(f"{key}: {len(const)}/{v.shape[1]} columns constant (sd<{CONST_SD})")
        score = v.mean(1)
        uniq = len(np.unique(np.round(score, 8)))
        facts[f"{key}_unique_scores"] = int(uniq)
        if uniq < 0.5 * n:
            issues.append(f"{key}: only {uniq} distinct per-example scores of {n} (heavy ties)")
    if "H" in a:
        H = a["H"]
        facts["H_range"] = [float(H.min()), float(H.max())]
        facts["H_distinct_values"] = int(len(np.unique(np.round(H, 6))))
        facts["H_share_zero"] = float((H == 0).mean())
        if H.min() < -1e-6 or H.max() > np.log(max(2, H.shape[-1] if H.ndim > 1 else 2)) + 5:
            issues.append(f"H outside expected range [{H.min():.3f}, {H.max():.3f}]")
    return {"n": n, "issues": issues, "facts": facts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    root = Path(p.parse_args().root)
    out = {d.name: check_run(d) for d in sorted(root.iterdir()) if (d / "test_arrays.npz").exists()}
    (root / "sanity_sweep.json").write_text(json.dumps(out, indent=2))
    for run, r in out.items():
        f = r["facts"]
        print(f"\n{run} (n={r['n']})")
        for i in r["issues"]:
            print(f"  ISSUE  {i}")
        keep = {k: v for k, v in f.items() if not k.endswith("sd_per_col")}
        print("  facts ", json.dumps(keep))


if __name__ == "__main__":
    main()

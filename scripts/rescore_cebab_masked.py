"""Rescore CEBaB results with unannotated / no-majority aspects masked out.

The 532fd05 loader maps aspects whose majority is '' (0 annotators) or
'no majority' to concept label 0 (Negative) and gives them entropy H=0
(scripts/audit_cebab_majority.py). This script rebuilds a per-aspect validity
mask from the raw Hugging Face annotations, verifies it is aligned with the
saved test arrays, and recomputes every H- or concept-label-dependent metric.

    python scripts/rescore_cebab_masked.py --root outputs/icml_2026_reeval

Writes <root>/cebab_masked_rescore.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ASPECTS = ["food", "service", "ambiance", "noise"]
LABEL = {"Negative": 0, "unknown": 1, "Positive": 2}
N_BOOT = 1000
COLLAPSED_SD = 1e-3  # per-example U_ale SD below this = collapsed head


def parse_dist(d):
    if isinstance(d, dict):
        return d
    try:
        return json.loads(d) if d else {}
    except Exception:
        return json.loads(d.replace("'", '"'))


def raw_masks():
    """Per-aspect masks for the test split, in loader order."""
    from datasets import load_dataset

    rows = [r for r in load_dataset("CEBaB/CEBaB")["test"]
            if str(r.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}
            and (r.get("description") or "").strip()]
    maj = np.array([[r[f"{a}_aspect_majority"] for a in ASPECTS] for r in rows], dtype=object)
    n_ann = np.array([[sum(parse_dist(r[f"{a}_aspect_label_distribution"]).values()) for a in ASPECTS]
                      for r in rows])
    h_valid = n_ann > 0                                   # entropy target is defined
    binary = np.isin(maj, ["Negative", "Positive"])       # binary concept target is defined
    unknown = maj == "unknown"
    expected_label = np.vectorize(lambda m: LABEL.get(m, 0))(maj)  # what the loader emitted
    return maj, h_valid, binary, unknown, expected_label


def rho(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(spearmanr(a, b)[0])


def boot_ci(fn, n, seed):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    vals = np.array([v for v in vals if math.isfinite(v)])
    if vals.size == 0:
        return [math.nan, math.nan]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def masked_mean(x, m):
    return np.where(m, x, 0).sum(1) / np.maximum(m.sum(1), 1)


def strata(H, err, eu, conf):
    cuts = np.quantile(H, [1 / 3, 2 / 3])
    bins = np.digitize(H, cuts)
    out = {"cuts": cuts.tolist()}
    for b, name in enumerate(("low", "med", "high")):
        m = bins == b
        out[name] = {
            "n": int(m.sum()), "errors": int(err[m].sum()),
            "H_range": [float(H[m].min()), float(H[m].max())] if m.any() else None,
            "auroc_eu": float(roc_auc_score(err[m], eu[m])) if 0 < err[m].sum() < m.sum() else math.nan,
            "auroc_maxprob": float(roc_auc_score(err[m], conf[m])) if 0 < err[m].sum() < m.sum() else math.nan,
        }
    return out


def concept_acc(probs, labels, mask):
    pred_pos = probs > 0.5
    per = {}
    for j, a in enumerate(ASPECTS):
        m = mask[:, j]
        per[a] = {"n": int(m.sum()), "acc": float((pred_pos[m, j] == (labels[m, j] == 2)).mean())}
    return per, float(np.mean([v["acc"] for v in per.values()]))


def rescore(run_dir, masks):
    maj, h_valid, binary, unknown, expected_label = masks
    a = np.load(run_dir / "test_arrays.npz")
    seed = json.loads((run_dir / "run_manifest.json").read_text())["seed"]
    H, au, eu, cl, cp = a["H"], a["au"], a["eu"], a["concept_labels"], a["concept_probs"]

    # Alignment checks: saved labels are what the loader emitted from these raw rows,
    # and every unannotated aspect carries the H=0 fallback.
    assert H.shape == maj.shape, (H.shape, maj.shape)
    assert np.array_equal(cl, expected_label), "concept labels not aligned with raw rows"
    assert np.all(H[~h_valid] == 0), "unannotated aspects expected to carry H=0"

    err = (a["y_true"] != a["y_pred"]).astype(float)
    eu_s, conf = eu.mean(1), 1 - a["probs"].max(1)
    au_all = au.mean(1)
    H_old = H.mean(1)
    H_new = masked_mean(H, h_valid)
    au_matched = masked_mean(au, h_valid)
    n = len(err)

    if au_all.std() < COLLAPSED_SD:
        # A constant ambiguity head (e.g. DeBERTa-v3 seed 2024) has no ranking;
        # any rank correlation with it reflects floating-point ties, not signal.
        au = np.full_like(au, np.nan)
        au_all = au_matched = np.full(n, np.nan)
    pooled = lambda idx: rho(au[idx][h_valid[idx]], H[idx][h_valid[idx]])
    out = {
        "run_id": run_dir.name, "seed": seed, "n": n,
        "unchanged": {
            "accuracy": float(1 - err.mean()),
            "rho_eu_au": rho(eu_s, au_all), "rho_eu_err": rho(eu_s, err),
            "auroc_eu_err": float(roc_auc_score(err, eu_s)),
            "auroc_maxprob_err": float(roc_auc_score(err, conf)),
        },
        "rho_au_H": {
            "old_all_aspects": {"value": rho(au_all, H_old),
                                "boot95": boot_ci(lambda i: rho(au_all[i], H_old[i]), n, seed)},
            "masked_matched": {"value": rho(au_matched, H_new),
                               "boot95": boot_ci(lambda i: rho(au_matched[i], H_new[i]), n, seed),
                               "note": "AU and H both averaged over annotated aspects"},
            "masked_H_only": {"value": rho(au_all, H_new),
                              "note": "U_ale over all concepts, H over annotated aspects"},
            "masked_pooled": {"value": pooled(np.arange(n)), "boot95": boot_ci(pooled, n, seed),
                              "n_entries": int(h_valid.sum())},
            "per_aspect_old": {asp: rho(au[:, j], H[:, j]) for j, asp in enumerate(ASPECTS)},
            "per_aspect_masked": {asp: rho(au[h_valid[:, j], j], H[h_valid[:, j], j])
                                  for j, asp in enumerate(ASPECTS)},
        },
        "strata_old": strata(H_old, err, eu_s, conf),
        "strata_masked": strata(H_new, err, eu_s, conf),
        "n_reviews_H0_old": int((H_old == 0).sum()),
        "n_reviews_H0_masked": int((H_new == 0).sum()),
    }
    old_known = cl != 1
    per_old, mean_old = concept_acc(cp, cl, old_known)
    per_new, mean_new = concept_acc(cp, cl, binary)
    out["concept_accuracy"] = {"old_known_incl_unannotated": {"mean": mean_old, "per_aspect": per_old},
                               "masked_binary_only": {"mean": mean_new, "per_aspect": per_new}}
    # What the model predicts on the aspects that were silently labelled Negative.
    phantom = old_known & ~binary
    out["phantom_negative_entries"] = {
        "n": int(phantom.sum()),
        "share_predicted_negative": float((cp[phantom] <= 0.5).mean()) if phantom.any() else math.nan,
    }
    return out


def seed_mean_sd(runs, getter):
    v = np.array([getter(r) for r in runs], dtype=float)
    ok = v[np.isfinite(v)]  # seeds with an undefined value (collapsed head) are dropped, not averaged in
    return {"per_seed": v.tolist(), "n_valid": int(ok.size),
            "mean": float(ok.mean()) if ok.size else math.nan,
            "sd": float(ok.std(ddof=1)) if ok.size > 1 else math.nan}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    masks = raw_masks()
    maj, h_valid, binary, unknown, _ = masks
    report = {
        "mask_summary": {
            asp: {"annotated": int(h_valid[:, j].sum()), "unannotated": int((~h_valid[:, j]).sum()),
                  "binary_target": int(binary[:, j].sum()), "unknown_majority": int(unknown[:, j].sum()),
                  "no_majority": int((maj[:, j] == "no majority").sum())}
            for j, asp in enumerate(ASPECTS)},
        "runs": {},
    }
    for run_dir in sorted(root.glob("cebab_3class*")):
        report["runs"][run_dir.name] = rescore(run_dir, masks)

    groups = {}
    for name, r in report["runs"].items():
        groups.setdefault(name.replace("_seed123", "").replace("_seed2024", ""), []).append(r)
    report["seed_summary"] = {
        g: {
            "seeds": [r["seed"] for r in rs],
            "rho_au_H_old": seed_mean_sd(rs, lambda r: r["rho_au_H"]["old_all_aspects"]["value"]),
            "rho_au_H_masked_matched": seed_mean_sd(rs, lambda r: r["rho_au_H"]["masked_matched"]["value"]),
            "rho_au_H_masked_pooled": seed_mean_sd(rs, lambda r: r["rho_au_H"]["masked_pooled"]["value"]),
            "concept_acc_old": seed_mean_sd(rs, lambda r: r["concept_accuracy"]["old_known_incl_unannotated"]["mean"]),
            "concept_acc_masked": seed_mean_sd(rs, lambda r: r["concept_accuracy"]["masked_binary_only"]["mean"]),
            **{k: seed_mean_sd(rs, lambda r, k=k: r["unchanged"][k]) for k in rs[0]["unchanged"]},
        }
        for g, rs in groups.items()
    }
    (root / "cebab_masked_rescore.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["mask_summary"], indent=1))
    for g, s in report["seed_summary"].items():
        print(f"\n{g}  seeds={s['seeds']}")
        for k, v in s.items():
            if k != "seeds":
                print(f"  {k:26s} {v['mean']:+.3f} ± {v['sd']:.3f}   {['%.3f' % x for x in v['per_seed']]}")


if __name__ == "__main__":
    main()

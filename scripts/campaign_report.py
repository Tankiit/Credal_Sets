"""Write CAMPAIGN_RESULTS.md from the re-evaluated test arrays.

    python scripts/campaign_report.py --root outputs/icml_2026_reeval --out CAMPAIGN_RESULTS.md

Every value is computed from saved per-example arrays. CEBaB rho(U_ale, H) and
concept accuracy come from cebab_masked_rescore.json (unannotated aspects
excluded). Correlations involving a collapsed ambiguity head are reported as
undefined, and seeds are averaged only over runs where a value is defined.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

COLLAPSED_SD = 1e-3
S3 = (42, 123, 2024)

GROUPS = [
    ("Retrained, corrected CEBaB labels (3 seeds)", [
        ("CEBaB DistilBERT", {s: f"cebab_3class_distilbert_seed{s}_100ep_fixed" for s in S3}),
        ("CEBaB RoBERTa", {s: f"cebab_3class_roberta_base_seed{s}_100ep_fixed" for s in S3}),
    ]),
    ("Ablations, DistilBERT, corrected labels (3 seeds)", [
        ("Default (λ_a = 2, λ_d = 0)", {s: f"cebab_3class_distilbert_seed{s}_100ep_fixed" for s in S3}),
        ("No ambiguity supervision (λ_a = 0)", {s: f"cebab_3class_distilbert_seed{s}_100ep_fixed_noale" for s in S3}),
        ("Decorrelation penalty (λ_d = 5)", {s: f"cebab_3class_distilbert_seed{s}_100ep_fixed_decorr5" for s in S3}),
    ]),
    ("Other datasets, seed 42 added (3 seeds)", [
        ("HateXplain", {42: "hatexplain_distilbert_seed42_100ep", 123: "hatexplain_seed123_100ep",
                        2024: "hatexplain_seed2024_100ep"}),
        ("GoEmotions", {42: "goemotions_distilbert_seed42_100ep", 123: "goemotions_seed123_100ep",
                        2024: "goemotions_seed2024_100ep"}),
        ("MAQA* (λ_decorr = 5)", {42: "maqa_distilbert_seed42_100ep", 123: "maqa_seed123_100ep",
                                  2024: "maqa_seed2024_100ep"}),
        ("AmbigQA* only (λ_decorr = 5)", {s: f"ambigqa_distilbert_seed{s}_100ep" for s in S3}),
    ]),
    ("Previous checkpoints, original labels, masked evaluation (2 seeds)", [
        ("CEBaB DistilBERT", {s: f"cebab_3class_seed{s}_100ep" for s in (123, 2024)}),
        ("CEBaB RoBERTa", {s: f"cebab_3class_roberta_base_seed{s}_100ep" for s in (123, 2024)}),
    ]),
]
COLS = [("acc", "Acc %"), ("concept_acc", "Concept acc %"), ("rho_eu_au", "ρ(EU,AU)"),
        ("rho_eu_err", "ρ(EU,Err)"), ("rho_au_H", "ρ(AU,H)"), ("auroc_eu", "AUROC EU"),
        ("auroc_maxprob", "AUROC MaxProb")]


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return math.nan
    return float(spearmanr(a[ok], b[ok])[0])


def run_metrics(root, run, masked):
    d = root / run
    if not (d / "test_arrays.npz").exists():
        return None
    a = dict(np.load(d / "test_arrays.npz"))
    kind = json.loads((d / "run_manifest.json").read_text())["kind"]
    err = (a["y_true"] != a["y_pred"]).astype(float)
    eu = a["eu"].mean(-1) if a["eu"].ndim > 1 else a["eu"]
    if kind == "maqa":
        au, conf = a["au_no_H_input"], (1 - a["maxprob"]) if "maxprob" in a else None
    else:
        au = a["au"].mean(-1) if a["au"].ndim > 1 else a["au"]
        conf = 1 - a["probs"].max(-1)
    collapsed = bool(np.std(au) < COLLAPSED_SD)
    eu_collapsed = bool(np.std(eu) < COLLAPSED_SD)  # e.g. EU driven constant by a decorrelation penalty
    H = a.get("H")
    m = {"acc": 100 * (1 - err.mean()), "collapsed": collapsed, "eu_collapsed": eu_collapsed,
         "rho_eu_err": math.nan if eu_collapsed else rho(eu, err),
         "auroc_eu": math.nan if eu_collapsed else float(roc_auc_score(err, eu)),
         "auroc_maxprob": float(roc_auc_score(err, conf)) if conf is not None else math.nan,
         "rho_eu_au": math.nan if (collapsed or eu_collapsed) else rho(eu, au), "rho_au_H": math.nan, "concept_acc": math.nan}
    if run in masked:  # CEBaB: masked values
        r = masked[run]
        m["rho_au_H"] = r["rho_au_H"]["masked_matched"]["value"]
        m["concept_acc"] = 100 * r["concept_accuracy"]["masked_binary_only"]["mean"]
    elif H is not None and not collapsed:
        m["rho_au_H"] = rho(au, H.mean(-1) if H.ndim > 1 else H)
    return m


def fmt(vals, digits):
    v = np.array([x for x in vals if x is not None and math.isfinite(x)])
    if v.size == 0:
        return "—"
    s = f"{v.mean():.{digits}f}"
    if v.size > 1:
        s += f" ± {v.std(ddof=1):.{digits}f}"
    if v.size < len(vals):
        s += f" (n={v.size})"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root)
    mpath = root / "cebab_masked_rescore.json"
    masked = json.loads(mpath.read_text())["runs"] if mpath.exists() else {}

    lines = ["# Retraining campaign results", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/campaign_report.py` "
             "from `outputs/icml_2026_reeval/`. Mean ± sample SD over seeds; `(n=k)` means only k seeds "
             "have a defined value. `—` means undefined, including correlations with a collapsed "
             "(constant) ambiguity head. CEBaB ρ(AU,H) and concept accuracy exclude unannotated aspects "
             "(`CEBAB_MASK_AUDIT.md`).", ""]
    missing, collapsed, eu_collapsed = [], [], []
    for title, rows in GROUPS:
        lines += [f"## {title}", "", "| Configuration | Seeds | " + " | ".join(c[1] for c in COLS) + " |",
                  "|" + "---|" * (len(COLS) + 2)]
        for name, runs in rows:
            ms = {}
            for s, r in runs.items():
                m = run_metrics(root, r, masked)
                if m is None:
                    missing.append(r)
                else:
                    ms[s] = m
                    if m["collapsed"]:
                        collapsed.append(r)
                    if m["eu_collapsed"]:
                        eu_collapsed.append(r)
            cells = [fmt([m[k] for m in ms.values()], 1 if k in ("acc", "concept_acc") else 2) for k, _ in COLS]
            lines.append(f"| {name} | {','.join(map(str, sorted(ms))) or 'none'} | " + " | ".join(cells) + " |")
        lines.append("")

    base = sorted(root.glob("baselines_*.json"))
    if base:
        lines += ["## Baselines (error-detection AUROC with bootstrap 95% CI)", "",
                  "| Group | Ensemble acc % | Ens. MaxProb | Ens. entropy | Ens. mutual info | Mean CBM EU | MC-dropout MI (per seed) |",
                  "|---|---|---|---|---|---|---|"]
        f = lambda v: f"{v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]"
        for b in base:
            d = json.loads(b.read_text())
            e = d["ensemble"]
            mc = ", ".join(f"{s}: {v['auroc_mc_mutual_info'][0]:.3f}" for s, v in d.get("mc_dropout", {}).items()) or "—"
            lines.append(f"| `{d['group']}` seeds {d['seeds']} | {100 * e['accuracy']:.1f} | {f(e['auroc_maxprob'])} | "
                         f"{f(e['auroc_pred_entropy'])} | {f(e['auroc_mutual_info'])} | {f(e['auroc_cbm_eu_mean'])} | {mc} |")
        lines.append("")

    missing, collapsed = list(dict.fromkeys(missing)), list(dict.fromkeys(collapsed))
    lines += ["## Status", ""]
    lines.append(f"- Missing runs (not trained or not yet collected): {', '.join(f'`{m}`' for m in missing) or 'none'}")
    lines.append(f"- Runs with a collapsed ambiguity head: {', '.join(f'`{c}`' for c in collapsed) or 'none'}")
    lines.append(f"- Runs with a collapsed epistemic head (EU metrics undefined): "
                 f"{', '.join(f'`{c}`' for c in dict.fromkeys(eu_collapsed)) or 'none'}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

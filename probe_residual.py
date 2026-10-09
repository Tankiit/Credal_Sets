"""Residual Probing & Leakage Audit script.

Audits concept and task leakage in raw invisible residuals vs corrected residuals r under twin reparameterizations.

Usage:
    python probe_residual.py --run runs/cebab-cbm-r16-logits-s0
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, f1_score

from sweep import get_sweep_features
from concept_models.models import load_model
from concept_models.reparam import make_twin
from concept_models.training import predict


def probe_auc(Ztr: np.ndarray, Ytr: np.ndarray, Zte: np.ndarray, Yte: np.ndarray, C_reg: float = 1e3) -> float:
    """Train one-vs-rest linear probes per target label (large C_reg for minimal regularization bias)."""
    if Ytr.ndim == 1:
        # Multi-class target label (task label y)
        if len(np.unique(Ytr)) < 2 or len(np.unique(Yte)) < 2:
            return 0.0
        try:
            clf = LogisticRegression(C=C_reg, max_iter=1000, random_state=0)
            clf.fit(Ztr, Ytr)
            preds = clf.predict(Zte)
            return float(f1_score(Yte, preds, average="macro"))
        except Exception:
            return 0.0

    # Binary matrix targets (concepts C)
    k_targets = Ytr.shape[1]
    aucs = []
    for j in range(k_targets):
        if len(np.unique(Ytr[:, j])) < 2 or len(np.unique(Yte[:, j])) < 2:
            continue
        try:
            clf = LogisticRegression(C=C_reg, max_iter=1000, random_state=0)
            clf.fit(Ztr, Ytr[:, j])
            probs = clf.predict_proba(Zte)[:, 1]
            aucs.append(roc_auc_score(Yte[:, j], probs))
        except Exception:
            continue

    return float(np.mean(aucs)) if aucs else 0.0


def residual_parts(model, feats_tr: dict, feats_te: dict):
    """Split z into z_vis and z_inv, fit affine mapping z_inv ~ z_vis on train, return raw z_inv and corrected r."""
    Z_tr = predict(model, feats_tr["X"])["z"]
    Z_te = predict(model, feats_te["X"])["z"]

    d_full = Z_tr.shape[1]
    k_vis = model.k if model.config["kind"] == "cbm" else 2 * model.k

    if d_full <= k_vis:
        # No invisible dimensions present
        return Z_tr, Z_tr, Z_te, Z_te

    z_vis_tr, z_inv_tr = Z_tr[:, :k_vis], Z_tr[:, k_vis:]
    z_vis_te, z_inv_te = Z_te[:, :k_vis], Z_te[:, k_vis:]

    # Fit Ridge regression z_inv ~ z_vis on Train set
    reg = Ridge(alpha=1e-3)
    reg.fit(z_vis_tr, z_inv_tr)

    pred_inv_tr = reg.predict(z_vis_tr)
    pred_inv_te = reg.predict(z_vis_te)

    r_tr = z_inv_tr - pred_inv_tr
    r_te = z_inv_te - pred_inv_te

    return z_inv_tr, r_tr, z_inv_te, r_te


def compare(model, twin, feats_tr: dict, feats_te: dict, C_reg: float = 1e3) -> dict:
    """Compare concept and task probing accuracy on raw z_inv vs corrected r between model and twin."""
    has_C = "C" in feats_tr and "C" in feats_te
    has_y = "y" in feats_tr and "y" in feats_te

    C_tr = feats_tr["C"].numpy() if has_C else None
    C_te = feats_te["C"].numpy() if has_C else None
    y_tr = feats_tr["y"].numpy() if has_y else None
    y_te = feats_te["y"].numpy() if has_y else None

    # Get residual parts for original model and twin
    z_inv_tr_m, r_tr_m, z_inv_te_m, r_te_m = residual_parts(model, feats_tr, feats_te)
    z_inv_tr_t, r_tr_t, z_inv_te_t, r_te_t = residual_parts(twin, feats_tr, feats_te)

    res = {}

    if has_C:
        # 1. Probe Concept Content (inter-concept leakage)
        res["concept_auc_raw_orig"] = probe_auc(z_inv_tr_m, C_tr, z_inv_te_m, C_te, C_reg)
        res["concept_auc_raw_twin"] = probe_auc(z_inv_tr_t, C_tr, z_inv_te_t, C_te, C_reg)
        res["concept_auc_r_orig"] = probe_auc(r_tr_m, C_tr, r_te_m, C_te, C_reg)
        res["concept_auc_r_twin"] = probe_auc(r_tr_t, C_tr, r_te_t, C_te, C_reg)

    if has_y:
        # 2. Probe Task Signal (concept-task leakage)
        res["task_f1_raw_orig"] = probe_auc(z_inv_tr_m, y_tr, z_inv_te_m, y_te, C_reg)
        res["task_f1_raw_twin"] = probe_auc(z_inv_tr_t, y_tr, z_inv_te_t, y_te, C_reg)
        res["task_f1_r_orig"] = probe_auc(r_tr_m, y_tr, r_te_m, y_te, C_reg)
        res["task_f1_r_twin"] = probe_auc(r_tr_t, y_tr, r_te_t, y_te, C_reg)

    return res


def main():
    parser = argparse.ArgumentParser(description="Residual Probing and Gauge Leakage Audit")
    parser.add_argument("--run", type=Path, default=Path("runs/cebab-cbm-r16-logits-s0"), help="Path to run directory")
    parser.add_argument("--mix", type=float, default=1.0, help="Twin mixing parameter")
    parser.add_argument("--rotate", type=float, default=1.0, help="Twin rotation parameter")
    parser.add_argument("--shift", type=float, default=0.0, help="Twin shift parameter")
    parser.add_argument("--seed", type=int, default=0, help="Twin random seed")
    parser.add_argument("--out", type=Path, default=Path("results/probe_residual.json"), help="Output JSON results path")
    args = parser.parse_args()

    model_path = args.run / "model.pt"
    if not model_path.exists():
        print(f"Model checkpoint {model_path} not found!")
        return

    model = load_model(model_path)
    cfg = model.config

    feats_tr = get_sweep_features(cfg["dataset"], "train", cfg["encoder"], cfg)
    feats_te = get_sweep_features(cfg["dataset"], "test", cfg["encoder"], cfg)

    twin, info = make_twin(model, args.rotate, args.mix, args.shift, args.seed)

    print(f"Auditing residual probing for {args.run.name} (mix={args.mix}, rotate={args.rotate})...")
    res = compare(model, twin, feats_tr, feats_te)
    res.update(run=args.run.name, mix=args.mix, rotate=args.rotate, shift=args.shift, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))

    print("\nResidual Probing Audit Results:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

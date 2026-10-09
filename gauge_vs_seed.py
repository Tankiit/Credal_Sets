"""Gauge vs. Seed comparison script.

Evaluates metric invariance under twin reparameterizations vs. seed-to-seed variance across independently trained models.

Usage:
    python gauge_vs_seed.py --run runs/cebab-cbm-r16-logits-s0
"""

import argparse
import json
import itertools
from pathlib import Path
import numpy as np
import torch

from concept_models.features import get_features
from concept_models.models import load_model
from concept_models.reparam import make_twin
from concept_models.training import predict


# ---- 1. Alignment Metrics --------------------------------------------------

def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear Center Kernel Alignment (CKA)."""
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)

    gram_cross = A_c.T @ B_c
    gram_aa = A_c.T @ A_c
    gram_bb = B_c.T @ B_c

    num = np.sum(gram_cross ** 2)
    denom = np.sqrt(np.sum(gram_aa ** 2) * np.sum(gram_bb ** 2)) + 1e-12
    return float(num / denom)


def regularized_cca(A: np.ndarray, B: np.ndarray, reg: float = 1e-5) -> float:
    """Mean Canonical Correlation Analysis (CCA) with L2 regularization."""
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)
    N = A.shape[0]

    C_aa = (A_c.T @ A_c) / (N - 1) + reg * np.eye(A.shape[1])
    C_bb = (B_c.T @ B_c) / (N - 1) + reg * np.eye(B.shape[1])
    C_ab = (A_c.T @ B_c) / (N - 1)

    try:
        # Use SVD of cov matrices for stable matrix square root inverse
        Ua, Sa, _ = np.linalg.svd(C_aa)
        Ub, Sb, _ = np.linalg.svd(C_bb)

        inv_sqrt_a = Ua @ np.diag(1.0 / np.sqrt(Sa)) @ Ua.T
        inv_sqrt_b = Ub @ np.diag(1.0 / np.sqrt(Sb)) @ Ub.T

        T = inv_sqrt_a @ C_ab @ inv_sqrt_b
        singular_values = np.linalg.svd(T, compute_uv=False)
        return float(np.mean(np.clip(singular_values, 0.0, 1.0)))
    except Exception:
        return 0.0


def mutual_knn(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    """Mutual k-nearest neighbours overlap after centering."""
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)
    n = len(A)
    if n <= k:
        return 1.0

    norm_a = np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-12
    norm_b = np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-12
    A_n = A_c / norm_a
    B_n = B_c / norm_b

    S_a = A_n @ A_n.T
    S_b = B_n @ B_n.T
    np.fill_diagonal(S_a, -np.inf)
    np.fill_diagonal(S_b, -np.inf)

    knn_a = np.argpartition(S_a, -k, axis=1)[:, -k:]
    knn_b = np.argpartition(S_b, -k, axis=1)[:, -k:]

    overlaps = [len(set(knn_a[i]) & set(knn_b[i])) / float(k) for i in range(n)]
    return float(np.mean(overlaps))


METRICS = {
    "cka": linear_cka,
    "cca": regularized_cca,
    "knn": mutual_knn,
}


# ---- 2. Twin Table & Seed Table --------------------------------------------

def twin_table(run_path: Path, split: str = "test"):
    """Evaluate alignment metrics across twin reparameterization grid for a single run."""
    model = load_model(run_path / "model.pt")
    cfg = model.config
    
    # Try loading cached/synthetic features
    try:
        feats = get_features(cfg["dataset"], split, cfg["encoder"])
    except Exception:
        gen = torch.Generator().manual_seed(0)
        feats = {"X": torch.randn(200, cfg["in_dim"], generator=gen)}

    Z_orig = predict(model, feats["X"])["z"]
    d_full = Z_orig.shape[1]
    k_vis = model.k if cfg["kind"] == "cbm" else 2 * model.k

    ROTATE = [0.0, 1.0, 2.0]
    MIX = [0.0, 0.5, 1.0, 2.0] if model.head_linear_in_z else [0.0]
    SHIFT = [0.0, 1.0]

    rows = []
    for rot, mix, shift in itertools.product(ROTATE, MIX, SHIFT):
        twin, info = make_twin(model, rot, mix, shift, seed=0)
        Z_twin = predict(twin, feats["X"])["z"]

        row = {
            "rot": rot, "mix": mix, "shift": shift,
            "cond_B": float(info["cond_B"]),
            "invisible_dims": int(info["invisible_dims"])
        }

        # Compute metrics on full z, visible z, and invisible z
        for name, fn in METRICS.items():
            row[f"{name}_full"] = fn(Z_orig, Z_twin)
            if k_vis < d_full:
                row[f"{name}_vis"] = fn(Z_orig[:, :k_vis], Z_twin[:, :k_vis])
                row[f"{name}_invis"] = fn(Z_orig[:, k_vis:], Z_twin[:, k_vis:])

        rows.append(row)
    return rows


def seed_table(run_paths: list[Path], split: str = "test"):
    """Evaluate metric similarities between independently trained seeds with identical configs."""
    models = [load_model(p / "model.pt") for p in run_paths if (p / "model.pt").exists()]
    if len(models) < 2:
        return []

    cfg = models[0].config
    feats = get_features(cfg["dataset"], split, cfg["encoder"])
    
    zs = [predict(m, feats["X"])["z"] for m in models]
    k_vis = models[0].k if cfg["kind"] == "cbm" else 2 * models[0].k
    d_full = zs[0].shape[1]

    rows = []
    for (i, m1), (j, m2) in itertools.combinations(enumerate(models), 2):
        Z1, Z2 = zs[i], zs[j]
        row = {"seed1": m1.config.get("seed", i), "seed2": m2.config.get("seed", j)}

        for name, fn in METRICS.items():
            row[f"{name}_full"] = fn(Z1, Z2)
            if k_vis < d_full:
                row[f"{name}_vis"] = fn(Z1[:, :k_vis], Z2[:, :k_vis])
                row[f"{name}_invis"] = fn(Z1[:, k_vis:], Z2[:, k_vis:])
        rows.append(row)

    return rows


def place_twins_on_seed_scale(twin_rows: list[dict], seed_rows: list[dict]):
    """Compute the cond_B at which twin similarity matches median seed-to-seed similarity."""
    if not seed_rows or not twin_rows:
        return {}

    seed_medians = {}
    for metric_key in ["cka_full", "cca_full", "knn_full"]:
        if metric_key in seed_rows[0]:
            seed_medians[metric_key] = float(np.median([r[metric_key] for r in seed_rows]))

    equivalent_cond_B = {}
    for metric_key, target_val in seed_medians.items():
        # Find twin row with closest metric value to median seed value
        closest_row = min(twin_rows, key=lambda r: abs(r.get(metric_key, 0.0) - target_val))
        equivalent_cond_B[metric_key] = {
            "median_seed_val": target_val,
            "matched_twin_cond_B": closest_row["cond_B"],
            "matched_twin_mix": closest_row["mix"],
            "matched_twin_val": closest_row.get(metric_key, 0.0)
        }

    return equivalent_cond_B


def main():
    parser = argparse.ArgumentParser(description="Gauge vs. Seed alignment scale evaluation")
    parser.add_argument("--run", type=Path, default=Path("runs/cebab-cbm-r16-logits-s0"), help="Path to run directory")
    args = parser.parse_args()

    if not (args.run / "model.pt").exists():
        print(f"Model path {args.run / 'model.pt'} not found!")
        return

    print(f"Evaluating twin grid on {args.run.name}...")
    twin_results = twin_table(args.run)
    print(f"Generated {len(twin_results)} twin evaluation rows.")
    print("\nSample Twin Row (mix=1.0):")
    sample_mix1 = [r for r in twin_results if r["mix"] == 1.0]
    if sample_mix1:
        print(json.dumps(sample_mix1[0], indent=2))


if __name__ == "__main__":
    main()

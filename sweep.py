"""Tier-0 Sweep script for evaluating reparameterized twin models across datasets, architectures, and transformation grids.

Run from repository root:
    python tier0_sweep.py --out results/tier0.jsonl
"""

import argparse
import json
import itertools
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from concept_datasets import load_split
from concept_models.features import get_features
from concept_models.models import load_model
from concept_models.reparam import make_twin
from concept_models.training import predict
from explain import contributions
from cebab_edits import compare_model_and_twin, load_edit_pairs

sig = lambda v: 1 / (1 + np.exp(-v))

# ---- 1. Alignment metrics between z and z_twin -----------------------------

def linear_cka(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Compute linear Center Kernel Alignment (CKA) between representations Z1 and Z2."""
    Z1_c = Z1 - Z1.mean(axis=0, keepdims=True)
    Z2_c = Z2 - Z2.mean(axis=0, keepdims=True)

    gram_cross = Z1_c.T @ Z2_c
    gram_11 = Z1_c.T @ Z1_c
    gram_22 = Z2_c.T @ Z2_c

    num = np.sum(gram_cross ** 2)
    denom = np.sqrt(np.sum(gram_11 ** 2) * np.sum(gram_22 ** 2)) + 1e-12
    return float(num / denom)


def mutual_knn(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """Compute the fraction of shared k-nearest neighbours between Z1 and Z2 under cosine similarity."""
    n = len(Z1)
    if n <= k:
        return 1.0

    # Normalize vectors to unit norm
    norm1 = np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-12
    norm2 = np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-12
    Z1_n = Z1 / norm1
    Z2_n = Z2 / norm2

    # Cosine similarity matrices
    S1 = Z1_n @ Z1_n.T
    S2 = Z2_n @ Z2_n.T

    # Exclude self-similarity by setting diagonal to -infinity
    np.fill_diagonal(S1, -np.inf)
    np.fill_diagonal(S2, -np.inf)

    # Top-k nearest neighbour indices
    knn1 = np.argpartition(S1, -k, axis=1)[:, -k:]
    knn2 = np.argpartition(S2, -k, axis=1)[:, -k:]

    # Calculate average Jaccard overlap of k-NN sets
    overlaps = []
    for i in range(n):
        set1 = set(knn1[i])
        set2 = set(knn2[i])
        overlaps.append(len(set1 & set2) / float(k))

    return float(np.mean(overlaps))


# ---- 2. Behavioral Diagnostics ---------------------------------------------

def d_logits(m, f):
    return predict(m, f["X"])["logits"]


def d_concept_probs(m, f):
    return sig(predict(m, f["X"])["concept_logits"])


def d_z(m, f):
    return predict(m, f["X"])["z"]


def d_intervened_behaviour(m, twin, f, frac: float = 0.5, seed: int = 0) -> dict:
    """Evaluate behavioural agreement under concept interventions."""
    if "C" not in f:
        return {"intervened_agree": 1.0, "intervened_acc_orig": 0.0, "intervened_acc_twin": 0.0, "intervened_med_abs_dlogit": 0.0}
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(f["C"].shape, generator=gen) < frac
    ai = predict(m, f["X"], f["C"], mask)
    bi = predict(twin, f["X"], f["C"], mask)
    
    pred_a = ai["logits"].argmax(1)
    pred_b = bi["logits"].argmax(1)
    agree = float((pred_a == pred_b).mean())
    
    res = {
        "intervened_agree": agree,
        "intervened_med_abs_dlogit": float(np.median(np.abs(ai["logits"] - bi["logits"]).max(axis=1)))
    }
    if "y" in f:
        y = f["y"].numpy()
        res["intervened_acc_orig"] = float((pred_a == y).mean())
        res["intervened_acc_twin"] = float((pred_b == y).mean())
    return res


def d_push_centered(m, twin, f) -> float:
    """Compute maximum absolute difference in per-concept pushes after centering by dataset mean."""
    if len(m.head) != 1:
        return 0.0
    try:
        terms_orig, _, _, _ = contributions(m, f["X"])
        terms_twin, _, _, _ = contributions(twin, f["X"])
        
        # Subtract per-concept dataset mean of terms BEFORE difference
        terms_orig_c = terms_orig - terms_orig.mean(dim=0, keepdim=True)
        terms_twin_c = terms_twin - terms_twin.mean(dim=0, keepdim=True)
        return float(torch.abs(terms_orig_c - terms_twin_c).max().item())
    except Exception:
        return 0.0


def cka_split(Z1: np.ndarray, Z2: np.ndarray, k_visible: int) -> dict:
    """Compute linear CKA on the visible block vs the invisible block separately."""
    d = Z1.shape[1]
    k_vis = min(k_visible, d)
    
    cka_vis = linear_cka(Z1[:, :k_vis], Z2[:, :k_vis]) if k_vis > 0 else 1.0
    cka_invis = linear_cka(Z1[:, k_vis:], Z2[:, k_vis:]) if d > k_vis else 1.0
    
    return {"cka_vis": cka_vis, "cka_invis": cka_invis}


def d_intervened_logits(m, twin, f, frac=0.5, seed=0):
    """Evaluate max absolute logit difference under random concept interventions."""
    if "C" not in f:
        return 0.0
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(f["C"].shape, generator=gen) < frac
    ai = predict(m, f["X"], f["C"], mask)
    bi = predict(twin, f["X"], f["C"], mask)
    return float(np.abs(ai["logits"] - bi["logits"]).max())


def d_probe_on_z(m, twin, f_train, f_test):
    """Train linear probes z -> concept on train set, report mean absolute AUC drift on test set between original and twin z."""
    if "C" not in f_train or "C" not in f_test:
        return 0.0

    z_train_orig = d_z(m, f_train)
    z_test_orig = d_z(m, f_test)
    z_test_twin = d_z(twin, f_test)

    C_train = f_train["C"].numpy()
    C_test = f_test["C"].numpy()

    k_concepts = C_train.shape[1]
    auc_diffs = []

    for j in range(k_concepts):
        # Skip concepts with no positive or negative examples in train/test
        if len(np.unique(C_train[:, j])) < 2 or len(np.unique(C_test[:, j])) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=500, random_state=0)
            clf.fit(z_train_orig, C_train[:, j])

            prob_orig = clf.predict_proba(z_test_orig)[:, 1]
            prob_twin = clf.predict_proba(z_test_twin)[:, 1]

            auc_orig = roc_auc_score(C_test[:, j], prob_orig)
            auc_twin = roc_auc_score(C_test[:, j], prob_twin)
            auc_diffs.append(abs(auc_orig - auc_twin))
        except Exception:
            continue

    return float(np.mean(auc_diffs)) if auc_diffs else 0.0


def d_push(m, twin, f):
    """Compute maximum absolute difference in per-concept logit pushes between original model and twin."""
    if len(m.head) != 1:
        return 0.0  # Only linear heads can be split into linear per-concept pushes
    try:
        terms_orig, _, _, _ = contributions(m, f["X"])
        terms_twin, _, _, _ = contributions(twin, f["X"])
        return float(torch.abs(terms_orig - terms_twin).max().item())
    except Exception:
        return 0.0


def cebab_edit_effect_rows(m, twin, dataset, split="test") -> list[dict]:
    """Evaluate validated aspect-specific CEBaB original-to-edit interventions.

    This is the B1 data source.  It never falls back to arbitrary row pairs or
    silently replaces a loader failure with a zero.
    """
    if dataset != "cebab":
        return []
    pairs = load_edit_pairs(split)
    feats = get_sweep_features(dataset, split, m.config["encoder"], m.config)
    return compare_model_and_twin(m, twin, feats, pairs)


def d_cebab_edit_effect(m, twin, dataset, split="test") -> float:
    """Maximum absolute twin-vs-model targeted CEBaB expected-effect gap."""
    rows = cebab_edit_effect_rows(m, twin, dataset, split)
    if not rows:
        return 0.0
    return float(max(abs(row["twin_minus_model_expected_effect"]) for row in rows))


# ---- 3. One cell of the sweep ----------------------------------------------

def one(run: Path, rot: float, mix: float, shift: float, tseed: int, feats_test: dict, feats_train: dict = None):
    model = load_model(run / "model.pt")
    if not model.head_linear_in_z and mix != 0:
        return None  # Skip invalid combinations

    twin, info = make_twin(model, rot, mix, shift, tseed)
    
    # Extract model structural configuration fields
    residual_dim = model.config.get("residual_dim", 0)
    head_input = model.config.get("head_input", "probs" if model.config["kind"] == "cbm" else "emb")

    rec = dict(
        run=run.name,
        dataset=model.config["dataset"],
        model_kind=model.config["kind"],
        head_input=head_input,
        residual_dim=residual_dim,
        rot=rot,
        mix=mix,
        shift=shift,
        tseed=tseed,
        cond_B=float(info["cond_B"]),
        invisible_dims=int(info["invisible_dims"])
    )

    # Sanity checks: outputs should remain invariant (~0 up to float precision)
    rec["max_d_logits"] = float(np.abs(d_logits(model, feats_test) - d_logits(twin, feats_test)).max())
    rec["max_d_cprobs"] = float(np.abs(d_concept_probs(model, feats_test) - d_concept_probs(twin, feats_test)).max())

    # Representation alignment
    Z1, Z2 = d_z(model, feats_test), d_z(twin, feats_test)
    rec["cka"] = linear_cka(Z1, Z2)
    rec["knn"] = mutual_knn(Z1, Z2, k=10)
    rec["rel_change_z"] = float(np.linalg.norm(Z2 - Z1) / (np.linalg.norm(Z1) + 1e-12))

    # Split CKA on visible vs invisible blocks
    k_vis = model.k if model.config["kind"] == "cbm" else 2 * model.k
    rec.update(cka_split(Z1, Z2, k_vis))

    # Behavioral & Diagnostic Metrics
    rec["d_intervened_logits"] = d_intervened_logits(model, twin, feats_test, frac=0.5, seed=tseed)
    rec.update(d_intervened_behaviour(model, twin, feats_test, frac=0.5, seed=tseed))
    rec["d_push"] = d_push(model, twin, feats_test)
    rec["d_push_centered"] = d_push_centered(model, twin, feats_test)
    rec["d_cebab_edit"] = d_cebab_edit_effect(model, twin, model.config["dataset"], split="test")

    if feats_train is not None:
        rec["d_probe_auc_drift"] = d_probe_on_z(model, twin, feats_train, feats_test)

    return rec


def get_sweep_features(dataset_name: str, split: str, encoder_name: str, config: dict) -> dict:
    """Load cached features or generate synthetic evaluation features if dataset is not downloaded."""
    try:
        return get_features(dataset_name, split, encoder_name)
    except Exception:
        gen = torch.Generator().manual_seed(0 if split == "test" else 1)
        n_samples = 200
        in_dim = config.get("in_dim", 768)
        n_concepts = config.get("n_concepts", 8)
        X = torch.randn(n_samples, in_dim, generator=gen)
        C = (torch.rand(n_samples, n_concepts, generator=gen) > 0.5).float()
        return {"X": X, "C": C}


# ---- 4. Driver --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier-0 Sweep for Twin Reparameterization Diagnostics")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing trained model runs")
    parser.add_argument("--out", type=Path, default=Path("results/tier0.jsonl"), help="Output JSONL filepath")
    parser.add_argument("--split", default="test", help="Dataset split for evaluation")
    parser.add_argument("--include_probe", action="store_true", help="Include train-set linear probe diagnostic (slower)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.runs_dir.glob("*-s0"))

    ROTATE_GRID = [0.0, 1.0, 2.0]
    MIX_GRID = [0.0, 0.5, 1.0, 2.0]
    SHIFT_GRID = [0.0, 1.0]
    TSEEDS_GRID = [0, 1, 2]

    print(f"Starting Tier-0 Sweep across {len(runs)} model runs...")
    print(f"Output path: {args.out}")

    count = 0
    with open(args.out, "w") as fh:
        for run in runs:
            model_path = run / "model.pt"
            if not model_path.exists():
                print(f"Skipping {run.name}: '{model_path}' not found on disk. (Train it first with `python train.py`) ")
                continue

            model = load_model(model_path)
            dataset_name = model.config["dataset"]
            encoder_name = model.config["encoder"]

            feats_test = get_sweep_features(dataset_name, args.split, encoder_name, model.config)
            feats_train = get_sweep_features(dataset_name, "train", encoder_name, model.config) if args.include_probe else None

            print(f"\nProcessing run: {run.name}")
            grid = list(itertools.product(ROTATE_GRID, MIX_GRID, SHIFT_GRID, TSEEDS_GRID))
            for rot, mix, shift, ts in tqdm(grid, desc=f"Sweep {run.name}", leave=False):
                r = one(run, rot, mix, shift, ts, feats_test, feats_train)
                if r:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    count += 1

    print(f"\nSweep complete! Saved {count} evaluation records to {args.out}")


if __name__ == "__main__":
    main()

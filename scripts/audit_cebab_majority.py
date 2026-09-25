"""Audit CEBaB aspect majorities, annotator counts, and entropy targets.

Runs the exact loader used by the ICML_2026 pipeline (load_cebab_direct at
commit 532fd05, patched to the three-class task as in reeval_icml_modal.py) and
reads every item through CEBaBDataset.__getitem__.

    python scripts/audit_cebab_majority.py --code-dir <532fd05 worktree> \
        [--saved-test outputs/icml_2026_reeval/cebab_3class_seed123_100ep/test_arrays.npz]
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ASPECTS = ["food", "service", "ambiance", "noise"]
LOADER_KEYS = ("Negative", "unknown", "Positive")
IDX_TO_NAME = {0: "Negative", 1: "unknown", 2: "Positive"}


def parse_dist(d):
    if isinstance(d, dict):
        return d
    if not d:
        return {}
    try:
        return json.loads(d)
    except Exception:
        try:
            return json.loads(d.replace("'", '"'))
        except Exception:
            return {}


def raw_entropy(dist):
    """Normalized (by log 3) entropy of the raw annotator labels, all keys kept."""
    counts = np.array([v for v in dist.values() if v > 0], dtype=float)
    if counts.sum() == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(3))


def load_splits(code_dir):
    """Return {split: (raw_rows_kept, dataset)} using the pipeline's own loader."""
    sys.path.insert(0, str(Path(code_dir).resolve()))
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import load_cebab_direct as cebab

    cebab = importlib.reload(cebab)
    ds = load_dataset("CEBaB/CEBaB")
    raw = {"train": list(ds["train_inclusive"]), "validation": list(ds["validation"]), "test": list(ds["test"])}

    cebab.load_cebab = lambda *a, **k: raw
    original_process = cebab.process_cebab_raw

    def process_ternary(raw_data):
        raw_data = [r for r in raw_data if str(r.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}]
        out = original_process(raw_data)
        for item in out:
            rating = item["label"] + 1
            item["label"] = 0 if rating <= 2 else (1 if rating == 3 else 2)
        return out

    cebab.process_cebab_raw = process_ternary
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train, val, test, _, _ = cebab.get_cebab_dataloaders(
        tokenizer=tok, batch_size=8, max_length=256, num_workers=0, include_edits=False
    )
    kept = {
        s: [r for r in rows
            if str(r.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}
            and (r.get("description") or "").strip()]
        for s, rows in raw.items()
    }
    return {"train": (kept["train"], train.dataset),
            "validation": (kept["validation"], val.dataset),
            "test": (kept["test"], test.dataset)}


def audit(split, rows, dataset, saved_H=None):
    n = len(dataset)
    assert len(rows) == n, f"{split}: raw/loader length mismatch {len(rows)} vs {n}"
    assert all(r["description"] == dataset.data[i]["text"] for i, r in enumerate(rows)), "row misalignment"
    items = [dataset[i] for i in range(n)]
    print(f"\n=== {split}: {n} reviews ===")
    loader_H = np.stack([it["annotator_entropy"].numpy() for it in items])
    summary = {}
    for a_idx, a in enumerate(ASPECTS):
        maj = collections.Counter(r[f"{a}_aspect_majority"] for r in rows)
        dists = [parse_dist(r[f"{a}_aspect_label_distribution"]) for r in rows]
        n_ann = np.array([sum(d.values()) for d in dists])
        extra_keys = collections.Counter(k for d in dists for k in d if k not in LOADER_KEYS)

        emitted = collections.defaultdict(collections.Counter)
        for r, it in zip(rows, items):
            emitted[r[f"{a}_aspect_majority"]][IDX_TO_NAME[int(it["concept_labels"][a_idx])]] += 1

        H_raw = np.array([raw_entropy(d) for d in dists])
        diff = np.abs(H_raw - loader_H[:, a_idx])
        finite = np.isfinite(diff)

        print(f"\n[{a}] majority values: {dict(maj)}")
        print(f"  annotators per review: {dict(sorted(collections.Counter(n_ann.tolist()).items()))}"
              f"  (<5: {int((n_ann < 5).sum())})")
        if extra_keys:
            print(f"  distribution keys the loader drops: {dict(extra_keys)}")
        for m, c in emitted.items():
            print(f"  majority={m!r:15s} -> loader concept label {dict(c)}")
        print(f"  H recomputed from raw labels vs loader target: max|diff|={np.nanmax(diff):.4f}, "
              f"rows with diff>1e-4: {int((diff[finite] > 1e-4).sum())}, empty dists: {int((~finite).sum())}")
        summary[a] = dict(maj)

    if saved_H is not None:
        d = np.abs(saved_H - loader_H)
        print(f"\n  saved test H (Table 1 arrays) vs loader target: max|diff|={d.max():.2e}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    p.add_argument("--saved-test", default="")
    args = p.parse_args()
    saved = np.load(args.saved_test)["H"] if args.saved_test else None
    for split, (rows, ds) in load_splits(args.code_dir).items():
        audit(split, rows, ds, saved if split == "test" else None)


if __name__ == "__main__":
    main()

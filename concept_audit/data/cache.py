"""
Load and split feature+concept caches for run.py.

A "cache" directory is the on-disk unit run.py consumes: four row-aligned
arrays -- embeddings.npy, labels.npy, ids.npy (from 01_extract_features.py)
and concepts.npy (from build_concept_cache.py).

load_cache() just loads one such directory as-is, with shape checks.

split_cache() turns ONE full cache into the --features-dir /
--eval-features-dir pair run.py expects: an 80% train split in randomly
shuffled row order, and a 20% eval split restored to ascending original
order (eval is read once for audit, not iterated over epochs, so it stays
unshuffled and reproducible id-for-id).

split_cache_4way() is the audit-pipeline version: it splits ONE full cache
into four disjoint parts instead of two --

  model-train      -- trains the model under audit. Shuffled.
  reference-train  -- trains the separate reference/baseline model the
                       audit compares against. Disjoint from model-train
                       so the comparison isn't biased by shared rows.
                       Shuffled.
  calibration      -- never trained on. Used to set the threshold / null
                       distribution that the certification statistic gets
                       compared against. Restored to original row order.
  certification    -- read exactly once, at the very end, to compute the
                       final reported statistic (the "readout"). Restored
                       to original row order, same reproducibility
                       guarantee as eval above.

Every row of the input cache lands in exactly one of the four output
directories -- the split is a partition, not four independent samples.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_cache(cache_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load (z, g, y) = (embeddings, concepts, labels) from one cache directory."""
    cache_dir = Path(cache_dir)
    for name in ("embeddings.npy", "concepts.npy", "labels.npy"):
        if not (cache_dir / name).exists():
            raise FileNotFoundError(
                f"{cache_dir} is missing {name} -- build it with build_concept_cache.py "
                "and split_cache() before passing it to run.py"
            )
    embeddings = np.load(cache_dir / "embeddings.npy")
    concepts = np.load(cache_dir / "concepts.npy")
    labels = np.load(cache_dir / "labels.npy")

    if embeddings.shape[0] != concepts.shape[0] or embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{cache_dir}: row count mismatch -- embeddings {embeddings.shape[0]}, "
            f"concepts {concepts.shape[0]}, labels {labels.shape[0]}"
        )

    z = torch.from_numpy(embeddings.astype(np.float32))
    g = torch.from_numpy(concepts.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    return z, g, y


def split_cache(cache_dir: Path, train_dir: Path, eval_dir: Path,
                 train_frac: float = 0.8, seed: int = 0) -> None:
    """
    Split one full cache into a shuffled train cache and an order-preserving
    eval cache.

    Train rows are kept in the shuffled order torch.randperm produces.
    Eval rows are the held-out remainder, sorted back to ascending original
    row index -- so eval is never shuffled.
    """
    cache_dir, train_dir, eval_dir = Path(cache_dir), Path(train_dir), Path(eval_dir)
    embeddings = np.load(cache_dir / "embeddings.npy")
    concepts = np.load(cache_dir / "concepts.npy")
    labels = np.load(cache_dir / "labels.npy")
    ids = np.load(cache_dir / "ids.npy", allow_pickle=True)

    n = embeddings.shape[0]
    n_train = int(round(n * train_frac))
    if not (0 < n_train < n):
        raise ValueError(f"train_frac={train_frac} gives an empty split for n={n} rows")

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    train_idx = perm[:n_train]              # shuffled order, kept as-is
    eval_idx = np.sort(perm[n_train:])      # held-out remainder, restored to original order

    for out_dir, idx in ((train_dir, train_idx), (eval_dir, eval_idx)):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", embeddings[idx])
        np.save(out_dir / "concepts.npy", concepts[idx])
        np.save(out_dir / "labels.npy", labels[idx])
        np.save(out_dir / "ids.npy", ids[idx])
    print(f"[saved] {train_dir} ({n_train} rows, shuffled) / {eval_dir} ({n - n_train} rows, original order)")


def split_cache_4way(cache_dir: Path,
                      model_train_dir: Path, reference_train_dir: Path,
                      calibration_dir: Path, certification_dir: Path,
                      model_train_frac: float = 0.4, reference_train_frac: float = 0.2,
                      calibration_frac: float = 0.2, certification_frac: float = 0.2,
                      seed: int = 0) -> None:
    """
    Split one full cache into four disjoint parts: model-train,
    reference-train, calibration, certification. See the module docstring
    for what each part is for.

    model-train and reference-train are kept in shuffled order (they're
    training sets). calibration and certification are restored to
    ascending original row order (they're read-only audit sets, so they
    stay reproducible id-for-id, same as split_cache()'s eval split).

    The four fractions must sum to 1.0; every row of the input cache ends
    up in exactly one output directory.
    """
    fracs = {
        "model_train": model_train_frac,
        "reference_train": reference_train_frac,
        "calibration": calibration_frac,
        "certification": certification_frac,
    }
    total = sum(fracs.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"fractions must sum to 1.0, got {total} ({fracs})")

    cache_dir = Path(cache_dir)
    embeddings = np.load(cache_dir / "embeddings.npy")
    concepts = np.load(cache_dir / "concepts.npy")
    labels = np.load(cache_dir / "labels.npy")
    ids = np.load(cache_dir / "ids.npy", allow_pickle=True)

    n = embeddings.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()

    names = ["model_train", "reference_train", "calibration", "certification"]
    counts = [int(round(n * fracs[name])) for name in names]
    counts[-1] = n - sum(counts[:-1])  # fix rounding drift so counts sum exactly to n
    if any(c <= 0 for c in counts):
        raise ValueError(f"fractions {fracs} give an empty split for n={n} rows: {dict(zip(names, counts))}")

    bounds = np.cumsum([0] + counts)
    idx_chunks = {name: perm[bounds[i]:bounds[i + 1]] for i, name in enumerate(names)}

    # calibration / certification are read-only audit sets: restore original order.
    idx_chunks["calibration"] = np.sort(idx_chunks["calibration"])
    idx_chunks["certification"] = np.sort(idx_chunks["certification"])

    out_dirs = {
        "model_train": Path(model_train_dir),
        "reference_train": Path(reference_train_dir),
        "calibration": Path(calibration_dir),
        "certification": Path(certification_dir),
    }

    saved = []
    for name in names:
        out_dir = out_dirs[name]
        idx = idx_chunks[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", embeddings[idx])
        np.save(out_dir / "concepts.npy", concepts[idx])
        np.save(out_dir / "labels.npy", labels[idx])
        np.save(out_dir / "ids.npy", ids[idx])
        order = "shuffled" if name in ("model_train", "reference_train") else "original order"
        saved.append(f"{out_dir} ({len(idx)} rows, {order})")
    print("[saved] " + " / ".join(saved))
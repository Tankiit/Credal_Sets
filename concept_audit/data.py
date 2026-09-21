"""
Load and split feature+concept caches for run.py.

A "cache" directory is the on-disk unit run.py consumes: four row-aligned
arrays -- embeddings.npy, labels.npy, ids.npy (from 01_extract_features.py)
and concepts.npy (from build_concept_cache.py).

load_cache() just loads one such directory as-is, with shape checks.
split_cache() is the piece that turns ONE full cache into the
--features-dir / --eval-features-dir pair run.py expects: an 80% train
split in randomly shuffled row order, and a 20% eval split restored to
ascending original order (eval is read once for audit, not iterated over
epochs, so it stays unshuffled and reproducible id-for-id).
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
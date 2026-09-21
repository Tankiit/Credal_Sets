"""
Build concepts.npy for a features cache, joining 01_extract_features.py's
embeddings/labels/ids against a dataset's own soft/credal annotation file.

Two annotation formats exist upstream (see common.py's module docstring):
  --soft-labels PATH        CIFAR-10H's soft_labels.csv (prob_0..prob_{K-1})
                             -> concepts.npy columns = prob_0..prob_{K-1}
  --attribute-credal PATH   CUB's attribute_credal.csv (lower_0..lower_{A-1},
                             upper_0..upper_{A-1})
                             -> concepts.npy columns = lower_0..lower_{A-1},
                                upper_0..upper_{A-1} (kept as an interval,
                                not collapsed to a point estimate -- see the
                                note at the bottom of this reply)

Exactly one of the two must be given. Rows whose id has no matching
annotation are dropped from ALL FOUR output arrays (embeddings/labels/
ids/concepts), so everything in --out-dir stays row-aligned.
"""
import argparse
from pathlib import Path

import numpy as np

from blindspot_analysis.common import load_soft_labels, load_attribute_credal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True,
                     help="output of 01_extract_features.py (embeddings.npy, labels.npy, ids.npy)")
    ap.add_argument("--soft-labels", type=Path, help="CIFAR-10H-style soft_labels.csv")
    ap.add_argument("--attribute-credal", type=Path, help="CUB-style attribute_credal.csv")
    ap.add_argument("--num-attributes", type=int,
                     help="required with --attribute-credal, e.g. 312 for CUB")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if bool(args.soft_labels) == bool(args.attribute_credal):
        ap.error("pass exactly one of --soft-labels / --attribute-credal")
    if args.attribute_credal and args.num_attributes is None:
        ap.error("--attribute-credal requires --num-attributes")

    embeddings = np.load(args.features_dir / "embeddings.npy")
    labels = np.load(args.features_dir / "labels.npy")
    ids = np.load(args.features_dir / "ids.npy", allow_pickle=True)
    id_list = [str(i) for i in ids]

    if args.soft_labels:
        concepts, aligned_ids = load_soft_labels(args.soft_labels, id_list)
    else:
        lower, upper, aligned_ids = load_attribute_credal(
            args.attribute_credal, id_list, args.num_attributes
        )
        concepts = np.concatenate([lower, upper], axis=1)

    # keep embeddings/labels/ids in sync with whatever the loader kept
    # (it may have dropped ids with no matching annotation row)
    pos = {id_: i for i, id_ in enumerate(id_list)}
    keep = [pos[i] for i in aligned_ids]
    embeddings = embeddings[keep]
    labels = labels[keep]
    ids = ids[keep]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", embeddings.astype(np.float32))
    np.save(args.out_dir / "concepts.npy", concepts.astype(np.float32))
    np.save(args.out_dir / "labels.npy", labels.astype(np.int64))
    np.save(args.out_dir / "ids.npy", ids)
    print(f"[saved] {args.out_dir} ({len(aligned_ids)} rows, {concepts.shape[1]} concept dims)")


if __name__ == "__main__":
    main()
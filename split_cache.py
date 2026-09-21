
"""CLI wrapper around concept_audit.data.split_cache."""
import argparse
from pathlib import Path

from concept_audit.data import split_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split_cache(args.cache_dir, args.train_dir, args.eval_dir, args.train_frac, args.seed)


if __name__ == "__main__":
    main()
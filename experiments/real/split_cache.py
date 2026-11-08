"""CLI wrapper around concept_audit.data.split_cache / split_cache_4way."""
import argparse
from pathlib import Path

from concept_audit.data import split_cache, split_cache_4way


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)

    two_way = ap.add_argument_group("two-way split (train / eval)")
    two_way.add_argument("--train-dir", type=Path)
    two_way.add_argument("--eval-dir", type=Path)
    two_way.add_argument("--train-frac", type=float, default=0.8)

    four_way = ap.add_argument_group(
        "four-way split (model-train / reference-train / calibration / certification)"
    )
    four_way.add_argument("--model-train-dir", type=Path)
    four_way.add_argument("--reference-train-dir", type=Path)
    four_way.add_argument("--calibration-dir", type=Path)
    four_way.add_argument("--certification-dir", type=Path)
    four_way.add_argument("--model-train-frac", type=float, default=0.4)
    four_way.add_argument("--reference-train-frac", type=float, default=0.2)
    four_way.add_argument("--calibration-frac", type=float, default=0.2)
    four_way.add_argument("--certification-frac", type=float, default=0.2)

    args = ap.parse_args()

    four_way_dirs = [
        args.model_train_dir, args.reference_train_dir,
        args.calibration_dir, args.certification_dir,
    ]
    if any(d is not None for d in four_way_dirs):
        if not all(d is not None for d in four_way_dirs):
            ap.error(
                "four-way split requires all of --model-train-dir "
                "--reference-train-dir --calibration-dir --certification-dir"
            )
        split_cache_4way(
            args.cache_dir,
            args.model_train_dir, args.reference_train_dir,
            args.calibration_dir, args.certification_dir,
            args.model_train_frac, args.reference_train_frac,
            args.calibration_frac, args.certification_frac,
            args.seed,
        )
    else:
        if not (args.train_dir and args.eval_dir):
            ap.error(
                "must supply either --train-dir/--eval-dir (two-way) or "
                "--model-train-dir/--reference-train-dir/--calibration-dir/"
                "--certification-dir (four-way)"
            )
        split_cache(args.cache_dir, args.train_dir, args.eval_dir, args.train_frac, args.seed)


if __name__ == "__main__":
    main()
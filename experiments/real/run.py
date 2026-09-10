"""Train and audit on separate, annotated real-data feature caches."""
import argparse
from pathlib import Path
import torch
from experiments.arguments import add_training_arguments, parse_arguments
from concept_audit.data import load_cache
from concept_audit.training import train_and_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "results/real/audit.json")
    parser.add_argument("--features-dir", type=Path, help="Training cache with concepts.npy")
    parser.add_argument("--eval-features-dir", type=Path, help="Separate held-out annotated cache")
    args = parse_arguments(parser, argv)
    if args.features_dir is None or args.eval_features_dir is None:
        parser.error("Real experiments require both --features-dir and --eval-features-dir; no synthetic fallback")
    if args.features_dir.resolve() == args.eval_features_dir.resolve():
        parser.error("Training and evaluation caches must be separate")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    z_train, g_train, y_train = load_cache(args.features_dir)
    z_eval, g_eval, y_eval = load_cache(args.eval_features_dir)
    if z_train.shape[1] != z_eval.shape[1] or g_train.shape[1] != g_eval.shape[1]:
        parser.error("Train/evaluation feature and concept dimensions must match")
    train_and_audit(torch.cat((z_train, z_eval)), torch.cat((g_train, g_eval)),
                           torch.cat((y_train, y_eval)), len(z_train), args, family="real",
                           provenance={"data": "cached_features", "train_cache": str(args.features_dir),
                                       "eval_cache": str(args.eval_features_dir)})


if __name__ == "__main__":
    main()

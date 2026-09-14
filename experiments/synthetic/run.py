"""Synthetic theory sanity checks and baseline diagnostic audits."""
import argparse
import torch
from experiments.arguments import add_training_arguments, parse_arguments
from experiments.synthetic.data import DATA_SEED, make_dataset
from concept_audit.training import train_and_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "results/synthetic/audit.json")
    parser.add_argument("--condition", default="complete",
                        choices=["complete", "incomplete_concepts"],
                        help="complete: y = g(c*). incomplete_concepts: y = g(c*, u).")
    parser.add_argument("--data-seed", type=int, default=DATA_SEED,
                        help="Dataset seed. Frozen by default so --seed moves only the model.")
    args = parse_arguments(parser, argv)

    # --seed is the MODEL seed: initialisation and batch order. The dataset is
    # drawn from --data-seed, so runs at different model seeds see identical
    # data and their representations are comparable.
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    z, g, y, n_train = make_dataset(args.data_seed, condition=args.condition)
    train_and_audit(z, g, y, n_train, args, family="synthetic",
                    provenance={"data": "synthetic", "condition": args.condition,
                                "data_seed": args.data_seed, "model_seed": args.seed,
                                "train_cache": None, "eval_cache": None})


if __name__ == "__main__":
    main()

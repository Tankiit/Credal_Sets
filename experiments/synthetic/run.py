"""Synthetic theory sanity checks and baseline diagnostic audits."""
import argparse
import torch
from experiments.arguments import add_training_arguments, parse_arguments
from experiments.synthetic.data import make_data
from concept_audit.training import train_and_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "results/synthetic/audit.json")
    args = parse_arguments(parser, argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    z, g, y, n_train = make_data(args.seed)
    train_and_audit(z, g, y, n_train, args, family="synthetic",
                           provenance={"data": "synthetic", "train_cache": None, "eval_cache": None})


if __name__ == "__main__":
    main()

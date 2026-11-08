"""Compatibility entry point; prefer experiments.synthetic.run or experiments.real.run."""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--features-dir")
    parser.add_argument("--eval-features-dir")
    args, _ = parser.parse_known_args(argv)
    if args.features_dir is not None or args.eval_features_dir is not None:
        from experiments.real.run import main as run
    else:
        from experiments.synthetic.run import main as run
    return run(argv)


if __name__ == "__main__":
    main()

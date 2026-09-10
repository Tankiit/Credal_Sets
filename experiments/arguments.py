"""Common CLI options; each family validates its own JSON configuration keys."""
import argparse
import json
from pathlib import Path


def add_training_arguments(parser, output):
    parser.add_argument("--config", type=Path, help="Family-specific JSON config; CLI flags override it")
    parser.add_argument("--backend", choices=["native", "pyc", "cem"], default="native")
    parser.add_argument("--readout", choices=["identity", "coordinates", "grouped"], default="coordinates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(output))


def parse_arguments(parser, argv):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        config = json.loads(config_args.config.read_text())
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
        if not isinstance(config, dict) or set(config) - actions.keys():
            parser.error("Config must be an object containing only this experiment family's options")
        for key, value in config.items():
            action = actions[key]
            if action.type is int and (not isinstance(value, int) or isinstance(value, bool)):
                parser.error(f"Config {key} must be an integer")
            if action.type is Path:
                if not isinstance(value, str):
                    parser.error(f"Config {key} must be a path string")
                config[key] = Path(value)
            if action.choices and value not in action.choices:
                parser.error(f"Invalid config {key}: {value}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    return args

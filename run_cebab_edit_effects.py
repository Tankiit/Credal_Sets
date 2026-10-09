"""Run the validated CEBaB edit-pair analysis for one model/twin pair.

Example:
  python run_cebab_edit_effects.py --run runs/cebab-cbm-r16-logits-s0 --mix 1 --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cebab_edits import compare_model_and_twin, load_edit_pairs
from concept_models.features import get_features
from concept_models.models import load_model
from concept_models.reparam import make_twin


def _correlation(x, y) -> float | None:
    result = spearmanr(x, y)
    return None if np.isnan(result.statistic) else float(result.statistic)


def main():
    parser = argparse.ArgumentParser(description="CEBaB targeted edit-pair effects")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rotate", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Twin random seed")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = load_model(args.run / "model.pt")
    if model.config.get("dataset") != "cebab":
        raise ValueError(f"{args.run} is not a CEBaB model")
    twin, transform = make_twin(model, args.rotate, args.mix, args.shift, args.seed)
    features = get_features("cebab", args.split, model.config["encoder"])
    pairs = load_edit_pairs(args.split)
    rows = compare_model_and_twin(model, twin, features, pairs)

    human = [r["human_effect"] for r in rows]
    model_effect = [r["model_expected_effect"] for r in rows]
    twin_effect = [r["twin_expected_effect"] for r in rows]
    gaps = [abs(r["twin_minus_model_expected_effect"]) for r in rows]
    summary = {
        "run": args.run.name,
        "split": args.split,
        "n_pairs": len(rows),
        "transform": transform,
        "human_effect_mean": float(np.mean(human)),
        "model_human_spearman": _correlation(model_effect, human),
        "twin_human_spearman": _correlation(twin_effect, human),
        "mean_abs_twin_model_effect_gap": float(np.mean(gaps)),
        "max_abs_twin_model_effect_gap": float(np.max(gaps)),
        "label_change_disagreements": int(sum(r["twins_disagree_on_label_change"] for r in rows)),
    }
    out = args.out or args.run / "cebab_edit_effects.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()

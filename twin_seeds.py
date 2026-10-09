"""Twin seeds script: evaluates twin reparameterizations across multiple twin random seeds (n_tseeds=20).

Usage:
    python twin_seeds.py --runs_dir runs --out results/twin_seeds.jsonl --n_tseeds 20
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from sweep import one, get_sweep_features
from concept_models.models import load_model

MIX = [0.0, 0.5, 1.0, 2.0]
ROT, SHIFT = 1.0, 0.0  # Fixed rotation and shift for scaling grid


def main():
    ap = argparse.ArgumentParser(description="Sweep over multiple twin random seeds")
    ap.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing model runs")
    ap.add_argument("--out", type=Path, default=Path("results/twin_seeds.jsonl"), help="Output JSONL path")
    ap.add_argument("--n_tseeds", type=int, default=20, help="Number of twin random seeds per cell")
    ap.add_argument("--run_glob", default="*-s0", help="Glob pattern for run directories (e.g., '*-s[0-4]')")
    a = ap.parse_args()

    a.out.parent.mkdir(exist_ok=True, parents=True)
    n = 0
    runs = sorted(a.runs_dir.glob(a.run_glob))
    print(f"Starting twin_seeds evaluation across {len(runs)} runs (n_tseeds={a.n_tseeds})...")

    with open(a.out, "w") as fh:
        for run in runs:
            model_path = run / "model.pt"
            if not model_path.exists():
                print(f"[skip] no model.pt in {run}")
                continue

            model = load_model(model_path)
            cfg = model.config
            feats = get_sweep_features(cfg["dataset"], "test", cfg["encoder"], cfg)

            print(f"Processing run: {run.name}")
            for mix in tqdm(MIX, desc=f"Twin seeds {run.name}", leave=False):
                for ts in range(a.n_tseeds):
                    r = one(run, ROT, mix, SHIFT, ts, feats)
                    if r:
                        r["train_seed"] = int(run.name.split("-s")[-1])
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        n += 1

    if n == 0:
        raise SystemExit("0 records written")
    print(f"Successfully wrote {n} evaluation records to {a.out}")


if __name__ == "__main__":
    main()

"""Train a CBM or CEM on one dataset (frozen-encoder features, cached on first use).

    python train.py --dataset cebab --model cem
    python train.py --dataset goemotions --model cbm --residual_dim 16
    python train.py --dataset civil_comments --model cbm --soft_concepts

Outputs runs/<name>/{model.pt, metrics.json}.  Load later with
concept_models.models.load_model("runs/<name>/model.pt").
"""

import argparse
import json
from pathlib import Path

import torch

from concept_datasets import DATASETS, load_meta
from concept_models.features import DEFAULT_ENCODER, get_features
from concept_models.models import build_model, save_model
from concept_models.training import evaluate, fit

# Per-dataset defaults; any of them can be overridden from the command line.
DEFAULTS = {
    "cebab": {"concept_weight": 5.0, "epochs": 60},
    "goemotions": {"concept_weight": 5.0, "epochs": 60},
    "civil_comments": {"concept_weight": 5.0, "epochs": 60},
    "imdb_cad": {"concept_weight": 5.0, "epochs": 100},
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--dataset", required=True, choices=DATASETS)
p.add_argument("--model", required=True, choices=["cbm", "cem"])
p.add_argument("--residual_dim", type=int, default=0, help="CBM: extra unsupervised dims in z (0 = vanilla CBM)")
p.add_argument("--head_input", default="probs", choices=["probs", "logits"],
               help="CBM: head reads concept probabilities (classic) or logits (allows --mix in twin.py)")
p.add_argument("--emb_dim", type=int, default=16, help="CEM: embedding size m per concept")
p.add_argument("--p_int", type=float, default=0.25, help="CEM: random intervention prob. during training")
p.add_argument("--head", default="linear", choices=["linear", "mlp"])
p.add_argument("--hidden", type=int, default=256, help="width of the shared trunk")
p.add_argument("--concept_weight", type=float)
p.add_argument("--soft_concepts", action="store_true", help="train concepts on soft scores (civil_comments)")
p.add_argument("--epochs", type=int)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--encoder", default=DEFAULT_ENCODER)
p.add_argument("--name", help="run name (default: <dataset>-<model>[-r<dim>][-logits]-s<seed>)")
args = p.parse_args()
for key, value in DEFAULTS[args.dataset].items():
    if getattr(args, key) is None:
        setattr(args, key, value)

meta = load_meta(args.dataset)
feats = {s: get_features(args.dataset, s, args.encoder) for s in ("train", "val", "test")}

torch.manual_seed(args.seed)
config = {"kind": args.model, "in_dim": feats["train"]["X"].shape[1],
          "n_concepts": len(meta["concept_names"]), "n_classes": len(meta["class_names"]),
          "hidden": args.hidden, "head": args.head, "dataset": args.dataset, "encoder": args.encoder}
if args.model == "cbm":
    config.update(residual_dim=args.residual_dim, head_input=args.head_input)
else:
    config.update(emb_dim=args.emb_dim, p_int=args.p_int)
model = build_model(config)

device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
print(f"\nTraining {args.model} on {args.dataset} ({config['n_concepts']} concepts, "
      f"{config['n_classes']} classes) on {device}")
history = fit(model, feats["train"], feats["val"], concept_weight=args.concept_weight,
              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
              soft_concepts=args.soft_concepts, device=device, seed=args.seed)

metrics = {s: evaluate(model, feats[s]) for s in ("val", "test")}
name = args.name or "-".join(
    [args.dataset, args.model]
    + ([f"r{args.residual_dim}"] if args.model == "cbm" and args.residual_dim else [])
    + (["logits"] if args.model == "cbm" and args.head_input == "logits" else [])
    + [f"s{args.seed}"])
out = Path("runs") / name
out.mkdir(parents=True, exist_ok=True)
save_model(model, out / "model.pt")
(out / "metrics.json").write_text(json.dumps(
    {"args": vars(args), "config": config, "metrics": metrics, "history": history}, indent=2))
print(f"\nSaved {out}/model.pt")
for split, m in metrics.items():
    print(f"  {split:<5} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

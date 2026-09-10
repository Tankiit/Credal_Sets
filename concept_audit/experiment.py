"""Train on frozen features, then audit. Default data are synthetic, not images."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from concept_audit.data import load_cache
from concept_audit.models import NativeCBM, NativeCEM, PyCAdapter
from concept_audit.readouts import IdentityReadout, CoordinateReadout, GroupReadout
from concept_audit.transforms import admissible_transform
from concept_audit.diagnostics import AuditState, default_registry
from concept_audit.audits import audit_equivalence, audit_structural, audit_informational, audit_consequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--eval-features-dir", type=Path, help="Separate held-out cache; required with --features-dir")
    parser.add_argument("--backend", choices=["native", "pyc", "cem"], default="native")
    parser.add_argument("--readout", choices=["identity", "coordinates", "grouped"], default="coordinates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/concept_audit.json"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if bool(args.features_dir) != bool(args.eval_features_dir):
        parser.error("Provide both train and evaluation feature directories")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.features_dir:
        if args.features_dir.resolve() == args.eval_features_dir.resolve():
            parser.error("Training and evaluation caches must be separate")
        z_train, g_train, y_train = load_cache(args.features_dir)
        z_eval, g_eval, y_eval = load_cache(args.eval_features_dir)
        if z_train.shape[1] != z_eval.shape[1] or g_train.shape[1] != g_eval.shape[1]:
            parser.error("Train/evaluation feature and concept dimensions must match")
        n_train = len(z_train)
        z, g, y = torch.cat((z_train, z_eval)), torch.cat((g_train, g_eval)), torch.cat((y_train, y_eval))
    else:
        z = torch.randn(512, 12)
        g = (z[:, :3] > 0).float()
        y = (g[:, 0] + g[:, 1] + (z[:, 3] > 0)).long() % 2
        n_train = 384
    k, classes = g.shape[1], int(y.max()) + 1
    train = torch.arange(len(z)) < n_train
    if args.backend == "cem":
        model = NativeCEM(z.shape[1], classes, k)
    else:
        if args.readout == "identity":
            readout, blocks = IdentityReadout(k), None
        elif args.readout == "coordinates":
            readout, blocks = CoordinateReadout(2*k, range(k)), [(j, k+j) for j in range(k)]
        else:
            blocks = [(2*j, 2*j+1) for j in range(k)]
            readout = GroupReadout(2*k, blocks)
        factory = NativeCBM if args.backend == "native" else PyCAdapter
        model = factory(z.shape[1], classes, readout, blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    model.train()
    for _ in range(args.epochs):
        for rows in torch.arange(n_train).split(128):
            concept_scores, logits = model(z[rows])
            loss = F.binary_cross_entropy_with_logits(concept_scores, g[rows]) + F.cross_entropy(logits, y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Double precision isolates mathematical invariance from float32 roundoff.
    model.eval().double()
    with torch.no_grad():
        state = AuditState(model.encode(z.double()), g.double(), y, model, train)
        registry = default_registry()
        a = admissible_transform(model.readout, seed=args.seed)
        report = {
            "seed": args.seed, "backend": args.backend, "epochs": args.epochs,
            "data": "cached_features" if args.features_dir else "synthetic",
            "train_cache": str(args.features_dir) if args.features_dir else None,
            "eval_cache": str(args.eval_features_dir) if args.eval_features_dir else None,
            "readout_matrix": model.readout.matrix.tolist(), "blocks": model.blocks,
            "transform_matrix": a.tolist(),
            "train_samples": n_train, "eval_samples": len(z)-n_train,
            "equivalence": audit_equivalence(state, a, registry, tol=1e-8),
            "structural": audit_structural(state, registry),
            "informational": audit_informational(state, registry, args.seed),
            "consequence": audit_consequence(state, registry, args.seed),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save({"model_state_dict": model.state_dict(), "backend": args.backend,
                "feature_dim": z.shape[1], "num_classes": classes,
                "blocks": model.blocks, "readout_matrix": model.readout.matrix,
                "transform_matrix": a, "seed": args.seed}, args.out.with_suffix(".pt"))
    print(f"Saved {args.out}; max logit error={report['equivalence']['max_logit_error']:.3g}")


if __name__ == "__main__":
    main()

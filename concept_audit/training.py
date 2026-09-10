"""Shared training and audit mechanics; experiment families own their data/configs."""
import json
import torch
from torch.nn import functional as F
from concept_audit.models import NativeCBM, NativeCEM, PyCAdapter
from concept_audit.readouts import IdentityReadout, CoordinateReadout, GroupReadout
from concept_audit.transforms import admissible_transform
from concept_audit.diagnostics import AuditState, default_registry
from concept_audit.audits import audit_equivalence, audit_structural, audit_informational, audit_consequence


def train_and_audit(z, g, y, n_train, args, *, family, provenance):
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
            "data": provenance["data"], "experiment_family": family,
            **provenance,
            "readout_matrix": model.readout.matrix.tolist(), "blocks": model.blocks,
            "readout_rank": int(torch.linalg.matrix_rank(model.readout.matrix)),
            "unconstrained_dim": model.readout.latent_dim - int(torch.linalg.matrix_rank(model.readout.matrix)),
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

    return report

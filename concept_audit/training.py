"""Shared training and audit mechanics; experiment families own their data/configs."""
import json
from datetime import datetime, timezone
from uuid import uuid4
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
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
    log_root = getattr(args, "log_dir", None) or args.out.parent / "tensorboard"
    run_name = f"{family}-{args.backend}-{args.readout}-seed{args.seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    log_dir = log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        writer.add_text("run/config", json.dumps({**vars(args), "family": family, **provenance}, default=str, indent=2), 0)
        epochs = tqdm(range(args.epochs), desc=f"{family}: training", unit="epoch", dynamic_ncols=True)
        for epoch in epochs:
            model.train()
            totals = {"loss": 0., "concept_loss": 0., "task_loss": 0., "task_accuracy": 0.}
            batches = torch.arange(n_train).split(128)
            for rows in tqdm(batches, desc="Training batches", unit="batch", leave=False, dynamic_ncols=True):
                concept_scores, logits = model(z[rows])
                concept_loss = F.binary_cross_entropy_with_logits(concept_scores, g[rows])
                task_loss = F.cross_entropy(logits, y[rows])
                loss = concept_loss + task_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals["loss"] += loss.item() * len(rows)
                totals["concept_loss"] += concept_loss.item() * len(rows)
                totals["task_loss"] += task_loss.item() * len(rows)
                totals["task_accuracy"] += (logits.detach().argmax(1) == y[rows]).sum().item()
            for name, value in totals.items():
                writer.add_scalar(f"train/{name}", value / n_train, epoch + 1)
            model.eval()
            evaluation = {"loss": 0., "concept_loss": 0., "task_loss": 0., "task_accuracy": 0.}
            with torch.no_grad():
                for rows in torch.arange(n_train, len(z)).split(128):
                    scores, logits = model(z[rows])
                    concept_loss = F.binary_cross_entropy_with_logits(scores, g[rows]).item()
                    task_loss = F.cross_entropy(logits, y[rows]).item()
                    evaluation["loss"] += (concept_loss + task_loss) * len(rows)
                    evaluation["concept_loss"] += concept_loss * len(rows)
                    evaluation["task_loss"] += task_loss * len(rows)
                    evaluation["task_accuracy"] += (logits.argmax(1) == y[rows]).sum().item()
            for name, value in evaluation.items():
                writer.add_scalar(f"eval/{name}", value / (len(z) - n_train), epoch + 1)
            epochs.set_postfix(loss=f"{totals['loss']/n_train:.4f}", eval_acc=f"{evaluation['task_accuracy']/(len(z)-n_train):.3f}")
        epochs.close()
        # Double precision isolates mathematical invariance from float32 roundoff.
        model.eval().double()
        with torch.no_grad():
            state = AuditState(model.encode(z.double()), g.double(), y, model, train)
            registry = default_registry()
            a = admissible_transform(model.readout, seed=args.seed)
            audit_results = {}
            audits = [
                ("equivalence", lambda: audit_equivalence(state, a, registry, tol=1e-8)),
                ("structural", lambda: audit_structural(state, registry)),
                ("informational", lambda: audit_informational(state, registry, args.seed)),
                ("consequence", lambda: audit_consequence(state, registry, args.seed)),
            ]
            for name, run in tqdm(audits, desc="Auditing", unit="audit", dynamic_ncols=True):
                audit_results[name] = run()
            report = {
                "seed": args.seed, "backend": args.backend, "epochs": args.epochs,
                "data": provenance["data"], "experiment_family": family,
                **provenance,
                "readout_matrix": model.readout.matrix.tolist(), "blocks": model.blocks,
                "readout_rank": int(torch.linalg.matrix_rank(model.readout.matrix)),
                "unconstrained_dim": model.readout.latent_dim - int(torch.linalg.matrix_rank(model.readout.matrix)),
                "transform_matrix": a.tolist(),
                "train_samples": n_train, "eval_samples": len(z)-n_train,
                **audit_results,
                "tensorboard_log_dir": str(log_dir),
            }
        for name in ("readout_rank", "unconstrained_dim"):
            writer.add_scalar(f"audit/{name}", report[name], args.epochs)
        _log_scalars(writer, "audit", audit_results, args.epochs)
        writer.flush()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        torch.save({"model_state_dict": model.state_dict(), "backend": args.backend,
                    "feature_dim": z.shape[1], "num_classes": classes,
                    "blocks": model.blocks, "readout_matrix": model.readout.matrix,
                    "transform_matrix": a, "seed": args.seed}, args.out.with_suffix(".pt"))
        print(f"Saved {args.out}; max logit error={report['equivalence']['max_logit_error']:.3g}")

        return report
    finally:
        writer.close()


def _log_scalars(writer, prefix, value, step):
    """Record audit scalars, including indexed per-concept diagnostics."""
    if isinstance(value, dict):
        for key, child in value.items():
            _log_scalars(writer, f"{prefix}/{key}", child, step)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _log_scalars(writer, f"{prefix}/{index}", child, step)
    elif isinstance(value, (int, float)):
        writer.add_scalar(prefix, value, step)

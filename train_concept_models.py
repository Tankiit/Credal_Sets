"""Train auditable CBM/residual-CBM/CEM on cached frozen encoder features for real text datasets (CEBaB, Civil Comments, GoEmotions).

Commands:
  python train_concept_models.py prepare --dataset cebab --out cebab_data
  python train_concept_models.py train --data cebab_data --arch cbm --out runs/cbm
  python train_concept_models.py train --data cebab_data --arch cem --out runs/cem
  python train_concept_models.py export --data cebab_data --checkpoint runs/cem/best.pt --out traces/cem
  python train_concept_models.py self-test

Required: PyTorch. Optional --backend torch-concepts uses Torch Concepts' low-level
linear layers. CEM mixing is implemented explicitly so state embeddings and the
actual mixed blocks are accessible without relying on private library methods.
This is a transparent CEM variant, not a reproduction of package default training.

DATA CONTRACT
  schema.json: {"concepts": [{"name":"food_pos", "type":"binary"}, ...],
                "task_type":"multiclass", "n_tasks":5,
                "encoder_metadata":{"model":"sentence-transformers/all-mpnet-base-v2", "pooling":"mean"}}
  train.pt, val.pt, test.pt: dictionaries of CPU tensors:
    x [N,D] float cached features, c [N,K] float concept targets, y task targets.
    Binary c: probability in [0,1]; categorical c: integer class index; continuous
    c: value in a fixed scale. NaN denotes missing concepts. Optional c_mask [N,K]
    boolean. Multiclass y: [N] integer; multilabel y: [N,T] 0/1. No missing task labels.
    Optional ids: list[str] unique across splits (checked if all splits provide it).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from concept_datasets import load_meta, DATASETS
from concept_models.features import get_features, DEFAULT_ENCODER


def validate_schema(schema):
    concepts = schema["concepts"]
    if not concepts or len({c["name"] for c in concepts}) != len(concepts):
        raise ValueError("Concepts must be nonempty with unique names.")
    for c in concepts:
        if c["type"] not in {"binary", "categorical", "continuous"}:
            raise ValueError(f"Unsupported concept: {c}")
        if c["type"] == "categorical" and int(c.get("cardinality", 0)) < 2:
            raise ValueError("Categorical concepts need cardinality >= 2.")
    if schema["task_type"] not in {"multiclass", "multilabel"}:
        raise ValueError("task_type must be multiclass or multilabel.")
    if int(schema["n_tasks"]) < (2 if schema["task_type"] == "multiclass" else 1):
        raise ValueError("Invalid n_tasks.")


class LinearStage(nn.Module):
    """Same numerical linear map, optionally using the documented low-level API."""
    def __init__(self, in_dim, out_dim, backend, role):
        super().__init__()
        self.backend, self.role = backend, role
        if backend == "torch-concepts":
            from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept
            self.layer = (LinearEmbeddingToConcept(in_embeddings=in_dim, out_concepts=out_dim)
                          if role == "readout" else
                          LinearConceptToConcept(in_concepts=in_dim, out_concepts=out_dim))
        else:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.backend == "torch":
            return self.layer(x)
        return self.layer(embeddings=x) if self.role == "readout" else self.layer(concepts=x)


class ConceptModel(nn.Module):
    def __init__(self, config, schema):
        super().__init__()
        validate_schema(schema)
        self.config, self.schema = dict(config), schema
        self.concepts = schema["concepts"]
        self.arch = config["arch"]
        if self.arch not in {"cbm", "residual-cbm", "cem"}:
            raise ValueError("Unknown architecture.")
        d, e = config["feature_dim"], config["embedding_dim"]
        self.register_buffer("feature_mean", torch.zeros(d))
        self.register_buffer("feature_scale", torch.ones(d))
        self.slices, start = {}, 0
        for c in self.concepts:
            width = int(c["cardinality"]) if c["type"] == "categorical" else 1
            self.slices[c["name"]] = slice(start, start + width)
            start += width
        self.total_concept_width = start
        self.feature_tap = nn.Identity()
        self.concept_values_tap = nn.Identity()
        self.head_input_tap = nn.Identity()
        backend = config["backend"]
        if self.arch == "cem":
            if any(c["type"] == "continuous" for c in self.concepts):
                raise ValueError("This CEM supports binary/categorical concepts, not magnitudes.")
            self.state_encoders, self.concept_readouts = nn.ModuleList(), nn.ModuleList()
            self.states = []
            for c in self.concepts:
                states = 2 if c["type"] == "binary" else int(c["cardinality"])
                width = 1 if c["type"] == "binary" else states
                self.states.append(states)
                self.state_encoders.append(nn.Sequential(nn.Linear(d, states * e), nn.LeakyReLU()))
                self.concept_readouts.append(LinearStage(states * e, width, backend, "readout"))
            head_dim = len(self.concepts) * e
        else:
            self.concept_readout = LinearStage(d, start, backend, "readout")
            head_dim = start
        self.residual = None
        if self.arch == "residual-cbm":
            if config["residual_dim"] < 1:
                raise ValueError("Residual CBM requires residual_dim >= 1.")
            self.residual = nn.Linear(d, config["residual_dim"])
            head_dim += config["residual_dim"]
        self.task_head = LinearStage(head_dim, schema["n_tasks"], backend, "task")

    def normalize_concepts(self, raw):
        parts = []
        for c in self.concepts:
            v = raw[:, self.slices[c["name"]]]
            parts.append(v.sigmoid() if c["type"] == "binary" else
                         v.softmax(-1) if c["type"] == "categorical" else v)
        return torch.cat(parts, -1)

    def edit(self, values, edits):
        edited = values.clone()
        by_name = {c["name"]: c for c in self.concepts}
        for name, replacement in (edits or {}).items():
            if name not in by_name:
                raise KeyError(name)
            sl = self.slices[name]
            target = edited[:, sl]
            v = torch.as_tensor(replacement, dtype=values.dtype, device=values.device)
            if target.shape[-1] == 1 and v.shape == target.shape[:-1]:
                v = v.unsqueeze(-1)
            v = torch.broadcast_to(v, target.shape)
            kind = by_name[name]["type"]
            if not bool(torch.isfinite(v).all()):
                raise ValueError(f"Nonfinite edit: {name}")
            if kind != "continuous" and bool(((v < 0) | (v > 1)).any()):
                raise ValueError("Discrete edits must be probabilities in [0,1].")
            if kind == "categorical" and not torch.allclose(v.sum(-1), torch.ones_like(v.sum(-1))):
                raise ValueError("Categorical edit must replace a whole probability vector summing to one.")
            edited[:, sl] = v
        return edited

    def forward(self, x, *, edits=None, return_trace=False):
        h = self.feature_tap((x - self.feature_mean) / self.feature_scale)
        states = None
        if self.arch == "cem":
            states = [enc(h).reshape(len(h), s, self.config["embedding_dim"])
                      for enc, s in zip(self.state_encoders, self.states)]
            raw = torch.cat([readout(state.flatten(1)) for readout, state in
                             zip(self.concept_readouts, states)], -1)
        else:
            raw = self.concept_readout(h)
        values = self.normalize_concepts(raw)
        edited = self.concept_values_tap(self.edit(values, edits))
        mixed, residual = None, None
        if self.arch == "cem":
            blocks = []
            for c, state in zip(self.concepts, states):
                p = edited[:, self.slices[c["name"]]]
                if c["type"] == "binary":
                    p = torch.cat([1 - p, p], -1)  # state 0 negative; state 1 positive
                blocks.append((p.unsqueeze(-1) * state).sum(1))
            mixed = torch.stack(blocks, 1)  # [B,K,E]
            head_input = mixed.flatten(1)
        else:
            residual = self.residual(h) if self.residual is not None else None
            head_input = edited if residual is None else torch.cat([edited, residual], -1)
        head_input = self.head_input_tap(head_input)
        logits = self.task_head(head_input)
        if not return_trace:
            return logits
        return logits, {"encoder_features": x, "normalized_features": h,
                        "concept_raw": raw, "concept_values": values,
                        "concept_edited": edited, "state_embeddings": states,
                        "mixed_embeddings": mixed, "residual": residual,
                        "head_input": head_input, "task_logits": logits}


def concept_loss(raw, c, mask, model):
    terms = []
    for j, spec in enumerate(model.concepts):
        observed = mask[:, j]
        if not bool(observed.any()):
            continue
        pred = raw[observed, model.slices[spec["name"]]]
        target = c[observed, j]
        kind = spec["type"]
        terms.append(F.cross_entropy(pred, target.long()) if kind == "categorical" else
                     F.binary_cross_entropy_with_logits(pred[:, 0], target) if kind == "binary" else
                     F.mse_loss(pred[:, 0], target))
    return torch.stack(terms).mean() if terms else raw.sum() * 0


def task_loss(logits, y, schema, loss_name: str = "cross_entropy"):
    if schema["task_type"] == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, y.float())
    if loss_name == "cross_entropy":
        return F.cross_entropy(logits, y.long())
    if loss_name == "mse":
        target = F.one_hot(y.long(), num_classes=schema["n_tasks"]).to(logits.dtype)
        # Sum per-class squared error, then average examples.  Averaging over
        # classes would reduce this term by 1 / n_tasks relative to CE and
        # unintentionally let concept supervision dominate the joint loss.
        return F.mse_loss(logits.softmax(-1), target, reduction="none").sum(-1).mean()
    raise ValueError(f"Unknown multiclass task loss: {loss_name}")


def load_data(folder):
    folder = Path(folder)
    schema = json.loads((folder / "schema.json").read_text())
    validate_schema(schema)
    splits, raw_splits = {}, {}
    for name in ("train", "val", "test"):
        data = torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True)
        x, c, y = data["x"].float(), data["c"].float(), data["y"]
        if x.ndim != 2 or not len(x) or c.shape != (len(x), len(schema["concepts"])) or len(y) != len(x):
            raise ValueError(f"Invalid feature/target shapes in {name}.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Features must be finite.")
        mask = data.get("c_mask", torch.ones_like(c, dtype=torch.bool)).bool()
        if mask.shape != c.shape:
            raise ValueError("c_mask must have the same shape as c.")
        mask = mask & torch.isfinite(c)
        for j, spec in enumerate(schema["concepts"]):
            t = c[mask[:, j], j]
            if spec["type"] == "binary" and bool(((t < 0) | (t > 1)).any()):
                raise ValueError("Binary concept targets must lie in [0,1].")
            if spec["type"] == "categorical" and bool(((t < 0) | (t >= spec["cardinality"]) | (t != t.round())).any()):
                raise ValueError("Categorical concept targets must be valid integer indices.")
        if schema["task_type"] == "multiclass":
            if y.shape != (len(x),) or not bool(torch.isfinite(y).all()) or bool(((y < 0) | (y >= schema["n_tasks"]) | (y != y.round())).any()):
                raise ValueError("Multiclass task targets must be valid integer indices.")
        elif y.shape != (len(x), schema["n_tasks"]) or not bool(torch.isfinite(y).all()) or bool(((y != 0) & (y != 1)).any()):
            raise ValueError("Multilabel task targets must be finite 0/1 arrays.")
        splits[name] = TensorDataset(x, torch.nan_to_num(c), y, mask)
        raw_splits[name] = data
    if len({ds.tensors[0].shape[1] for ds in splits.values()}) != 1:
        raise ValueError("Feature dimensions differ across splits.")
    if all("ids" in d for d in raw_splits.values()):
        ids = [i for d in raw_splits.values() for i in d["ids"]]
        if any(len(d["ids"]) != len(d["x"]) for d in raw_splits.values()) or len(set(ids)) != len(ids):
            raise ValueError("Example IDs are duplicated or have incorrect lengths.")
    return schema, splits, raw_splits


@torch.no_grad()
def evaluate(model, loader, device, desc: str = "Evaluating"):
    model.eval()
    task_sum, n, correct, tp, fp, fn = 0., 0, 0, 0, 0, 0
    concept_sums = [0.] * len(model.concepts)
    counts = [0] * len(model.concepts)
    for x, c, y, mask in tqdm(loader, desc=desc, leave=False):
        x, c, y, mask = [v.to(device) for v in (x, c, y, mask)]
        logits, trace = model(x, return_trace=True)
        task_sum += float(task_loss(logits, y, model.schema, model.config.get("task_loss", "cross_entropy"))) * len(x)
        n += len(x)
        if model.schema["task_type"] == "multiclass":
            correct += int((logits.argmax(-1) == y).sum())
        else:
            pred, target = logits >= 0, y.bool()
            correct += int((pred == target).all(-1).sum())
            tp += int((pred & target).sum()); fp += int((pred & ~target).sum()); fn += int((~pred & target).sum())
        for j, spec in enumerate(model.concepts):
            observed = mask[:, j]
            if bool(observed.any()):
                one_mask = torch.zeros_like(mask); one_mask[:, j] = observed
                count = int(observed.sum())
                concept_sums[j] += float(concept_loss(trace["concept_raw"], c, one_mask, model)) * count
                counts[j] += count
    measured = [s / count for s, count in zip(concept_sums, counts) if count]
    result = {"task_loss": task_sum / n, "concept_loss": sum(measured) / len(measured) if measured else None,
              "accuracy" if model.schema["task_type"] == "multiclass" else "exact_match": correct / n}
    if model.schema["task_type"] == "multilabel":
        result["micro_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return result


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ConceptModel(checkpoint["config"], checkpoint["schema"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lambda_concept < 0:
        raise ValueError("Invalid training arguments.")
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(args.seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    schema, splits, _ = load_data(args.data)
    task_loss_name = getattr(args, "task_loss", "cross_entropy")
    config = {"arch": args.arch, "feature_dim": splits["train"].tensors[0].shape[1],
              "embedding_dim": args.embedding_dim, "residual_dim": args.residual_dim,
              "backend": args.backend, "task_loss": task_loss_name}
    if args.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive.")
    model = ConceptModel(config, schema).to(args.device)
    x = splits["train"].tensors[0]
    with torch.no_grad():
        model.feature_mean.copy_(x.mean(0).to(args.device))
        model.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(1e-6).to(args.device))
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train",
                               generator=generator if name == "train" else None)
               for name, ds in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError("Use a fresh output folder to avoid replacing an existing run.")
    writer = SummaryWriter(log_dir=getattr(args, "tensorboard_dir", None) or str(out / "tensorboard"))
    writer.add_text("run/config", json.dumps({"training_args": vars(args), "model_config": config,
                                                "schema": schema}, indent=2))
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_examples = 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch:03d}/{args.epochs:03d}", leave=False)
        for x, c, y, mask in pbar:
            x, c, y, mask = [v.to(args.device) for v in (x, c, y, mask)]
            logits, trace = model(x, return_trace=True)
            loss = task_loss(logits, y, schema, task_loss_name) + args.lambda_concept * concept_loss(trace["concept_raw"], c, mask, model)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(x)
            train_examples += len(x)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        val = evaluate(model, loaders["val"], args.device, desc=f"Val Epoch {epoch:03d}")
        row = {"epoch": epoch, "train_loss": train_loss_sum / train_examples, "val": val}; history.append(row)
        writer.add_scalar("loss/train", row["train_loss"], epoch)
        for name, value in val.items():
            if value is not None:
                writer.add_scalar(f"validation/{name}", value, epoch)
        print(json.dumps(row), flush=True)
        if val["task_loss"] < best:
            best, stale = val["task_loss"], 0
            checkpoint = {"format_version": 1, "config": config, "schema": schema,
                          "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                          "val_metrics": val, "seed": args.seed,
                          "training_args": vars(args).copy(), "torch_version": str(torch.__version__)}
            torch.save(checkpoint, out / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    model, checkpoint = load_checkpoint(out / "best.pt", args.device)
    test = evaluate(model, loaders["test"], args.device, desc="Test Eval")
    report = {"best_epoch": checkpoint["epoch"], "test": test, "history": history}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    for name, value in test.items():
        if value is not None:
            writer.add_scalar(f"test/{name}", value, checkpoint["epoch"])
    writer.flush(); writer.close()
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test": test}), flush=True)


def cpu_tree(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, list):
        return [cpu_tree(x) for x in v]
    if isinstance(v, dict):
        return {k: cpu_tree(x) for k, x in v.items()}
    return v


@torch.no_grad()
def export(args):
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    schema, splits, originals = load_data(args.data)
    if schema != checkpoint["schema"]:
        raise ValueError("Data schema differs from checkpoint, including encoder metadata.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Trace output folder must be empty.")
    out.mkdir(parents=True, exist_ok=True)
    edits = json.loads(Path(args.edits).read_text()) if args.edits else None
    loader = DataLoader(splits[args.split], batch_size=args.batch_size, shuffle=False)
    offset = 0
    for batch_index, (x, c, y, mask) in enumerate(tqdm(loader, desc=f"Exporting {args.split}", leave=False)):
        _, trace = model(x.to(args.device), edits=edits, return_trace=True)
        ids = originals[args.split].get("ids", list(range(len(splits[args.split]))))
        torch.save({"trace": cpu_tree(trace), "ids": ids[offset:offset + len(x)],
                    "c": c, "y": y, "c_mask": mask, "edits": edits}, out / f"batch_{batch_index:05d}.pt")
        offset += len(x)
    (out / "metadata.json").write_text(json.dumps({"schema": schema, "config": checkpoint["config"],
        "seed": checkpoint["seed"], "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split, "edits": edits, "n_examples": offset}, indent=2))


def prepare_real_dataset(dataset_name: str, out_folder: Path, encoder: str = DEFAULT_ENCODER,
                         embed_batch_size: int = 64):
    """Build real dataset tensor folder (schema.json and train/val/test.pt) from concept_datasets."""
    from concept_datasets import download as download_ds
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    try:
        meta = load_meta(dataset_name)
    except FileNotFoundError:
        print(f"Dataset '{dataset_name}' not found locally. Downloading...")
        download_ds(dataset_name)
        meta = load_meta(dataset_name)

    concepts_spec = [{"name": cname, "type": "binary"} for cname in meta["concept_names"]]
    schema = {
        "dataset": dataset_name,
        "concepts": concepts_spec,
        "task_type": "multiclass",
        "n_tasks": len(meta["class_names"]),
        "encoder_metadata": {"model": encoder, "pooling": "mean"}
    }
    (out_folder / "schema.json").write_text(json.dumps(schema, indent=2))

    for split in tqdm(("train", "val", "test"), desc=f"Preparing {dataset_name}", leave=False):
        feats = get_features(dataset_name, split, encoder, batch_size=embed_batch_size)
        x = feats["X"]
        y = feats["y"]
        c = feats["C"] if "C" in feats else torch.zeros(len(x), len(meta["concept_names"]))
        c_mask = torch.ones_like(c, dtype=torch.bool)
        torch.save({"x": x, "c": c, "y": y, "c_mask": c_mask}, out_folder / f"{split}.pt")

    print(f"Successfully prepared real dataset '{dataset_name}' in {out_folder}")


def self_test():
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as folder:
        # Build synthetic test schema & tensors directly for self-test
        schema = {"concepts": [{"name": "binary", "type": "binary"},
                                {"name": "category", "type": "categorical", "cardinality": 3}],
                  "task_type": "multiclass", "n_tasks": 2,
                  "encoder_metadata": {"model": "synthetic", "pooling": "none"}}
        (Path(folder) / "schema.json").write_text(json.dumps(schema, indent=2))
        g = torch.Generator().manual_seed(13)
        for split, n in (("train", 512), ("val", 128), ("test", 128)):
            x = torch.randn(n, 8, generator=g)
            binary = (x[:, 0] > 0).float()
            category = x[:, 1:4].argmax(-1)
            c = torch.stack([binary, category.float()], -1)
            y = ((binary.bool()) ^ (category == 1)).long()
            torch.save({"x": x, "c": c, "y": y, "ids": [f"{split}_{i}" for i in range(n)]}, Path(folder) / f"{split}.pt")

        schema, splits, _ = load_data(folder)
        x, c, y, mask = [t[:8] for t in splits["train"].tensors]
        for arch in ("cbm", "residual-cbm", "cem"):
            config = {"arch": arch, "feature_dim": 8, "embedding_dim": 4,
                      "residual_dim": 3, "backend": "torch"}
            model = ConceptModel(config, schema)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            old = [p.detach().clone() for p in model.parameters()]
            logits, trace = model(x, return_trace=True)
            loss = task_loss(logits, y, schema) + concept_loss(trace["concept_raw"], c, mask, model)
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            assert any(not torch.equal(a, b) for a, b in zip(old, model.parameters()))
            model.eval()
            _, base = model(x, return_trace=True)
            _, edited = model(x, edits={"binary": 1., "category": [0., 1., 0.]}, return_trace=True)
            assert torch.equal(base["concept_values"], edited["concept_values"])
            assert torch.equal(edited["concept_edited"], torch.tensor([1., 0., 1., 0.]).expand(8, -1))
            if arch == "cem":
                assert base["mixed_embeddings"].shape == (8, 2, 4)
                assert torch.equal(edited["mixed_embeddings"][:, 0], base["state_embeddings"][0][:, 1])
            if arch == "residual-cbm":
                assert torch.equal(base["residual"], edited["residual"])
            path = Path(folder) / "checkpoint.pt"
            torch.save({"config": config, "schema": schema, "model_state": model.state_dict()}, path)
            reloaded, _ = load_checkpoint(path)
            assert torch.equal(model(x), reloaded(x))
            assert torch.isfinite(concept_loss(base["concept_raw"], c, torch.zeros_like(mask), model))
            print(f"PASS {arch}: update, edits, trace shapes, missing labels, checkpoint round trip")
        run = Path(folder) / "run"
        args = argparse.Namespace(data=folder, arch="cem", out=str(run), backend="torch",
            embedding_dim=4, residual_dim=3, epochs=2, batch_size=128, lr=1e-3,
            weight_decay=1e-4, lambda_concept=1., patience=2, seed=0, device="cpu", command="train")
        train(args)
        export(argparse.Namespace(checkpoint=str(run / "best.pt"), data=folder,
            out=str(Path(folder) / "traces"), split="test", batch_size=32, device="cpu", edits=None))
        assert len(list((Path(folder) / "traces").glob("batch_*.pt"))) == 4
        print("PASS trainer, validation selection, test evaluation, trace export")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", choices=DATASETS, required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--encoder", default=DEFAULT_ENCODER)
    prep.add_argument("--embed-batch-size", type=int, default=64)
    t = commands.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--arch", choices=["cbm", "residual-cbm", "cem"], required=True)
    t.add_argument("--backend", choices=["torch", "torch-concepts"], default="torch")
    t.add_argument("--embedding-dim", type=int, default=16); t.add_argument("--residual-dim", type=int, default=16)
    t.add_argument("--epochs", type=int, default=100); t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--lambda-concept", type=float, default=1.); t.add_argument("--patience", type=int, default=10)
    t.add_argument("--task-loss", choices=["cross_entropy", "mse"], default="cross_entropy",
                   help="multiclass task objective; MSE compares softmax probabilities to one-hot labels")
    default_dev = get_default_device()
    t.add_argument("--seed", type=int, default=0); t.add_argument("--device", default=default_dev)
    t.add_argument("--tensorboard-dir", help="Event-file directory (default: <out>/tensorboard)")
    e = commands.add_parser("export")
    e.add_argument("--data", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True); e.add_argument("--split", choices=["train", "val", "test"], default="test")
    e.add_argument("--batch-size", type=int, default=128); e.add_argument("--device", default=default_dev)
    e.add_argument("--edits", help="JSON name->scalar/probability-vector for uniform edits")
    args = p.parse_args()
    if args.command == "prepare":
        prepare_real_dataset(args.dataset, Path(args.out), args.encoder, args.embed_batch_size)
    elif args.command == "train": train(args)
    elif args.command == "export": export(args)
    elif args.command == "self-test": self_test()


if __name__ == "__main__":
    main()

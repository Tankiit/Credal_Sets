"""Joint training (task CE + concept_weight * concept BCE) and evaluation."""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .models import ConceptModel


def loss_fn(out, y, c_target, concept_weight):
    task = F.cross_entropy(out["logits"], y)
    concept = F.binary_cross_entropy_with_logits(out["concept_logits"], c_target)
    return task + concept_weight * concept, task, concept


def fit(model: ConceptModel, train: dict, val: dict, concept_weight: float = 1.0,
        epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256,
        patience: int = 8, soft_concepts: bool = False, device: str = "cuda", seed: int = 0):
    """Train with early stopping on the validation joint loss; returns the history."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    target = "C_soft" if soft_concepts else "C"
    X, y, C = (train[k].to(device) for k in ("X", "y", target))
    Xv, yv, Cv = (val[k].to(device) for k in ("X", "y", target))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(X), device=device).split(batch_size):
            loss, _, _ = loss_fn(model(X[idx], C[idx]), y[idx], C[idx], concept_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss, vtask, vconcept = (v.item() for v in loss_fn(model(Xv), yv, Cv, concept_weight))
        history.append({"epoch": epoch, "val_loss": vloss, "val_task": vtask, "val_concept": vconcept})
        print(f"  epoch {epoch:3d}  val loss {vloss:.4f} (task {vtask:.4f}, concept {vconcept:.4f})")
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if hasattr(model, "calibrate_interventions"):
        model.calibrate_interventions(X, train["C"].to(device))
    return history


@torch.no_grad()
def predict(model: ConceptModel, X, C=None, intervene=None, batch_size: int = 4096) -> dict:
    """Run the model in batches; returns numpy arrays for every output key."""
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch_size):
        sl = slice(i, i + batch_size)
        outs.append(model(X[sl].to(device),
                          None if C is None else C[sl].to(device),
                          None if intervene is None else intervene[sl].to(device)))
    return {k: torch.cat([o[k] for o in outs]).cpu().numpy() for k in outs[0]}


def evaluate(model: ConceptModel, feats: dict) -> dict:
    out = predict(model, feats["X"])
    y = feats["y"].numpy()
    pred = out["logits"].argmax(1)
    metrics = {"task_acc": float((pred == y).mean()),
               "task_macro_f1": float(f1_score(y, pred, average="macro"))}
    if "C" in feats:
        C = feats["C"].numpy()
        probs = 1 / (1 + np.exp(-out["concept_logits"]))
        aucs = [roc_auc_score(C[:, j], probs[:, j]) for j in range(C.shape[1])
                if 0 < C[:, j].sum() < len(C)]
        metrics["concept_acc"] = float(((probs > 0.5) == C).mean())
        metrics["concept_mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        # Task accuracy when every concept is replaced by its true value
        full = predict(model, feats["X"], feats["C"], torch.ones_like(feats["C"], dtype=torch.bool))
        metrics["task_acc_full_intervention"] = float((full["logits"].argmax(1) == y).mean())
    return metrics

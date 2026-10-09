"""CEBaB B1 evaluation for the probability-space residual CBM used on Modal.

The residual gauge twin is implemented as an evaluation wrapper, not by changing
the trained checkpoint: for an unedited input it is exactly function preserving.
For an edit it retains the residual's dependence on the *pre-edit* concept
values, yielding the intervention discrepancy predicted by the residual gauge.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from cebab_edits import expected_rating, load_edit_pairs
from train_concept_models import load_checkpoint


def _task_delta(model, head_input: torch.Tensor, residual_delta: torch.Tensor) -> torch.Tensor:
    """Apply the linear task head to a residual-only input, removing its bias."""
    delta = torch.zeros_like(head_input)
    delta[:, -residual_delta.shape[1]:] = residual_delta
    zeros = torch.zeros_like(head_input)
    return model.task_head(delta) - model.task_head(zeros)


@torch.no_grad()
def gauge_logits(model, x: torch.Tensor, edits: dict[str, torch.Tensor] | None, matrix: torch.Tensor):
    """Exact unedited functional twin and its intervened logits.

    If c is the original predicted concept vector, c' its intervention, r the
    residual, and H=[H_c,H_r], this evaluates
    H_c c' + H_r r + H_r M(c-c').  At c'=c it equals the original logits.
    """
    logits, trace = model(x, edits=edits, return_trace=True)
    correction = _task_delta(model, trace["head_input"],
                             (trace["concept_values"] - trace["concept_edited"]) @ matrix.T)
    return logits + correction


def _spearman(x, y):
    value = spearmanr(x, y).statistic
    return None if np.isnan(value) else float(value)


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, data_file: Path, *, n_tseeds: int = 20,
                        mix: float = 1.0, device: str = "cuda") -> tuple[list[dict], list[dict]]:
    """Return per-twin summaries and per-edit rows for one residual-CBM checkpoint."""
    model, ckpt = load_checkpoint(checkpoint, device)
    if model.arch != "residual-cbm":
        raise ValueError(f"B1 residual gauge requires residual-cbm, found {model.arch}")
    data = torch.load(data_file, map_location="cpu", weights_only=True)
    pairs = load_edit_pairs("test")
    if len(data["x"]) != 1689 or max(p.edit_index for p in pairs) >= len(data["x"]):
        raise ValueError("Prepared CEBaB test data are not aligned with the verified raw split")

    x = data["x"].to(device)
    c = data["c"].to(device)
    original_indices = torch.tensor([p.original_index for p in pairs], device=device)
    edit_indices = torch.tensor([p.edit_index for p in pairs], device=device)
    xp, cp = x[original_indices], c[edit_indices]
    names = [spec["name"] for spec in model.concepts]
    by_aspect = {aspect: [i for i, p in enumerate(pairs) if p.aspect == aspect]
                 for aspect in ("food", "service", "ambiance", "noise")}
    baseline = model(xp).detach().cpu().numpy()

    # Original model's targeted effects are identical across twin seeds.
    edited_base = np.empty_like(baseline)
    for aspect, indices in by_aspect.items():
        idx = torch.tensor(indices, device=device)
        edits = {f"{aspect}_{suffix}": cp[idx, names.index(f"{aspect}_{suffix}")]
                 for suffix in ("pos", "neg")}
        edited_base[indices] = model(xp[idx], edits=edits).detach().cpu().numpy()
    model_effect = expected_rating(edited_base) - expected_rating(baseline)
    human = np.asarray([p.human_effect for p in pairs])

    summaries, rows = [], []
    residual_dim, k = model.config["residual_dim"], len(names)
    for twin_seed in range(n_tseeds):
        gen = torch.Generator(device="cpu").manual_seed(twin_seed)
        matrix = (mix * torch.randn(residual_dim, k, generator=gen)
                  / max(k, 1) ** 0.5).to(device)
        twin_base = gauge_logits(model, xp, None, matrix).cpu().numpy()
        if not np.allclose(twin_base, baseline, atol=1e-5, rtol=1e-5):
            raise AssertionError("Gauge twin changed unedited logits")
        edited_twin = np.empty_like(baseline)
        for aspect, indices in by_aspect.items():
            idx = torch.tensor(indices, device=device)
            edits = {f"{aspect}_{suffix}": cp[idx, names.index(f"{aspect}_{suffix}")]
                     for suffix in ("pos", "neg")}
            edited_twin[indices] = gauge_logits(model, xp[idx], edits, matrix).cpu().numpy()
        twin_effect = expected_rating(edited_twin) - expected_rating(twin_base)
        model_changed = baseline.argmax(1) != edited_base.argmax(1)
        twin_changed = twin_base.argmax(1) != edited_twin.argmax(1)
        disagreements = model_changed != twin_changed
        gap = twin_effect - model_effect
        summaries.append({
            "train_seed": ckpt["seed"], "twin_seed": twin_seed, "n_pairs": len(pairs), "mix": mix,
            "model_human_spearman": _spearman(model_effect, human),
            "twin_human_spearman": _spearman(twin_effect, human),
            "mean_abs_twin_model_effect_gap": float(np.abs(gap).mean()),
            "max_abs_twin_model_effect_gap": float(np.abs(gap).max()),
            "label_change_disagreements": int(disagreements.sum()),
        })
        rows.extend({"train_seed": ckpt["seed"], "twin_seed": twin_seed,
                     "original_id": pair.original_id, "edit_id": pair.edit_id,
                     "aspect": pair.aspect, "goal": pair.goal, "human_effect": pair.human_effect,
                     "model_expected_effect": float(model_effect[i]),
                     "twin_expected_effect": float(twin_effect[i]),
                     "twin_minus_model_expected_effect": float(gap[i]),
                     "twins_disagree_on_label_change": bool(disagreements[i])}
                    for i, pair in enumerate(pairs))
    return summaries, rows


def save_jsonl(rows: list[dict], path: Path):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

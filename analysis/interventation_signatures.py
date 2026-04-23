"""
Intervention signature measurements on checkpoints.

Complements snapshot_grad_iso.py. Gradients tell us how the heads update;
intervention signatures tell us whether the heads track what they should.

Two measurements per checkpoint:
  ρ(σ_epi, concept_error)     — epistemic tracks model ignorance
  ρ(σ_ale, annotator_entropy) — aleatoric tracks data ambiguity

These are Spearman correlations on the val split. The paper reports them
as the validity check on the uncertainty decomposition.

Usage:
    python -m analysis.intervention_signatures \
        --checkpoint checkpoints/hybrid_credal_cebab/best_model.pt \
        --dataset cebab \
        --output outputs/interventions/cebab.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats

from analysis.snapshot_grad_iso import load_checkpoint
from loaders import LOADERS


@dataclass
class InterventionResult:
    dataset: str
    checkpoint: str
    num_samples: int

    # Per-concept and aggregate correlations
    rho_epi_error_agg: float
    p_epi_error_agg: float
    rho_ale_entropy_agg: float
    p_ale_entropy_agg: float
    rho_epi_ale_agg: float       # cross-uncertainty correlation; the paper's headline metric
    p_epi_ale_agg: float

    rho_epi_error_per_concept: dict  # {concept_name: rho}
    rho_ale_entropy_per_concept: dict

    # Distributional summaries for the paper's table
    mean_sigma_epi: float
    mean_sigma_ale: float
    std_sigma_epi: float
    std_sigma_ale: float


def evaluate_checkpoint(
    model,
    bundle,
    device: torch.device,
) -> InterventionResult:
    """Single-pass evaluation producing all intervention signatures."""
    model.eval()

    all_sigma_epi = []
    all_sigma_ale = []
    all_concept_preds = []       # mu → concept predictions
    all_concept_labels = []
    all_annotator_entropy = []
    has_entropy = False
    has_concepts = False

    with torch.no_grad():
        for batch in bundle.val_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "concept_labels" in batch and batch["concept_labels"] is not None:
                inputs["concept_labels"] = batch["concept_labels"].to(device)
            if "annotator_entropy" in batch and batch["annotator_entropy"] is not None:
                inputs["annotator_entropy"] = batch["annotator_entropy"].to(device)

            outputs = model(**inputs)

            # Model outputs epistemic/aleatoric; rename for consistency
            all_sigma_epi.append(outputs.get("epistemic", outputs.get("sigma_epi")).cpu().numpy())
            all_sigma_ale.append(outputs.get("aleatoric", outputs.get("sigma_ale")).cpu().numpy())

            # Concept predictions from μ — used to compute per-concept errors.
            # NOTE: the paper convention is concept_error = 1 if argmax(μ) differs
            # from concept_label, else 0. Adjust if your forward returns predictions
            # under a different key (concept_preds vs concept_probs, etc.).
            if "concept_probs" in outputs:
                preds = outputs["concept_probs"].argmax(dim=-1).cpu().numpy()
                all_concept_preds.append(preds)
            elif "mu" in outputs:
                # Fall back: use μ as logits, argmax per concept
                preds = outputs["mu"].argmax(dim=-1).cpu().numpy() \
                    if outputs["mu"].dim() == 3 else outputs["mu"].cpu().numpy()
                all_concept_preds.append(preds)

            if "concept_labels" in inputs:
                all_concept_labels.append(inputs["concept_labels"].cpu().numpy())
                has_concepts = True
            if "annotator_entropy" in inputs:
                all_annotator_entropy.append(inputs["annotator_entropy"].cpu().numpy())
                has_entropy = True

    sigma_epi = np.concatenate(all_sigma_epi, axis=0)   # [N, K]
    sigma_ale = np.concatenate(all_sigma_ale, axis=0)   # [N, K]

    # Aggregate uncertainties per sample
    epi_agg = sigma_epi.mean(axis=-1)  # [N]
    ale_agg = sigma_ale.mean(axis=-1)  # [N]

    # Concept errors (if concept labels available)
    rho_epi_error_agg = float("nan")
    p_epi_error_agg = float("nan")
    rho_epi_error_per_concept = {}
    if has_concepts and all_concept_preds:
        concept_preds = np.concatenate(all_concept_preds, axis=0)
        concept_labels = np.concatenate(all_concept_labels, axis=0)
        # Per-sample mean error rate
        errors = (concept_preds != concept_labels).astype(np.float32)  # [N, K]
        error_agg = errors.mean(axis=-1)
        if error_agg.std() > 0 and epi_agg.std() > 0:
            rho_epi_error_agg, p_epi_error_agg = stats.spearmanr(epi_agg, error_agg)
            rho_epi_error_agg = float(rho_epi_error_agg)
            p_epi_error_agg = float(p_epi_error_agg)

        # Per-concept
        for k, name in enumerate(bundle.concept_names):
            if errors[:, k].std() > 0 and sigma_epi[:, k].std() > 0:
                r, _ = stats.spearmanr(sigma_epi[:, k], errors[:, k])
                rho_epi_error_per_concept[name] = float(r)

    # Annotator entropy (if available)
    rho_ale_entropy_agg = float("nan")
    p_ale_entropy_agg = float("nan")
    rho_ale_entropy_per_concept = {}
    if has_entropy:
        annotator_entropy = np.concatenate(all_annotator_entropy, axis=0)  # [N, K]
        entropy_agg = annotator_entropy.mean(axis=-1)
        if entropy_agg.std() > 0 and ale_agg.std() > 0:
            rho_ale_entropy_agg, p_ale_entropy_agg = stats.spearmanr(ale_agg, entropy_agg)
            rho_ale_entropy_agg = float(rho_ale_entropy_agg)
            p_ale_entropy_agg = float(p_ale_entropy_agg)

        for k, name in enumerate(bundle.concept_names):
            if annotator_entropy[:, k].std() > 0 and sigma_ale[:, k].std() > 0:
                r, _ = stats.spearmanr(sigma_ale[:, k], annotator_entropy[:, k])
                rho_ale_entropy_per_concept[name] = float(r)

    # Cross-uncertainty correlation — the paper's ρ(EU, AU)
    rho_epi_ale_agg = float("nan")
    p_epi_ale_agg = float("nan")
    if epi_agg.std() > 0 and ale_agg.std() > 0:
        rho_epi_ale_agg, p_epi_ale_agg = stats.spearmanr(epi_agg, ale_agg)
        rho_epi_ale_agg = float(rho_epi_ale_agg)
        p_epi_ale_agg = float(p_epi_ale_agg)

    return InterventionResult(
        dataset=bundle.name,
        checkpoint="",
        num_samples=int(len(epi_agg)),
        rho_epi_error_agg=rho_epi_error_agg,
        p_epi_error_agg=p_epi_error_agg,
        rho_ale_entropy_agg=rho_ale_entropy_agg,
        p_ale_entropy_agg=p_ale_entropy_agg,
        rho_epi_ale_agg=rho_epi_ale_agg,
        p_epi_ale_agg=p_epi_ale_agg,
        rho_epi_error_per_concept=rho_epi_error_per_concept,
        rho_ale_entropy_per_concept=rho_ale_entropy_per_concept,
        mean_sigma_epi=float(sigma_epi.mean()),
        mean_sigma_ale=float(sigma_ale.mean()),
        std_sigma_epi=float(sigma_epi.std()),
        std_sigma_ale=float(sigma_ale.std()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Intervention signature measurement.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True,
                   choices=["cebab", "hatexplain", "goemotions", "sst2"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    bundle = LOADERS[args.dataset].load(
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    model = load_checkpoint(args.checkpoint, bundle, device)
    result = evaluate_checkpoint(model, bundle, device)
    result.checkpoint = str(args.checkpoint)

    print(f"\n{'=' * 60}")
    print(f"INTERVENTION SIGNATURES — {bundle.name}")
    print(f"{'=' * 60}")
    print(f"  ρ(EU, AU)            = {result.rho_epi_ale_agg:+.3f}  (target: < 0.30)")
    print(f"  ρ(σ_epi, error)      = {result.rho_epi_error_agg:+.3f}  (target: > 0.20, positive)")
    print(f"  ρ(σ_ale, entropy)    = {result.rho_ale_entropy_agg:+.3f}  (target: > 0.30, positive)")
    if result.rho_epi_error_per_concept:
        print(f"  per-concept ρ(σ_epi, error):")
        for c, r in result.rho_epi_error_per_concept.items():
            print(f"    {c:12s}: {r:+.3f}")
    if result.rho_ale_entropy_per_concept:
        print(f"  per-concept ρ(σ_ale, entropy):")
        for c, r in result.rho_ale_entropy_per_concept.items():
            print(f"    {c:12s}: {r:+.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(asdict(result), f, indent=2, default=float)
    print(f"\n[wrote] {args.output}")


if __name__ == "__main__":
    main()
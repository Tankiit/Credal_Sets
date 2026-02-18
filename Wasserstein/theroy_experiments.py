"""
Theoretical Validation Experiments for Credal DRO (UAI 2026)
=============================================================

Three experiments that validate the theory from the paper:

Experiment 1 — DRO Equivalence:
  Shows that PGD worst-case loss ≈ first-order dual approximation.
  Validates: credal ellipsoid → Mahalanobis norm regularization.
  
Experiment 2 — Width-Margin Equilibrium:
  Tracks ε·margin product over training, shows convergence to β/λ_dro.
  Validates: Theorem (dual calibration equilibrium).

Experiment 3 — Margin Coupling:
  Scatter plot: concept margin vs learned width at convergence.
  Shows inverse relationship for Mode C only (not A or B).
  Validates: why joint training matters.

All experiments use the same model interface as eval_credal_dro.py:
  model(features, labels, concept_labels) -> {mu, sigma_sq, epsilon, logits}

Usage:
  # After training, run all experiments on a trained model:
  python credal_theory_experiments.py \\
      --model_dir results/cebab_joint/ \\
      --cache_dir latent_cache/roberta_cebab \\
      --dataset cebab \\
      --output_dir results/theory_validation/

  # For 3-panel margin coupling (needs all 3 modes trained):
  python credal_theory_experiments.py \\
      --three_way_dir results/three_way_comparison/ \\
      --cache_dir latent_cache/roberta_cebab \\
      --dataset cebab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from dataclasses import dataclass, field
import json
import math


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def extract_classifier_weights(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract task classifier W, b from model.
    Tries common paths: model.W, model.classifier.weight, model.label_head.weight
    """
    for w_path in ["W", "classifier.weight", "label_head.weight",
                    "task_head.weight", "label_head.linear.weight"]:
        try:
            obj = model
            for p in w_path.split("."):
                obj = getattr(obj, p)
            W = obj.detach()
            break
        except AttributeError:
            continue
    else:
        raise ValueError("Cannot find classifier weight W in model")

    for b_path in ["b", "classifier.bias", "label_head.bias",
                    "task_head.bias", "label_head.linear.bias"]:
        try:
            obj = model
            for p in b_path.split("."):
                obj = getattr(obj, p)
            b = obj.detach()
            break
        except AttributeError:
            continue
    else:
        raise ValueError("Cannot find classifier bias b in model")

    return W, b


def compute_task_margin(
    mu: torch.Tensor,   # [N, K] concept means
    W: torch.Tensor,    # [C, K] classifier weight
    b: torch.Tensor,    # [C] classifier bias
    y: torch.Tensor,    # [N] true labels
) -> torch.Tensor:
    """
    Compute the task classifier margin in concept space.
    
    margin(x) = f_{y_true}(μ) - max_{y' ≠ y_true} f_{y'}(μ)
    
    where f_y(p) = W_y · p + b_y
    
    Positive margin → correct prediction.
    Large margin → confident and robust.
    The equilibrium theorem predicts ε(x) ∝ 1/margin(x).
    
    Returns: [N] margin values (can be negative)
    """
    # All class scores: [N, C]
    logits = F.linear(mu, W, b)
    N, C = logits.shape
    
    # Score of the true class: [N]
    true_scores = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    
    # Max score of any OTHER class: [N]
    # Mask out the true class with -inf
    mask = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), float("-inf"))
    other_scores = (logits + mask).max(dim=1).values
    
    margin = true_scores - other_scores
    return margin


# ============================================================================
# EXPERIMENT 1: DRO EQUIVALENCE
# ============================================================================
# Validates that the PGD inner maximization over the credal ellipsoid
# produces the same result as the first-order dual norm approximation.
#
# Theory: For linear task head f(p) = Wp + b with ellipsoidal credal set
# C(x) = {p : (p-μ)^T Σ_epi^{-1} (p-μ) ≤ 1}, the worst-case loss is:
#
#   sup_{p ∈ C(x)} ℓ(p;y) ≈ ℓ(μ;y) + ‖∇_p ℓ(μ;y)‖_{Σ_epi}}
#
# where ‖v‖_Σ = √(v^T Σ v) is the Mahalanobis norm.
# This is exact when ℓ is linear in p (before softmax in CE loss).


@torch.no_grad()
def experiment_1_dro_equivalence(
    model,
    data_loader,
    device: str = "cuda",
    pgd_steps: int = 50,
    pgd_lr: float = 0.01,
) -> Dict:
    """
    Compare three quantities for each test example:
      1. Exact worst-case loss via PGD over the credal ellipsoid
      2. First-order dual approximation: ℓ(μ) + ‖∇ℓ‖_Σ
      3. Nominal loss: ℓ(μ)
    
    The scatter plot of (1) vs (2) should cluster near the diagonal,
    validating that the ellipsoidal credal set → Mahalanobis regularization.
    """
    model.eval()
    W, b = extract_classifier_weights(model)
    W, b = W.to(device), b.to(device)
    
    all_pgd_loss = []       # exact worst-case from PGD
    all_dual_loss = []      # first-order approximation
    all_nominal_loss = []   # loss at μ (no perturbation)
    all_epsilon = []        # credal width ε(x)
    all_margin = []         # task margin m(x)
    all_gap = []            # |PGD - dual| (approximation gap)
    
    for batch in data_loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        concept_labels = batch.get("concept_labels")
        if concept_labels is not None:
            concept_labels = concept_labels.to(device)
        
        output = model(features, labels, concept_labels)
        mu = output["mu"]             # [B, K]
        sigma_sq = output["sigma_sq"] # [B, K]
        
        B, K = mu.shape
        
        # ─── (1) Nominal loss ℓ(μ; y) ────────────────────────────
        logits_nom = F.linear(mu, W, b)
        loss_nom = F.cross_entropy(logits_nom, labels, reduction="none")  # [B]
        
        # ─── (2) Dual approximation ──────────────────────────────
        # ℓ(μ;y) + ‖∇_p ℓ‖_{Σ_epi}
        #
        # For CE loss with linear head: ∇_p ℓ(μ;y) = W^T (softmax(Wμ+b) - e_y)
        # This is the gradient of CE w.r.t. concept predictions p, evaluated at μ.
        #
        # The Mahalanobis norm: ‖g‖_Σ = √(g^T Σ_epi g) = √(Σ_k g_k² σ_k²)
        # since Σ_epi = diag(σ_1², ..., σ_K²)
        
        probs = F.softmax(logits_nom, dim=-1)                   # [B, C]
        one_hot = F.one_hot(labels, probs.shape[1]).float()     # [B, C]
        grad_logits = probs - one_hot                           # [B, C]
        
        # ∇_p ℓ = W^T · (softmax - one_hot)
        # W: [C, K], grad_logits: [B, C]
        grad_p = grad_logits @ W                                # [B, K]
        
        # Mahalanobis norm: ‖grad_p‖_Σ = sqrt(Σ_k (grad_p_k)² · σ_k²)
        mahal_sq = (grad_p ** 2 * sigma_sq).sum(dim=-1)         # [B]
        mahal_norm = torch.sqrt(mahal_sq.clamp(min=1e-12))      # [B]
        
        loss_dual = loss_nom + mahal_norm                       # [B]
        
        # ─── (3) Exact PGD worst-case ────────────────────────────
        # Maximize ℓ(p; y) subject to (p-μ)^T Σ^{-1} (p-μ) ≤ 1 and p ∈ [0,1]^K
        loss_pgd = _pgd_worst_case_loss(
            mu, sigma_sq, W, b, labels,
            steps=pgd_steps, lr=pgd_lr,
        )  # [B]
        
        # ─── Collect ─────────────────────────────────────────────
        eps = torch.sqrt(sigma_sq.sum(dim=-1))
        margin = compute_task_margin(mu, W, b, labels)
        gap = (loss_pgd - loss_dual).abs()
        
        all_pgd_loss.append(loss_pgd.cpu())
        all_dual_loss.append(loss_dual.cpu())
        all_nominal_loss.append(loss_nom.cpu())
        all_epsilon.append(eps.cpu())
        all_margin.append(margin.cpu())
        all_gap.append(gap.cpu())
    
    # Concatenate
    pgd_loss = torch.cat(all_pgd_loss).numpy()
    dual_loss = torch.cat(all_dual_loss).numpy()
    nom_loss = torch.cat(all_nominal_loss).numpy()
    epsilon = torch.cat(all_epsilon).numpy()
    margin = torch.cat(all_margin).numpy()
    gap = torch.cat(all_gap).numpy()
    
    # ─── Compute summary statistics ─────────────────────────────
    # How well does the dual approximation match PGD?
    rho_pgd_dual, _ = spearmanr(pgd_loss, dual_loss)
    
    # Mean absolute gap
    mean_gap = gap.mean()
    relative_gap = (gap / (pgd_loss + 1e-8)).mean()
    
    # R² of linear fit
    from numpy.polynomial.polynomial import polyfit
    coeffs = np.polyfit(dual_loss, pgd_loss, 1)
    predicted = np.polyval(coeffs, dual_loss)
    ss_res = ((pgd_loss - predicted) ** 2).sum()
    ss_tot = ((pgd_loss - pgd_loss.mean()) ** 2).sum()
    r_squared = 1 - ss_res / (ss_tot + 1e-8)
    
    # Does the gap correlate with ε? (Larger credal sets → more linearization error)
    rho_gap_eps, _ = spearmanr(gap, epsilon)
    
    results = {
        # Correlations
        "spearman_pgd_dual": float(rho_pgd_dual),
        "r_squared_pgd_dual": float(r_squared),
        "slope_pgd_dual": float(coeffs[0]),
        "intercept_pgd_dual": float(coeffs[1]),
        
        # Gap statistics
        "mean_gap": float(mean_gap),
        "relative_gap": float(relative_gap),
        "max_gap": float(gap.max()),
        "spearman_gap_epsilon": float(rho_gap_eps),
        
        # Loss statistics
        "nominal_loss_mean": float(nom_loss.mean()),
        "dual_loss_mean": float(dual_loss.mean()),
        "pgd_loss_mean": float(pgd_loss.mean()),
        "robustness_premium_mean": float((pgd_loss - nom_loss).mean()),
        
        "n_examples": len(pgd_loss),
        
        # Raw arrays for plotting
        "_arrays": {
            "pgd_loss": pgd_loss,
            "dual_loss": dual_loss,
            "nominal_loss": nom_loss,
            "epsilon": epsilon,
            "margin": margin,
            "gap": gap,
        },
    }
    
    # Report
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: DRO EQUIVALENCE")
    print("=" * 65)
    print(f"  N = {len(pgd_loss)} examples")
    print(f"")
    print(f"  ── Loss comparison ──")
    print(f"  Nominal ℓ(μ):       {nom_loss.mean():.4f} ± {nom_loss.std():.4f}")
    print(f"  Dual approx:        {dual_loss.mean():.4f} ± {dual_loss.std():.4f}")
    print(f"  PGD exact:          {pgd_loss.mean():.4f} ± {pgd_loss.std():.4f}")
    print(f"  Robustness premium: {(pgd_loss - nom_loss).mean():.4f}")
    print(f"")
    print(f"  ── PGD vs Dual match ──")
    print(f"  Spearman ρ:         {rho_pgd_dual:.4f}")
    print(f"  R²:                 {r_squared:.4f}")
    print(f"  Slope (should ≈ 1): {coeffs[0]:.4f}")
    print(f"  Mean |gap|:         {mean_gap:.6f}")
    print(f"  Relative gap:       {relative_gap:.4f}")
    print(f"  ρ(gap, ε):          {rho_gap_eps:.4f}")
    print("=" * 65)
    
    return results


def _pgd_worst_case_loss(
    mu: torch.Tensor,         # [B, K]
    sigma_sq: torch.Tensor,   # [B, K]
    W: torch.Tensor,          # [C, K]
    b: torch.Tensor,          # [C]
    labels: torch.Tensor,     # [B]
    steps: int = 50,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    PGD inner maximization: find p* = argmax_{p ∈ C(x)} ℓ(p; y)

    The credal ellipsoid constraint is:
      (p - μ)^T Σ_epi^{-1} (p - μ) ≤ 1

    which in our diagonal case (Σ_epi = diag(σ²)) simplifies to:
      Σ_k (p_k - μ_k)² / σ²_k ≤ 1

    We also enforce p ∈ [0, 1]^K.

    Returns: [B] worst-case loss per example
    """
    B, K = mu.shape
    orig_device = mu.device

    # Move to CPU to avoid MPS gradient issues
    mu = mu.detach().cpu().float()
    sigma_sq = sigma_sq.detach().cpu().float()
    W = W.detach().cpu().float()
    b = b.detach().cpu().float()
    labels = labels.detach().cpu()

    # Inverse variances for projection (avoid div by zero)
    inv_sigma_sq = 1.0 / sigma_sq.clamp(min=1e-8)  # [B, K]

    # Initialize p at mu (the center of the ellipsoid)
    # Use p.data for in-place updates to avoid gradient tracking issues
    p = mu.clone()

    for step in range(steps):
        # Compute logits: z = p @ W^T + b
        logits = torch.matmul(p, W.t()) + b  # [B, C]

        # Compute softmax and gradient of cross-entropy w.r.t. logits
        # For cross-entropy: ℓ = -log(softmax[z_y])
        # ∂ℓ/∂z_i = softmax(z_i) - 1{i=y}
        probs = F.softmax(logits, dim=-1)  # [B, C]
        grad_logits = probs.clone()
        grad_logits[torch.arange(B), labels] -= 1.0  # [B, C]

        # Gradient w.r.t. p: ∂ℓ/∂p = ∂ℓ/∂z @ W
        grad_p = torch.matmul(grad_logits, W)  # [B, K]

        # Gradient ascent step (maximize loss)
        p = p + lr * grad_p

        # Project back onto [0,1]^K and ellipsoid
        # Enforce p ∈ [0, 1]^K first
        p = p.clamp(0.0, 1.0)

        # Project onto ellipsoid: Σ_k (p_k - μ_k)² / σ_k² ≤ 1
        delta = p - mu
        mahal_sq = (delta ** 2 * inv_sigma_sq).sum(dim=-1, keepdim=True)  # [B, 1]
        # If outside ellipsoid, scale delta to lie on the boundary
        scale = torch.where(
            mahal_sq > 1.0,
            1.0 / torch.sqrt(mahal_sq.clamp(min=1e-8)),
            torch.ones_like(mahal_sq),
        )
        p = mu + delta * scale

    # Final worst-case loss
    with torch.no_grad():
        p_star = p.clamp(0.0, 1.0)
        logits_star = torch.matmul(p_star, W.t()) + b
        loss_star = F.cross_entropy(logits_star, labels, reduction="none")

    return loss_star.to(orig_device)


# ============================================================================
# EXPERIMENT 2: WIDTH-MARGIN EQUILIBRIUM
# ============================================================================
# Tracks ε(x) · margin(x) over training. The equilibrium theorem predicts
# this product converges to a constant ∝ β/λ_dro.
#
# This experiment requires logging during training (not just at test time).
# We provide (a) a callback to add to your training loop, and (b) a 
# post-hoc analysis that runs over saved checkpoints.


@dataclass
class EquilibriumSnapshot:
    """One snapshot of the width-margin relationship during training."""
    epoch: int
    phase: str
    lambda_dro: float
    beta: float
    
    # Per-example statistics (means over the validation set)
    eps_mean: float = 0.0
    eps_std: float = 0.0
    margin_mean: float = 0.0
    margin_std: float = 0.0
    product_mean: float = 0.0       # E[ε · margin] — should converge
    product_std: float = 0.0
    predicted_product: float = 0.0  # β/λ_dro — theoretical prediction
    
    # Per-concept breakdown
    per_concept_sigma_sq: List[float] = field(default_factory=list)
    per_concept_gradient_norm: List[float] = field(default_factory=list)
    per_concept_product: List[float] = field(default_factory=list)
    
    def summary(self) -> str:
        return (
            f"Ep {self.epoch:3d} [{self.phase}] "
            f"ε={self.eps_mean:.4f} margin={self.margin_mean:.4f} "
            f"ε·m={self.product_mean:.4f} "
            f"(theory: {self.predicted_product:.4f})"
        )


@torch.no_grad()
def compute_equilibrium_snapshot(
    model,
    data_loader,
    epoch: int,
    phase: str,
    lambda_dro: float,
    beta: float,
    device: str = "cuda",
) -> EquilibriumSnapshot:
    """
    Compute one snapshot of the width-margin equilibrium.
    
    Call this at the end of each epoch (or every N epochs) during training
    to track how the ε·margin product evolves.
    
    The key prediction: at convergence,
      E[ε(x) · margin(x)] → constant ∝ β/λ_dro
    
    And per-concept (Corollary: marginal cost equalization):
      σ_k² · ‖∂ℓ/∂p_k‖ → β/λ_dro  for all k
    """
    model.eval()
    W, b = extract_classifier_weights(model)
    W, b = W.to(device), b.to(device)
    
    all_eps = []
    all_margin = []
    all_products = []
    all_sigma_sq = []
    all_grad_norms = []
    
    for batch in data_loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        concept_labels = batch.get("concept_labels")
        if concept_labels is not None:
            concept_labels = concept_labels.to(device)
        
        output = model(features, labels, concept_labels)
        mu = output["mu"]
        sigma_sq = output["sigma_sq"]
        
        # ε(x)
        eps = torch.sqrt(sigma_sq.sum(dim=-1))  # [B]
        
        # margin(x) = f_y(μ) - max_{y'≠y} f_{y'}(μ)
        margin = compute_task_margin(mu, W, b, labels)  # [B]
        
        # ε · margin product
        product = eps * margin.abs()  # use |margin| since some may be negative
        
        # Per-concept gradient norms (for marginal cost equalization)
        # ∂ℓ/∂p_k evaluated at μ
        logits = F.linear(mu, W, b)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, probs.shape[1]).float()
        grad_logits = probs - one_hot                  # [B, C]
        grad_p = grad_logits @ W                       # [B, K]
        grad_norms_per_k = grad_p.abs()                # [B, K]
        
        all_eps.append(eps.cpu())
        all_margin.append(margin.cpu())
        all_products.append(product.cpu())
        all_sigma_sq.append(sigma_sq.cpu())
        all_grad_norms.append(grad_norms_per_k.cpu())
    
    eps = torch.cat(all_eps).numpy()
    margin = torch.cat(all_margin).numpy()
    products = torch.cat(all_products).numpy()
    sigma_sq = torch.cat(all_sigma_sq).numpy()      # [N, K]
    grad_norms = torch.cat(all_grad_norms).numpy()  # [N, K]
    
    K = sigma_sq.shape[1]
    
    # Per-concept σ_k² · |∂ℓ/∂p_k| — should equalize across k
    per_concept_products = []
    for k in range(K):
        pk = (sigma_sq[:, k] * grad_norms[:, k]).mean()
        per_concept_products.append(float(pk))
    
    # Theoretical prediction
    predicted = beta / lambda_dro if lambda_dro > 0 else float("inf")
    
    snap = EquilibriumSnapshot(
        epoch=epoch,
        phase=phase,
        lambda_dro=lambda_dro,
        beta=beta,
        eps_mean=float(eps.mean()),
        eps_std=float(eps.std()),
        margin_mean=float(margin.mean()),
        margin_std=float(margin.std()),
        product_mean=float(products.mean()),
        product_std=float(products.std()),
        predicted_product=predicted,
        per_concept_sigma_sq=[float(sigma_sq[:, k].mean()) for k in range(K)],
        per_concept_gradient_norm=[float(grad_norms[:, k].mean()) for k in range(K)],
        per_concept_product=per_concept_products,
    )
    
    return snap


def experiment_2_equilibrium_from_checkpoints(
    checkpoint_dir: str,
    data_loader,
    model_class,
    model_config,
    lambda_dro: float,
    beta: float,
    device: str = "cuda",
    every_n_epochs: int = 1,
) -> List[EquilibriumSnapshot]:
    """
    Post-hoc equilibrium analysis from saved checkpoints.
    
    Expects checkpoints named: epoch_{N}.pt or checkpoint_{N}.pt
    Each checkpoint should contain model state dict.
    
    Returns list of EquilibriumSnapshot for each checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    
    # Find checkpoint files
    ckpt_files = sorted(ckpt_dir.glob("*epoch*.pt")) + \
                 sorted(ckpt_dir.glob("*checkpoint*.pt"))
    
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return []
    
    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")
    
    snapshots = []
    for ckpt_path in ckpt_files:
        # Parse epoch from filename
        name = ckpt_path.stem
        epoch = int("".join(c for c in name if c.isdigit()) or "0")
        
        if epoch % every_n_epochs != 0:
            continue
        
        # Load model
        model = model_class(model_config).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        
        # Determine phase (rough heuristic from epoch number)
        # You should adapt this to match your PhasedSchedule
        phase = "unknown"
        
        snap = compute_equilibrium_snapshot(
            model, data_loader, epoch, phase,
            lambda_dro, beta, device,
        )
        snapshots.append(snap)
        print(f"  {snap.summary()}")
    
    return snapshots


@torch.no_grad()
def experiment_2_single_model(
    model,
    data_loader,
    lambda_dro: float,
    beta: float,
    device: str = "cuda",
) -> Dict:
    """
    If you don't have checkpoints, evaluate the equilibrium at the final model.
    
    Returns stats about the ε·margin product and per-concept marginal costs.
    """
    snap = compute_equilibrium_snapshot(
        model, data_loader, epoch=-1, phase="final",
        lambda_dro=lambda_dro, beta=beta, device=device,
    )
    
    K = len(snap.per_concept_product)
    
    results = {
        "eps_mean": snap.eps_mean,
        "margin_mean": snap.margin_mean,
        "product_mean": snap.product_mean,
        "product_std": snap.product_std,
        "predicted_product": snap.predicted_product,
        "product_ratio": (snap.product_mean / snap.predicted_product
                          if snap.predicted_product < float("inf") else None),
        "per_concept_sigma_sq": snap.per_concept_sigma_sq,
        "per_concept_gradient_norm": snap.per_concept_gradient_norm,
        "per_concept_product": snap.per_concept_product,
        # How equalized are the per-concept products?
        # CV (coefficient of variation) → 0 at perfect equalization
        "marginal_cost_cv": (float(np.std(snap.per_concept_product) /
                                   (np.mean(snap.per_concept_product) + 1e-8))),
    }
    
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: WIDTH-MARGIN EQUILIBRIUM")
    print("=" * 65)
    print(f"  ε mean:         {snap.eps_mean:.4f} ± {snap.eps_std:.4f}")
    print(f"  margin mean:    {snap.margin_mean:.4f} ± {snap.margin_std:.4f}")
    print(f"  ε·|m| product:  {snap.product_mean:.4f} ± {snap.product_std:.4f}")
    print(f"  Theory (β/λ):   {snap.predicted_product:.4f}")
    if results["product_ratio"] is not None:
        print(f"  Ratio emp/thy:  {results['product_ratio']:.3f}")
    print(f"")
    print(f"  ── Marginal cost equalization (Corollary) ──")
    print(f"  Per-concept σ²_k · |∂ℓ/∂p_k|:")
    for k in range(K):
        print(f"    k={k}: σ²={snap.per_concept_sigma_sq[k]:.5f} × "
              f"|∇|={snap.per_concept_gradient_norm[k]:.5f} = "
              f"{snap.per_concept_product[k]:.5f}")
    print(f"  CV of marginal costs: {results['marginal_cost_cv']:.4f} "
          f"(0 = perfect equalization)")
    print("=" * 65)
    
    return results


# ============================================================================
# EXPERIMENT 3: MARGIN-WIDTH COUPLING (3-panel across modes)
# ============================================================================
# The key visual result: scatter plot of concept margin vs learned width.
# Mode A (post-hoc): blob (no correlation)
# Mode B (fixed-ε): blob (no correlation)  
# Mode C (joint): hyperbolic curve (inverse relationship)


@torch.no_grad()
def compute_margin_width_data(
    model,
    data_loader,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Collect per-example margin and width for the scatter plot.
    
    Returns dict with:
      margin: [N] task margin
      epsilon: [N] credal width ε(x)
      sigma_sq: [N, K] per-concept variance
      correct: [N] whether prediction is correct
    """
    model.eval()
    W, b = extract_classifier_weights(model)
    W, b = W.to(device), b.to(device)
    
    all_margin, all_eps, all_sigma_sq, all_correct = [], [], [], []
    
    for batch in data_loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        concept_labels = batch.get("concept_labels")
        if concept_labels is not None:
            concept_labels = concept_labels.to(device)
        
        output = model(features, labels, concept_labels)
        mu = output["mu"]
        sigma_sq = output["sigma_sq"]
        logits = output.get("logits", output.get("label_logits"))
        
        eps = torch.sqrt(sigma_sq.sum(dim=-1))
        margin = compute_task_margin(mu, W, b, labels)
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).float()
        
        all_margin.append(margin.cpu())
        all_eps.append(eps.cpu())
        all_sigma_sq.append(sigma_sq.cpu())
        all_correct.append(correct.cpu())
    
    return {
        "margin": torch.cat(all_margin).numpy(),
        "epsilon": torch.cat(all_eps).numpy(),
        "sigma_sq": torch.cat(all_sigma_sq).numpy(),
        "correct": torch.cat(all_correct).numpy(),
    }


def experiment_3_margin_coupling(
    models: Dict[str, object],
    data_loader,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """
    Run the margin-width analysis for multiple modes.
    
    Args:
        models: dict mapping mode name to model, e.g.:
          {"Post-hoc (A)": model_a, "Fixed-ε (B)": model_b, "Joint (C)": model_c}
    
    Returns dict mapping mode name to results.
    """
    results = {}
    
    for mode_name, model in models.items():
        print(f"\n  Computing margin-width for {mode_name}...")
        data = compute_margin_width_data(model, data_loader, device)
        
        # Spearman(margin, ε) — should be strongly negative for Mode C
        rho, p = spearmanr(data["margin"], data["epsilon"])
        
        # Fit hyperbola: ε ≈ a / margin + c
        # (Only on correctly classified examples with positive margin)
        mask = (data["correct"] > 0.5) & (data["margin"] > 0.01)
        if mask.sum() > 10:
            coeffs = np.polyfit(1.0 / data["margin"][mask],
                               data["epsilon"][mask], 1)
            hyperbola_a = coeffs[0]
        else:
            hyperbola_a = 0.0
        
        results[mode_name] = {
            "spearman_margin_eps": float(rho),
            "spearman_p": float(p),
            "hyperbola_coefficient": float(hyperbola_a),
            "accuracy": float(data["correct"].mean()),
            "n_examples": len(data["margin"]),
            "_arrays": data,
        }
        
        print(f"    ρ(margin, ε) = {rho:+.4f}  (p={p:.2e})")
        print(f"    Accuracy: {data['correct'].mean():.4f}")
    
    # Report comparison
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: MARGIN-WIDTH COUPLING")
    print("=" * 65)
    print(f"  {'Mode':<20s}  {'ρ(margin,ε)':>12s}  {'Acc':>6s}  {'Hyperbola a':>12s}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*6}  {'─'*12}")
    for mode_name, r in results.items():
        print(f"  {mode_name:<20s}  {r['spearman_margin_eps']:>+12.4f}  "
              f"{r['accuracy']:>6.4f}  {r['hyperbola_coefficient']:>12.4f}")
    print("=" * 65)
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_experiment_1(results: Dict, save_path: Optional[str] = None):
    """
    Scatter: PGD worst-case loss vs dual approximation.
    Points near diagonal → theory validated.
    Color by ε(x) to show gap grows with credal width.
    """
    import matplotlib.pyplot as plt
    
    arr = results["_arrays"]
    pgd = arr["pgd_loss"]
    dual = arr["dual_loss"]
    nom = arr["nominal_loss"]
    eps = arr["epsilon"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: PGD vs Dual (the key validation)
    ax = axes[0]
    sc = ax.scatter(dual, pgd, c=eps, cmap="viridis", s=8, alpha=0.5)
    plt.colorbar(sc, ax=ax, label=r"$\varepsilon(\mathbf{x})$")
    
    # Diagonal reference
    lims = [min(dual.min(), pgd.min()), max(dual.max(), pgd.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
    
    ax.text(0.05, 0.92,
            f"$\\rho = {results['spearman_pgd_dual']:.3f}$\n"
            f"$R^2 = {results['r_squared_pgd_dual']:.3f}$",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    ax.set_xlabel(r"Dual approximation $\ell(\mu) + \|\nabla\ell\|_{\Sigma}$",
                  fontsize=10)
    ax.set_ylabel(r"PGD worst-case $\sup_{p \in \mathcal{C}} \ell(p)$",
                  fontsize=10)
    ax.set_title("(a) DRO Equivalence", fontsize=12)
    
    # Panel B: Three losses compared
    ax = axes[1]
    idx = np.argsort(eps)
    ax.plot(eps[idx], nom[idx], ".", alpha=0.2, ms=3, label="Nominal", color="green")
    ax.plot(eps[idx], dual[idx], ".", alpha=0.2, ms=3, label="Dual approx", color="blue")
    ax.plot(eps[idx], pgd[idx], ".", alpha=0.2, ms=3, label="PGD exact", color="red")
    
    ax.set_xlabel(r"$\varepsilon(\mathbf{x})$", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("(b) Loss by credal width", fontsize=12)
    ax.legend(fontsize=9, markerscale=3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_experiment_2_trajectory(
    snapshots: List[EquilibriumSnapshot],
    save_path: Optional[str] = None,
):
    """
    Training dynamics: ε, margin, and ε·margin product over epochs.
    The product should converge to β/λ_dro (dotted horizontal line).
    """
    import matplotlib.pyplot as plt
    
    epochs = [s.epoch for s in snapshots]
    eps_means = [s.eps_mean for s in snapshots]
    margin_means = [s.margin_mean for s in snapshots]
    products = [s.product_mean for s in snapshots]
    predicted = snapshots[-1].predicted_product  # β/λ_dro
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Panel A: ε over training
    ax = axes[0]
    ax.plot(epochs, eps_means, "o-", color="#2196F3", ms=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\bar{\varepsilon}$")
    ax.set_title(r"(a) Credal width $\varepsilon$")
    ax.grid(True, alpha=0.3)
    
    # Panel B: margin over training
    ax = axes[1]
    ax.plot(epochs, margin_means, "o-", color="#FF9800", ms=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\bar{m}$")
    ax.set_title("(b) Task margin")
    ax.grid(True, alpha=0.3)
    
    # Panel C: product (should converge to horizontal line)
    ax = axes[2]
    ax.plot(epochs, products, "o-", color="#4CAF50", ms=4,
            label=r"$\bar{\varepsilon} \cdot |\bar{m}|$ (empirical)")
    if predicted < float("inf"):
        ax.axhline(y=predicted, color="red", ls="--", lw=1.5,
                    label=rf"$\beta/\lambda_{{dro}} = {predicted:.4f}$ (theory)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\varepsilon \cdot |m|$")
    ax.set_title("(c) Width-margin equilibrium")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Equilibrium Convergence During Training", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_experiment_2_marginal_costs(
    results: Dict,
    concept_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Bar plot: per-concept marginal cost σ²_k · |∂ℓ/∂p_k|.
    All bars should be approximately equal height (equalization).
    """
    import matplotlib.pyplot as plt
    
    products = results["per_concept_product"]
    K = len(products)
    if concept_names is None:
        concept_names = [f"Concept {k}" for k in range(K)]
    
    fig, ax = plt.subplots(figsize=(max(5, K * 1.2), 4))
    
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    bars = ax.bar(concept_names, products, color=colors, edgecolor="gray")
    
    # Reference line at the mean (theoretical equalized value)
    mean_val = np.mean(products)
    ax.axhline(y=mean_val, color="red", ls="--", lw=1.5,
               label=f"Mean = {mean_val:.5f}")
    
    ax.set_ylabel(r"$\sigma^2_k \cdot |\partial\ell/\partial p_k|$", fontsize=11)
    ax.set_title(f"Marginal Cost Equalization (CV = {results['marginal_cost_cv']:.3f})",
                 fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_experiment_3(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    3-panel scatter: margin vs ε for each training mode.
    Mode A/B: blob (no coupling)
    Mode C: hyperbolic curve (ε ∝ 1/margin)
    """
    import matplotlib.pyplot as plt
    
    modes = list(results.keys())
    n_modes = len(modes)
    
    fig, axes = plt.subplots(1, n_modes, figsize=(5.5 * n_modes, 4.5))
    if n_modes == 1:
        axes = [axes]
    
    for ax, mode_name in zip(axes, modes):
        data = results[mode_name]["_arrays"]
        margin = data["margin"]
        eps = data["epsilon"]
        correct = data["correct"]
        rho = results[mode_name]["spearman_margin_eps"]
        
        # Color by correctness
        colors = np.where(correct > 0.5, "#2196F3", "#F44336")
        ax.scatter(margin, eps, c=colors, s=8, alpha=0.4)
        
        # Fit and plot trend for correct examples
        mask = correct > 0.5
        if mask.sum() > 10:
            z = np.polyfit(margin[mask], eps[mask], 1)
            x_line = np.linspace(
                np.percentile(margin[mask], 5),
                np.percentile(margin[mask], 95), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5, alpha=0.7)
        
        ax.text(0.05, 0.95, f"$\\rho = {rho:+.3f}$",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        
        ax.set_xlabel("Task margin $m(\\mathbf{x})$", fontsize=10)
        ax.set_ylabel(r"$\varepsilon(\mathbf{x})$", fontsize=10)
        ax.set_title(mode_name, fontsize=12)
        
        # Add legend for colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
                   ms=6, label="Correct"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
                   ms=6, label="Wrong"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    
    plt.suptitle("Margin-Width Coupling", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ============================================================================
# COMBINED RUNNER
# ============================================================================

@torch.no_grad()
def run_all_theory_experiments(
    model,
    data_loader,
    lambda_dro: float,
    beta: float,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    concept_names: Optional[List[str]] = None,
    pgd_steps: int = 50,
) -> Dict:
    """
    Run all three theory experiments on a single trained model.
    
    For the full 3-panel Experiment 3, call experiment_3_margin_coupling
    separately with all three mode models.
    """
    print("\n" + "=" * 65)
    print("  RUNNING THEORY VALIDATION EXPERIMENTS")
    print("=" * 65)
    
    # Experiment 1: DRO Equivalence
    print("\n── Experiment 1: DRO Equivalence ──")
    exp1 = experiment_1_dro_equivalence(
        model, data_loader, device, pgd_steps=pgd_steps)
    
    # Experiment 2: Equilibrium (single snapshot at final model)
    print("\n── Experiment 2: Width-Margin Equilibrium ──")
    exp2 = experiment_2_single_model(
        model, data_loader, lambda_dro, beta, device)
    
    # Experiment 3: Margin coupling (single mode)
    print("\n── Experiment 3: Margin Coupling (single mode) ──")
    data = compute_margin_width_data(model, data_loader, device)
    rho, p = spearmanr(data["margin"], data["epsilon"])
    exp3_single = {
        "spearman_margin_eps": float(rho),
        "accuracy": float(data["correct"].mean()),
        "_arrays": data,
    }
    print(f"  ρ(margin, ε) = {rho:+.4f}")
    
    all_results = {
        "experiment_1_dro_equivalence": exp1,
        "experiment_2_equilibrium": exp2,
        "experiment_3_margin_coupling": exp3_single,
    }
    
    # Save plots if output_dir given
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        plot_experiment_1(exp1, save_path=str(out / "exp1_dro_equivalence.pdf"))
        plot_experiment_2_marginal_costs(
            exp2, concept_names=concept_names,
            save_path=str(out / "exp2_marginal_costs.pdf"))
        
        # Save JSON results (without arrays)
        json_safe = {}
        for key, val in all_results.items():
            json_safe[key] = {k: v for k, v in val.items() if k != "_arrays"}
        with open(out / "theory_results.json", "w") as f:
            json.dump(json_safe, f, indent=2)
        print(f"\nResults saved to {out}/")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Theory validation experiments for Credal DRO")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cebab")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_dro", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pgd_steps", type=int, default=50)
    
    # For three-way comparison (Experiment 3)
    parser.add_argument("--three_way_dir", type=str, default=None,
                        help="Dir with post_hoc/, fixed_eps/, joint/ subdirs")
    args = parser.parse_args()
    
    print("Theory experiment runner.")
    print("Import and call run_all_theory_experiments() from your training code,")
    print("or load your model here and call it.")
    print()
    print("Example:")
    print("  from credal_theory_experiments import run_all_theory_experiments")
    print("  results = run_all_theory_experiments(")
    print("      model, val_loader, lambda_dro=0.1, beta=0.005,")
    print("      output_dir='results/theory/', concept_names=['food','ambiance','service','noise'])")
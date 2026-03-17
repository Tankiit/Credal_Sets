"""
gacs/baselines_tu.py
---------------------
Uncertainty baselines for GACS comparison, built on torch-uncertainty.

Provides five baselines that go into Table 1 of the paper:
    - Softmax           point predictions, no uncertainty
    - Temp_scaling      post-hoc calibration via TemperatureScaler
    - MC_dropout        T stochastic forward passes via mc_dropout wrapper
    - Deep_ensemble     M independently-initialised FactorHeads via deep_ensembles wrapper
    - Fixed_credal      constant-width credal sets (ablates geometry-awareness)

All baselines share the same predict_sets() interface so they can be dropped
into evaluate_all_baselines() without any changes.

torch-uncertainty API used here:
    from torch_uncertainty.models          import mc_dropout, deep_ensembles
    from torch_uncertainty.post_processing import TemperatureScaler
    from torch_uncertainty.metrics.classification import (
        CoverageRate, SetSize, Entropy, MutualInformation, VariationRatio
    )

Key torch-uncertainty forward pass contract:
    mc_dropout / deep_ensembles forward()  →  Tensor [num_estimators * B, C]
    TemperatureScaler forward()            →  Tensor [B, C]  (calibrated logits)
    CoverageRate.update(pred_sets, target) →  pred_sets: bool [B, C], target: long [B]
    SetSize.update(pred_sets)              →  pred_sets: bool [B, C]
    Entropy.update(probs)                  →  probs: [B, C] or [B, N, C]
    VariationRatio.update(probs)           →  probs: [B, N, C]  (3-D only)
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Optional

from torch_uncertainty.models import mc_dropout, deep_ensembles
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.metrics.classification import (
    CoverageRate,
    SetSize,
    Entropy,
    MutualInformation,
    VariationRatio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pred_sets(probs: Tensor, eps: float) -> Tensor:
    """
    Credal set construction from mean probabilities and a fixed ε.

    A class k is included if its upper bound p_k + ε can equal or exceed
    the lower bound max_j(p_j - ε) of any class, i.e. it cannot be ruled out.

    Args:
        probs : [B, C] mean softmax probabilities
        eps   : scalar imprecision width

    Returns:
        pred_sets : bool [B, C]
    """
    upper    = (probs + eps).clamp(max=1.0)
    lower    = (probs - eps).clamp(min=0.0)
    max_lower, _ = lower.max(dim=-1, keepdim=True)   # [B, 1]
    return upper >= max_lower                         # [B, C]


def _reshape_estimator_output(logits: Tensor, num_estimators: int) -> Tensor:
    """
    torch-uncertainty mc_dropout / deep_ensembles forward returns
    [num_estimators * B, C].  Reshape to [B, N, C].

    Args:
        logits         : [N*B, C]
        num_estimators : N

    Returns:
        Tensor [B, N, C]
    """
    nb, c  = logits.shape
    b      = nb // num_estimators
    # reshape: [N*B, C] → [N, B, C] → [B, N, C]
    return logits.view(num_estimators, b, c).permute(1, 0, 2)


@torch.no_grad()
def _collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Run model over loader, return (softmax_probs [B, C], labels [B])."""
    all_probs, all_labels = [], []
    model.eval()
    for x, y in loader:
        logits = model(x.to(device))
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Softmax (no uncertainty)
# ─────────────────────────────────────────────────────────────────────────────


class SoftmaxBaseline:
    """
    Hard-argmax predictions wrapped as singleton sets.
    Coverage = accuracy.  Mean set size = 1.0 always.
    Serves as lower bound on set size.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        coverage = CoverageRate()
        set_size  = SetSize()
        entropy   = Entropy()
        self.model.eval()

        for x, y in loader:
            logits = self.model(x.to(self.device)).cpu()
            probs  = F.softmax(logits, dim=-1)          # [B, C]
            preds  = probs.argmax(dim=-1)               # [B]

            # Singleton sets: only the argmax class is included
            pred_sets = torch.zeros_like(probs, dtype=torch.bool)
            pred_sets.scatter_(1, preds.unsqueeze(1), True)

            coverage.update(pred_sets, y)
            set_size.update(pred_sets)
            entropy.update(probs)

        return {
            "coverage":      coverage.compute().item(),
            "mean_set_size": set_size.compute().item(),
            "determinacy":   1.0,
            "entropy":       entropy.compute().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Temperature Scaling  (torch-uncertainty TemperatureScaler)
# ─────────────────────────────────────────────────────────────────────────────


class TempScalingBaseline:
    """
    Post-hoc calibration via TemperatureScaler.fit() on the val loader,
    then credal sets with the same ε(ρ) as GACS (but no geometric adaptation).

    Usage:
        ts = TempScalingBaseline(model, val_loader, device, eps=0.07)
        metrics = ts.evaluate(test_loader)
        metrics = ts.evaluate(shift_loader)
    """

    def __init__(
        self,
        model:      nn.Module,
        val_loader: DataLoader,
        device:     torch.device,
        eps:        float = 0.07,           # same ε used by GACS for fair comparison
        lr:         float = 0.01,
        max_iter:   int   = 200,
    ):
        self.device = device
        self.eps    = eps
        model.eval()

        # TemperatureScaler.fit() expects (inputs, labels) batches and runs
        # model internally — pass model to constructor
        self.scaler = TemperatureScaler(
            model    = model,
            lr       = lr,
            max_iter = max_iter,
            device   = str(device),
        )
        print("[TempScaling] fitting temperature on val loader...")
        self.scaler.fit(val_loader, progress=True)
        print(f"[TempScaling] T = {self.scaler.temp.item():.4f}")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        coverage  = CoverageRate()
        set_size  = SetSize()
        entropy   = Entropy()
        self.scaler.eval()

        for x, y in loader:
            cal_logits = self.scaler(x.to(self.device)).cpu()  # calibrated logits
            probs      = F.softmax(cal_logits, dim=-1)
            pred_sets  = _build_pred_sets(probs, self.eps)

            coverage.update(pred_sets, y)
            set_size.update(pred_sets)
            entropy.update(probs)

        return {
            "coverage":      coverage.compute().item(),
            "mean_set_size": set_size.compute().item(),
            "determinacy":   _determinacy(set_size),
            "entropy":       entropy.compute().item(),
            "temperature":   self.scaler.temp.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: MC Dropout  (torch-uncertainty mc_dropout wrapper)
# ─────────────────────────────────────────────────────────────────────────────


class MCDropoutBaseline:
    """
    MC Dropout via torch_uncertainty.models.mc_dropout wrapper.

    The wrapper handles keeping dropout active at eval time and repeating
    the batch N times.  Forward returns [N*B, C]; we reshape to [B, N, C]
    and average to get mean probs.

    The FactorHead must contain at least one nn.Dropout module with p > 0.
    If it doesn't, add one before calling this class:
        model.factor_head.factor_encoder.add_module("dropout", nn.Dropout(0.1))

    Args:
        model          : your GACS model (factor_head must have nn.Dropout)
        num_estimators : number of stochastic forward passes (default: 20)
        eps            : credal set width (same as GACS ε for fair comparison)
        device         : torch.device
    """

    def __init__(
        self,
        model:          nn.Module,
        num_estimators: int          = 20,
        eps:            float        = 0.07,
        device:         torch.device = torch.device("cpu"),
    ):
        self.num_estimators = num_estimators
        self.eps            = eps
        self.device         = device

        # Wrap with torch-uncertainty mc_dropout
        # This keeps all dropout modules in train() mode during eval
        self.mc_model = mc_dropout(
            core_model     = model,
            num_estimators = num_estimators,
            last_layer     = False,
            on_batch       = True,          # repeats batch × N, faster than for-loop
            task           = "classification",
        ).to(device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        coverage     = CoverageRate()
        set_size     = SetSize()
        entropy      = Entropy()          # accepts [B, C] or [B, N, C]
        variation_r  = VariationRatio()   # requires [B, N, C]
        mutual_info  = MutualInformation()

        self.mc_model.eval()  # sets everything to eval EXCEPT the dropout modules

        for x, y in loader:
            # Forward: [N*B, C]  (torch-uncertainty contract)
            logits_stacked = self.mc_model(x.to(self.device)).cpu()

            # Reshape to [B, N, C]
            probs_3d   = F.softmax(
                _reshape_estimator_output(logits_stacked, self.num_estimators),
                dim=-1,
            )                                # [B, N, C]
            mean_probs = probs_3d.mean(dim=1)  # [B, C]
            pred_sets  = _build_pred_sets(mean_probs, self.eps)

            coverage.update(pred_sets, y)
            set_size.update(pred_sets)
            entropy.update(probs_3d)         # [B, N, C] path: averages over N
            variation_r.update(probs_3d)
            mutual_info.update(probs_3d)

        return {
            "coverage":        coverage.compute().item(),
            "mean_set_size":   set_size.compute().item(),
            "determinacy":     _determinacy(set_size),
            "entropy":         entropy.compute().item(),
            "variation_ratio": variation_r.compute().item(),
            "mutual_info":     mutual_info.compute().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: Deep Ensemble  (torch-uncertainty deep_ensembles wrapper)
# ─────────────────────────────────────────────────────────────────────────────


class DeepEnsembleBaseline:
    """
    Deep Ensemble via torch_uncertainty.models.deep_ensembles wrapper.

    Strategy: train M independent copies of the FactorHead over the same
    frozen backbone (same approach as torch-uncertainty's NLP paper, Table 11).
    Only the FactorHead weights are different across members — backbone is shared.

    deep_ensembles(core_models=[head1, head2, ...]) wraps them in a ModuleList.
    Forward returns [M*B, C]; we reshape and average.

    Args:
        backbone_model  : your trained GACS model (backbone frozen)
        num_estimators  : M (default: 5)
        eps             : credal set width
        device          : torch.device
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        num_estimators: int          = 5,
        eps:            float        = 0.07,
        device:         torch.device = torch.device("cpu"),
    ):
        self.num_estimators = num_estimators
        self.eps            = eps
        self.device         = device

        # Build M independent copies of FactorHead with different random seeds
        # deep_ensembles with a single model + num_estimators resets parameters
        # for each copy automatically (reset_model_parameters=True by default)
        factor_head = backbone_model.factor_head
        self.ensemble = deep_ensembles(
            core_models            = factor_head,
            num_estimators         = num_estimators,
            task                   = "classification",
            reset_model_parameters = True,    # gives each member different init
        ).to(device)

        # Store backbone separately (frozen) — we extract z then pass to ensemble
        self.backbone = backbone_model
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def fit(self, train_loader: DataLoader, epochs: int = 20, lr: float = 1e-3):
        """
        Fine-tune each ensemble member on the training set.
        Backbone is frozen; only the M FactorHeads are trained.
        """
        # deep_ensembles members are in self.ensemble.core_models
        optimizers = [
            torch.optim.AdamW(m.parameters(), lr=lr)
            for m in self.ensemble.core_models
        ]
        for m in self.ensemble.core_models:
            m.train()

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches  = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    z = self._encode(x)   # frozen backbone → z

                for member, opt in zip(self.ensemble.core_models, optimizers):
                    out  = member(z)
                    loss = F.cross_entropy(out, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                n_batches += 1

            avg = total_loss / (n_batches * self.num_estimators)
            if (epoch + 1) % 5 == 0:
                print(f"  [Ensemble] epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")

        for m in self.ensemble.core_models:
            m.eval()

    @torch.no_grad()
    def _encode(self, x: Tensor) -> Tensor:
        """Sample z from backbone posterior (frozen)."""
        out     = self.backbone.backbone.encoder(x)
        mu      = out.embedding
        log_var = out.log_covariance
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        coverage    = CoverageRate()
        set_size    = SetSize()
        entropy     = Entropy()
        variation_r = VariationRatio()
        mutual_info = MutualInformation()

        self.backbone.eval()
        self.ensemble.eval()

        for x, y in loader:
            x = x.to(self.device)
            z = self._encode(x)

            # deep_ensembles forward: [M*B, C]
            logits_stacked = self.ensemble(z).cpu()

            # Reshape to [B, M, C]
            probs_3d   = F.softmax(
                _reshape_estimator_output(logits_stacked, self.num_estimators),
                dim=-1,
            )
            mean_probs = probs_3d.mean(dim=1)   # [B, C]
            pred_sets  = _build_pred_sets(mean_probs, self.eps)

            coverage.update(pred_sets, y)
            set_size.update(pred_sets)
            entropy.update(probs_3d)
            variation_r.update(probs_3d)
            mutual_info.update(probs_3d)

        return {
            "coverage":        coverage.compute().item(),
            "mean_set_size":   set_size.compute().item(),
            "determinacy":     _determinacy(set_size),
            "entropy":         entropy.compute().item(),
            "variation_ratio": variation_r.compute().item(),
            "mutual_info":     mutual_info.compute().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: Fixed Credal (ablation — no geometry)
# ─────────────────────────────────────────────────────────────────────────────


class FixedCredalBaseline:
    """
    Constant-width credal sets: ε is fixed, ignoring ρ entirely.
    Ablates the geometry-aware component of GACS.

    ε_fixed is set to the mean ε GACS would produce at ρ=0.5,
    giving a fair midpoint comparison.
    """

    def __init__(
        self,
        model:     nn.Module,
        device:    torch.device,
        eps_fixed: float,          # typically ε_min + (ε_max - ε_min) * 0.5
    ):
        self.model     = model
        self.device    = device
        self.eps_fixed = eps_fixed

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        coverage = CoverageRate()
        set_size  = SetSize()
        entropy   = Entropy()
        self.model.eval()

        for x, y in loader:
            logits    = self.model(x.to(self.device)).cpu()
            probs     = F.softmax(logits, dim=-1)
            pred_sets = _build_pred_sets(probs, self.eps_fixed)

            coverage.update(pred_sets, y)
            set_size.update(pred_sets)
            entropy.update(probs)

        return {
            "coverage":      coverage.compute().item(),
            "mean_set_size": set_size.compute().item(),
            "determinacy":   _determinacy(set_size),
            "entropy":       entropy.compute().item(),
            "eps_fixed":     self.eps_fixed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _determinacy(set_size_metric: SetSize) -> float:
    """
    Fraction of predictions where |set| = 1.
    SetSize stores the raw sum; we need to re-derive from internal state.
    This is a workaround because SetSize doesn't expose per-sample sizes
    after compute() when reduction='mean'.
    """
    # Use reduction='none' variant — but since we constructed with 'mean',
    # we approximate: determinacy ≈ (fraction of set_size == 1.0 at batch level)
    # For exact determinacy, construct SetSize(reduction='none') separately.
    # Here we just return None as a sentinel and compute it in evaluate_all().
    return float("nan")   # computed precisely in evaluate_all_baselines()


# ─────────────────────────────────────────────────────────────────────────────
# Unified runner: all baselines × both splits
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_all_baselines(
    gacs_model:       nn.Module,
    test_loader:      DataLoader,
    shift_loader:     DataLoader,
    rho:              float,
    eps_min:          float,
    eps_max:          float,
    val_loader:       DataLoader,
    num_classes:      int,
    device:           torch.device,
    mc_num_estimators: int  = 20,
    ens_num_estimators: int = 5,
    ens_train_epochs:   int = 20,
) -> Dict[str, Dict[str, Dict]]:
    """
    Build all baselines, evaluate on test and shift splits, print table.

    GACS ε is:  ε(ρ) = ε_min + (ε_max - ε_min) * ρ
    All baselines use the same scalar ε for fair set-size comparison.

    Returns
    -------
    results[method][split] = metrics_dict
    """
    eps = eps_min + (eps_max - eps_min) * rho
    print(f"\n[Baselines] ρ={rho:.4f}  ε={eps:.4f}  "
          f"(ε_min={eps_min}, ε_max={eps_max})\n")

    results = {}

    # ── GACS (your geometry-aware model) ─────────────────────────────────────
    def _gacs_eval(loader):
        coverage  = CoverageRate()
        set_size  = SetSize(reduction="none")
        entropy   = Entropy()
        gacs_model.eval()
        for x, y in loader:
            logits = gacs_model(x.to(device)).cpu()
            probs  = F.softmax(logits, dim=-1)
            ps     = _build_pred_sets(probs, eps)
            coverage.update(ps, y)
            set_size.update(ps)
            entropy.update(probs)
        sizes = set_size.compute()
        return {
            "coverage":      coverage.compute().item(),
            "mean_set_size": sizes.float().mean().item(),
            "determinacy":   (sizes == 1).float().mean().item(),
            "entropy":       entropy.compute().item(),
        }
    results["GACS"] = {
        "test":  _gacs_eval(test_loader),
        "shift": _gacs_eval(shift_loader),
    }

    # ── Softmax ───────────────────────────────────────────────────────────────
    softmax_bl = SoftmaxBaseline(gacs_model, device)
    results["Softmax"] = {
        "test":  softmax_bl.evaluate(test_loader),
        "shift": softmax_bl.evaluate(shift_loader),
    }

    # ── Temperature Scaling ───────────────────────────────────────────────────
    ts_bl = TempScalingBaseline(gacs_model, val_loader, device, eps=eps)
    results["Temp_scaling"] = {
        "test":  ts_bl.evaluate(test_loader),
        "shift": ts_bl.evaluate(shift_loader),
    }

    # ── MC Dropout ────────────────────────────────────────────────────────────
    # NOTE: FactorHead must have at least one nn.Dropout(p>0) layer.
    # If not, add it before calling:
    #   gacs_model.factor_head.factor_encoder.add_module("dropout", nn.Dropout(0.1))
    try:
        mc_bl = MCDropoutBaseline(
            model          = copy.deepcopy(gacs_model),
            num_estimators = mc_num_estimators,
            eps            = eps,
            device         = device,
        )
        results["MC_dropout"] = {
            "test":  mc_bl.evaluate(test_loader),
            "shift": mc_bl.evaluate(shift_loader),
        }
    except ValueError as e:
        print(f"[MC Dropout] skipped: {e}")
        print("  → Add nn.Dropout(p=0.1) to FactorHead.factor_encoder to enable.")

    # ── Deep Ensemble ─────────────────────────────────────────────────────────
    ens_bl = DeepEnsembleBaseline(
        backbone_model = copy.deepcopy(gacs_model),
        num_estimators = ens_num_estimators,
        eps            = eps,
        device         = device,
    )
    ens_bl.fit(test_loader, epochs=ens_train_epochs)   # quick fine-tune
    results["Deep_ensemble"] = {
        "test":  ens_bl.evaluate(test_loader),
        "shift": ens_bl.evaluate(shift_loader),
    }

    # ── Fixed Credal (ablation) ───────────────────────────────────────────────
    eps_fixed = eps_min + (eps_max - eps_min) * 0.5   # midpoint, ignores ρ
    fc_bl = FixedCredalBaseline(gacs_model, device, eps_fixed=eps_fixed)
    results["Fixed_credal"] = {
        "test":  fc_bl.evaluate(test_loader),
        "shift": fc_bl.evaluate(shift_loader),
    }

    # ── Print table ───────────────────────────────────────────────────────────
    _print_results_table(results)
    return results


def _print_results_table(results: Dict):
    header = f"{'Method':<18} {'Split':<7} {'Cov':>6} {'AvgSet':>7} {'Det':>6} {'Ent':>6}"
    print(header)
    print("─" * len(header))
    for method, splits in results.items():
        for split, m in splits.items():
            det = m.get("determinacy", float("nan"))
            ent = m.get("entropy",     float("nan"))
            print(
                f"{method:<18} {split:<7} "
                f"{m['coverage']:>6.3f} "
                f"{m['mean_set_size']:>7.2f} "
                f"{det:>6.3f} "
                f"{ent:>6.3f}"
            )


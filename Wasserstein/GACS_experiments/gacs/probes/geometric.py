"""
GACS Geometric Probes
======================

Two probes for measuring the degeneracy ratio ρ:

1. Perturbation Sensitivity (primary, cheap):
   Sample m random directions, perturb θ*, measure Δ loss.
   ρ_pert = fraction of insensitive directions.

2. Hessian Spectral Analysis (optional, expensive):
   Compute top-k eigenvalues via Lanczos iteration.
   ρ_hess = 1 - r_eff / d, where r_eff = exp(H_spec).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from copy import deepcopy


class PerturbationProbe:
    """
    Perturbation Sensitivity Analysis.

    Measures the fraction of random weight-space directions
    along which the loss is insensitive (flat).

    ρ_pert = (1/m) Σ_i 1[|L(θ* + ε·v_i) - L(θ*)| < τ]

    - ρ → 0: sharp minimum, all directions matter → low epistemic uncertainty
    - ρ → 1: flat minimum, many redundant directions → high epistemic uncertainty
    """

    def __init__(self, config):
        self.num_directions = config.probe.num_directions
        self.epsilon = config.probe.perturbation_epsilon
        self.threshold = config.probe.insensitivity_threshold
        self.probe_scope = config.probe.probe_scope

    @torch.no_grad()
    def compute_degeneracy_ratio(
        self,
        model: nn.Module,
        loss_fn,
        dataloader,
        device: torch.device,
        num_batches: int = 5,
    ) -> Dict[str, float]:
        """
        Compute the perturbation-based degeneracy ratio.

        Args:
            model: trained model
            loss_fn: function(model_outputs, batch) → scalar loss
            dataloader: data to evaluate loss on
            device: computation device
            num_batches: number of batches to average loss over

        Returns:
            dict with:
                rho: degeneracy ratio ∈ [0, 1]
                deltas: list of loss deltas for each direction
                base_loss: loss at θ*
                num_insensitive: count of insensitive directions
        """
        model.eval()

        # 1. Compute base loss L(θ*)
        base_loss = self._compute_avg_loss(model, loss_fn, dataloader, device, num_batches)

        # 2. Get parameters to probe
        params = model.get_probeable_parameters(self.probe_scope)
        param_shapes = [p.shape for p in params]
        total_params = sum(p.numel() for p in params)

        # 3. Sample random directions and measure sensitivity
        deltas = []
        num_insensitive = 0

        for i in range(self.num_directions):
            # Sample random unit direction in parameter space
            direction = self._sample_unit_direction(param_shapes, device)

            # Perturb: θ* + ε * v_i
            self._apply_perturbation(params, direction, self.epsilon)

            # Compute perturbed loss
            perturbed_loss = self._compute_avg_loss(
                model, loss_fn, dataloader, device, num_batches
            )

            # Undo perturbation
            self._apply_perturbation(params, direction, -self.epsilon)

            # Record delta
            delta = abs(perturbed_loss - base_loss)
            deltas.append(delta)

            if delta < self.threshold:
                num_insensitive += 1

        rho = num_insensitive / self.num_directions

        return {
            "rho": rho,
            "deltas": deltas,
            "base_loss": base_loss,
            "num_insensitive": num_insensitive,
            "num_directions": self.num_directions,
            "total_probeable_params": total_params,
            "mean_delta": np.mean(deltas),
            "std_delta": np.std(deltas),
            "median_delta": np.median(deltas),
        }

    def _sample_unit_direction(
        self, param_shapes: List[torch.Size], device: torch.device
    ) -> List[torch.Tensor]:
        """Sample a random unit direction in parameter space."""
        direction = []
        for shape in param_shapes:
            d = torch.randn(shape, device=device)
            direction.append(d)

        # Normalize to unit length
        total_norm = torch.sqrt(
            sum(d.pow(2).sum() for d in direction)
        )
        direction = [d / (total_norm + 1e-10) for d in direction]
        return direction

    def _apply_perturbation(
        self, params: List[torch.Tensor], direction: List[torch.Tensor], epsilon: float
    ):
        """Add epsilon * direction to parameters in-place."""
        for p, d in zip(params, direction):
            p.data.add_(d, alpha=epsilon)

    def _compute_avg_loss(
        self,
        model: nn.Module,
        loss_fn,
        dataloader,
        device: torch.device,
        num_batches: int,
    ) -> float:
        """Compute average loss over num_batches."""
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # loss_fn should return a scalar
            loss = loss_fn(outputs, batch)
            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)


class HessianProbe:
    """
    Hessian Spectral Analysis via Lanczos iteration.

    Computes top-k eigenvalues of the loss Hessian, then derives:
        r_eff = exp(H_spec)  where H_spec = -Σ λ̂_i log λ̂_i
        ρ_hess = 1 - r_eff / d

    More expensive but captures local quadratic curvature precisely.
    """

    def __init__(self, config):
        self.top_k = config.probe.hessian_top_k
        self.num_batches = config.probe.hessian_num_batches
        self.probe_scope = config.probe.probe_scope

    def compute_degeneracy_ratio(
        self,
        model: nn.Module,
        loss_fn,
        dataloader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute Hessian-based degeneracy ratio using Lanczos iteration.

        Returns:
            dict with rho, effective_rank, top eigenvalues, spectral entropy
        """
        model.eval()

        params = model.get_probeable_parameters(self.probe_scope)
        total_params = sum(p.numel() for p in params)

        # Compute top-k eigenvalues via Lanczos
        eigenvalues = self._lanczos_eigenvalues(
            model, loss_fn, dataloader, params, device
        )

        # Compute spectral entropy and effective rank
        eigenvalues_pos = np.maximum(eigenvalues, 1e-10)
        eigenvalues_norm = eigenvalues_pos / eigenvalues_pos.sum()

        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        effective_rank = np.exp(spectral_entropy)

        rho = 1.0 - effective_rank / total_params

        return {
            "rho": max(0.0, min(1.0, rho)),
            "effective_rank": effective_rank,
            "spectral_entropy": spectral_entropy,
            "top_eigenvalues": eigenvalues.tolist(),
            "total_params": total_params,
            "max_eigenvalue": float(eigenvalues.max()),
            "trace_estimate": float(eigenvalues.sum()),
        }

    def _lanczos_eigenvalues(
        self,
        model: nn.Module,
        loss_fn,
        dataloader,
        params: List[nn.Parameter],
        device: torch.device,
    ) -> np.ndarray:
        """
        Compute top-k eigenvalues of the Hessian via Lanczos iteration
        using Hessian-vector products (no explicit Hessian storage).
        """
        def hvp(v_flat: torch.Tensor) -> torch.Tensor:
            """Hessian-vector product via double backprop."""
            # Reshape flat vector to parameter shapes
            v_params = self._unflatten(v_flat, params)

            # Compute gradient
            model.zero_grad()
            total_loss = 0.0
            count = 0
            for i, batch in enumerate(dataloader):
                if i >= self.num_batches:
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss_fn(outputs, batch)
                total_loss += loss
                count += 1

            avg_loss = total_loss / max(count, 1)

            # First-order gradients
            grads = torch.autograd.grad(avg_loss, params, create_graph=True)

            # Hessian-vector product: ∇²L · v = ∂/∂θ (∇L · v)
            grad_dot_v = sum((g * v).sum() for g, v in zip(grads, v_params))
            hvp_result = torch.autograd.grad(grad_dot_v, params)

            return self._flatten([h.detach() for h in hvp_result])

        # Lanczos iteration
        d = sum(p.numel() for p in params)
        k = min(self.top_k, d)

        # Initialize
        alpha = []  # diagonal of tridiagonal matrix
        beta = []   # off-diagonal
        V = []      # Lanczos vectors

        v = torch.randn(d, device=device)
        v = v / v.norm()
        V.append(v)

        w = hvp(v)
        a = w.dot(v).item()
        alpha.append(a)
        w = w - a * v

        for j in range(1, k):
            b = w.norm().item()
            if b < 1e-10:
                break
            beta.append(b)

            v_prev = V[-1]
            v = w / b
            V.append(v)

            w = hvp(v)
            a = w.dot(v).item()
            alpha.append(a)
            w = w - a * v - b * v_prev

        # Eigenvalues of tridiagonal matrix
        T = np.zeros((len(alpha), len(alpha)))
        for i in range(len(alpha)):
            T[i, i] = alpha[i]
        for i in range(len(beta)):
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

        eigenvalues = np.linalg.eigvalsh(T)
        return np.sort(eigenvalues)[::-1]  # descending

    def _flatten(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    def _unflatten(
        self, flat: torch.Tensor, params: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        result = []
        offset = 0
        for p in params:
            n = p.numel()
            result.append(flat[offset:offset + n].reshape(p.shape))
            offset += n
        return result


# ---------------------------------------------------------------------------
# Unified probe interface
# ---------------------------------------------------------------------------

def compute_geometric_probe(
    model: nn.Module,
    loss_fn,
    dataloader,
    config,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run the configured geometric probe and return degeneracy ratio.

    Uses perturbation probe by default; Hessian probe if configured.
    """
    if config.probe.use_hessian:
        probe = HessianProbe(config)
    else:
        probe = PerturbationProbe(config)

    return probe.compute_degeneracy_ratio(model, loss_fn, dataloader, device)

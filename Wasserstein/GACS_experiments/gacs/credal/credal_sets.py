"""
GACS Credal Set Construction and Evaluation
=============================================

Maps degeneracy ratio ρ → credal set width ε,
then constructs credal prediction sets from softmax probabilities.

A credal set C_ε is the set of all probability distributions consistent
with the interval constraints [p_k - ε, p_k + ε] ∩ [0,1] for each class k.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class CredalSetConstructor:
    """
    Constructs credal prediction sets from model outputs + degeneracy ratio.

    The mapping ρ → ε → credal set follows:
        ε(ρ) = ε_min + (ε_max - ε_min) · g(ρ)

    where g is a monotone calibration function (linear, sigmoid, or sqrt).
    """

    def __init__(self, config):
        self.epsilon_min = config.credal.epsilon_min
        self.epsilon_max = config.credal.epsilon_max
        self.mapping_fn = config.credal.mapping_fn

    def rho_to_epsilon(self, rho: float) -> float:
        """Map degeneracy ratio ρ ∈ [0,1] to credal set width ε."""
        rho = max(0.0, min(1.0, rho))

        if self.mapping_fn == "linear":
            g = rho
        elif self.mapping_fn == "sigmoid":
            g = 1.0 / (1.0 + np.exp(-10 * (rho - 0.5)))
        elif self.mapping_fn == "sqrt":
            g = np.sqrt(rho)
        else:
            g = rho

        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * g

    def construct_credal_set(
        self,
        probs: torch.Tensor,  # [B, K] softmax probabilities
        epsilon: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct credal prediction sets from softmax probabilities.

        For each predicted probability p_k:
            lower_k = max(0, p_k - ε)
            upper_k = min(1, p_k + ε)

        The credal set C_ε is the convex hull of all distributions
        consistent with these interval constraints.

        Returns:
            lower: lower bounds [B, K]
            upper: upper bounds [B, K]
            predicted_set: which classes are in the credal set [B, K] (bool)
            set_size: size of the predicted set per example [B]
        """
        B, K = probs.shape

        lower = torch.clamp(probs - epsilon, min=0.0)
        upper = torch.clamp(probs + epsilon, max=1.0)

        # A class is in the credal prediction set if its upper bound
        # is >= the lower bound of the maximum probability class,
        # i.e., it could plausibly be the true class under any
        # distribution in the credal set.
        max_lower = lower.max(dim=1, keepdim=True).values
        predicted_set = upper >= max_lower

        # Alternative (simpler): class is in set if its interval overlaps
        # with the top class's interval
        top_class = probs.argmax(dim=1, keepdim=True)
        top_lower = lower.gather(1, top_class)
        predicted_set_v2 = upper >= top_lower

        set_size = predicted_set.float().sum(dim=1)

        return {
            "lower": lower,
            "upper": upper,
            "predicted_set": predicted_set,
            "set_size": set_size,
            "epsilon": epsilon,
        }

    def predict_with_credal(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        rho: float,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end: model forward → softmax → credal set.

        Returns model outputs augmented with credal set information.
        """
        model.eval()
        with torch.no_grad():
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )

        probs = F.softmax(outputs["logits"], dim=-1)
        epsilon = self.rho_to_epsilon(rho)
        credal = self.construct_credal_set(probs, epsilon)

        return {
            **outputs,
            "probs": probs,
            **credal,
        }


class CredalEvaluator:
    """
    Evaluate credal prediction sets.

    Metrics:
        - set_accuracy: fraction where true label ∈ predicted set (coverage)
        - mean_set_size: average |predicted set|
        - determinacy: fraction of singleton predictions
        - conditional_coverage: coverage stratified by subgroups
    """

    @staticmethod
    def evaluate(
        predicted_sets: torch.Tensor,  # [N, K] bool
        labels: torch.Tensor,  # [N] long
        set_sizes: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute credal set evaluation metrics.

        Args:
            predicted_sets: boolean mask of which classes are in the set [N, K]
            labels: true class labels [N]
            set_sizes: precomputed set sizes [N] (optional)

        Returns:
            dict of metrics
        """
        N, K = predicted_sets.shape

        # Coverage (set accuracy): is the true label in the predicted set?
        label_one_hot = F.one_hot(labels, num_classes=K).bool()
        covered = (predicted_sets & label_one_hot).any(dim=1)
        coverage = covered.float().mean().item()

        # Set size
        if set_sizes is None:
            set_sizes = predicted_sets.float().sum(dim=1)
        mean_set_size = set_sizes.float().mean().item()
        median_set_size = set_sizes.float().median().item()

        # Determinacy: fraction of singleton predictions
        determinacy = (set_sizes == 1).float().mean().item()

        # Empty sets (should be rare/zero)
        empty_rate = (set_sizes == 0).float().mean().item()

        # Full sets (all classes — maximum imprecision)
        full_rate = (set_sizes == K).float().mean().item()

        # Coverage-efficiency trade-off
        # Ideal: high coverage, small set size
        if mean_set_size > 0:
            efficiency = coverage / mean_set_size
        else:
            efficiency = 0.0

        return {
            "coverage": coverage,
            "mean_set_size": mean_set_size,
            "median_set_size": median_set_size,
            "determinacy": determinacy,
            "empty_rate": empty_rate,
            "full_rate": full_rate,
            "efficiency": efficiency,
        }

    @staticmethod
    def evaluate_coverage_by_group(
        predicted_sets: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,  # [N] group assignment
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate coverage stratified by group membership.

        Useful for checking conditional coverage across subpopulations
        (e.g., counterfactual vs. original in CEBaB).
        """
        unique_groups = groups.unique()
        results = {}

        for g in unique_groups:
            mask = groups == g
            group_sets = predicted_sets[mask]
            group_labels = labels[mask]

            N_g, K = group_sets.shape
            label_one_hot = F.one_hot(group_labels, num_classes=K).bool()
            covered = (group_sets & label_one_hot).any(dim=1)

            set_sizes = group_sets.float().sum(dim=1)

            results[f"group_{g.item()}"] = {
                "coverage": covered.float().mean().item(),
                "mean_set_size": set_sizes.float().mean().item(),
                "n_examples": N_g,
            }

        return results


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------

class FixedCredalBaseline:
    """Fixed-width credal sets (no geometry, constant ε)."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def predict(self, probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K = probs.shape
        lower = torch.clamp(probs - self.epsilon, min=0.0)
        upper = torch.clamp(probs + self.epsilon, max=1.0)

        max_lower = lower.max(dim=1, keepdim=True).values
        predicted_set = upper >= max_lower
        set_size = predicted_set.float().sum(dim=1)

        return {
            "lower": lower,
            "upper": upper,
            "predicted_set": predicted_set,
            "set_size": set_size,
        }


class MCDropoutBaseline:
    """MC Dropout uncertainty → prediction sets via thresholding."""

    def __init__(self, num_samples: int = 10, coverage_target: float = 0.9):
        self.num_samples = num_samples
        self.coverage_target = coverage_target

    @torch.no_grad()
    def predict(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Run MC Dropout and construct prediction sets."""
        model.train()  # enable dropout

        all_probs = []
        for _ in range(self.num_samples):
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            probs = F.softmax(outputs["logits"], dim=-1)
            all_probs.append(probs)

        model.eval()

        # Stack: [num_samples, B, K]
        all_probs = torch.stack(all_probs, dim=0)

        # Mean prediction and uncertainty
        mean_probs = all_probs.mean(dim=0)  # [B, K]
        std_probs = all_probs.std(dim=0)  # [B, K]

        # Construct prediction sets: include class if
        # mean_prob + z * std_prob > threshold
        # Using a simple threshold-based approach
        upper = mean_probs + 2 * std_probs  # ~95% interval
        lower = mean_probs - 2 * std_probs

        lower = torch.clamp(lower, min=0.0)
        upper = torch.clamp(upper, max=1.0)

        max_lower = lower.max(dim=1, keepdim=True).values
        predicted_set = upper >= max_lower
        set_size = predicted_set.float().sum(dim=1)

        return {
            "mean_probs": mean_probs,
            "std_probs": std_probs,
            "lower": lower,
            "upper": upper,
            "predicted_set": predicted_set,
            "set_size": set_size,
        }


class TemperatureScalingBaseline:
    """Temperature scaling → prediction sets."""

    def __init__(self, temperature: float = 1.0, epsilon: float = 0.1):
        self.temperature = temperature
        self.epsilon = epsilon

    def calibrate(
        self,
        model,
        val_loader,
        device: torch.device,
    ):
        """Find optimal temperature on validation set."""
        from torch.optim import LBFGS

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_logits.append(outputs["logits"])
                all_labels.append(batch["labels"])

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Optimize temperature
        temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = LBFGS([temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = temperature.item()
        print(f"Calibrated temperature: {self.temperature:.3f}")

    def predict(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = F.softmax(logits / self.temperature, dim=-1)

        lower = torch.clamp(probs - self.epsilon, min=0.0)
        upper = torch.clamp(probs + self.epsilon, max=1.0)

        max_lower = lower.max(dim=1, keepdim=True).values
        predicted_set = upper >= max_lower
        set_size = predicted_set.float().sum(dim=1)

        return {
            "probs": probs,
            "lower": lower,
            "upper": upper,
            "predicted_set": predicted_set,
            "set_size": set_size,
        }

"""
Credal Set Construction and Calibration

Maps model predictions + geometric degeneracy ratio → credal predictive sets.

A credal set C_ε replaces point probabilities p_k with intervals:
    [p_k - ε, p_k + ε] ∩ [0, 1]

The imprecision ε is calibrated by the degeneracy ratio ρ:
    ε(ρ) = ε_min + (ε_max - ε_min) · g(ρ)

where g is a monotone calibration function.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CredalPrediction:
    """A single credal predictive set."""
    lower_probs: np.ndarray  # [K] lower bounds on class probabilities
    upper_probs: np.ndarray  # [K] upper bounds on class probabilities
    point_probs: np.ndarray  # [K] point prediction (center)
    epsilon: float           # imprecision parameter
    predicted_set: List[int] # classes consistent with the credal set
    is_determinate: bool     # True if predicted_set has exactly 1 element


class CredalCalibrator:
    """
    Maps degeneracy ratio ρ to credal set imprecision ε.
    
    Supports two calibration functions g(ρ):
        - linear: g(ρ) = ρ
        - sigmoid: g(ρ) = σ(a·(ρ - b))
    """
    
    def __init__(self, config):
        self.epsilon_min = config.credal.epsilon_min
        self.epsilon_max = config.credal.epsilon_max
        self.calibration_type = config.credal.calibration_type
        self.sigmoid_steepness = config.credal.sigmoid_steepness
        self.sigmoid_midpoint = config.credal.sigmoid_midpoint
        self.target_coverage = config.credal.target_coverage
    
    def calibration_function(self, rho: float) -> float:
        """
        Monotone calibration function g: [0,1] → [0,1].
        Maps degeneracy ratio to normalized imprecision.
        """
        rho = np.clip(rho, 0.0, 1.0)
        
        if self.calibration_type == "linear":
            return rho
        elif self.calibration_type == "sigmoid":
            # g(ρ) = σ(a·(ρ - b))
            return 1.0 / (1.0 + np.exp(
                -self.sigmoid_steepness * (rho - self.sigmoid_midpoint)
            ))
        else:
            raise ValueError(f"Unknown calibration type: {self.calibration_type}")
    
    def rho_to_epsilon(self, rho: float) -> float:
        """Map degeneracy ratio ρ to imprecision parameter ε."""
        g = self.calibration_function(rho)
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * g
    
    def calibrate_on_validation(
        self,
        point_probs: np.ndarray,
        true_labels: np.ndarray,
        rho: float,
    ) -> Dict[str, float]:
        """
        Calibrate ε_min and ε_max on validation data to achieve target coverage.
        
        Uses binary search to find ε_max such that coverage ≈ target_coverage
        when ρ maps to the full ε range.
        
        Args:
            point_probs: [N, C] predicted probabilities
            true_labels: [N] true class labels
            rho: degeneracy ratio from geometric probe
        
        Returns:
            Dict with calibrated parameters and achieved coverage
        """
        # Binary search for epsilon_max
        lo, hi = 0.01, 0.5
        target = self.target_coverage
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            self.epsilon_max = mid
            
            eps = self.rho_to_epsilon(rho)
            coverage = self._compute_coverage(point_probs, true_labels, eps)
            
            if coverage < target:
                lo = mid
            else:
                hi = mid
        
        self.epsilon_max = (lo + hi) / 2.0
        final_eps = self.rho_to_epsilon(rho)
        final_coverage = self._compute_coverage(point_probs, true_labels, final_eps)
        
        return {
            "epsilon_min": self.epsilon_min,
            "epsilon_max": self.epsilon_max,
            "calibrated_epsilon": final_eps,
            "achieved_coverage": final_coverage,
            "target_coverage": target,
            "rho": rho,
        }
    
    def _compute_coverage(
        self,
        point_probs: np.ndarray,
        true_labels: np.ndarray,
        epsilon: float,
    ) -> float:
        """Compute coverage: fraction of true labels in predicted credal sets."""
        N = len(true_labels)
        covered = 0
        for i in range(N):
            pred = self.predict_credal(point_probs[i], epsilon)
            if true_labels[i] in pred.predicted_set:
                covered += 1
        return covered / N
    
    def predict_credal(
        self,
        point_probs: np.ndarray,
        epsilon: float,
    ) -> CredalPrediction:
        """
        Construct a credal prediction from point probabilities and imprecision.
        
        Args:
            point_probs: [C] point probability prediction
            epsilon: imprecision parameter
        
        Returns:
            CredalPrediction with interval probabilities and predicted set
        """
        C = len(point_probs)
        
        # Interval probabilities
        lower = np.clip(point_probs - epsilon, 0.0, 1.0)
        upper = np.clip(point_probs + epsilon, 0.0, 1.0)
        
        # Predicted set: classes that could be argmax under some
        # distribution in the credal set
        # A class k is in the set if upper[k] >= max(lower[j] for j != k)
        predicted_set = []
        for k in range(C):
            # Can k be the argmax? Yes if its upper bound exceeds
            # all other classes' lower bounds
            others_max_lower = max(
                lower[j] for j in range(C) if j != k
            ) if C > 1 else 0.0
            if upper[k] >= others_max_lower:
                predicted_set.append(k)
        
        # If empty (shouldn't happen), include argmax
        if len(predicted_set) == 0:
            predicted_set = [int(np.argmax(point_probs))]
        
        return CredalPrediction(
            lower_probs=lower,
            upper_probs=upper,
            point_probs=point_probs,
            epsilon=epsilon,
            predicted_set=predicted_set,
            is_determinate=(len(predicted_set) == 1),
        )
    
    def predict_batch(
        self,
        point_probs: np.ndarray,
        rho: float,
    ) -> List[CredalPrediction]:
        """Predict credal sets for a batch of samples."""
        epsilon = self.rho_to_epsilon(rho)
        return [
            self.predict_credal(point_probs[i], epsilon)
            for i in range(len(point_probs))
        ]
    
    def evaluate(
        self,
        point_probs: np.ndarray,
        true_labels: np.ndarray,
        rho: float,
    ) -> Dict[str, float]:
        """
        Evaluate credal predictions comprehensively.
        
        Returns standard credal set metrics:
            - set_accuracy: fraction where true label ∈ predicted set
            - mean_set_size: average |predicted set|
            - determinacy: fraction of single-class predictions
            - avg_epsilon: the imprecision parameter used
        """
        epsilon = self.rho_to_epsilon(rho)
        predictions = self.predict_batch(point_probs, rho)
        
        N = len(predictions)
        
        # Coverage / set accuracy
        covered = sum(
            1 for pred, y in zip(predictions, true_labels)
            if y in pred.predicted_set
        )
        set_accuracy = covered / N
        
        # Mean set size
        mean_set_size = np.mean([len(p.predicted_set) for p in predictions])
        
        # Determinacy
        determinacy = np.mean([1.0 if p.is_determinate else 0.0 for p in predictions])
        
        # Point accuracy (for comparison)
        point_preds = np.argmax(point_probs, axis=1)
        point_accuracy = np.mean(point_preds == true_labels)
        
        return {
            "set_accuracy": float(set_accuracy),
            "mean_set_size": float(mean_set_size),
            "determinacy": float(determinacy),
            "point_accuracy": float(point_accuracy),
            "epsilon": float(epsilon),
            "rho": float(rho),
            "n_samples": N,
        }


class BaselineCredalMethods:
    """
    Baseline credal set methods for comparison.
    """
    
    @staticmethod
    def fixed_epsilon(
        point_probs: np.ndarray,
        true_labels: np.ndarray,
        epsilon: float,
    ) -> Dict[str, float]:
        """Fixed-width credal sets (no geometry)."""
        calibrator = CredalCalibrator.__new__(CredalCalibrator)
        calibrator.epsilon_min = epsilon
        calibrator.epsilon_max = epsilon
        calibrator.calibration_type = "linear"
        predictions = [
            calibrator.predict_credal(point_probs[i], epsilon)
            for i in range(len(point_probs))
        ]
        covered = sum(
            1 for p, y in zip(predictions, true_labels)
            if y in p.predicted_set
        )
        return {
            "set_accuracy": covered / len(true_labels),
            "mean_set_size": np.mean([len(p.predicted_set) for p in predictions]),
            "determinacy": np.mean([1.0 if p.is_determinate else 0.0 for p in predictions]),
            "epsilon": epsilon,
            "method": "fixed_epsilon",
        }
    
    @staticmethod
    def mc_dropout_credal(
        model,
        dataloader,
        device,
        num_samples: int = 20,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty → credal sets via prediction interval.
        
        Returns:
            mean_probs, labels, std_probs
        """
        model.train()  # enable dropout
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]
                
                # Multiple forward passes
                batch_probs = []
                for _ in range(num_samples):
                    outputs = model(input_ids, attention_mask)
                    probs = F.softmax(outputs["logits"], dim=-1)
                    batch_probs.append(probs.cpu().numpy())
                
                batch_probs = np.stack(batch_probs)  # [S, B, C]
                all_probs.append(batch_probs)
                all_labels.append(labels.numpy())
        
        model.eval()
        
        all_probs = np.concatenate(all_probs, axis=1)  # [S, N, C]
        all_labels = np.concatenate(all_labels)  # [N]
        
        mean_probs = all_probs.mean(axis=0)  # [N, C]
        std_probs = all_probs.std(axis=0)  # [N, C]
        
        return mean_probs, all_labels, std_probs

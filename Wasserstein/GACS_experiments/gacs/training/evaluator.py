"""
GACS Evaluation Pipeline

Runs the full evaluation:
  1. Train model
  2. Run geometric probes
  3. Calibrate credal sets
  4. Evaluate under distribution shift
  5. Compare with baselines
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from collections import defaultdict


class GACSEvaluator:
    """
    Full evaluation pipeline for GACS.
    
    Computes and compares:
      - GACS (geometry-adapted credal sets)
      - Fixed-ε credal sets (no geometry)
      - MC Dropout
      - Temperature scaling
      - Raw softmax (point predictions)
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.results = {}
    
    @torch.no_grad()
    def collect_predictions(
        self,
        dataloader,
        label: str = "test",
    ) -> Dict[str, np.ndarray]:
        """Collect model predictions on a dataset."""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        all_concepts = []
        all_concept_labels = []
        all_mu = []
        all_logvar = []
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            probs = F.softmax(outputs["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
            all_concepts.append(outputs["concept_probs"].cpu().numpy())
            all_concept_labels.append(batch["concepts"].numpy())
            all_mu.append(outputs["mu"].cpu().numpy())
            all_logvar.append(outputs["logvar"].cpu().numpy())
        
        result = {
            "probs": np.concatenate(all_probs),
            "labels": np.concatenate(all_labels),
            "concepts": np.concatenate(all_concepts),
            "concept_labels": np.concatenate(all_concept_labels),
            "mu": np.concatenate(all_mu),
            "logvar": np.concatenate(all_logvar),
        }
        
        print(f"[{label}] Collected {len(result['labels'])} predictions, "
              f"point accuracy={np.mean(np.argmax(result['probs'], axis=1) == result['labels']):.4f}")
        
        return result
    
    def run_geometric_probe(
        self,
        probe,
        dataloader,
        loss_fn,
    ) -> Dict:
        """Run the geometric probe and return degeneracy ratio."""
        from gacs.training.losses import GACSLoss
        
        # We need a wrapper that matches the probe's expected interface
        def probe_loss_fn(outputs, labels, concepts, epoch):
            return loss_fn.compute(outputs, labels, concepts, epoch)
        
        result = probe.compute_degeneracy_ratio(
            model=self.model,
            dataloader=dataloader,
            loss_fn=probe_loss_fn,
            device=self.device,
        )
        
        print(f"Degeneracy ratio ρ = {result['degeneracy_ratio']:.4f}")
        return result
    
    def evaluate_gacs(
        self,
        predictions: Dict[str, np.ndarray],
        rho: float,
        calibrator,
        label: str = "test",
    ) -> Dict[str, float]:
        """Evaluate GACS credal predictions."""
        metrics = calibrator.evaluate(
            point_probs=predictions["probs"],
            true_labels=predictions["labels"],
            rho=rho,
        )
        metrics["method"] = "GACS"
        metrics["label"] = label
        return metrics
    
    def evaluate_fixed_credal(
        self,
        predictions: Dict[str, np.ndarray],
        epsilon: float,
        label: str = "test",
    ) -> Dict[str, float]:
        """Evaluate fixed-ε credal sets (no geometry)."""
        from gacs.credal.calibration import BaselineCredalMethods
        metrics = BaselineCredalMethods.fixed_epsilon(
            predictions["probs"],
            predictions["labels"],
            epsilon=epsilon,
        )
        metrics["label"] = label
        return metrics
    
    def evaluate_mc_dropout(
        self,
        dataloader,
        num_samples: int = 20,
        label: str = "test",
    ) -> Dict[str, float]:
        """Evaluate MC Dropout uncertainty."""
        self.model.train()  # enable dropout
        
        all_sample_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                batch_probs = []
                for _ in range(num_samples):
                    outputs = self.model(input_ids, attention_mask)
                    probs = F.softmax(outputs["logits"], dim=-1).cpu().numpy()
                    batch_probs.append(probs)
                
                all_sample_probs.append(np.stack(batch_probs))  # [S, B, C]
                all_labels.append(batch["label"].numpy())
        
        self.model.eval()
        
        sample_probs = np.concatenate(all_sample_probs, axis=1)  # [S, N, C]
        labels = np.concatenate(all_labels)
        
        mean_probs = sample_probs.mean(axis=0)  # [N, C]
        std_probs = sample_probs.std(axis=0)    # [N, C]
        
        # Point accuracy from mean
        point_acc = np.mean(np.argmax(mean_probs, axis=1) == labels)
        
        # Convert MC uncertainty to credal-like sets:
        # Class k in set if mean_prob[k] + 2*std[k] > max(mean_prob[j] - 2*std[j])
        set_sizes = []
        covered = 0
        determinate = 0
        
        for i in range(len(labels)):
            pred_set = []
            for k in range(mean_probs.shape[1]):
                upper_k = mean_probs[i, k] + 2 * std_probs[i, k]
                others_lower = [
                    mean_probs[i, j] - 2 * std_probs[i, j]
                    for j in range(mean_probs.shape[1]) if j != k
                ]
                if upper_k >= max(others_lower) if others_lower else True:
                    pred_set.append(k)
            
            if not pred_set:
                pred_set = [int(np.argmax(mean_probs[i]))]
            
            set_sizes.append(len(pred_set))
            if labels[i] in pred_set:
                covered += 1
            if len(pred_set) == 1:
                determinate += 1
        
        return {
            "method": "MC_Dropout",
            "label": label,
            "set_accuracy": covered / len(labels),
            "mean_set_size": np.mean(set_sizes),
            "determinacy": determinate / len(labels),
            "point_accuracy": point_acc,
            "num_samples": num_samples,
        }
    
    def evaluate_temperature_scaling(
        self,
        val_predictions: Dict[str, np.ndarray],
        test_predictions: Dict[str, np.ndarray],
        label: str = "test",
    ) -> Dict[str, float]:
        """
        Temperature scaling: calibrate T on val, apply to test.
        Then convert calibrated probabilities to credal-like sets
        using entropy-based thresholding.
        """
        # Optimize temperature on validation logits
        # Since we have probs, convert back to logits
        val_logits = np.log(np.clip(val_predictions["probs"], 1e-10, 1.0))
        val_labels = val_predictions["labels"]
        
        # Grid search for best temperature
        best_T = 1.0
        best_ece = float("inf")
        
        for T in np.arange(0.1, 5.0, 0.1):
            scaled_probs = self._softmax_with_temp(val_logits, T)
            ece = self._compute_ece(scaled_probs, val_labels)
            if ece < best_ece:
                best_ece = ece
                best_T = T
        
        # Apply to test
        test_logits = np.log(np.clip(test_predictions["probs"], 1e-10, 1.0))
        test_labels = test_predictions["labels"]
        calibrated_probs = self._softmax_with_temp(test_logits, best_T)
        
        # Point accuracy
        point_acc = np.mean(np.argmax(calibrated_probs, axis=1) == test_labels)
        
        # Convert to prediction sets using entropy threshold
        # Higher entropy → larger set
        entropies = -np.sum(
            calibrated_probs * np.log(np.clip(calibrated_probs, 1e-10, 1.0)),
            axis=1
        )
        max_entropy = np.log(calibrated_probs.shape[1])
        normalized_entropy = entropies / max_entropy
        
        set_sizes = []
        covered = 0
        determinate = 0
        
        for i in range(len(test_labels)):
            # Include classes with prob > threshold
            # threshold decreases with entropy (more uncertain → more classes)
            threshold = max(0.05, 0.5 * (1 - normalized_entropy[i]))
            pred_set = [k for k in range(calibrated_probs.shape[1])
                       if calibrated_probs[i, k] > threshold]
            
            if not pred_set:
                pred_set = [int(np.argmax(calibrated_probs[i]))]
            
            set_sizes.append(len(pred_set))
            if test_labels[i] in pred_set:
                covered += 1
            if len(pred_set) == 1:
                determinate += 1
        
        return {
            "method": "Temp_Scaling",
            "label": label,
            "set_accuracy": covered / len(test_labels),
            "mean_set_size": np.mean(set_sizes),
            "determinacy": determinate / len(test_labels),
            "point_accuracy": point_acc,
            "temperature": best_T,
            "ece": best_ece,
        }
    
    def _softmax_with_temp(self, logits: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled = logits / T
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    
    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
        
        return ece / len(labels)
    
    def run_full_evaluation(
        self,
        probe,
        loss_fn,
        calibrator,
        val_loader,
        test_loader,
        shift_loader=None,
        shift_label: str = "shift",
    ) -> Dict:
        """
        Run the complete evaluation pipeline.
        
        Returns a dict with all results for all methods.
        """
        results = {"methods": {}, "meta": {}}
        
        # 1. Collect predictions
        print("\n--- Collecting predictions ---")
        val_preds = self.collect_predictions(val_loader, "val")
        test_preds = self.collect_predictions(test_loader, "test")
        
        shift_preds = None
        if shift_loader is not None:
            shift_preds = self.collect_predictions(shift_loader, shift_label)
        
        # 2. Run geometric probe
        print("\n--- Running geometric probe ---")
        probe_result = self.run_geometric_probe(probe, val_loader, loss_fn)
        rho = probe_result["degeneracy_ratio"]
        results["meta"]["probe"] = probe_result
        
        # 3. Calibrate credal sets on validation
        print("\n--- Calibrating credal sets ---")
        cal_result = calibrator.calibrate_on_validation(
            val_preds["probs"], val_preds["labels"], rho
        )
        results["meta"]["calibration"] = cal_result
        print(f"Calibrated: ε_max={cal_result['epsilon_max']:.4f}, "
              f"coverage={cal_result['achieved_coverage']:.4f}")
        
        # 4. Evaluate all methods
        print("\n--- Evaluating methods ---")
        
        datasets = {"test": test_preds}
        loaders = {"test": test_loader}
        if shift_preds is not None:
            datasets[shift_label] = shift_preds
            loaders[shift_label] = shift_loader
        
        for ds_name, preds in datasets.items():
            results["methods"][ds_name] = {}
            
            # GACS
            gacs_metrics = self.evaluate_gacs(preds, rho, calibrator, ds_name)
            results["methods"][ds_name]["GACS"] = gacs_metrics
            
            # Fixed credal (use same ε as GACS for fair comparison)
            fixed_metrics = self.evaluate_fixed_credal(
                preds, epsilon=cal_result["calibrated_epsilon"], label=ds_name
            )
            results["methods"][ds_name]["Fixed_Credal"] = fixed_metrics
            
            # MC Dropout
            mc_metrics = self.evaluate_mc_dropout(
                loaders[ds_name], num_samples=20, label=ds_name
            )
            results["methods"][ds_name]["MC_Dropout"] = mc_metrics
            
            # Temperature scaling
            ts_metrics = self.evaluate_temperature_scaling(
                val_preds, preds, label=ds_name
            )
            results["methods"][ds_name]["Temp_Scaling"] = ts_metrics
            
            # Raw softmax (point predictions as degenerate credal sets)
            raw_acc = np.mean(np.argmax(preds["probs"], axis=1) == preds["labels"])
            results["methods"][ds_name]["Softmax"] = {
                "method": "Softmax",
                "label": ds_name,
                "set_accuracy": raw_acc,  # = point accuracy for size-1 sets
                "mean_set_size": 1.0,
                "determinacy": 1.0,
                "point_accuracy": raw_acc,
            }
        
        # Print comparison table
        self._print_results_table(results)
        
        return results
    
    def _print_results_table(self, results: Dict):
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON")
        print("=" * 80)
        
        methods = ["Softmax", "Temp_Scaling", "MC_Dropout", "Fixed_Credal", "GACS"]
        
        for ds_name, ds_results in results["methods"].items():
            print(f"\n--- {ds_name.upper()} ---")
            print(f"{'Method':<16} {'Set Acc':>10} {'Set Size':>10} {'Determ.':>10} {'Point Acc':>10}")
            print("-" * 56)
            
            for method in methods:
                if method in ds_results:
                    m = ds_results[method]
                    print(
                        f"{method:<16} "
                        f"{m['set_accuracy']:>10.4f} "
                        f"{m['mean_set_size']:>10.2f} "
                        f"{m.get('determinacy', 1.0):>10.4f} "
                        f"{m['point_accuracy']:>10.4f}"
                    )
        
        print("\n" + "=" * 80)
    
    def save_results(self, results: Dict, path: str):
        """Save results to JSON."""
        # Convert numpy types to python types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)
        print(f"Results saved to {path}")

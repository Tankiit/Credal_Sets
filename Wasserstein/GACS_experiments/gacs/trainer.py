"""
GACS Trainer
=============
Full training loop with:
- Multi-component loss (3 desiderata)
- Periodic geometric probing
- Credal set evaluation on val/test/shift splits
- Baseline comparisons
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Dict, Optional
import json
import time
from pathlib import Path
import numpy as np

from gacs.models.vae import GACSModel
from gacs.losses.gacs_loss import GACSLoss
from gacs.probes.geometric import PerturbationProbe, compute_geometric_probe
from gacs.credal.credal_sets import (
    CredalSetConstructor,
    CredalEvaluator,
    FixedCredalBaseline,
    MCDropoutBaseline,
    TemperatureScalingBaseline,
)


class GACSTrainer:
    """End-to-end trainer for GACS experiments."""

    def __init__(self, config, model, dataloaders, tokenizer):
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.tokenizer = tokenizer
        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Loss
        self.loss_fn = GACSLoss(config)

        # Credal
        self.credal_constructor = CredalSetConstructor(config)
        self.credal_evaluator = CredalEvaluator()

        # Optimizer: separate LR for encoder vs. heads
        encoder_params = list(model.encoder.parameters())
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]

        self.optimizer = AdamW([
            {"params": encoder_params, "lr": config.training.encoder_lr},
            {"params": head_params, "lr": config.training.head_lr},
        ], weight_decay=config.training.weight_decay)

        # Scheduler
        total_steps = len(dataloaders["train"]) * config.training.epochs
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-7
            )
        else:
            self.scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01,
                total_iters=total_steps
            )

        # Tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train": [], "val": [], "probes": [], "credal": []}
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"GACS Training — {self.config.data.name}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Probeable ({self.config.probe.probe_scope}): {self.model.num_probeable_parameters(self.config.probe.probe_scope):,}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.training.epochs):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)
            self.history["train"].append(train_metrics)

            # Validate
            val_metrics = self._evaluate("val", epoch)
            self.history["val"].append(val_metrics)

            elapsed = time.time() - t0

            # Print
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['total_loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"kl_w={train_metrics['kl_weight']:.3f} | "
                f"{elapsed:.1f}s"
            )

            # Save checkpoint for every epoch
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_acc': val_metrics["accuracy"],
                'val_loss': val_metrics["total_loss"],
                'train_acc': train_metrics["accuracy"],
                'train_loss': train_metrics["total_loss"],
            }, checkpoint_path)

            # Save best model separately
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                if self.config.training.save_best:
                    torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model for final evaluation
        if self.config.training.save_best and (self.output_dir / "best_model.pt").exists():
            self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))
            print(f"\nLoaded best model (val_acc={self.best_val_acc:.4f})")

    def run_geometric_probes(self):
        """Run geometric probes after training."""
        print(f"\n{'='*60}")
        print("Running Geometric Probes")
        print(f"{'='*60}\n")

        def probe_loss_fn(outputs, batch):
            """Scalar loss for probing."""
            loss_dict = self.loss_fn.compute(outputs, batch, epoch=self.config.training.epochs)
            return loss_dict["total_loss"]

        # Probe on training data
        probe_results = compute_geometric_probe(
            self.model,
            probe_loss_fn,
            self.dataloaders["train"],
            self.config,
            self.device,
        )

        self.history["probes"].append({"split": "train", **probe_results})

        print(f"Degeneracy ratio ρ = {probe_results['rho']:.4f}")
        print(f"  Mean Δ loss = {probe_results.get('mean_delta', 0):.6f}")
        print(f"  Insensitive directions: {probe_results.get('num_insensitive', 0)}/{probe_results.get('num_directions', 0)}")

        return probe_results

    def run_credal_evaluation(self, rho: float):
        """
        Evaluate credal sets on test and shifted data.

        Compares:
            1. GACS (geometry-adapted ε)
            2. Fixed credal sets (constant ε)
            3. MC Dropout prediction sets
            4. Temperature scaling prediction sets
        """
        print(f"\n{'='*60}")
        print(f"Credal Set Evaluation (ρ = {rho:.4f})")
        print(f"{'='*60}\n")

        epsilon_gacs = self.credal_constructor.rho_to_epsilon(rho)
        print(f"GACS ε = {epsilon_gacs:.4f}")

        results = {}

        # Baselines
        fixed_baseline = FixedCredalBaseline(epsilon=0.1)
        mc_baseline = MCDropoutBaseline(num_samples=10)
        temp_baseline = TemperatureScalingBaseline()

        # Calibrate temperature on val
        if "val" in self.dataloaders:
            temp_baseline.calibrate(self.model, self.dataloaders["val"], self.device)

        # Evaluate on each split
        for split_name in ["test", "shift"]:
            if split_name not in self.dataloaders:
                continue

            print(f"\n--- {split_name.upper()} ---")
            loader = self.dataloaders[split_name]

            split_results = self._evaluate_credal_on_split(
                loader, epsilon_gacs, fixed_baseline, mc_baseline, temp_baseline
            )
            results[split_name] = split_results

            for method, metrics in split_results.items():
                print(
                    f"  {method:20s} | "
                    f"coverage={metrics['coverage']:.3f}  "
                    f"set_size={metrics['mean_set_size']:.2f}  "
                    f"determinacy={metrics['determinacy']:.3f}"
                )

        self.history["credal"].append({"rho": rho, "epsilon": epsilon_gacs, **results})

        return results

    def _evaluate_credal_on_split(
        self, loader, epsilon_gacs, fixed_baseline, mc_baseline, temp_baseline
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all methods on a single data split."""
        self.model.eval()

        all_probs = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                all_probs.append(F.softmax(outputs["logits"], dim=-1))
                all_logits.append(outputs["logits"])
                all_labels.append(batch["labels"])

        probs = torch.cat(all_probs, dim=0)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        results = {}

        # 1. GACS
        gacs_credal = self.credal_constructor.construct_credal_set(probs, epsilon_gacs)
        results["GACS"] = self.credal_evaluator.evaluate(
            gacs_credal["predicted_set"], labels
        )

        # 2. Fixed credal
        fixed_credal = fixed_baseline.predict(probs)
        results["Fixed_credal"] = self.credal_evaluator.evaluate(
            fixed_credal["predicted_set"], labels
        )

        # 3. Temperature scaling
        temp_credal = temp_baseline.predict(logits)
        results["Temp_scaling"] = self.credal_evaluator.evaluate(
            temp_credal["predicted_set"], labels
        )

        # 4. Softmax argmax (point prediction, set_size = 1 always)
        argmax_set = F.one_hot(probs.argmax(dim=1), num_classes=probs.size(1)).bool()
        results["Softmax"] = self.credal_evaluator.evaluate(argmax_set, labels)

        # 5. MC Dropout
        # (skip in this batch evaluation — requires model forward passes per batch)
        # We'll compute it separately if needed

        return results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_sum = {}
        correct = 0
        total = 0

        for batch in self.dataloaders["train"]:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss_dict = self.loss_fn.compute(outputs, batch, epoch)
            loss = loss_dict["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.loss.grad_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            for k, v in loss_dict.items():
                if k != "total_loss":
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            metrics_sum["total_loss"] = metrics_sum.get("total_loss", 0.0) + loss.item()

            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        n_batches = len(self.dataloaders["train"])
        metrics = {k: v / n_batches for k, v in metrics_sum.items()}
        metrics["accuracy"] = correct / max(total, 1)
        return metrics

    @torch.no_grad()
    def _evaluate(self, split: str, epoch: int) -> Dict[str, float]:
        """Evaluate on a data split."""
        self.model.eval()
        metrics_sum = {}
        correct = 0
        total = 0

        loader = self.dataloaders.get(split)
        if loader is None:
            return {"total_loss": 0, "accuracy": 0}

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss_dict = self.loss_fn.compute(outputs, batch, epoch)

            for k, v in loss_dict.items():
                if k != "total_loss":
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            metrics_sum["total_loss"] = metrics_sum.get("total_loss", 0.0) + loss_dict["total_loss"].item()

            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        n_batches = len(loader)
        metrics = {k: v / n_batches for k, v in metrics_sum.items()}
        metrics["accuracy"] = correct / max(total, 1)
        return metrics

    def save_results(self):
        """Save all experiment results to JSON."""
        results = {
            "config": {
                "dataset": self.config.data.name,
                "z_dim": self.config.model.z_dim,
                "concept_dim": self.config.model.concept_dim,
                "num_classes": self.config.data.num_classes,
                "epochs": self.config.training.epochs,
                "probe_scope": self.config.probe.probe_scope,
                "num_directions": self.config.probe.num_directions,
            },
            "history": self.history,
            "best_val_acc": self.best_val_acc,
        }

        out_path = self.output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

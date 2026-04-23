"""
HybridCredalCBM trainer.

Single-model, single-loader trainer. Assumes the dataset provides a
DatasetBundle (see loaders/interface.py) and the model is a HybridCredalCBM
that returns a dict with keys: predictions, loss, sigma_epi, sigma_ale,
epistemic, aleatoric, and the separable loss components (task_loss,
concept_bce, error_supervision, credal_kl, aleatoric_loss, aleatoric_unknown,
orth_penalty).
"""
from __future__ import annotations

import json
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loaders.interface import DatasetBundle
from training.logging_utils import ExperimentLogger
from training.grad_iso_mixin import GradientIsolationMixin
from training.metrics import UncertaintyMetrics, compute_uncertainty_correlations


class HybridCredalCBMTrainer:
    """Plain trainer. Inherit with GradientIsolationMixin for probe logging."""

    def __init__(
        self,
        model,                  # HybridCredalCBM
        config,                 # HybridCredalConfig
        bundle: DatasetBundle,
        device: str = "auto",
        save_dir: str | Path = "./checkpoints",
        log_dir: str | Path | None = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_mode: str = "offline",
    ):
        self.model = model
        self.config = config
        self.bundle = bundle
        self.device = self._resolve_device(device)
        self.model.to(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir) if log_dir is not None else self.save_dir / "logs"
        self.logger = ExperimentLogger(
            log_dir=self.log_dir,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_mode=wandb_mode,
        )

        self.best_val_acc = 0.0
        self.current_epoch = 0
        self._forward_arg_names = set(inspect.signature(self.model.forward).parameters)
        self._forward_arg_names.discard("self")

    def _log_metrics(self, metrics: Dict, step: int, prefix: str = "") -> None:
        if getattr(self, "logger", None) is not None and self.logger.enabled:
            self.logger.log_metrics(metrics, step=step, prefix=prefix)

    def close(self) -> None:
        if getattr(self, "logger", None) is not None:
            self.logger.close()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ----------------------------------------------------------------------
    def train_epoch(self, train_loader, optimizer, scheduler=None) -> Dict:
        """One training epoch. Override via mixin for probe logging."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch in pbar:
            inputs = self._batch_to_device(batch)
            outputs = self.model(**self._forward_inputs(inputs))
            loss = outputs["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += float(loss)
            preds = outputs["predictions"].detach()
            labels = inputs["labels"].detach()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            pbar.set_postfix({"loss": float(loss)})

        return {
            "loss": total_loss / max(1, len(train_loader)),
            "accuracy": float(correct / max(1, total)),
        }

    # ----------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self, loader) -> UncertaintyMetrics:
        """Full validation pass with correlation statistics."""
        self.model.eval()

        all_preds, all_labels = [], []
        all_sigma_epi, all_sigma_ale = [], []
        all_eu, all_au = [], []
        all_entropy = []
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {self.current_epoch} [Val]"):
            inputs = self._batch_to_device(batch)
            outputs = self.model(**self._forward_inputs(inputs))

            total_loss += float(outputs["loss"])
            all_preds.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(inputs["labels"].cpu().numpy())
            all_sigma_epi.append(outputs["sigma_epi"].cpu().numpy())
            all_sigma_ale.append(outputs["aleatoric"].cpu().numpy())
            all_eu.append(outputs["epistemic"].cpu().numpy())
            all_au.append(outputs["aleatoric"].cpu().numpy())
            if "annotator_entropy" in inputs:
                all_entropy.append(inputs["annotator_entropy"].cpu().numpy())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        sigma_epi_arr = np.concatenate(all_sigma_epi, axis=0)
        sigma_ale_arr = np.concatenate(all_sigma_ale, axis=0)
        eu_arr = np.concatenate(all_eu, axis=0)
        au_arr = np.concatenate(all_au, axis=0)
        entropy_arr = np.concatenate(all_entropy, axis=0) if all_entropy else None
        errors = (preds_arr != labels_arr).astype(float)

        corr = compute_uncertainty_correlations(eu_arr, au_arr, errors, entropy_arr)

        eu_sample = eu_arr.mean(axis=-1) if eu_arr.ndim > 1 else eu_arr
        au_sample = au_arr.mean(axis=-1) if au_arr.ndim > 1 else au_arr

        m = UncertaintyMetrics(
            accuracy=float((preds_arr == labels_arr).mean()),
            loss=total_loss / max(1, len(loader)),
            mean_sigma_epi=float(sigma_epi_arr.mean()),
            std_sigma_epi=float(sigma_epi_arr.std()),
            mean_sigma_ale=float(sigma_ale_arr.mean()),
            std_sigma_ale=float(sigma_ale_arr.std()),
            mean_eu=float(eu_sample.mean()),
            std_eu=float(eu_sample.std()),
            mean_au=float(au_sample.mean()),
            std_au=float(au_sample.std()),
            rho_eu_au=corr["rho_eu_au"], p_eu_au=corr["p_eu_au"],
            rho_eu_error=corr["rho_eu_error"], p_eu_error=corr["p_eu_error"],
            rho_ale_entropy=corr["rho_ale_entropy"], p_ale_entropy=corr["p_ale_entropy"],
        )
        return m

    # ----------------------------------------------------------------------
    def fit(
        self,
        num_epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        save_every: int = 5,
    ) -> Dict:
        """Full training loop. Saves best by val accuracy."""
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        total_steps = len(self.bundle.train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        history = []
        best_metrics: Optional[UncertaintyMetrics] = None
        start = time.time()

        print(f"\n{'=' * 60}")
        print(f"Training {self.bundle.name} — {num_epochs} epochs, lr={lr:.0e}")
        print(f"Device: {self.device}, save_dir: {self.save_dir}")
        if self.logger.enabled:
            print(f"Logging to: {self.log_dir}")
        print(f"{'=' * 60}")

        self.logger.log_hyperparams({
            "dataset": self.bundle.name,
            "device": str(self.device),
            "save_dir": str(self.save_dir),
            "log_dir": str(self.log_dir),
            "config": self.config,
            "training": {
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "save_every": save_every,
            },
        })

        self._log_metrics(
            {
                "dataset/train_size": self.bundle.train_size,
                "dataset/val_size": self.bundle.val_size,
                "dataset/test_size": self.bundle.test_size,
                "dataset/num_concepts": self.bundle.num_concepts,
                "dataset/num_classes": self.bundle.num_classes,
                "optimizer/lr": lr,
                "optimizer/weight_decay": weight_decay,
                "scheduler/warmup_steps": warmup_steps,
                "training/num_epochs": num_epochs,
            },
            step=0,
        )

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(self.bundle.train_loader, optimizer, scheduler)
            val_metrics = self.evaluate(self.bundle.val_loader)

            print(f"\nEpoch {epoch}: "
                  f"train_loss={train_metrics['loss']:.4f}  "
                  f"val_acc={val_metrics.accuracy:.4f}  "
                  f"ρ(EU,AU)={val_metrics.rho_eu_au:+.3f}  "
                  f"ρ(σ_epi,err)={val_metrics.rho_eu_error:+.3f}  "
                  f"ρ(σ_ale,H)={val_metrics.rho_ale_entropy:+.3f}")

            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics.to_dict(),
                "timestamp": datetime.now().isoformat(),
            })

            self._log_metrics({"train": train_metrics, "val": val_metrics.to_dict()}, step=epoch)

            if val_metrics.accuracy > self.best_val_acc:
                self.best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "metrics": val_metrics.to_dict(),
                    "dataset": self.bundle.name,
                }, self.save_dir / "best_model.pt")
                print(f"  ✓ new best (acc={val_metrics.accuracy:.4f})")
                self._log_metrics({"val_accuracy": self.best_val_acc}, step=epoch, prefix="best/")

            if epoch % save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }, self.save_dir / f"checkpoint_epoch_{epoch}.pt")

        duration = time.time() - start
        print(f"\n{'=' * 60}")
        print(f"Done. Best val acc: {self.best_val_acc:.4f}  ({duration/60:.1f} min)")
        print(f"{'=' * 60}")

        with (self.save_dir / "training_history.json").open("w") as f:
            json.dump({
                "dataset": self.bundle.name,
                "history": history,
                "best_val_accuracy": self.best_val_acc,
            }, f, indent=2, default=float)

        self._log_metrics({"best_val_accuracy": self.best_val_acc, "duration_seconds": duration}, step=num_epochs, prefix="summary/")

        return {"history": history, "best_metrics": best_metrics}

    # ----------------------------------------------------------------------
    def _batch_to_device(self, batch: dict) -> dict:
        """Move batch to device, preserving optional keys."""
        out = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }
        if "concept_labels" in batch and batch["concept_labels"] is not None:
            out["concept_labels"] = batch["concept_labels"].to(self.device)
        if "annotator_entropy" in batch and batch["annotator_entropy"] is not None:
            out["annotator_entropy"] = batch["annotator_entropy"].to(self.device)
        return out

    def _forward_inputs(self, batch: dict) -> dict:
        """Filter device-moved batch down to the model's forward signature."""
        return {key: value for key, value in batch.items() if key in self._forward_arg_names}


# ----------------------------------------------------------------------
class InstrumentedTrainer(GradientIsolationMixin, HybridCredalCBMTrainer):
    """
    Same trainer, but overrides train_epoch with the three-phase gradient
    probe. Writes grad_iso_log.jsonl in save_dir at every step.
    """

    def train_epoch(self, train_loader, optimizer, scheduler=None) -> Dict:
        self._grad_iso_setup()
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {self.current_epoch} [Train+GradIso]")
        for batch_idx, batch in enumerate(pbar):
            inputs = self._batch_to_device(batch)
            model_inputs = self._forward_inputs(inputs)

            # One forward; reuse graph across three backwards.
            outputs = self.model(**model_inputs)

            # Probe (phases 1 and 2)
            rec = self._probe_step(outputs, self.current_epoch, batch_idx)
            if rec is not None:
                self._persist_record(rec)
                self._log_metrics({"grad_iso": rec}, step=self._grad_iso_global_step, prefix="grad_iso/")

            # Phase 3: combined backward + optimizer step
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            self._grad_iso_global_step += 1
            total_loss += float(outputs["loss"])
            preds = outputs["predictions"].detach()
            labels = model_inputs["labels"].detach()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            pbar.set_postfix({"loss": float(outputs["loss"])})

        return {
            "loss": total_loss / max(1, len(train_loader)),
            "accuracy": float(correct / max(1, total)),
        }

"""
gacs/training/shared_trainer.py
--------------------------------
Shared training loop for both VisionGACSModel (MedMNIST)
and GACSModel (CEBaB/text).

The two models differ in:
  - Input format: image tensor vs (input_ids, attention_mask)
  - Reconstruction loss: BCE on pixels vs contrastive on embeddings
  - Factor supervision: unsupervised (MedMNIST) vs supervised (derm7pt, CEBaB)

Everything else is identical:
  - KL divergence
  - D3 diversity + sparsity losses
  - Optimiser + scheduler
  - Early stopping
  - Checkpoint saving
  - Probe handoff

Design: the trainer takes a ModelWrapper that abstracts the forward pass
and reconstruction loss. This avoids a large if/else tree inside the loop.

Usage:
    # Vision (MedMNIST)
    model   = VisionGACSModel(in_channels=1, num_classes=11)
    wrapper = VisionModelWrapper(model)
    trainer = GACSSharedTrainer(wrapper, config, train_loader, val_loader)
    trainer.train()
    model, probe_params = trainer.get_probe_ready_model()

    # Text (CEBaB)
    model   = GACSModel(config)
    wrapper = TextModelWrapper(model)
    trainer = GACSSharedTrainer(wrapper, config, train_loader, val_loader)
    trainer.train()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Model wrappers — abstract forward pass and reconstruction loss
# ══════════════════════════════════════════════════════════════════════════════

class VisionModelWrapper:
    """
    Wraps VisionGACSModel for the shared training loop.

    forward_batch   : runs model on a batch, returns output dict
    recon_loss      : BCE on pixels (D1 compliant)
    get_probe_params: delegates to model.get_probeable_parameters()
    """

    def __init__(self, model):
        self.model = model

    def forward_batch(self, batch: dict, device: torch.device) -> dict:
        """
        MedMNIST batch format: (images, labels) tuple from DataLoader.
        medmnist returns labels as [B, 1] — squeeze to [B].
        """
        x, y = batch
        x    = x.to(device)
        y    = y.to(device).squeeze(-1).long()   # [B,1] → [B]
        out  = self.model(x)
        out["labels"] = y
        out["x"]      = x   # keep input for reconstruction loss
        return out

    def recon_loss(self, out: dict) -> torch.Tensor:
        """
        D1: BCE on [0,1] pixel values.
        out["recon"] ∈ [0,1]^{B,C,H,W} (Sigmoid output from decoder).
        out["x"]     ∈ [0,1]^{B,C,H,W} (input image after ToTensor).
        """
        return F.binary_cross_entropy(
            out["recon"], out["x"], reduction="mean"
        )

    def get_model(self):
        return self.model

    def get_probe_params(self, scope: str = "decoder") -> list:
        return self.model.get_probeable_parameters(scope)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)


class TextModelWrapper:
    """
    Wraps GACSModel (text/CEBaB) for the shared training loop.

    forward_batch   : runs model on tokenised batch
    recon_loss      : NLPContrastiveReconLoss on L2-normalised CLS embeddings
    """

    def __init__(self, model, recon_loss_fn=None):
        """
        Parameters
        ----------
        model         : GACSModel instance
        recon_loss_fn : NLPContrastiveReconLoss instance.
                        If None, reconstruction loss is skipped (ablation).
        """
        self.model         = model
        self.recon_loss_fn = recon_loss_fn

    def forward_batch(self, batch: dict, device: torch.device) -> dict:
        """
        CEBaB batch format: dict with input_ids, attention_mask, label,
        and optionally concepts (supervised factor labels).
        """
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        out          = self.model(input_ids, attention_mask)
        out["labels"] = labels

        # Supervised factor labels (optional)
        out["criteria_labels"] = batch.get("concepts")
        if out["criteria_labels"] is not None:
            out["criteria_labels"] = out["criteria_labels"].to(device)

        return out

    def recon_loss(self, out: dict) -> torch.Tensor:
        """
        D1: contrastive reconstruction loss on L2-normalised CLS embeddings.
        Requires NLPContrastiveReconLoss from gacs/losses.py.
        """
        if self.recon_loss_fn is None:
            return torch.tensor(0.0)
        loss, _ = self.recon_loss_fn(out["recon"], out["h_norm"])
        return loss

    def get_model(self):
        return self.model

    def get_probe_params(self, scope: str = "decoder") -> list:
        return self.model.get_probeable_parameters(scope)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Shared loss computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_gacs_loss(
    out:            dict,
    wrapper,
    epoch:          int,
    warmup_epochs:  int,
    recon_weight:   float = 1.0,
    kl_max:         float = 1.0,
    div_weight:     float = 0.1,
    sparse_weight:  float = 0.05,
    criteria_loss_fn = None,   # Derm7ptFactorHeadLoss or None
) -> Tuple[torch.Tensor, dict]:
    """
    Compute full GACS loss for one batch.

    Works for both vision and text — the only difference is
    how recon_loss is computed (handled by the wrapper).

    Returns (total_loss, metrics_dict).

    D2: sigmoid KL annealing — β(epoch) rises from ~0 to kl_max
        over warmup_epochs. Vanishing derivative at both ends ensures
        the loss landscape is stationary at convergence.
    """
    from gacs.losses import sigmoid_kl_weight

    B = out["logits"].size(0)

    # ── Classification ────────────────────────────────────────────────────────
    cls_loss = F.cross_entropy(out["logits"], out["labels"])

    # ── Reconstruction (D1) ───────────────────────────────────────────────────
    recon_loss = wrapper.recon_loss(out)

    # ── KL divergence (D2: sigmoid annealing) ────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["z_logvar"] - out["z_mu"].pow(2) - out["z_logvar"].exp()
    )
    kl_w = kl_max * sigmoid_kl_weight(epoch, warmup_epochs)

    # ── D3: factor diversity (covariance → identity) ──────────────────────────
    s = out["s"]   # [B, K]
    K = s.size(1)
    if B > 1:
        s_centered = s - s.mean(dim=0, keepdim=True)
        cov        = (s_centered.T @ s_centered) / (B - 1 + 1e-8)
        eye_K      = torch.eye(K, device=s.device)
        div_loss   = ((cov - eye_K) ** 2).mean()
    else:
        div_loss = torch.tensor(0.0, device=s.device)

    # ── D3: factor sparsity (L1) ──────────────────────────────────────────────
    sparse_loss = s.abs().mean()

    # ── Supervised criteria (derm7pt only) ────────────────────────────────────
    criteria_loss = torch.tensor(0.0, device=s.device)
    if criteria_loss_fn is not None and out.get("criteria_labels") is not None:
        _, criteria_metrics = criteria_loss_fn(
            out["logits"], s,
            out["labels"], out["criteria_labels"],
        )
        criteria_loss = torch.tensor(
            criteria_metrics["criteria_loss"], device=s.device
        )
        # TODO: use the full weighted loss from criteria_loss_fn directly
        # rather than extracting from metrics dict. Refactor in next session.

    # ── Total ─────────────────────────────────────────────────────────────────
    total_loss = (
        cls_loss
        + recon_weight  * recon_loss
        + kl_w          * kl_loss
        + div_weight    * div_loss
        + sparse_weight * sparse_loss
        + criteria_loss
    )

    metrics = {
        "total_loss":    total_loss.item(),
        "cls_loss":      cls_loss.item(),
        "recon_loss":    recon_loss.item(),
        "kl_loss":       kl_loss.item(),
        "kl_weight":     kl_w,
        "div_loss":      div_loss.item(),
        "sparse_loss":   sparse_loss.item(),
        "criteria_loss": criteria_loss.item(),
    }
    return total_loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Shared trainer
# ══════════════════════════════════════════════════════════════════════════════

class GACSSharedTrainer:
    """
    Shared training loop for VisionGACSModel and GACSModel.

    Differential learning rates:
      - Encoder (CNN or BERT): lr * encoder_lr_scale  (default 0.1)
      - All other parameters:  lr  (full rate)

    For vision: encoder_lr_scale=1.0 (CNN trained from scratch, no reason to slow it)
    For text:   encoder_lr_scale=0.1 (BERT fine-tuning, standard practice)

    TODO: pass encoder_lr_scale=0.1 for text, 1.0 for vision.
    """

    def __init__(
        self,
        wrapper,                      # VisionModelWrapper or TextModelWrapper
        config,                       # ExperimentConfig or your config object
        train_loader,
        val_loader,
        test_loader       = None,
        criteria_loss_fn  = None,     # Derm7ptFactorHeadLoss, or None
        encoder_lr_scale: float = 1.0,
        # Loss weights
        recon_weight:  float = 1.0,
        kl_max:        float = 1.0,
        div_weight:    float = 0.1,
        sparse_weight: float = 0.05,
    ):
        self.wrapper         = wrapper
        self.config          = config
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.test_loader     = test_loader
        self.criteria_loss_fn = criteria_loss_fn
        self.device          = torch.device(config.device)
        self.recon_weight    = recon_weight
        self.kl_max          = kl_max
        self.div_weight      = div_weight
        self.sparse_weight   = sparse_weight

        wrapper.to(self.device)

        # Differential learning rates
        encoder_params = []
        other_params   = []
        for name, p in wrapper.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = AdamW([
            {"params": encoder_params, "lr": config.lr * encoder_lr_scale},
            {"params": other_params,   "lr": config.lr},
        ], weight_decay=getattr(config, "weight_decay", 1e-4))

        total_steps = len(train_loader) * config.epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr    = [config.lr * encoder_lr_scale, config.lr],
            total_steps = total_steps,
            pct_start   = getattr(config, "warmup_ratio", 0.1),
        )

        self.best_val_acc      = 0.0
        self.patience_counter  = 0
        self.history           = {"train": [], "val": []}
        self.global_step       = 0

        self.output_dir = Path(getattr(config, "output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Full training loop with early stopping."""
        logger.info(
            f"Training | device={self.device} | "
            f"epochs={self.config.epochs} | "
            f"train_batches={len(self.train_loader)}"
        )

        for epoch in range(self.config.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._validate(epoch)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['total_loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"kl_w={train_metrics['kl_weight']:.3f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc    = val_metrics["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            save_every = getattr(self.config, "save_every_n_epochs", 10)
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            patience = getattr(self.config, "patience", 10)
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self._save_history()
        return {
            "best_val_acc": self.best_val_acc,
            "history":      self.history,
        }

    def _train_epoch(self, epoch: int) -> dict:
        self.wrapper.train()
        running   = {}
        n_correct = 0
        n_total   = 0
        warmup_ep = getattr(self.config, "warmup_epochs", 15)

        for batch in self.train_loader:
            out  = self.wrapper.forward_batch(batch, self.device)
            loss, metrics = compute_gacs_loss(
                out, self.wrapper, epoch, warmup_ep,
                self.recon_weight, self.kl_max,
                self.div_weight, self.sparse_weight,
                self.criteria_loss_fn,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.wrapper.get_model().parameters(),
                getattr(self.config, "max_grad_norm", 1.0),
            )
            self.optimizer.step()
            self.scheduler.step()

            preds     = out["logits"].argmax(dim=-1)
            n_correct += (preds == out["labels"]).sum().item()
            n_total   += out["labels"].size(0)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            self.global_step += 1

        n_batches = len(self.train_loader)
        result    = {k: v / n_batches for k, v in running.items()}
        result["accuracy"] = n_correct / n_total
        return result

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.wrapper.eval()
        running   = {}
        n_correct = 0
        n_total   = 0
        all_probs  = []
        all_labels = []
        warmup_ep  = getattr(self.config, "warmup_epochs", 15)

        for batch in self.val_loader:
            out  = self.wrapper.forward_batch(batch, self.device)
            _, metrics = compute_gacs_loss(
                out, self.wrapper, epoch, warmup_ep,
                self.recon_weight, self.kl_max,
                self.div_weight, self.sparse_weight,
                self.criteria_loss_fn,
            )

            preds     = out["logits"].argmax(dim=-1)
            n_correct += (preds == out["labels"]).sum().item()
            n_total   += out["labels"].size(0)

            all_probs.append(F.softmax(out["logits"], dim=-1).cpu())
            all_labels.append(out["labels"].cpu())

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

        n_batches = len(self.val_loader)
        result    = {k: v / n_batches for k, v in running.items()}
        result["accuracy"] = n_correct / n_total

        self.val_probs  = torch.cat(all_probs,  dim=0)
        self.val_labels = torch.cat(all_labels, dim=0)
        return result

    # ── Checkpoint and probe handoff ──────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        name = "best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt"
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.wrapper.get_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc":     self.best_val_acc,
        }, self.output_dir / name)

    def _save_history(self):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def load_best_model(self):
        path = self.output_dir / "best_model.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.wrapper.get_model().load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded best model | epoch={ckpt['epoch']} | "
            f"val_acc={ckpt['best_val_acc']:.4f}"
        )

    def get_probe_ready_model(self, scope: str = "decoder"):
        """
        Load best checkpoint and return model + probe parameters.

        Call after train():
            model, probe_params = trainer.get_probe_ready_model()
            rho = perturbation_probe(model, probe_params, probe_loader)
        """
        self.load_best_model()
        self.wrapper.eval()
        model        = self.wrapper.get_model()
        probe_params = self.wrapper.get_probe_params(scope)
        n = sum(p.numel() for p in probe_params)
        logger.info(
            f"Probe-ready | scope={scope} | "
            f"n_params={n:,} | val_acc={self.best_val_acc:.4f}"
        )
        return model, probe_params

    def get_val_arrays_for_calibration(self):
        """Return val probs and labels for calibrate_eps()."""
        if not hasattr(self, "val_probs"):
            raise RuntimeError("Call train() before get_val_arrays_for_calibration()")
        return self.val_probs.numpy(), self.val_labels.numpy()

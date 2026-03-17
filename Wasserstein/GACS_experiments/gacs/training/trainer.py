"""
GACS Training Loop

Handles:
  - Training with all loss components
  - Validation with early stopping
  - Metric logging
  - Checkpoint saving
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, Optional
from pathlib import Path
import json
import time
import numpy as np


class GACSTrainer:
    """
    Trainer for the Stochastic Concept Bottleneck Model.
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        config,
        train_loader,
        val_loader,
        test_loader=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        # Optimizer: differential LR for encoder vs heads
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = AdamW([
            {"params": encoder_params, "lr": config.training.learning_rate * 0.1},
            {"params": head_params, "lr": config.training.learning_rate},
        ], weight_decay=config.training.weight_decay)
        
        # Scheduler
        total_steps = len(train_loader) * config.training.epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[config.training.learning_rate * 0.1, config.training.learning_rate],
            total_steps=total_steps,
            pct_start=config.training.warmup_ratio,
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train": [], "val": []}
        self.global_step = 0
        
        # Output dir
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training GACS — {self.config.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.training.epochs):
            t0 = time.time()
            
            # Train
            train_metrics = self._train_epoch(epoch)
            self.history["train"].append(train_metrics)
            
            # Validate
            val_metrics = self._validate(epoch)
            self.history["val"].append(val_metrics)
            
            elapsed = time.time() - t0
            
            # Print summary
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['total_loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"kl_w={train_metrics['kl_weight']:.4f} | "
                f"{elapsed:.1f}s"
            )
            
            # Checkpointing
            # Save checkpoint for every epoch
            self._save_checkpoint(epoch, is_best=False)

            # Also save best model separately
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save history
        self._save_history()
        
        return {
            "best_val_acc": self.best_val_acc,
            "final_epoch": epoch,
            "history": self.history,
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        
        running = {}
        n_correct = 0
        n_total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            concepts = batch["concepts"].to(self.device)
            
            # Forward
            outputs = self.model(input_ids, attention_mask)
            
            # Loss
            loss_dict = self.loss_fn.compute(outputs, labels, concepts, epoch)
            loss = loss_dict["total_loss"]
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
            
            for k, v in loss_dict.items():
                if k != "total_loss":
                    running[k] = running.get(k, 0.0) + (v if isinstance(v, float) else v)
                else:
                    running[k] = running.get(k, 0.0) + loss.item()
            
            self.global_step += 1
            
            # Log periodically
            if (batch_idx + 1) % self.config.training.log_every == 0:
                print(
                    f"  batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"loss={loss.item():.4f} "
                    f"recon={loss_dict['recon_loss']:.4f} "
                    f"kl={loss_dict['kl_loss']:.4f} "
                    f"cls={loss_dict['classify_loss']:.4f}"
                )
        
        # Average
        n_batches = len(self.train_loader)
        metrics = {k: v / n_batches for k, v in running.items()}
        metrics["accuracy"] = n_correct / n_total
        metrics["epoch"] = epoch
        
        return metrics
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()
        
        running = {}
        n_correct = 0
        n_total = 0
        all_probs = []
        all_labels = []
        all_concept_probs = []
        all_concept_labels = []
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            concepts = batch["concepts"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss_dict = self.loss_fn.compute(outputs, labels, concepts, epoch)
            
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
            
            # Store for credal evaluation later
            probs = F.softmax(outputs["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_concept_probs.append(outputs["concept_probs"].cpu().numpy())
            all_concept_labels.append(concepts.cpu().numpy())
            
            for k, v in loss_dict.items():
                if k != "total_loss":
                    running[k] = running.get(k, 0.0) + (v if isinstance(v, float) else v)
                else:
                    running[k] = running.get(k, 0.0) + loss_dict["total_loss"].item()
        
        n_batches = len(self.val_loader)
        metrics = {k: v / n_batches for k, v in running.items()}
        metrics["accuracy"] = n_correct / n_total
        metrics["epoch"] = epoch
        
        # Store arrays for later use
        self.val_probs = np.concatenate(all_probs)
        self.val_labels = np.concatenate(all_labels)
        self.val_concept_probs = np.concatenate(all_concept_probs)
        self.val_concept_labels = np.concatenate(all_concept_labels)
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        path = self.output_dir / ("best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)
    
    def _save_history(self):
        """Save training history to JSON."""
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_best_model(self):
        """Load the best checkpoint."""
        path = self.output_dir / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best model from epoch {ckpt['epoch']} "
                  f"(val_acc={ckpt['best_val_acc']:.4f})")
        else:
            print("No checkpoint found, using current model")

"""
Vision Trainer for GACS

Optimized for fast MedMNIST experiments on a single GPU.
Key differences from NLP trainer:
  - No tokenizer/encoder unfreezing dance
  - Single LR (no differential rates)
  - Faster batches (images, not text)
  - Handles image/label dict format
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
from pathlib import Path
import json
import time
import numpy as np


class VisionTrainer:
    """Fast trainer for vision VAE on MedMNIST."""
    
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
        
        # Simple optimizer — no differential LR needed for CNN
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6,
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train": [], "val": []}
        
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict:
        """Full training loop."""
        print(f"\nTraining {self.config.experiment_name} on {self.device}")
        print(f"Train: {len(self.train_loader)} batches, "
              f"Val: {len(self.val_loader)} batches, "
              f"Epochs: {self.config.training.epochs}\n")
        
        for epoch in range(self.config.training.epochs):
            t0 = time.time()
            
            train_m = self._train_epoch(epoch)
            val_m = self._validate(epoch)
            
            self.scheduler.step()
            self.history["train"].append(train_m)
            self.history["val"].append(val_m)
            
            elapsed = time.time() - t0
            
            print(
                f"Ep {epoch:3d} | "
                f"trn_loss={train_m['total_loss']:.4f} "
                f"trn_acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['total_loss']:.4f} "
                f"val_acc={val_m['accuracy']:.4f} | "
                f"recon={val_m['recon_loss']:.4f} "
                f"kl={val_m['kl_loss']:.4f} | "
                f"{elapsed:.1f}s"
            )
            
            # Save checkpoint for every epoch
            self._save(f"checkpoint_epoch_{epoch:03d}.pt", epoch,
                      train_acc=train_m['accuracy'], train_loss=train_m['total_loss'],
                      val_acc=val_m['accuracy'], val_loss=val_m['total_loss'])

            # Save best model separately
            if val_m["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_m["accuracy"]
                self.patience_counter = 0
                self._save("best_model.pt", epoch)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self._save_history()
        return {"best_val_acc": self.best_val_acc, "history": self.history}
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {}
        n_correct = 0
        n_total = 0
        
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(images)
            loss_dict = self.loss_fn.compute(outputs, labels, epoch=epoch)
            loss = loss_dict["total_loss"]
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
            
            for k, v in loss_dict.items():
                val = v if isinstance(v, (int, float)) else v
                running[k] = running.get(k, 0.0) + (loss.item() if k == "total_loss" else val)
        
        nb = len(self.train_loader)
        metrics = {k: v / nb for k, v in running.items()}
        metrics["accuracy"] = n_correct / n_total
        return metrics
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running = {}
        n_correct = 0
        n_total = 0
        
        all_probs = []
        all_labels = []
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(images)
            loss_dict = self.loss_fn.compute(outputs, labels, epoch=epoch)
            
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
            
            all_probs.append(F.softmax(outputs["logits"], dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            for k, v in loss_dict.items():
                val = v if isinstance(v, (int, float)) else v
                running[k] = running.get(k, 0.0) + (loss_dict["total_loss"].item() if k == "total_loss" else val)
        
        nb = len(self.val_loader)
        metrics = {k: v / nb for k, v in running.items()}
        metrics["accuracy"] = n_correct / n_total
        
        self.val_probs = np.concatenate(all_probs)
        self.val_labels = np.concatenate(all_labels)
        
        return metrics
    
    def _save(self, name: str, epoch: int, train_acc=None, train_loss=None,
              val_acc=None, val_loss=None):
        """Save checkpoint with optional metrics."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
        }

        # Add metrics if provided
        if train_acc is not None:
            checkpoint["train_acc"] = train_acc
        if train_loss is not None:
            checkpoint["train_loss"] = train_loss
        if val_acc is not None:
            checkpoint["val_acc"] = val_acc
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        torch.save(checkpoint, self.output_dir / name)
    
    def load_best(self):
        path = self.output_dir / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best model (epoch {ckpt['epoch']}, acc={ckpt['best_val_acc']:.4f})")
    
    def _save_history(self):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

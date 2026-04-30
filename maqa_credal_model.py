"""
Credal QA model and trainer for MAQA/AmbigQA-style datasets.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class CredalQAParameters:
    mu: torch.Tensor
    sigma_epi: torch.Tensor
    sigma_ale: torch.Tensor


class MAQADataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        use_paired: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_paired = use_paired
        self.pairs = self._create_pairs() if use_paired else None

    def _create_pairs(self) -> List[Tuple[int, int]]:
        ambiguous = [i for i, d in enumerate(self.data) if d["ambiguity_level"] >= 2]
        clear = [i for i, d in enumerate(self.data) if d["ambiguity_level"] == 0]
        pairs = []
        for i in range(min(len(ambiguous), len(clear))):
            pairs.append((ambiguous[i], clear[i]))
        return pairs

    def __len__(self):
        return len(self.pairs) if self.use_paired and self.pairs else len(self.data)

    def __getitem__(self, idx):
        if self.use_paired and self.pairs:
            amb_idx, clear_idx = self.pairs[idx]
            return {"amb": self._get_item(amb_idx), "clear": self._get_item(clear_idx)}
        return self._get_item(idx)

    def _get_item(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "p_star": torch.tensor(item["p_star"], dtype=torch.float32),
            "entropy": torch.tensor(item["entropy"], dtype=torch.float32),
            "ambiguity_level": torch.tensor(item["ambiguity_level"], dtype=torch.long),
            "num_answers": len(item["answers"]),
            "dominant_answer_idx": torch.tensor(item["dominant_answer_idx"], dtype=torch.long),
        }


def collate_single(batch):
    max_answers = max(item["p_star"].size(0) for item in batch)
    p_stars = []
    for item in batch:
        p_star = item["p_star"]
        if p_star.size(0) < max_answers:
            p_star = F.pad(p_star, (0, max_answers - p_star.size(0)))
        p_stars.append(p_star)
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "p_star": torch.stack(p_stars),
        "entropy": torch.stack([item["entropy"] for item in batch]),
        "ambiguity_level": torch.stack([item["ambiguity_level"] for item in batch]),
        "num_answers": torch.tensor([item["num_answers"] for item in batch]),
        "dominant_answer_idx": torch.stack([item["dominant_answer_idx"] for item in batch]),
    }


def maqa_collate_fn(batch):
    if batch and isinstance(batch[0], dict) and "amb" in batch[0]:
        return {
            "amb": collate_single([item["amb"] for item in batch]),
            "clear": collate_single([item["clear"] for item in batch]),
        }
    return collate_single(batch)


class CredalMAQA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_answers: int = 10,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.num_answers = num_answers
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_answers),
        )
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
        )
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_embeddings: bool = False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(hidden)
        mu = F.softmax(self.mu_head(projected), dim=-1)
        sigma_epi = F.softplus(self.sigma_epi_head(projected)).squeeze(-1)
        sigma_ale = F.softplus(self.sigma_ale_head(projected)).squeeze(-1)
        params = CredalQAParameters(mu=mu, sigma_epi=sigma_epi, sigma_ale=sigma_ale)
        if return_embeddings:
            return params, hidden
        return params


class MAQACredalLoss(nn.Module):
    def __init__(self, alpha_kl: float = 1.0, alpha_reg: float = 0.1, alpha_cal: float = 0.5, alpha_cont: float = 0.3):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.alpha_reg = alpha_reg
        self.alpha_cal = alpha_cal
        self.alpha_cont = alpha_cont

    def forward(
        self,
        params: CredalQAParameters,
        p_star: torch.Tensor,
        entropy_gt: torch.Tensor,
        params_clear: Optional[CredalQAParameters] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        losses: Dict[str, torch.Tensor] = {}
        kl_losses = []
        for i in range(p_star.size(0)):
            target = p_star[i]
            if float(target.sum()) <= 0:
                continue
            kl_losses.append(F.kl_div(torch.log(params.mu[i].clamp(min=1e-8)), target, reduction="batchmean"))
        loss_kl = torch.stack(kl_losses).mean() if kl_losses else torch.zeros((), device=p_star.device)
        losses["loss_kl"] = loss_kl

        margin = torch.tensor(0.1, device=params.sigma_epi.device)
        loss_reg = F.relu(margin - torch.abs(params.sigma_epi - params.sigma_ale)).mean()
        losses["loss_reg"] = loss_reg

        loss_cal = F.mse_loss(params.sigma_ale, entropy_gt)
        losses["loss_cal"] = loss_cal

        if params_clear is not None:
            loss_cont = F.relu(params_clear.sigma_ale - params.sigma_ale).mean()
        else:
            loss_cont = torch.zeros((), device=params.sigma_ale.device)
        losses["loss_cont"] = loss_cont

        total = (
            self.alpha_kl * loss_kl
            + self.alpha_reg * loss_reg
            + self.alpha_cal * loss_cal
            + self.alpha_cont * loss_cont
        )
        losses["loss_total"] = total
        return total, losses


class MAQACredalTrainer:
    def __init__(
        self,
        model: CredalMAQA,
        train_loader,
        val_loader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = MAQACredalLoss()
        self.best_val_score = -math.inf
        self.history: list[dict] = []
        self.save_dir = Path("./checkpoints/maqa_credal")

    def _step(self, batch):
        if "amb" in batch:
            amb = {k: v.to(self.device) for k, v in batch["amb"].items()}
            clear = {k: v.to(self.device) for k, v in batch["clear"].items()}
            params_amb = self.model(amb["input_ids"], amb["attention_mask"])
            params_clear = self.model(clear["input_ids"], clear["attention_mask"])
            loss, losses = self.criterion(params_amb, amb["p_star"], amb["entropy"], params_clear=params_clear)
            batch_metrics = amb
            params = params_amb
        else:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            params = self.model(batch["input_ids"], batch["attention_mask"])
            loss, losses = self.criterion(params, batch["p_star"], batch["entropy"])
            batch_metrics = batch
        return loss, losses, params, batch_metrics

    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = total_answer = total_reg = total_cal = total_cont = 0.0
        total_correct = total = 0
        all_sigma_epi, all_sigma_ale, all_entropy = [], [], []
        for batch in tqdm(self.train_loader, desc="MAQA train"):
            self.optimizer.zero_grad(set_to_none=True)
            loss, losses, params, batch_metrics = self._step(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            total_answer += float(losses["loss_kl"].item())
            total_reg += float(losses["loss_reg"].item())
            total_cal += float(losses["loss_cal"].item())
            total_cont += float(losses["loss_cont"].item())
            preds = params.mu.argmax(dim=-1).detach().cpu()
            labels = batch_metrics["dominant_answer_idx"].detach().cpu()
            total_correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            all_sigma_epi.append(params.sigma_epi.detach().cpu())
            all_sigma_ale.append(params.sigma_ale.detach().cpu())
            all_entropy.append(batch_metrics["entropy"].detach().cpu())
        all_sigma_epi = torch.cat(all_sigma_epi, dim=0)
        all_sigma_ale = torch.cat(all_sigma_ale, dim=0)
        all_entropy = torch.cat(all_entropy, dim=0)
        return {
            "train_loss": total_loss / max(1, len(self.train_loader)),
            "answer_loss": total_answer / max(1, len(self.train_loader)),
            "kl_reg_loss": total_reg / max(1, len(self.train_loader)),
            "calibration_loss": total_cal / max(1, len(self.train_loader)),
            "contrastive_loss": total_cont / max(1, len(self.train_loader)),
            "task_accuracy": total_correct / max(1, total),
            "mean_sigma_epi": float(all_sigma_epi.mean().item()),
            "mean_sigma_ale": float(all_sigma_ale.mean().item()),
            "mean_entropy_gt": float(all_entropy.mean().item()),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict:
        self.model.eval()
        total_loss = total_answer = total_reg = total_cal = total_cont = 0.0
        total_correct = total = 0
        all_sigma_epi, all_sigma_ale, all_entropy = [], [], []
        for batch in tqdm(loader, desc="MAQA eval"):
            loss, losses, params, batch_metrics = self._step(batch)
            total_loss += float(loss.item())
            total_answer += float(losses["loss_kl"].item())
            total_reg += float(losses["loss_reg"].item())
            total_cal += float(losses["loss_cal"].item())
            total_cont += float(losses["loss_cont"].item())
            preds = params.mu.argmax(dim=-1).detach().cpu()
            labels = batch_metrics["dominant_answer_idx"].detach().cpu()
            total_correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            all_sigma_epi.append(params.sigma_epi.detach().cpu())
            all_sigma_ale.append(params.sigma_ale.detach().cpu())
            all_entropy.append(batch_metrics["entropy"].detach().cpu())
        all_sigma_epi = torch.cat(all_sigma_epi, dim=0)
        all_sigma_ale = torch.cat(all_sigma_ale, dim=0)
        all_entropy = torch.cat(all_entropy, dim=0)

        metrics = {
            "loss": total_loss / max(1, len(loader)),
            "answer_loss": total_answer / max(1, len(loader)),
            "kl_reg_loss": total_reg / max(1, len(loader)),
            "calibration_loss": total_cal / max(1, len(loader)),
            "contrastive_loss": total_cont / max(1, len(loader)),
            "task_accuracy": total_correct / max(1, total),
            "mean_sigma_epi": float(all_sigma_epi.mean().item()),
            "mean_sigma_ale": float(all_sigma_ale.mean().item()),
            "mean_entropy_gt": float(all_entropy.mean().item()),
        }
        if len(all_sigma_epi) > 10:
            from scipy import stats
            metrics["rho_eu_au"] = float(stats.pearsonr(all_sigma_epi.numpy(), all_sigma_ale.numpy())[0])
            metrics["rho_au_entropy"] = float(stats.pearsonr(all_sigma_ale.numpy(), all_entropy.numpy())[0])
        return metrics

    def fit(self, num_epochs: int, save_dir: Path, eval_every: int = 1):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        best = -math.inf
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader) if (epoch % eval_every == 0) else {}
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            self.history.append(row)
            score = val_metrics.get("task_accuracy", train_metrics["task_accuracy"])
            if score > best:
                best = score
                torch.save({"model_state_dict": self.model.state_dict()}, self.save_dir / "best_model.pt")
        (self.save_dir / "training_history.json").write_text(json.dumps({"history": self.history}, indent=2, default=float))

    def dump_eval_arrays(self, loader, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_sigma_epi, all_sigma_ale, all_entropy = [], [], []
        for batch in loader:
            if "amb" in batch:
                batch = batch["amb"]
            batch = {k: v.to(self.device) for k, v in batch.items()}
            params = self.model(batch["input_ids"], batch["attention_mask"])
            all_sigma_epi.append(params.sigma_epi.detach().cpu())
            all_sigma_ale.append(params.sigma_ale.detach().cpu())
            all_entropy.append(batch["entropy"].detach().cpu())
        np.savez(
            out_dir / "test_arrays.npz",
            sigma_epi=torch.cat(all_sigma_epi, dim=0).numpy(),
            sigma_ale=torch.cat(all_sigma_ale, dim=0).numpy(),
            entropy=torch.cat(all_entropy, dim=0).numpy(),
        )
        (out_dir / "metadata.json").write_text(json.dumps({"dataset": "maqa"}, indent=2))

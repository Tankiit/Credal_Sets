"""
Gradient Isolation Snapshot Analysis
====================================

Measures how well the model disentangles epistemic and aleatoric gradients.
Specifically, checks that ∇_KL and ∇_aleatoric on the shared μ-head are orthogonal.

Usage:
    python -m analysis.snapshot_grad_iso \\
        --checkpoint checkpoints/hybrid_credal_cebab/best_model.pt \\
        --dataset cebab \\
        --num_batches 50 \\
        --output outputs/grad_iso_snapshots/cebab.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel

from VCBM import VariationalCredalCBM, VariationalCredalConfig
from loaders import LOADERS


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    candidate_keys = (
        "model_state_dict",
        "state_dict",
        "model",
        "weights",
        "module",
    )
    state_dict = None
    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if state_dict is None:
        tensor_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
        if tensor_items:
            state_dict = tensor_items
        else:
            return None

    cleaned_state_dict = {}
    for name, tensor in state_dict.items():
        clean_name = name
        for prefix in ("module.", "model.", "net."):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        cleaned_state_dict[clean_name] = tensor

    return cleaned_state_dict


def _extract_config(checkpoint):
    if not isinstance(checkpoint, dict):
        return VariationalCredalConfig()

    for key in ("config", "model_config", "hparams", "hyperparameters"):
        config_value = checkpoint.get(key)
        if config_value is None:
            continue
        if isinstance(config_value, VariationalCredalConfig):
            return config_value
        if isinstance(config_value, dict):
            return VariationalCredalConfig.from_dict(config_value)
        if hasattr(config_value, "__dict__"):
            return VariationalCredalConfig.from_dict(vars(config_value))

    return VariationalCredalConfig()


def _flatten_grads(grads, params):
    flat_parts = []
    per_param_norms = []
    for param, grad in zip(params, grads):
        if grad is None:
            grad = torch.zeros_like(param)
        flat_parts.append(grad.reshape(-1))
        per_param_norms.append(grad.norm())

    if flat_parts:
        return torch.cat(flat_parts), torch.stack(per_param_norms)
    return torch.empty(0), torch.empty(0)


def _group_params(model, *needles):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(needle in name for needle in needles):
            params.append(param)
    return params


class _LinearHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class _DeepLinearHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, middle_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, middle_features),
            nn.ReLU(),
            nn.Linear(middle_features, out_features),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class _LegacyProjection(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.W_epi = nn.Linear(hidden_size, proj_dim, bias=False)
        self.W_ale = nn.Linear(hidden_size, proj_dim, bias=False)

    def forward(self, hidden: torch.Tensor):
        return self.W_epi(hidden), self.W_ale(hidden)


class _LegacyTaskClassifier(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, concept_probs: torch.Tensor):
        logits = self.linear(concept_probs)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class LegacyHybridCredalCBM(nn.Module):
    """Compatibility wrapper for older hybrid credal checkpoints."""

    def __init__(self, hidden_size: int, num_concepts: int, num_classes: int, proj_dim: int, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.projection = _LegacyProjection(hidden_size, proj_dim)
        self.concept_head = _LinearHead(proj_dim, 128, num_concepts)
        self.epistemic_head = _DeepLinearHead(proj_dim, 128, 64, num_concepts)
        self.aleatoric_head = _DeepLinearHead(proj_dim, 128, 64, num_concepts)
        self.task_classifier = _LegacyTaskClassifier(num_concepts, num_classes)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None, concept_labels=None, **kwargs):
        hidden = self.encode(input_ids, attention_mask)
        h_epi, h_ale = self.projection(hidden)

        concept_logits = self.concept_head(h_epi)
        concept_probs = torch.softmax(concept_logits, dim=-1)

        epistemic = F.softplus(self.epistemic_head(h_epi))
        aleatoric = F.softplus(self.aleatoric_head(h_ale))

        logits, probs = self.task_classifier(concept_probs)

        result = {
            'predictions': logits.argmax(dim=-1),
            'logits': logits,
            'probs': probs,
            'label_logits': logits,
            'label_probs': probs,
            'concept_probs': concept_probs,
            'mu': concept_probs,
            'concept_logits': concept_logits,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'sigma_epi': epistemic.mean(dim=-1),
            'sigma_ale': aleatoric.mean(dim=-1),
        }

        if labels is not None:
            result['ce_loss'] = F.cross_entropy(logits, labels)
        if concept_labels is not None:
            result['concept_loss'] = F.cross_entropy(
                concept_logits,
                concept_labels.argmax(dim=-1) if concept_labels.dim() > 1 else concept_labels,
            )
        result['kl_loss'] = epistemic.mean()
        return result


def load_checkpoint(checkpoint_path: str, device=None) -> VariationalCredalCBM:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, VariationalCredalCBM):
        return checkpoint.to(device)

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a recognizable state dict")

    if 'epistemic_head.net.0.weight' in state_dict and 'projection.W_epi.weight' in state_dict:
        hidden_size = state_dict['encoder.embeddings.word_embeddings.weight'].shape[1]
        task_weight_key = 'task_classifier.weight' if 'task_classifier.weight' in state_dict else 'task_classifier.linear.weight'
        num_concepts = state_dict[task_weight_key].shape[1]
        num_classes = state_dict[task_weight_key].shape[0]
        proj_dim = state_dict['projection.W_epi.weight'].shape[0]
        encoder_name = 'distilbert-base-uncased'
        model = LegacyHybridCredalCBM(hidden_size, num_concepts, num_classes, proj_dim, encoder_name)
        model.load_state_dict(state_dict, strict=False)
    else:
        config = _extract_config(checkpoint)
        model = VariationalCredalCBM(config)
        checkpoint_state = {}
        model_state = model.state_dict()
        for name, tensor in state_dict.items():
            if name in model_state and model_state[name].shape == tensor.shape:
                checkpoint_state[name] = tensor
        model.load_state_dict(checkpoint_state, strict=False)
    model = model.to(device)
    model.eval()

    return model


@dataclass
class GradIsoSnapshot:
    """Gradient isolation metrics for a checkpoint."""
    dataset: str
    checkpoint: str
    num_batches: int
    
    # Per-batch cosine similarities between KL and aleatoric gradients
    cos_sim_kl_ale_per_batch: list  # [B,]
    
    # Aggregate statistics
    mean_cos_sim_mu_head: float
    std_cos_sim_mu_head: float
    abs_max_cos_sim_mu_head: float
    
    # Cross-head gradient norms (verify disjoint tensor design)
    max_g_kl_on_sigma_ale: float     # max ||∇_KL w.r.t. σ_ale||
    max_g_ale_on_sigma_epi: float    # max ||∇_ale w.r.t. σ_epi||
    
    @property
    def summary(self):
        """Return summary dict for the paper table."""
        return {
            "mean_cos_sim_mu_head": self.mean_cos_sim_mu_head,
            "std_cos_sim_mu_head": self.std_cos_sim_mu_head,
            "abs_max_cos_sim_mu_head": self.abs_max_cos_sim_mu_head,
            "max_g_kl_on_sigma_ale": self.max_g_kl_on_sigma_ale,
            "max_g_ale_on_sigma_epi": self.max_g_ale_on_sigma_epi,
        }


def analyze_checkpoint(
    model,
    bundle,
    num_batches: int = 50,
    device: Optional[torch.device] = None,
) -> GradIsoSnapshot:
    """Analyze gradient isolation for a checkpoint."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.train()  # Need gradients

    epi_params = _group_params(model, "concept_encoder")
    ale_params = _group_params(model, "aleatoric_head")
    full_params = [param for param in model.parameters() if param.requires_grad]

    cos_sims = []
    max_g_kl_ale = 0.0
    max_g_ale_epi = 0.0
    
    with torch.enable_grad():
        for batch_idx, batch in enumerate(bundle.train_loader):
            if batch_idx >= num_batches:
                break
            
            # Zero grads
            model.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "concept_labels" in batch:
                inputs["concept_labels"] = batch["concept_labels"].to(device)
            
            # Forward pass
            outputs = model(**inputs)

            sigma_epi = outputs.get("sigma_epi", outputs.get("epistemic"))
            sigma_ale = outputs.get("sigma_ale", outputs.get("aleatoric"))

            # Compute KL loss
            kl_loss = outputs.get("kl_loss", outputs.get("kl", torch.tensor(0.0, device=device)))
            if isinstance(kl_loss, (int, float)):
                kl_loss = torch.tensor(float(kl_loss), device=device, requires_grad=True)

            # Compute aleatoric loss (prediction of unknown ratio)
            if "concept_labels" in inputs:
                unknown_ratio = (inputs["concept_labels"] == 1).float()
                aleatoric_pred = sigma_ale if sigma_ale is not None else outputs.get("aleatoric", torch.zeros_like(unknown_ratio))
                if aleatoric_pred.dim() == 1 and unknown_ratio.dim() > 1:
                    unknown_ratio = unknown_ratio.mean(dim=-1)
                elif aleatoric_pred.dim() > 1 and unknown_ratio.dim() == 1:
                    unknown_ratio = unknown_ratio.unsqueeze(-1).expand_as(aleatoric_pred)
                aleatoric_loss = torch.nn.functional.mse_loss(aleatoric_pred, unknown_ratio)
            else:
                aleatoric_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if kl_loss.requires_grad and aleatoric_loss.requires_grad:
                kl_full = torch.autograd.grad(
                    kl_loss, full_params, retain_graph=True, allow_unused=True
                )
                ale_full = torch.autograd.grad(
                    aleatoric_loss, full_params, retain_graph=True, allow_unused=True
                )
                kl_full_vec, _ = _flatten_grads(kl_full, full_params)
                ale_full_vec, _ = _flatten_grads(ale_full, full_params)

                if kl_full_vec.numel() > 0 and ale_full_vec.numel() > 0:
                    denom = kl_full_vec.norm() * ale_full_vec.norm()
                    cos_sims.append(0.0 if denom.item() == 0 else F.cosine_similarity(
                        kl_full_vec.unsqueeze(0), ale_full_vec.unsqueeze(0)
                    ).item())

                if ale_params:
                    kl_on_ale = torch.autograd.grad(
                        kl_loss, ale_params, retain_graph=True, allow_unused=True
                    )
                    _, kl_on_ale_norms = _flatten_grads(kl_on_ale, ale_params)
                    if kl_on_ale_norms.numel() > 0:
                        max_g_kl_ale = max(max_g_kl_ale, float(kl_on_ale_norms.max().item()))

                if epi_params:
                    ale_on_epi = torch.autograd.grad(
                        aleatoric_loss, epi_params, retain_graph=True, allow_unused=True
                    )
                    _, ale_on_epi_norms = _flatten_grads(ale_on_epi, epi_params)
                    if ale_on_epi_norms.numel() > 0:
                        max_g_ale_epi = max(max_g_ale_epi, float(ale_on_epi_norms.max().item()))

    # Statistics
    cos_sims = cos_sims if cos_sims else [0.0]
    import numpy as np
    cos_sims_np = np.array(cos_sims)
    mean_cos = float(cos_sims_np.mean())
    std_cos = float(cos_sims_np.std())
    abs_max_cos = float(np.abs(cos_sims_np).max())
    
    snapshot = GradIsoSnapshot(
        dataset=bundle.name,
        checkpoint="",
        num_batches=num_batches,
        cos_sim_kl_ale_per_batch=cos_sims,
        mean_cos_sim_mu_head=mean_cos,
        std_cos_sim_mu_head=std_cos,
        abs_max_cos_sim_mu_head=abs_max_cos,
        max_g_kl_on_sigma_ale=max_g_kl_ale,
        max_g_ale_on_sigma_epi=max_g_ale_epi,
    )
    
    return snapshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["cebab", "hatexplain", "goemotions", "sst2"])
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    bundle = LOADERS[args.dataset].load(
        tokenizer_name="distilbert-base-uncased",
        batch_size=4,
        max_length=128,
    )
    
    # Analyze
    print(f"Analyzing {args.num_batches} batches...")
    snapshot = analyze_checkpoint(model, bundle, num_batches=args.num_batches, device=device)
    snapshot.checkpoint = str(args.checkpoint)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_dict = {
        **asdict(snapshot),
        "summary": snapshot.summary,
    }
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

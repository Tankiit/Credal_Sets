"""SENN body for the HybridCredalSLVM shell."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.slvm_base import SLVMBody


class SENNBody(SLVMBody):
    def __init__(
        self,
        proj_dim: int,
        num_concepts: int,
        num_classes: int,
        hidden_dim: int = 128,
        stability_weight: float = 2e-4,
        stability_epsilon: float = 0.01,
    ):
        super().__init__()
        self.stability_weight = stability_weight
        self.stability_epsilon = stability_epsilon

        self.concept_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )
        self.relevance_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes * num_concepts),
        )

        self.num_concepts = num_concepts
        self.num_classes = num_classes

    def forward(self, h_concept, input_ids=None, attention_mask=None):
        mu = self.concept_net(h_concept)
        concept_scores = torch.sigmoid(mu)

        relevance = self.relevance_net(h_concept)
        relevance = relevance.view(-1, self.num_classes, self.num_concepts)
        relevance = torch.softmax(relevance, dim=-1)

        task_logits = torch.bmm(relevance, concept_scores.unsqueeze(-1)).squeeze(-1)

        stability = relevance.pow(2).mean(dim=(-1, -2))
        aux_loss = self.stability_weight * stability.mean()

        return {
            "mu": mu,
            "concept_scores": concept_scores,
            "task_logits": task_logits,
            "aux_loss": aux_loss,
        }

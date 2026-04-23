"""ProtoP body for the HybridCredalSLVM shell."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.slvm_base import SLVMBody


class ProtoPBody(SLVMBody):
    def __init__(
        self,
        proj_dim: int,
        num_concepts: int,
        num_classes: int,
        proto_dim: int = 128,
        num_protos_per_concept: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.num_protos_per_concept = num_protos_per_concept

        self.projector = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proto_dim),
        )
        self.prototypes = nn.Parameter(
            torch.randn(num_concepts, num_protos_per_concept, proto_dim) * 0.02
        )
        self.task_classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, h_concept, input_ids=None, attention_mask=None):
        z = self.projector(h_concept)
        z = torch.nn.functional.normalize(z, dim=-1)

        prototypes = torch.nn.functional.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bd,kpd->bkp", z, prototypes)
        concept_similarity, _ = similarity.max(dim=-1)

        concept_scores = torch.sigmoid(5.0 * concept_similarity)
        mu = torch.logit(concept_scores.clamp(1e-6, 1 - 1e-6))
        task_logits = self.task_classifier(concept_scores)

        return {
            "mu": mu,
            "concept_scores": concept_scores,
            "task_logits": task_logits,
            "aux_loss": None,
        }

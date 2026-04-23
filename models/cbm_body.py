"""
CBM body: standard concept bottleneck.

concept_scores = sigmoid(MLP(h_concept))
task_logits = linear(concept_scores)

This is the standard CBM formulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slvm_base import SLVMBody


class CBMBody(SLVMBody):
    def __init__(self, proj_dim: int, num_concepts: int, num_classes: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.concept_net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts),
        )
        self.task_classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, h_concept, input_ids=None, attention_mask=None):
        mu = self.concept_net(h_concept)
        concept_scores = torch.sigmoid(mu)
        task_logits = self.task_classifier(concept_scores)
        return {
            "mu": mu,
            "concept_scores": concept_scores,
            "task_logits": task_logits,
            "aux_loss": None,
        }
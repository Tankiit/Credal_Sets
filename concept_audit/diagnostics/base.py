from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass
class AuditState:
    c: torch.Tensor
    concepts: torch.Tensor
    labels: torch.Tensor
    model: object
    train_mask: torch.Tensor

    def __post_init__(self):
        n = len(self.c)
        if self.c.ndim != 2 or self.concepts.shape != (n, len(self.model.blocks)) or self.labels.shape != (n,):
            raise ValueError("Misaligned audit features, concept targets, or labels")
        if self.train_mask.shape != (n,) or self.train_mask.dtype != torch.bool:
            raise ValueError("train_mask must be a boolean sample mask")
        if not self.train_mask.any() or self.train_mask.all():
            raise ValueError("Audits require nonempty disjoint train and evaluation samples")


class Diagnostic(Protocol):
    name: str

    def compute(self, state: AuditState) -> torch.Tensor:
        """One scalar per declared concept block."""
        ...


class DiagnosticRegistry:
    def __init__(self, diagnostics=()):
        self.diagnostics = {}
        for diagnostic in diagnostics:
            self.register(diagnostic)

    def register(self, diagnostic):
        if diagnostic.name in self.diagnostics:
            raise ValueError(f"Duplicate diagnostic: {diagnostic.name}")
        self.diagnostics[diagnostic.name] = diagnostic

    def compute(self, state):
        result = {}
        for name, diagnostic in self.diagnostics.items():
            value = diagnostic.compute(state).detach().cpu()
            if value.shape != (len(state.model.blocks),) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must return one finite value per concept")
            result[name] = value.tolist()
        return result

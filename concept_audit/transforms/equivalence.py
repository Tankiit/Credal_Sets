"""Column convention c'=Ac; batches store rows, hence c' = c @ A.T."""
from copy import deepcopy
import math
import torch
from torch import nn
from concept_audit.models.base import ConceptModel


def admissible_transform(readout, strength=0.5, seed=0):
    """A = exp(N B) with R N = 0. Invertible in exact arithmetic; validated later."""
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    r = readout.matrix
    _, s, vh = torch.linalg.svd(r, full_matrices=True)
    threshold = max(r.shape) * torch.finfo(r.dtype).eps * s.max()
    rank = int((s > threshold).sum())
    n = vh[rank:].T
    if n.shape[1] == 0:
        return torch.eye(r.shape[1], dtype=r.dtype, device=r.device)
    generator = torch.Generator(device=r.device).manual_seed(seed)
    b = torch.randn(n.shape[1], r.shape[1], generator=generator, device=r.device, dtype=r.dtype)
    return torch.matrix_exp(strength * (n @ b) / math.sqrt(r.shape[1]))


class ReparameterizedModel(ConceptModel):
    """Owns a frozen snapshot; transports interventions through A by default.

    Coordinate-defined interventions can separately be called on this model's
    readout/blocks to investigate their parameterization dependence.
    """
    def __init__(self, model, a, tol=1e-5):
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if not isinstance(model.head, nn.Linear):
            raise TypeError("Exact compensation requires an explicit nn.Linear task head")
        r = model.readout.matrix
        a = torch.as_tensor(a, dtype=r.dtype, device=r.device).detach().clone()
        if a.shape != (r.shape[1], r.shape[1]) or not torch.isfinite(a).all():
            raise ValueError("A must be a finite square latent transformation")
        if not torch.allclose(r @ a, r, atol=tol, rtol=0):
            raise ValueError("A is not admissible: R A != R")
        try:
            inv = torch.linalg.solve(a, torch.eye(len(a), dtype=a.dtype, device=a.device))
        except RuntimeError as exc:
            raise ValueError("A must be invertible") from exc
        if not torch.isfinite(inv).all() or float(torch.linalg.cond(a)) > 1 / max(tol, torch.finfo(a.dtype).eps):
            raise ValueError("A is too ill-conditioned for the requested tolerance")
        head = deepcopy(model.head)
        with torch.no_grad():
            head.weight.copy_(torch.linalg.solve(a.T, model.head.weight.T).T)
        super().__init__(deepcopy(model.readout), head, model.blocks)
        self.source = deepcopy(model).eval()
        self.register_buffer("a", a)
        self.register_buffer("inverse", inv)
        self.requires_grad_(False)
        self.eval()

    def encode(self, z):
        return self.source.encode(z) @ self.a.T

    def intervene(self, c, concept_id, value):
        return self.source.intervene(c @ self.inverse.T, concept_id, value) @ self.a.T

    def substitute(self, c, concept_id, value):
        """Transport replacement; value is a block in the original source basis."""
        return self.source.substitute(c @ self.inverse.T, concept_id, value) @ self.a.T

    def substitute_donor(self, c, concept_id, donor):
        old_c, old_donor = c @ self.inverse.T, donor @ self.inverse.T
        return self.source.substitute_donor(old_c, concept_id, old_donor) @ self.a.T

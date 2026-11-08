"""NMF as a latent model: X ~ C H, everything nonnegative.

Observable: the factor product. Constraint: nonnegativity of both factors.

Known admissible transformations **under these conditions**: positive diagonal
rescaling and permutation, i.e. the positive monomial matrices, with
``C' = C D P`` and ``H' = P^{-1} D^{-1} H``.

Deliberately *not* a claim that these are the only NMF ambiguities in general.
Whether a factorisation is essentially unique depends on the geometry of the
data cone (separability, sufficient spread); for many matrices there are more
admissible transforms and for some there are none beyond scaling. What this
provides is a family whose *known* transformations can be audited, not a
uniqueness theorem.
"""
import torch

from concept_audit.core.constraints import Nonnegative
from concept_audit.core.latent_model import LatentModel, apply_transform
from concept_audit.core.observables import ReconstructionObservable


class NMFModel(LatentModel):
    family = "nmf"
    variant = "multiplicative"

    def __init__(self, h, reference_codes=None):
        super().__init__()
        h = torch.as_tensor(h).double()
        if (h < 0).any():
            raise ValueError("H must be nonnegative")
        self.register_buffer("h", h)                      # (k, d)
        self.reference_codes = None if reference_codes is None else torch.as_tensor(reference_codes).double()

    def latent_dim(self):
        return int(self.h.shape[0])

    def encode(self, z):
        raise NotImplementedError("NMF codes come from the fit; pass C directly")

    def decode(self, c):
        return torch.as_tensor(c).double() @ self.h

    def observables(self, c):
        return {"reconstruction": self.decode(c)}

    def observable_maps(self):
        return [ReconstructionObservable(self.h)]

    def constraints(self):
        return [Nonnegative("c", "h")]

    def constraint_parts(self, a, c=None):
        moved = self.compensate(a)
        parts = {"h": None if moved is None else moved.h}
        source = c if c is not None else self.reference_codes
        if source is not None:
            parts["c"] = apply_transform(source, a)
        return parts

    def compensate(self, a):
        """``H' = A^{-T} H`` so ``C' H' = C H``; None if H leaves the cone.

        The product may be right while the factorisation has negative entries --
        and that is not an NMF.
        """
        a = torch.as_tensor(a).double()
        try:
            h_new = torch.linalg.inv(a).T @ self.h
        except RuntimeError:
            return None
        if (h_new < -1e-12).any():
            return None
        return NMFModel(h_new.clamp(min=0.0), self.reference_codes)

    def rescale(self, scale):
        scale = torch.as_tensor(scale).double()
        if (scale <= 0).any():
            raise ValueError("NMF rescaling must be strictly positive")
        return torch.diag(scale)

    def permutation(self, order):
        return torch.eye(self.latent_dim()).double()[torch.as_tensor(order)]

    def monomial(self, seed=0, low=0.5, high=2.0):
        """Positive diagonal times permutation: the known NMF ambiguity."""
        generator = torch.Generator().manual_seed(seed)
        k = self.latent_dim()
        order = torch.randperm(k, generator=generator)
        scale = low + (high - low) * torch.rand(k, generator=generator).double()
        return self.permutation(order) @ torch.diag(scale)

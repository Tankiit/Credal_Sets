"""Observables: what training and evaluation hold fixed.

A model's equivalence class is defined *relative to* its observables. For a CBM
the observable is the supervision readout R c; for PCA it is the reconstruction;
for NMF the factor product. One concept, so families need not each invent their
own notion of "what must not change".

Two kinds, and conflating them is a real bug:

* ``fixed_map=True`` -- the map itself is part of the specification, so it must
  not move. Supervision *is* R, so admissibility means ``R A = R``, not merely
  that some value happens to be preserved.
* ``fixed_map=False`` -- only the value is held fixed and the map moves with the
  model. PCA's decoder becomes ``A^{-T} V`` precisely so the output is
  unchanged; checking it with the *old* decoder compares the wrong things.
"""
from abc import ABC, abstractmethod

import torch


class Observable(ABC):
    name = "observable"
    fixed_map = False

    @abstractmethod
    def __call__(self, c):
        """Evaluate on (n, latent_dim) latents."""

    def error(self, before, after):
        before, after = torch.as_tensor(before), torch.as_tensor(after)
        if before.shape != after.shape:
            raise ValueError(f"{self.name}: shape changed {tuple(before.shape)} -> {tuple(after.shape)}")
        return float((before - after).abs().max()) if before.numel() else 0.0


class LinearObservable(Observable):
    """``c -> c M^T`` for a fixed (m, latent_dim) matrix."""

    def __init__(self, matrix, name="linear", fixed_map=True):
        matrix = torch.as_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got {tuple(matrix.shape)}")
        self.matrix = matrix.detach().clone()
        self.name = name
        self.fixed_map = fixed_map

    def __call__(self, c):
        return torch.as_tensor(c) @ self.matrix.T.to(torch.as_tensor(c).dtype)

    @property
    def latent_dim(self):
        return int(self.matrix.shape[1])

    @property
    def rank(self):
        return int(torch.linalg.matrix_rank(self.matrix)) if self.matrix.numel() else 0

    @property
    def nullity(self):
        """dim ker(M): latent directions this observable cannot see."""
        return self.latent_dim - self.rank

    def matrix_error(self, a):
        """``max |M A - M|``: holds for every latent, not only sampled ones."""
        if not self.matrix.numel():
            return 0.0
        a = torch.as_tensor(a, dtype=self.matrix.dtype)
        return float((self.matrix @ a - self.matrix).abs().max())


class ReconstructionObservable(Observable):
    """``c -> c D`` for a decoder D. PCA, NMF and SAE all hold this fixed."""

    name = "reconstruction"
    fixed_map = False

    def __init__(self, decoder, mean=None):
        self.decoder = torch.as_tensor(decoder)
        self.mean = None if mean is None else torch.as_tensor(mean)

    def __call__(self, c):
        out = torch.as_tensor(c, dtype=self.decoder.dtype) @ self.decoder
        return out if self.mean is None else out + self.mean


def from_readout(readout, name=None):
    """Wrap an existing ``LinearReadout`` (it already exposes ``.matrix``)."""
    return LinearObservable(readout.matrix, name=name or type(readout).__name__, fixed_map=True)

"""PCA as a latent model.

Observable: the reconstruction. Constraint: an orthonormal basis.

The admissible class depends entirely on the eigenvalue spectrum:

* **distinct eigenvalues** -- only sign flips, plus permutations if component
  order is treated as unobservable. A finite group.
* **repeated eigenvalues** -- any orthogonal rotation *within* an eigenspace: a
  continuous group of dimension m(m-1)/2 for a multiplicity-m block.

So how non-identified PCA is depends on the data, not the fitting code. Note
that exact multiplicity never occurs in a finite sample, so "repeated" is always
a tolerance choice -- which ``rtol`` makes explicit rather than hiding.
"""
import torch

from concept_audit.core.constraints import Orthonormal
from concept_audit.core.latent_model import LatentModel
from concept_audit.core.observables import ReconstructionObservable


class PCAModel(LatentModel):
    family = "pca"
    variant = "linear"

    def __init__(self, components, mean=None, eigenvalues=None):
        super().__init__()
        components = torch.as_tensor(components).double()
        self.register_buffer("components", components)          # (k, d), rows orthonormal
        self.register_buffer("mean", torch.zeros(components.shape[1]).double()
                             if mean is None else torch.as_tensor(mean).double())
        self.eigenvalues = None if eigenvalues is None else torch.as_tensor(eigenvalues).double()

    @classmethod
    def fit(cls, x, n_components):
        x = torch.as_tensor(x).double()
        mean = x.mean(0)
        _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
        k = n_components
        return cls(vh[:k], mean, (s[:k] ** 2) / max(len(x) - 1, 1))

    def latent_dim(self):
        return int(self.components.shape[0])

    def encode(self, z):
        return (torch.as_tensor(z).double() - self.mean) @ self.components.T

    def decode(self, c):
        return torch.as_tensor(c).double() @ self.components + self.mean

    def observables(self, c):
        return {"reconstruction": self.decode(c)}

    def observable_maps(self):
        return [ReconstructionObservable(self.components, self.mean)]

    def constraints(self):
        return [Orthonormal("components")]

    def constraint_parts(self, a, c=None):
        moved = self.compensate(a)
        return {"components": None if moved is None else moved.components}

    def compensate(self, a):
        """``V' = A^{-T} V`` keeps ``c' V' = (A c)(A^{-T} V) = c V``."""
        a = torch.as_tensor(a).double()
        try:
            inv_t = torch.linalg.inv(a).T
        except RuntimeError:
            return None
        return PCAModel(inv_t @ self.components, self.mean, self.eigenvalues)

    def eigenvalue_blocks(self, rtol=1e-6):
        """Group component indices by (near-)equal eigenvalue."""
        if self.eigenvalues is None:
            return [[i] for i in range(self.latent_dim())]
        blocks = []
        for i, lam in enumerate(self.eigenvalues.tolist()):
            if blocks and abs(lam - self.eigenvalues[blocks[-1][0]].item()) <= rtol * max(abs(lam), 1e-12):
                blocks[-1].append(i)
            else:
                blocks.append([i])
        return blocks

    def admissible_dimension(self, rtol=1e-6):
        """Dimension of the continuous part: sum of m(m-1)/2 over blocks."""
        return sum(len(b) * (len(b) - 1) // 2 for b in self.eigenvalue_blocks(rtol))

    def sign_flip(self, signs):
        signs = torch.as_tensor(signs).double()
        if not torch.isin(signs, torch.tensor([-1.0, 1.0]).double()).all():
            raise ValueError("signs must be +1 or -1")
        return torch.diag(signs)

    def permutation(self, order):
        return torch.eye(self.latent_dim()).double()[torch.as_tensor(order)]

    def block_rotation(self, seed=0, rtol=1e-6):
        """A random rotation inside each repeated-eigenvalue block.

        The transform that makes PCA non-identified in a way sign flips do not:
        it mixes components continuously while leaving every reconstruction and
        every eigenvalue untouched.
        """
        generator = torch.Generator().manual_seed(seed)
        k = self.latent_dim()
        a = torch.eye(k).double()
        for block in self.eigenvalue_blocks(rtol):
            m = len(block)
            if m < 2:
                continue
            q, r = torch.linalg.qr(torch.randn(m, m, generator=generator).double())
            q = q * torch.sign(torch.diag(r))
            a[torch.tensor(block)[:, None], torch.tensor(block)[None, :]] = q
        return a

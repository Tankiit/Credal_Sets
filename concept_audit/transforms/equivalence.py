"""Column convention c'=Ac; batches store rows, hence c' = c @ A.T."""
from copy import deepcopy
import math
import torch
from torch import nn
from concept_audit.models.base import ConceptModel


def nullspace_basis(matrix):
    """Orthonormal columns spanning ker(matrix), via the trailing right singular vectors."""
    m = torch.as_tensor(matrix)
    _, s, vh = torch.linalg.svd(m, full_matrices=True)
    threshold = max(m.shape) * torch.finfo(m.dtype).eps * (s.max() if s.numel() else 1.0)
    rank = int((s > threshold).sum())
    return vh[rank:].T


def stabilizer_parameterization(readout):
    """The stabilizer of R: every A with R A = R is A = I + N C.

    Columns of N span ker(R), so R N = 0 and R(I + N C) = R for any C.
    Conversely R A = R means R(A - I) = 0, so every column of A - I
    lies in ker(R) and is therefore N c. The parameterisation is exact and
    complete: it is the whole stabilizer, not a subset.

    Returns (N, build); build(C) takes C of shape (dim ker(R), k).
    """
    r = torch.as_tensor(readout.matrix if hasattr(readout, "matrix") else readout)
    k = r.shape[1]
    n = nullspace_basis(r)
    identity = torch.eye(k, dtype=r.dtype, device=r.device)

    def build(c):
        if n.shape[1] == 0:
            return identity.clone()
        c = torch.as_tensor(c, dtype=r.dtype, device=r.device)
        if tuple(c.shape) != (n.shape[1], k):
            raise ValueError(f"C must have shape {(n.shape[1], k)}, got {tuple(c.shape)}")
        return identity + n @ c

    return n, build


def admissible_transform(readout, strength=0.5, seed=0, method="stabilizer",
                         condition_limit=1e3, max_tries=200):
    """Sample A with R A = R.

    method="stabilizer" draws A = I + N C with C Gaussian, covering the
    whole stabilizer (see :func:). Ill-conditioned
    draws are rejected, so the caller gets an invertible A.

    method="exponential" draws A = exp(N B) instead. Every such A is
    admissible and invertible by construction and so never needs rejecting --
    but the image of the exponential map is only the identity component, so it
    samples a strictly smaller part of the same group. Prefer it when a
    continuous path from the identity matters; do not use it to ask how large
    the equivalence class is.
    """
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    if method not in {"stabilizer", "exponential"}:
        raise ValueError(f"method must be 'stabilizer' or 'exponential', got {method!r}")

    r = readout.matrix if hasattr(readout, "matrix") else torch.as_tensor(readout)
    n, build = stabilizer_parameterization(r)
    if n.shape[1] == 0:
        return torch.eye(r.shape[1], dtype=r.dtype, device=r.device)

    generator = torch.Generator(device=r.device).manual_seed(seed)
    for _ in range(max_tries):
        b = torch.randn(n.shape[1], r.shape[1], generator=generator, device=r.device, dtype=r.dtype)
        scaled = strength * b / math.sqrt(r.shape[1])
        a = build(scaled) if method == "stabilizer" else torch.matrix_exp(n @ scaled)
        cond = float(torch.linalg.cond(a))
        if math.isfinite(cond) and cond <= condition_limit and torch.allclose(r @ a, r, atol=1e-6, rtol=0):
            return a
    raise RuntimeError("Could not sample a well-conditioned admissible transform")


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

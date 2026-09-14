"""Sparse autoencoder: c = relu(zW + b), reconstruction c D.

Observable: reconstruction. Constraints: the decoder norm convention, the
sparsity penalty, and -- the one that bites -- whether the transformed
*encoder* still emits ``A c``.

Preserving reconstruction only needs ``D' = A^{-T} D``, which any invertible A
can do. But ReLU does not commute with a general A: ``relu(zW')`` equals
``relu(zW) A^T`` only when ``A^T`` is a nonnegative monomial matrix. A dense A
yields the same reconstruction from a model that is no longer the same encoder.

Two further conditions are choices, not facts, and the class changes with them:

* **Which penalty.** ``l0`` is invariant to positive rescaling; ``l1`` is not,
  since the sum of activations scales. This is why L1 SAEs pin decoder norms --
  the norm constraint is what removes the scaling ambiguity.
* **Whether unit identity is observable.** If units are anonymous, permutation
  is admissible. If a unit has been labelled -- "unit 7 detects the Golden Gate
  Bridge" -- permuting changes a claim someone relies on. Set
  ``unit_identity_observable=True`` for that reading.

So an SAE's admissible class is much smaller than its reconstruction-preserving
class, and auditing reconstruction alone would overstate its non-identifiability.
"""
import torch

from concept_audit.core.constraints import PrecomputedError, PreservesSupport, UnitNormRows
from concept_audit.core.latent_model import LatentModel, apply_transform
from concept_audit.core.observables import ReconstructionObservable


class SAEModel(LatentModel):
    family = "sae"
    variant = "relu"

    def __init__(self, w, b, d, reference_inputs=None, enforce_unit_norm=False,
                 penalty="l1", unit_identity_observable=False):
        super().__init__()
        if penalty not in {"l0", "l1", "none"}:
            raise ValueError(f"penalty must be 'l0', 'l1' or 'none', got {penalty!r}")
        self.register_buffer("w", torch.as_tensor(w).double())     # (d, k)
        self.register_buffer("b", torch.as_tensor(b).double())     # (k,)
        self.register_buffer("d", torch.as_tensor(d).double())     # (k, d)
        self.reference_inputs = None if reference_inputs is None else torch.as_tensor(reference_inputs).double()
        self.enforce_unit_norm = enforce_unit_norm
        self.penalty = penalty
        self.unit_identity_observable = unit_identity_observable

    def latent_dim(self):
        return int(self.w.shape[1])

    def encode(self, z):
        return torch.relu(torch.as_tensor(z).double() @ self.w + self.b)

    def decode(self, c):
        return torch.as_tensor(c).double() @ self.d

    def observables(self, c):
        return {"reconstruction": self.decode(c)}

    def observable_maps(self):
        return [ReconstructionObservable(self.d)]

    def constraints(self):
        items = [PrecomputedError("encoder_consistency"), PrecomputedError("penalty_change")]
        if self.unit_identity_observable:
            items.append(PreservesSupport())
        if self.enforce_unit_norm:
            items.append(UnitNormRows("decoder"))
        return items

    def penalty_value(self, c):
        c = torch.as_tensor(c)
        if self.penalty == "l0":
            return float((c.abs() > 1e-8).sum())
        if self.penalty == "l1":
            return float(c.abs().sum())
        return 0.0

    def penalty_change(self, a, c=None):
        """Relative change in the sparsity penalty under ``c -> A c``."""
        if self.penalty == "none":
            return 0.0
        if c is None:
            if self.reference_inputs is None:
                return 0.0
            c = self.encode(self.reference_inputs)
        before = self.penalty_value(c)
        after = self.penalty_value(apply_transform(c, a))
        return 0.0 if before == 0 else abs(after - before) / abs(before)

    def encoder_consistency_error(self, a):
        """``max |encode'(z) - A encode(z)|``: does the moved encoder emit A c?"""
        if self.reference_inputs is None:
            return 0.0
        moved = self.compensate(a)
        if moved is None:
            return float("inf")
        target = apply_transform(self.encode(self.reference_inputs), a)
        return float((moved.encode(self.reference_inputs) - target).abs().max())

    def constraint_parts(self, a, c=None):
        moved = self.compensate(a)
        parts = {"decoder": None if moved is None else moved.d,
                 "encoder_consistency": self.encoder_consistency_error(a)}
        source = c
        if source is None and self.reference_inputs is not None:
            source = self.encode(self.reference_inputs)
        parts["penalty_change"] = self.penalty_change(a, source)
        if source is not None:
            parts["c"] = source
            parts["c_transformed"] = apply_transform(source, a)
        return parts

    def compensate(self, a):
        """``D' = A^{-T} D``, ``W' = W A^T``, ``b' = b A^T``.

        Always reconstruction-preserving. Whether the result is a valid SAE is
        decided by the constraints, not here.
        """
        a = torch.as_tensor(a).double()
        try:
            inv_t = torch.linalg.inv(a).T
        except RuntimeError:
            return None
        return SAEModel(self.w @ a.T, self.b @ a.T, inv_t @ self.d,
                        self.reference_inputs, self.enforce_unit_norm,
                        self.penalty, self.unit_identity_observable)

    def permutation(self, order):
        return torch.eye(self.latent_dim()).double()[torch.as_tensor(order)]

    def positive_scaling(self, scale):
        scale = torch.as_tensor(scale).double()
        if (scale <= 0).any():
            raise ValueError("scaling must be strictly positive to commute with ReLU")
        return torch.diag(scale)

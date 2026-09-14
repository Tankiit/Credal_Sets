"""Family membership conditions: the other half of admissibility.

Observable invariance is not sufficient. A transformed NMF can reproduce ``Z H``
exactly while containing negative entries, which puts it outside the NMF family
altogether. Constraints are what actually differ between families, and so are
the reason PCA, NMF, SAE and CBM have different equivalence classes despite
sharing this audit machinery.
"""
from abc import ABC, abstractmethod

import torch


class Constraint(ABC):
    name = "constraint"

    @abstractmethod
    def violation(self, **parts):
        """0.0 when satisfied; larger means further outside the family."""


class Nonnegative(Constraint):
    """Every named part must stay >= 0. The NMF condition."""

    name = "nonnegative"

    def __init__(self, *keys):
        self.keys = keys or ("c", "h")

    def violation(self, **parts):
        worst = 0.0
        for key in self.keys:
            value = parts.get(key)
            if value is not None:
                worst = max(worst, float(torch.as_tensor(value).clamp(max=0.0).abs().max()))
        return worst


class Orthonormal(Constraint):
    """Rows of ``key`` must be orthonormal. The PCA basis condition."""

    name = "orthonormal"

    def __init__(self, key="components"):
        self.key = key

    def violation(self, **parts):
        v = parts.get(self.key)
        if v is None:
            return 0.0
        v = torch.as_tensor(v)
        return float((v @ v.T - torch.eye(v.shape[0], dtype=v.dtype)).abs().max())


class UnitNormRows(Constraint):
    """Rows of ``key`` must have unit norm. Common SAE dictionary condition."""

    name = "unit_norm_rows"

    def __init__(self, key="decoder"):
        self.key = key

    def violation(self, **parts):
        d = parts.get(self.key)
        if d is None:
            return 0.0
        norms = torch.as_tensor(d).norm(dim=1)
        return float((norms - 1.0).abs().max()) if norms.numel() else 0.0


class PreservesSupport(Constraint):
    """The sparsity *pattern* must be unchanged, entry by entry.

    Only meaningful where a unit's index is itself a claim -- "unit 7 detects
    the Golden Gate Bridge". Where units are anonymous, a permutation changes
    this while changing nothing anyone relies on, so use a penalty check instead.
    """

    name = "preserves_support"

    def __init__(self, tol=1e-8):
        self.tol = tol

    def violation(self, **parts):
        before, after = parts.get("c"), parts.get("c_transformed")
        if before is None or after is None:
            return 0.0
        a = torch.as_tensor(before).abs() > self.tol
        b = torch.as_tensor(after).abs() > self.tol
        return float((a != b).float().mean())


class PrecomputedError(Constraint):
    """Reads a violation the model already computed, e.g. encoder consistency.

    Some conditions need a forward pass rather than a predicate over parameters.
    The model computes the number and reports it into the same dict.
    """

    def __init__(self, name, key=None):
        self.name = name
        self.key = key or name

    def violation(self, **parts):
        value = parts.get(self.key)
        return 0.0 if value is None else float(value)


def evaluate(constraints, **parts):
    return {c.name: c.violation(**parts) for c in constraints}

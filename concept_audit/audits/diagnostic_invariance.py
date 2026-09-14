"""The core object: is a diagnostic identifiable under a model's equivalence class?

    D is identifiable under the model  <=>  D(c) = D(A c) for all A in G.

One admissible A that moves D is a counterexample, and that is the experiment.
Works for any family, because admissibility is decided by the model
(``LatentModel.admissible``) while the diagnostics only see latents.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from concept_audit.core.latent_model import apply_transform


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@dataclass
class DiagnosticDelta:
    name: str
    before: np.ndarray
    after: np.ndarray
    max_abs_change: float
    relative_change: float
    spearman: float = None

    @property
    def invariant(self):
        return self.max_abs_change <= 1e-8

    def summary(self):
        return {"max_abs_change": self.max_abs_change, "relative_change": self.relative_change,
                "spearman": self.spearman, "invariant": self.invariant}


@dataclass
class InvarianceReport:
    equivalence: object
    deltas: dict = field(default_factory=dict)

    @property
    def moved(self):
        """Diagnostics that changed under a transform the model calls admissible."""
        return sorted(n for n, d in self.deltas.items() if not d.invariant)

    def to_dict(self):
        return {"equivalence": self.equivalence.to_dict(),
                "diagnostics": {n: d.summary() for n, d in self.deltas.items()},
                "moved": self.moved}


def _spearman(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    from scipy.stats import rankdata

    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def audit_diagnostic_invariance(model, c, a, diagnostics, tol=1e-8, require_admissible=True):
    """Compare every diagnostic before and after an admissible transform.

    ``diagnostics`` maps latents to values: ``{name: f(c)}``. Raises when A is
    inadmissible unless ``require_admissible=False`` -- a diagnostic moving
    under an *inadmissible* transform says nothing, and reporting it as a
    finding would be wrong.
    """
    equivalence = model.admissible(a, c=c, tol=tol)
    if require_admissible and not equivalence.admissible:
        raise ValueError(
            f"A is not admissible for this model ({equivalence}); a diagnostic "
            "changing under it is not evidence of non-identifiability"
        )
    if not isinstance(diagnostics, dict):
        diagnostics = {getattr(d, "name", f"d{i}"): d for i, d in enumerate(diagnostics)}

    moved_c = apply_transform(c, a)
    deltas = {}
    for name, fn in diagnostics.items():
        before, after = _as_numpy(fn(c)).astype(float), _as_numpy(fn(moved_c)).astype(float)
        denom = max(float(np.abs(before).max()) if before.size else 0.0, 1e-12)
        deltas[name] = DiagnosticDelta(
            name=name, before=before, after=after,
            max_abs_change=float(np.abs(after - before).max()) if before.size else 0.0,
            relative_change=float(np.abs(after - before).max() / denom) if before.size else 0.0,
            spearman=_spearman(before, after),
        )
    return InvarianceReport(equivalence=equivalence, deltas=deltas)


def audit_over_transforms(model, c, transforms, diagnostics, **kwargs):
    return [audit_diagnostic_invariance(model, c, a, diagnostics, **kwargs) for a in transforms]

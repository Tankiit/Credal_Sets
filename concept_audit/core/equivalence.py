"""The generic equivalence result, shared by every family."""
from dataclasses import asdict, dataclass, field

import torch


@dataclass
class EquivalenceResult:
    """Whether A is admissible, and exactly how that was checked."""

    admissible: bool
    observable_error: float
    prediction_error: float = None
    constraint_error: dict = field(default_factory=dict)
    condition_number: float = float("nan")
    tol: float = 1e-8
    detail: dict = field(default_factory=dict)

    @property
    def worst_constraint(self):
        return max(self.constraint_error.values(), default=0.0)

    def to_dict(self):
        return asdict(self)

    def __bool__(self):
        return self.admissible

    def __str__(self):
        verdict = "admissible" if self.admissible else "NOT admissible"
        pred = "n/a" if self.prediction_error is None else f"{self.prediction_error:.2e}"
        return (f"{verdict}: observable_error={self.observable_error:.2e} "
                f"prediction_error={pred} worst_constraint={self.worst_constraint:.2e} "
                f"cond={self.condition_number:.3g}")


def decide(observable_error, prediction_error=None, constraint_error=None,
           condition_number=float("nan"), tol=1e-8, detail=None):
    """Apply the admissibility rule in one place.

    Admissible means every observable unchanged, family constraints still met,
    and predictions unchanged. All three, not any of them.
    """
    constraint_error = dict(constraint_error or {})
    ok = observable_error <= tol and max(constraint_error.values(), default=0.0) <= tol
    if prediction_error is not None:
        ok = ok and prediction_error <= tol
    return EquivalenceResult(
        admissible=bool(ok),
        observable_error=float(observable_error),
        prediction_error=None if prediction_error is None else float(prediction_error),
        constraint_error=constraint_error,
        condition_number=float(condition_number),
        tol=tol,
        detail=dict(detail or {}),
    )


def audit_equivalence(model, c, a, tol=1e-8):
    """Generic entry point: ``audit_equivalence(model, c, A)``."""
    return model.admissible(a, c=c, tol=tol)

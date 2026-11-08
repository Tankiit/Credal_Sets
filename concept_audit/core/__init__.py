"""model family -> admissible equivalence class -> diagnostic invariance."""
from concept_audit.core.constraints import (
    Constraint, Nonnegative, Orthonormal, PrecomputedError, PreservesSupport, UnitNormRows, evaluate,
)
from concept_audit.core.equivalence import EquivalenceResult, audit_equivalence, decide
from concept_audit.core.latent_model import LatentModel, apply_transform
from concept_audit.core.observables import (
    LinearObservable, Observable, ReconstructionObservable, from_readout,
)

__all__ = [
    "LatentModel", "apply_transform",
    "Observable", "LinearObservable", "ReconstructionObservable", "from_readout",
    "Constraint", "Nonnegative", "Orthonormal", "UnitNormRows", "PreservesSupport",
    "PrecomputedError", "evaluate",
    "EquivalenceResult", "audit_equivalence", "decide",
]

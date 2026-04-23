"""
Uncertainty metrics and correlation utilities shared across trainers and
analysis scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

import numpy as np
from scipy import stats


@dataclass
class UncertaintyMetrics:
    """Single-pass evaluation metrics for a HybridCredalCBM checkpoint."""
    accuracy: float = 0.0
    loss: float = 0.0

    # Sigma summary statistics
    mean_sigma_epi: float = 0.0
    std_sigma_epi: float = 0.0
    mean_sigma_ale: float = 0.0
    std_sigma_ale: float = 0.0

    # Aggregate uncertainties per sample
    mean_eu: float = 0.0
    std_eu: float = 0.0
    mean_au: float = 0.0
    std_au: float = 0.0

    # Correlations — the validation signals for the paper
    rho_eu_au: float = 0.0
    p_eu_au: float = 1.0
    rho_eu_error: float = 0.0
    p_eu_error: float = 1.0
    rho_ale_entropy: float = 0.0
    p_ale_entropy: float = 1.0

    # Per-concept breakdown
    concept_accs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, dict) else v)
                for k, v in asdict(self).items()}


def compute_uncertainty_correlations(
    eu: np.ndarray,          # [N] or [N, K]
    au: np.ndarray,          # [N] or [N, K]
    errors: np.ndarray,      # [N] task errors (0/1)
    annotator_entropy: np.ndarray | None = None,  # [N] or [N, K]
) -> Dict[str, float]:
    """
    Compute Spearman correlations for the three headline relationships.
    Returns NaN-safe dict: if a statistic is undefined (constant series),
    returns 0.0 for rho and 1.0 for p-value.
    """
    eu_s = eu.mean(axis=-1) if eu.ndim > 1 else eu
    au_s = au.mean(axis=-1) if au.ndim > 1 else au

    def _spearman(x, y):
        if x.std() == 0 or y.std() == 0:
            return 0.0, 1.0
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    rho_eu_au, p_eu_au = _spearman(eu_s, au_s)
    rho_eu_err, p_eu_err = _spearman(eu_s, errors.astype(float))

    rho_ale_ent, p_ale_ent = 0.0, 1.0
    if annotator_entropy is not None:
        ent_s = annotator_entropy.mean(axis=-1) if annotator_entropy.ndim > 1 else annotator_entropy
        rho_ale_ent, p_ale_ent = _spearman(au_s, ent_s)

    return {
        "rho_eu_au": rho_eu_au, "p_eu_au": p_eu_au,
        "rho_eu_error": rho_eu_err, "p_eu_error": p_eu_err,
        "rho_ale_entropy": rho_ale_ent, "p_ale_entropy": p_ale_ent,
    }
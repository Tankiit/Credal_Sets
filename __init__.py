"""
Variational Credal Concept Bottleneck Model

A unified framework for interpretable ML with robust uncertainty quantification.
"""

__version__ = "0.1.0"

# Import key classes and functions for easy access
from .f_divergences import (
    FDivergence,
    KLDivergence,
    ReverseKLDivergence,
    HellingerDivergence,
    AlphaDivergence,
    ChiSquareDivergence,
    TotalVariationDivergence,
    get_divergence,
    compare_divergences,
)

from .credal_sets import (
    CredalSet,
    credal_kl_divergence,
    closed_form_credal_kl_binary,
    hellinger_distance,
    interval_propagate_linear,
    interval_propagate_monotone,
    compute_coverage,
    compute_average_width,
    compute_sharpness,
)

from .variational_credal_cbm import (
    VariationalLinear,
    VariationalCredalCBM,
    FDivergenceLoss,
    TheoreticalGuarantees,
    create_encoder,
)

__all__ = [
    # f-divergences
    "FDivergence",
    "KLDivergence",
    "ReverseKLDivergence",
    "HellingerDivergence",
    "AlphaDivergence",
    "ChiSquareDivergence",
    "TotalVariationDivergence",
    "get_divergence",
    "compare_divergences",
    
    # Credal sets
    "CredalSet",
    "credal_kl_divergence",
    "closed_form_credal_kl_binary",
    "hellinger_distance",
    "interval_propagate_linear",
    "interval_propagate_monotone",
    "compute_coverage",
    "compute_average_width",
    "compute_sharpness",
    
    # Model
    "VariationalLinear",
    "VariationalCredalCBM",
    "FDivergenceLoss",
    "TheoreticalGuarantees",
    "create_encoder",
]

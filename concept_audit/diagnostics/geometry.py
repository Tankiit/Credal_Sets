"""Geometry diagnostics: kNN structure and the blindspot score.

The single implementation. ``blindspot_analysis`` imports from here rather than
keeping a parallel copy, so the standalone exploratory pipeline and the
identifiability audit compute the same numbers by construction.

These are the diagnostics whose invariance is most in doubt: a transform that
supervision cannot see still moves the latent geometry, so anything built from
distances or neighbourhoods can change while every observable is fixed.

**On ranks and ties.** ``rankdata(..., method="ordinal")`` breaks ties by row
order, so two points with *identical* metric values can land in opposite
quadrants purely because of their position in the array -- and that reads as a
finding. Everything here uses ``method="average"``, the primary result is the
continuous score, and the band labels are for visualisation only.
"""
import numpy as np
from scipy.stats import rankdata

DEFAULT_METRIC = "cosine"


def _neighbors(embeddings, k, metric=DEFAULT_METRIC):
    from sklearn.neighbors import NearestNeighbors

    embeddings = np.asarray(embeddings)
    if k >= len(embeddings):
        raise ValueError(f"k={k} needs more than {len(embeddings)} points")
    index = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(embeddings)
    distances, indices = index.kneighbors(embeddings)
    return distances[:, 1:], indices[:, 1:]      # drop each point itself


def knn_label_purity(embeddings, labels, k, metric=DEFAULT_METRIC):
    """Fraction of each point's k nearest OTHER embeddings sharing its label.

    A local, label-aware, aleatoric-flavoured proxy: low purity means a locally
    confused region. It says nothing about whether the model has seen enough
    data near the point, which is what :func:`knn_mean_distance` is for.
    """
    labels = np.asarray(labels)
    _, indices = _neighbors(embeddings, k, metric)
    return (labels[indices] == labels[:, None]).mean(axis=1)


def knn_mean_distance(embeddings, k, metric=DEFAULT_METRIC):
    """Mean distance to each point's k nearest OTHER embeddings, label-agnostic.

    A density-based, epistemic-flavoured proxy: points far from everything else
    are where a frozen deterministic model has least basis for any prediction,
    independent of whether nearby points agree on a label.
    """
    distances, _ = _neighbors(embeddings, k, metric)
    return distances.mean(axis=1)


def knn_indices(embeddings, k, metric=DEFAULT_METRIC):
    return _neighbors(embeddings, k, metric)[1]


def neighborhood_overlap(embeddings_a, embeddings_b, k, metric=DEFAULT_METRIC):
    """Per-point overlap of the kNN sets between two representations.

    1.0 everywhere means a transform left neighbourhoods intact; lower means the
    same model, under an admissible reparameterisation, now calls different
    points similar.
    """
    a = knn_indices(embeddings_a, k, metric)
    b = knn_indices(embeddings_b, k, metric)
    return np.array([len(set(ra) & set(rb)) / k for ra, rb in zip(a, b)])


def pairwise_distance_matrix(embeddings, metric=DEFAULT_METRIC):
    from sklearn.metrics import pairwise_distances

    return pairwise_distances(np.asarray(embeddings), metric=metric)


def standardized_ranks(values):
    """Fractional ranks in [0, 1] with ties averaged, so equal values tie."""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def blindspot_score(x_signal, y_signal):
    """``B_i = r_y(i) - r_x(i)``: high in y, low in x.

    A continuous, tie-safe disagreement score in [-1, 1]. Positive means the
    point ranks higher on ``y_signal`` than on ``x_signal``; swap the arguments
    for the opposite direction. This is the primary quantity -- quadrants are a
    rendering of it, not a separate result.
    """
    return standardized_ranks(y_signal) - standardized_ranks(x_signal)


def blindspot_bands(score, low=-0.25, high=0.25):
    """Three states -- ``low`` / ``boundary`` / ``high`` -- for visualisation.

    Three, not two: a hard split at the median puts points differing by one
    floating-point step into opposite categories.
    """
    score = np.asarray(score, dtype=float)
    out = np.full(score.shape, "boundary", dtype=object)
    out[score <= low] = "low"
    out[score >= high] = "high"
    return out


def rank_quadrants(x, y, x_name="x", y_name="y", band=0.25):
    """Rank-based split on both axes, tie-safe.

    Replaces the ordinal-rank median split. Returns
    ``(quadrant, x_rank, y_rank, score)``; ``score`` is the continuous
    disagreement and should be preferred for any inference. Points near the
    boundary on either axis are labelled ``boundary_*`` rather than forced to a
    side, because with heavy ties -- which discrete uncertainty metrics produce
    constantly -- a forced side is an artefact of row order.
    """
    x_rank, y_rank = standardized_ranks(x), standardized_ranks(y)
    score = y_rank - x_rank
    quadrant = np.empty(len(x_rank), dtype=object)
    for i, (xr, yr) in enumerate(zip(x_rank, y_rank)):
        x_side = "low" if xr < 0.5 - band / 2 else ("high" if xr > 0.5 + band / 2 else "boundary")
        y_side = "low" if yr < 0.5 - band / 2 else ("high" if yr > 0.5 + band / 2 else "boundary")
        quadrant[i] = f"{x_side}_{x_name}_{y_side}_{y_name}"
    return quadrant, x_rank, y_rank, score

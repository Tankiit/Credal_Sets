"""The §14 tests, in order: what must hold before more datasets are added."""
import numpy as np
import pytest
import torch

from concept_audit.audits.diagnostic_invariance import audit_diagnostic_invariance
from concept_audit.audits.record import RunRecord
from concept_audit.diagnostics.geometry import (
    blindspot_bands, blindspot_score, knn_mean_distance, neighborhood_overlap,
    rank_quadrants, standardized_ranks,
)
from concept_audit.families import NMFModel, PCAModel, SAEModel
from concept_audit.models.native_cbm import NativeCBM
from concept_audit.readouts.coordinate_readout import CoordinateReadout
from concept_audit.readouts.group_readout import GroupReadout
from concept_audit.readouts.identity_readout import IdentityReadout
from concept_audit.transforms.equivalence import (
    admissible_transform, nullspace_basis, stabilizer_parameterization,
)

LATENT, N = 6, 120


def _cbm(readout, seed=0):
    """NativeCBM(feature_dim, num_classes, readout); double precision for exact checks."""
    torch.manual_seed(seed)
    return NativeCBM(8, 3, readout).double().eval()


# -- 1. CBM stabilizer: R A = R ------------------------------------------------

@pytest.mark.parametrize("readout", [
    CoordinateReadout(LATENT, [0, 1, 2]),
    GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]]),
])
@pytest.mark.parametrize("method", ["stabilizer", "exponential"])
def test_admissible_transforms_fix_the_readout(readout, method):
    a = admissible_transform(readout, strength=0.5, seed=0, method=method)
    assert torch.allclose(readout.matrix @ a, readout.matrix, atol=1e-6)


def test_stabilizer_parameterization_is_complete():
    """Every A with R A = R is I + N C, not merely some of them."""
    readout = CoordinateReadout(LATENT, [0, 1, 2])
    n, build = stabilizer_parameterization(readout)
    torch.manual_seed(0)
    arbitrary = torch.eye(LATENT) + n @ torch.randn(n.shape[1], LATENT)
    assert torch.allclose(readout.matrix @ arbitrary, readout.matrix, atol=1e-5)
    # N has orthonormal columns, so N^T recovers C.
    assert torch.allclose(build(n.T @ (arbitrary - torch.eye(LATENT))), arbitrary, atol=1e-5)


def test_nullspace_basis_spans_the_kernel():
    readout = GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]])
    n = nullspace_basis(readout.matrix)
    assert n.shape[1] == LATENT - int(torch.linalg.matrix_rank(readout.matrix))
    assert torch.allclose(readout.matrix @ n, torch.zeros(readout.matrix.shape[0], n.shape[1]), atol=1e-6)


# -- 2. CBM task compensation: W' A = W ----------------------------------------

def test_task_head_compensation_keeps_predictions_identical():
    model = _cbm(GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]]))
    a = admissible_transform(model.readout, strength=0.4, seed=1)
    moved = model.compensate(a)
    assert moved is not None

    assert torch.allclose(moved.head.weight @ a, model.head.weight, atol=1e-9)
    c = torch.randn(N, LATENT).double()
    assert torch.allclose(model.predict_from_concepts(c),
                          moved.predict_from_concepts(c @ a.T), atol=1e-4)


# -- 3. Diagnostic non-invariance ----------------------------------------------

def test_a_geometry_diagnostic_moves_while_outputs_do_not():
    """The phenomenon: exact equivalence, different geometry."""
    model = _cbm(GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]]))
    a = admissible_transform(model.readout, strength=0.8, seed=2)
    torch.manual_seed(0)
    c = torch.randn(N, LATENT).double()

    report = audit_diagnostic_invariance(
        model, c, a, {"knn": lambda x: knn_mean_distance(x.detach().numpy(), 10)}, tol=1e-8
    )
    assert report.equivalence.admissible
    assert report.equivalence.observable_error < 1e-8
    assert report.equivalence.prediction_error < 1e-8
    assert "knn" in report.moved
    overlap = neighborhood_overlap(c.numpy(), (c @ a.T).numpy(), 10)
    assert overlap.mean() < 1.0


def test_full_supervision_is_the_negative_control():
    model = _cbm(IdentityReadout(LATENT))
    a = admissible_transform(model.readout, strength=0.8, seed=3)
    assert torch.allclose(a, torch.eye(LATENT, dtype=a.dtype), atol=1e-6)
    torch.manual_seed(0)
    c = torch.randn(N, LATENT).double()
    report = audit_diagnostic_invariance(
        model, c, a, {"knn": lambda x: knn_mean_distance(x.detach().numpy(), 10)}, tol=1e-8
    )
    assert report.moved == []


def test_inadmissible_transforms_are_refused():
    """A diagnostic moving under an inadmissible A proves nothing."""
    model = _cbm(GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]]))
    bad = torch.eye(LATENT).double().clone()
    bad[0, 1] = 0.7
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="not admissible"):
        audit_diagnostic_invariance(
            model, torch.randn(N, LATENT).double(), bad,
            {"knn": lambda x: knn_mean_distance(x.detach().numpy(), 10)},
        )


# -- 4. PCA reconstruction invariance ------------------------------------------

@pytest.fixture
def pca():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(6, 6).double())[0][:3]
    return PCAModel(basis, torch.zeros(6).double(), torch.tensor([16.0, 1.0, 1.0]).double())


def test_repeated_eigenvalue_rotation_preserves_reconstruction(pca):
    torch.manual_seed(1)
    c = torch.randn(N, 3).double()
    assert pca.eigenvalue_blocks() == [[0], [1, 2]]
    assert pca.admissible_dimension() == 1          # one 2x2 block -> 2*1/2

    a = pca.block_rotation(seed=2)
    assert pca.admissible(a, c=c).admissible
    assert torch.allclose(pca.decode(c), pca.compensate(a).decode(c @ a.T), atol=1e-9)
    assert not torch.allclose(c, c @ a.T), "the rotation must actually move coordinates"


def test_pca_sign_flip_is_admissible(pca):
    torch.manual_seed(1)
    assert pca.admissible(pca.sign_flip([1, -1, 1]), c=torch.randn(N, 3).double()).admissible


def test_pca_scaling_preserves_reconstruction_but_leaves_the_family(pca):
    """Constraints are the other half of admissibility, not a formality."""
    torch.manual_seed(1)
    result = pca.admissible(torch.diag(torch.tensor([2.0, 1.0, 1.0]).double()),
                            c=torch.randn(N, 3).double())
    assert result.observable_error < 1e-10, "reconstruction survives"
    assert not result.admissible, "but the basis is no longer orthonormal"
    assert result.constraint_error["orthonormal"] > 0.1


def test_distinct_eigenvalues_leave_no_continuous_freedom():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 5).double())[0][:3]
    model = PCAModel(basis, torch.zeros(5).double(), torch.tensor([9.0, 4.0, 1.0]).double())
    assert model.admissible_dimension() == 0
    assert torch.allclose(model.block_rotation(seed=0), torch.eye(3).double())


# -- 5. NMF rescaling -----------------------------------------------------------

@pytest.fixture
def nmf():
    torch.manual_seed(0)
    codes = torch.rand(N, 4).double()
    return NMFModel(torch.rand(4, 9).double(), reference_codes=codes), codes


def test_nmf_rescaling_leaves_the_factor_product_unchanged(nmf):
    model, codes = nmf
    for a in (model.rescale([2.0, 0.5, 1.0, 3.0]),
              model.permutation([2, 0, 3, 1]),
              model.monomial(seed=1)):
        assert model.admissible(a, c=codes).admissible
        assert torch.allclose(codes @ model.h, (codes @ a.T) @ model.compensate(a).h, atol=1e-10)


def test_nmf_rejects_transforms_that_leave_the_nonnegative_cone(nmf):
    model, codes = nmf
    torch.manual_seed(5)
    for a in (torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]).double()),
              torch.linalg.qr(torch.randn(4, 4).double())[0]):
        assert not model.admissible(a, c=codes).admissible
        assert model.compensate(a) is None


# -- 6. Blindspot ties ----------------------------------------------------------

def test_tied_values_never_get_opposite_labels_from_row_order():
    """The ordinal-rank bug: identical metrics must not land in opposite quadrants."""
    tied = np.array([2.0, 2.0, 2.0, 2.0])
    ranks = standardized_ranks(tied)
    assert np.allclose(ranks, ranks[0]), "ties must share a rank"

    quadrant, _, _, score = rank_quadrants(tied, tied)
    assert len(set(quadrant)) == 1, "identical points must get identical labels"
    np.testing.assert_allclose(score, 0.0, atol=1e-12)


def test_blindspot_score_is_permutation_equivariant():
    x = np.array([1.0, 1.0, 3.0, 5.0])
    y = np.array([4.0, 2.0, 2.0, 1.0])
    order = np.array([3, 1, 0, 2])
    np.testing.assert_allclose(
        blindspot_score(x, y)[order], blindspot_score(x[order], y[order]), atol=1e-12
    )
    q_a, _, _, _ = rank_quadrants(x, y)
    q_b, _, _, _ = rank_quadrants(x[order], y[order])
    assert list(q_a[order]) == list(q_b)


def test_bands_have_three_states_not_two():
    bands = blindspot_bands(np.array([-0.9, -0.01, 0.0, 0.01, 0.9]))
    assert set(bands) == {"low", "boundary", "high"}
    assert bands[1] == bands[2] == bands[3] == "boundary"


# -- 7. Synthetic seed separation ----------------------------------------------

def test_dataset_is_fixed_across_model_seeds():
    """Model seeds must not move the data, or cross-seed runs aren't comparable."""
    from experiments.synthetic.data import make_dataset

    z0, c0, y0, _ = make_dataset(0)
    for model_seed in (1, 7, 123):
        torch.manual_seed(model_seed)
        torch.randn(64)                       # model-side randomness
        z1, c1, y1, _ = make_dataset(0)
        assert torch.equal(z0, z1) and torch.equal(c0, c1) and torch.equal(y0, y1)
    assert not torch.equal(z0, make_dataset(1)[0])


def test_complete_condition_labels_depend_only_on_concepts():
    """The baseline must not draw y from an unsupervised input coordinate."""
    from experiments.synthetic.data import make_dataset

    _, concepts, labels, _ = make_dataset(0, condition="complete")
    seen = {}
    for row, label in zip(concepts, labels):
        key = tuple(row.tolist())
        assert seen.setdefault(key, int(label)) == int(label), "y is not a function of c*"


def test_incomplete_condition_is_kept_separate():
    from experiments.synthetic.data import make_dataset

    _, concepts, labels, _ = make_dataset(0, condition="incomplete_concepts")
    seen, ambiguous = {}, False
    for row, label in zip(concepts, labels):
        key = tuple(row.tolist())
        if seen.setdefault(key, int(label)) != int(label):
            ambiguous = True
    assert ambiguous, "incomplete_concepts must have an unobserved cause"


# -- schema ---------------------------------------------------------------------

def test_every_family_emits_the_same_record_shape(pca, nmf):
    torch.manual_seed(0)
    cbm = _cbm(GroupReadout(LATENT, [[0, 1], [2, 3], [4, 5]]))
    nmf_model, codes = nmf
    sae = SAEModel(torch.randn(7, 5).double(), torch.randn(5).double(), torch.randn(5, 7).double(),
                   reference_inputs=torch.randn(50, 7).double(), penalty="l0")

    cases = [
        (cbm, torch.randn(N, LATENT).double(), admissible_transform(cbm.readout, strength=0.4, seed=0), 1e-8),
        (pca, torch.randn(N, 3).double(), pca.block_rotation(seed=2), 1e-8),
        (nmf_model, codes, nmf_model.monomial(seed=3), 1e-8),
        (sae, sae.encode(sae.reference_inputs), sae.permutation([2, 0, 4, 1, 3]), 1e-8),
    ]
    families = set()
    for model, c, a, tol in cases:
        report = audit_diagnostic_invariance(
            model, c, a, {"knn": lambda x: knn_mean_distance(x.detach().numpy(), 5)}, tol=tol
        )
        record = RunRecord.from_report(model, report).to_dict()
        assert set(record) == {"model_family", "model_variant", "latent_dim", "observables",
                               "equivalence", "diagnostics", "dataset", "provenance"}
        assert record["equivalence"]["admissible"] is True
        families.add(record["model_family"])
    assert families == {"cbm", "pca", "nmf", "sae"}

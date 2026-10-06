"""The construction: move the concept layer along directions the readout R ignores,
compensate in the head, and obtain a twin model with identical task logits and identical
concept probabilities but a different concept layer z.

For one readout block R (r x d) with orthonormal row-space basis U (d x r) and null-space
basis N (d x n), write z in coordinates (a, b) = (U^T z, N^T z). The map

    a' = a                         (what R sees: untouched)
    b' = M a + G b + t_null         (invisible part: mixed with a, rotated, shifted)

i.e.  z' = B z + t  with  B = [U N] [[I, 0], [M, G]] [U N]^T,  t = N t_null,
satisfies R B = R and R t = 0, so R(z') = R(z). G = expm(rotate * skew) is orthogonal,
M = mix * Gaussian, t_null = shift * Gaussian. The head is compensated by
head'(z') = head(B^{-1}(z' - t)), folded into its first linear layer.

mix != 0 changes the residual as a function of the concept logits, so it can only be
compensated when the head is linear in all of z (not for a CBM whose head reads
sigmoid(concept logits)).
"""

import copy

import torch

from .models import ConceptModel


def invisible_map(R: torch.Tensor, rotate: float = 1.0, mix: float = 1.0, shift: float = 0.0,
                  generator: torch.Generator | None = None, tol: float = 1e-8):
    """Random (B, t) in float64 with R @ B == R and R @ t == 0. Returns (B, t, n_null)."""
    R = R.double()
    d = R.shape[1]
    _, S, Vh = torch.linalg.svd(R, full_matrices=True)
    r = int((S > tol * max(S.max().item(), 1.0)).sum())
    U, N = Vh[:r].T, Vh[r:].T
    n = d - r
    if n == 0:
        return torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64), 0
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    A = randn(n, n)
    G = torch.linalg.matrix_exp(rotate * (A - A.T) / (2 * n) ** 0.5)
    M = mix * randn(n, r) / max(r, 1) ** 0.5
    Bt = torch.eye(d, dtype=torch.float64)
    Bt[r:, :r], Bt[r:, r:] = M, G
    Q = torch.cat([U, N], dim=1)
    B = Q @ Bt @ Q.T
    t = N @ (shift * randn(n) / n**0.5)
    return B, t, n


@torch.no_grad()
def fold_head(model: ConceptModel, B: torch.Tensor, t: torch.Tensor) -> None:
    """head(z) -> head(B^{-1}(z - t)), folded into the head's first linear layer."""
    first = model.head[0]
    W = first.weight.double() @ torch.linalg.inv(B)
    first.bias.copy_(first.bias.double() - W @ t)
    first.weight.copy_(W)


def make_twin(model: ConceptModel, rotate: float = 1.0, mix: float = 1.0, shift: float = 0.0,
              seed: int = 0) -> tuple[ConceptModel, dict]:
    """Return (twin, info). The twin is a deep copy on CPU with folded weights."""
    if mix and not model.head_linear_in_z:
        raise ValueError("this model's head is not linear in z (CBM with head_input='probs'), "
                         "so mixing concept logits into the residual cannot be folded; use mix=0")
    twin = copy.deepcopy(model).cpu().eval()
    gen = torch.Generator().manual_seed(seed)
    blocks = twin.readout_blocks()
    D = blocks[-1][0].stop
    B_full = torch.eye(D, dtype=torch.float64)
    t_full = torch.zeros(D, dtype=torch.float64)
    Bs, ts, n_null = [], [], 0
    for sl, R in blocks:
        B, t, n = invisible_map(R, rotate, mix, shift, gen)
        Bs.append(B), ts.append(t)
        B_full[sl, sl], t_full[sl] = B, t
        n_null += n
    twin.fold_concept_layer(Bs, ts)
    fold_head(twin, B_full, t_full)
    info = {"invisible_dims": n_null, "concept_layer_dim": D, "rotate": rotate, "mix": mix,
            "shift": shift, "seed": seed, "cond_B": torch.linalg.cond(B_full).item()}
    return twin, info

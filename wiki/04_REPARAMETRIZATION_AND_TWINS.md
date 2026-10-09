# 04. Construction & Twin Model Verification

This document details the reparameterization construction in [`concept_models/reparam.py`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/reparam.py) and the twin verification workflow in [`twin.py`](file:///Users/cril/tanmoy/research/Credal_Sets/twin.py).

---

## 1. Mathematical Construction (`invisible_map`)

The function `invisible_map(R, rotate, mix, shift, generator, tol=1e-8)` constructs an affine transformation $(B, t)$ acting on concept layer $z$:

$$z' = B z + t$$

```python
def invisible_map(R, rotate=1.0, mix=1.0, shift=0.0, generator=None, tol=1e-8):
    R = R.double()
    d = R.shape[1]
    _, S, Vh = torch.linalg.svd(R, full_matrices=True)
    r = int((S > tol * max(S.max().item(), 1.0)).sum())
    U, N = Vh[:r].T, Vh[r:].T
    n = d - r
    if n == 0:
        return torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64), 0
    
    A = torch.randn(n, n, generator=generator, dtype=torch.float64)
    G = torch.linalg.matrix_exp(rotate * (A - A.T) / (2 * n) ** 0.5)
    M = mix * torch.randn(n, r, generator=generator, dtype=torch.float64) / max(r, 1) ** 0.5
    
    Bt = torch.eye(d, dtype=torch.float64)
    Bt[r:, :r], Bt[r:, r:] = M, G
    Q = torch.cat([U, N], dim=1)
    B = Q @ Bt @ Q.T
    t = N @ (shift * torch.randn(n, generator=generator, dtype=torch.float64) / n**0.5)
    return B, t, n
```

---

## 2. Transformation Components Explained

1. **Rotation ($G$)**:
   - $A - A^T$ is a random skew-symmetric matrix.
   - Matrix exponential $\exp(\cdot)$ projects it onto $SO(n)$, generating a random uniform rotation of the null space.
   - Controlled by `--rotate` (default `1.0`).

2. **Mixing ($M$)**:
   - Mixes visible concept features into invisible null-space coordinates.
   - Controlled by `--mix` (default `1.0`).
   - Requires `head_linear_in_z = True`.

3. **Shift ($t$)**:
   - Translates the internal representation along an invisible direction ($R t = 0$).
   - Controlled by `--shift` (default `0.0`).

---

## 3. Creating a Twin Model (`make_twin`)

The API function `make_twin(model, rotate, mix, shift, seed)` returns `(twin_model, info)`:

```python
from concept_models.models import load_model
from concept_models.reparam import make_twin

model = load_model("runs/cebab-cem-s0/model.pt")
twin, info = make_twin(model, rotate=1.0, mix=1.0, shift=0.0, seed=42)
```

### Process Flow
1. **Copy Model**: Creates a deep copy of the original model.
2. **Iterate Readout Blocks**: Obtains $(B_i, t_i)$ for each readout block $i$.
3. **Fold Generator**: Calls `twin.fold_concept_layer(Bs, ts)` to update internal linear generator weights.
4. **Fold Task Head**: Calls `fold_head(twin, B_full, t_full)` to multiply the first layer of the task head by $B^{-1}$ and adjust bias by $-W B^{-1} t$.

---

## 4. Verification & Diagnostic Metrics

When running `twin.py`, the following metrics are evaluated on test features:

| Metric | Target Value | Meaning |
| :--- | :--- | :--- |
| `max_abs_d_logits` | $< 10^{-5}$ | Maximum absolute change in output task logits $y$. |
| `max_abs_d_concept_probs` | $< 10^{-5}$ | Maximum absolute change in concept probabilities $\hat{p}$. |
| `rel_change_z` | $> 0.5$ (large) | Relative Frobenius change in concept layer $\|z' - z\| / \|z\|$. |
| `mean_cos_z` | $< 0.99$ | Mean cosine similarity between original $z$ and twin $z'$. |
| `cond_B` | Safe $(< 100)$ | Condition number $\kappa(B) = \|B\| \|B^{-1}\|$. |

### Concept Interventions Diagnostic
Under concept interventions:
- For **CEM**: Interventions are **blind** to reparameterization because the same transformation $B_i$ acts on both active $c_i^+$ and inactive $c_i^-$ embeddings.
- For **CBM + Residual** (`head_input="logits"`): Interventions reveal the twin model when $\text{mix} \neq 0$, because changing concept logits indirectly perturbs the residual vector fed into the task head.

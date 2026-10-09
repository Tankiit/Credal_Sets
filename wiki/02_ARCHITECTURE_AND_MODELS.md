# 02. Architecture & Model Specifications

This document describes the design and implementation of the neural network architectures in [`concept_models/models.py`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py).

---

## 1. Network Overview & Pipeline

Both CBM and CEM architectures follow a unified modular design:

```
[ Input Text Features x ]
          │
          ▼
   [ Shared Trunk ]  ─── Linear(in_dim, hidden) + LeakyReLU
          │
          ▼
   [ Concept Layer z ] ─── (CBM: concept logits + residual | CEM: concatenated z_i)
      │          │
      │          └─── [ Readout R ] ─── Concept Logits ─── Sigmoid ─── Concept Probs
      ▼
   [ Task Head ] ────── Linear / MLP ─── Task Logits y
```

---

## 2. Base Class: `ConceptModel`

Defined in [`concept_models/models.py`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L51):

```python
class ConceptModel(nn.Module):
    head_linear_in_z = True

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.k = config["n_concepts"]
        self.trunk = nn.Sequential(
            nn.Linear(config["in_dim"], config["hidden"]),
            nn.LeakyReLU()
        )

    def readout_blocks((self) -> list[tuple[slice, torch.Tensor]]:
        raise NotImplementedError

    def fold_concept_layer(self, B: list[torch.Tensor], t: list[torch.Tensor]) -> None:
        raise NotImplementedError
```

---

## 3. Concept Bottleneck Model (CBM)

Implemented in [`CBM`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L71).

### Architecture Details
- **Concept Layer Dim**: $D = k + r$, where $k$ is the number of concepts and $r$ is `residual_dim`.
- **Readout Matrix $R$**: $k \times (k+r)$ matrix with $R[:, :k] = I_k$ and $R[:, k:] = 0$.
- **`head_input` Modes**:
  1. `"probs"` (default, Koh et al. 2020): Head receives $[\sigma(\text{concept\_logits}) \,;\, \text{residual}]$.
     - `head_linear_in_z = False`.
     - Mixing concept logits into residual (`mix > 0`) is **not allowed** because $\sigma(\cdot)$ is non-linear.
  2. `"logits"`: Head receives $z = [\text{concept\_logits} \,;\, \text{residual}]$ directly.
     - `head_linear_in_z = True`.
     - Allows full reparameterization including `mix > 0`.

### Calibrating Interventions
Koh et al. set interventions using the 5th/95th percentile over all examples. In rare concepts, this can result in inaccurate values.
`CBM.calibrate_interventions(x, concepts)` computes class-conditional medians:

$$\text{logit\_on}[j] = \text{median}(\{ \hat{c}_{i,j} \mid c_{i,j} = 1 \})$$
$$\text{logit\_off}[j] = \text{median}(\{ \hat{c}_{i,j} \mid c_{i,j} = 0 \})$$

---

## 4. Concept Embedding Model (CEM)

Implemented in [`CEM`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L122).

### Architecture Details
- **Concept Embedding Dimension**: $m = \text{emb\_dim}$ (default 16).
- **Concept Generators**: For each concept $i \in \{1,\dots,k\}$, weights `gen_w` (shape $k \times 2 \times m \times h$) and bias `gen_b` (shape $k \times 2 \times m$) map trunk output $h(x)$ linearly to active embedding $c_i^+$ and inactive embedding $c_i^-$.
- **Scoring Layer**: Shared linear layer `score` ($2m \to 1$) computes concept logit:
  $$\hat{c}_i = \text{score}([c_i^+ ; c_i^-])$$
- **Concept Layer $z$**: Concatenation of $z_i = p_i c_i^+ + (1 - p_i) c_i^-$, total dimension $D = k \cdot m$.
- **RandInt Training**: Random concept interventions applied during training with probability `p_int` (default 0.25).

---

## 5. Task Head Implementations

The head is created via [`make_head`](file:///Users/cril/tanmoy/research/Credal_Sets/concept_models/models.py#L43):
- `"linear"`: Single `nn.Linear(in_dim, n_classes)` layer.
- `"mlp"`: `nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, n_classes))`.

> [!IMPORTANT]
> Weight folding for twin models requires the head's **first layer** to be linear. Both `"linear"` and `"mlp"` start with a `Linear` layer, allowing $B^{-1}$ folding into `head[0]`.

---

## 6. Model Comparison Table

| Feature | Vanilla CBM | CBM + Residual ($r>0$) | CEM ($m=16$) |
| :--- | :--- | :--- | :--- |
| **Concept Layer Dim $D$** | $k$ | $k + r$ | $k \cdot m$ |
| **Readout Rank per Block** | $1$ per concept | $1$ per concept | $\le 2$ per concept |
| **Invisible Dims** | $0$ | $r$ | $k \cdot (m - 2)$ |
| **`head_input` options** | `"probs"`, `"logits"` | `"probs"`, `"logits"` | Concatenated embeddings |
| **Twin Mixing Allowed?** | No (`mix=0`) | Yes (if `head_input="logits"`) | Yes |

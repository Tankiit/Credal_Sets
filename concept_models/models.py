"""CBM and CEM on top of frozen text features.

Both models share one structure, which is what the reparameterization relies on:

    x  --trunk-->  h  --concept layer-->  z  --head-->  task logits
                                          |
                                          +--readout R-->  concept logits  (sigmoid -> probs)

* ``z`` is the *concept layer*: exactly what the head reads.
* The readout ``R`` is the only thing compared to the concept labels during training.
  Every model exposes it as linear constraints on ``z`` through ``readout_blocks()``:
  a list of ``(slice_of_z, R_block)``.  Any change of ``z`` inside the null space of
  each ``R_block`` leaves the concept probabilities untouched.
* The last op producing ``z`` and the first op of the head are linear, so an invertible
  map ``z -> B z + t`` can be folded into the weights exactly (``fold_concept_layer`` +
  ``reparam.fold_head``), giving a model of the same architecture.

CBM (Koh et al., 2020), joint training. ``z = [concept logits (k) ; residual (r)]``,
    ``R = [I_k, 0]``.  With ``residual_dim = 0`` (vanilla CBM) R is invertible on z and
    nothing can move; with ``residual_dim > 0`` (hybrid CBM / side channel) the residual
    is invisible to R.  ``head_input`` chooses what the head reads:
      "probs"  (default, as in Koh et al.): [sigmoid(concept logits) ; residual].
               Interventions set probabilities to 0/1.  Only maps that leave the concept
               coordinates alone can be folded (no "mix", see reparam.py).
      "logits" : z itself, so z -> head is linear and every invisible map folds,
               including mixing concept logits into the residual.  Interventions write
               the median training logit of true positives / negatives.

CEM (Espinosa Zarlenga et al., 2022). Per concept i, two embeddings ``c_i+ , c_i-`` in
    R^m and ``p_i = sigmoid(s([c_i+; c_i-]))`` with a scoring layer s shared across
    concepts (as in the paper). ``z_i = p_i c_i+ + (1 - p_i) c_i-``, z = concat_i z_i.
    For a map ``z_i -> B_i z_i`` applied to both c_i+ and c_i-, p_i is unchanged iff
    ``s+^T B_i = s+^T`` and ``s-^T B_i = s-^T``: R_block_i = [s+; s-] (2 x m), leaving
    m - 2 invisible directions per concept.  Deviation from the paper: c_i+/- are a
    linear map of a shared trunk (the paper uses a per-concept Linear + LeakyReLU), so
    that the map can be folded exactly.
"""

import torch
from torch import nn


def make_head(in_dim: int, n_classes: int, kind: str, hidden: int = 128) -> nn.Sequential:
    if kind == "linear":
        return nn.Sequential(nn.Linear(in_dim, n_classes))
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, n_classes))
    raise ValueError(f"unknown head {kind!r}")


class ConceptModel(nn.Module):
    """Common interface. ``forward`` returns a dict with z, concept_logits, logits."""

    # Whether the head is a linear function of the whole of z (needed to fold "mix").
    head_linear_in_z = True

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.k = config["n_concepts"]
        self.trunk = nn.Sequential(nn.Linear(config["in_dim"], config["hidden"]), nn.LeakyReLU())

    def readout_blocks(self) -> list[tuple[slice, torch.Tensor]]:
        raise NotImplementedError

    def fold_concept_layer(self, B: list[torch.Tensor], t: list[torch.Tensor]) -> None:
        """Replace z by B z + t (one (B, t) per readout block) by editing the weights."""
        raise NotImplementedError


class CBM(ConceptModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.r = config.get("residual_dim", 0)
        self.head_input = config.get("head_input", "probs")
        self.head_linear_in_z = self.head_input == "logits"
        self.concept_layer = nn.Linear(config["hidden"], self.k + self.r)
        self.head = make_head(self.k + self.r, config["n_classes"], config["head"])
        # Logit written into z when a concept is intervened on/off; set by
        # calibrate_interventions() after training.
        self.register_buffer("logit_on", torch.full((self.k,), 3.0))
        self.register_buffer("logit_off", torch.full((self.k,), -3.0))

    def forward(self, x, concepts=None, intervene=None):
        z = self.concept_layer(self.trunk(x))
        concept_logits, residual = z[:, : self.k], z[:, self.k:]
        if self.head_input == "probs":
            c = torch.sigmoid(concept_logits)
            if intervene is not None:
                c = torch.where(intervene, concepts.float(), c)
        else:
            c = concept_logits
            if intervene is not None:
                c = torch.where(intervene, torch.where(concepts > 0.5, self.logit_on, self.logit_off), c)
        return {"z": z, "concept_logits": concept_logits,
                "logits": self.head(torch.cat([c, residual], dim=1))}

    @torch.no_grad()
    def calibrate_interventions(self, x: torch.Tensor, concepts: torch.Tensor) -> None:
        """on/off = median training logit among examples where the concept is truly on/off.
        (Koh et al. use the 95th/5th percentile over all examples, which for rare concepts
        is still a confidently "off" logit.)"""
        logits = self(x)["concept_logits"]
        for j in range(self.k):
            on, off = logits[concepts[:, j] > 0.5, j], logits[concepts[:, j] <= 0.5, j]
            self.logit_on[j] = on.median() if len(on) else logits[:, j].max()
            self.logit_off[j] = off.median() if len(off) else logits[:, j].min()

    def readout_blocks(self):
        R = torch.zeros(self.k, self.k + self.r)
        R[:, : self.k] = torch.eye(self.k)
        return [(slice(0, self.k + self.r), R)]

    @torch.no_grad()
    def fold_concept_layer(self, B, t):
        (B,), (t,) = B, t
        W, b = self.concept_layer.weight.double(), self.concept_layer.bias.double()
        self.concept_layer.weight.copy_(B @ W)
        self.concept_layer.bias.copy_(B @ b + t)


class CEM(ConceptModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.m = m = config.get("emb_dim", 16)
        self.p_int = config.get("p_int", 0.25)
        h = config["hidden"]
        # gen_w[i, 0] produces c_i+, gen_w[i, 1] produces c_i-
        self.gen_w = nn.Parameter(torch.randn(self.k, 2, m, h) / h**0.5)
        self.gen_b = nn.Parameter(torch.zeros(self.k, 2, m))
        self.score = nn.Linear(2 * m, 1)
        self.head = make_head(self.k * m, config["n_classes"], config["head"])

    def embeddings(self, x):
        """Return (c_plus, c_minus), each (batch, k, m)."""
        c = torch.einsum("ksmh,bh->bksm", self.gen_w, self.trunk(x)) + self.gen_b
        return c[:, :, 0], c[:, :, 1]

    def forward(self, x, concepts=None, intervene=None):
        c_plus, c_minus = self.embeddings(x)
        concept_logits = self.score(torch.cat([c_plus, c_minus], dim=-1)).squeeze(-1)
        p = torch.sigmoid(concept_logits)
        if intervene is None and self.training and self.p_int > 0 and concepts is not None:
            intervene = torch.rand_like(p) < self.p_int  # RandInt training (CEM paper)
        if intervene is not None:
            p = torch.where(intervene, concepts.float(), p)
        z = (p.unsqueeze(-1) * c_plus + (1 - p).unsqueeze(-1) * c_minus).flatten(1)
        return {"z": z, "concept_logits": concept_logits, "logits": self.head(z),
                "c_plus": c_plus, "c_minus": c_minus}

    def readout_blocks(self):
        s = self.score.weight.detach().cpu().view(2, self.m)  # rows: s+, s-
        return [(slice(i * self.m, (i + 1) * self.m), s.clone()) for i in range(self.k)]

    @torch.no_grad()
    def fold_concept_layer(self, B, t):
        for i, (Bi, ti) in enumerate(zip(B, t)):
            for sign in (0, 1):
                self.gen_w[i, sign].copy_(Bi @ self.gen_w[i, sign].double())
                self.gen_b[i, sign].copy_(Bi @ self.gen_b[i, sign].double() + ti)


MODELS = {"cbm": CBM, "cem": CEM}


def build_model(config: dict) -> ConceptModel:
    if config["kind"].startswith("pyc-"):  # imported lazily: needs pytorch-concepts
        from .pyc_models import PYC_MODELS
        return PYC_MODELS[config["kind"]](config)
    return MODELS[config["kind"]](config)


def save_model(model: ConceptModel, path) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path, device="cpu") -> ConceptModel:
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()

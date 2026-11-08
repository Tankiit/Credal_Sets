"""Fit an attribute readout on a frozen representation, and audit what it cannot see.

The bridge between the blindspot work and supervised concept models.

Given frozen features ``z = f_DINO(x)``, fit ``R_attr z ~ c``. That splits the
representation in two:

* ``row(R_attr)`` -- the directions attributes pin down;
* ``ker(R_attr)`` -- the directions no attribute constrains.

Transforms ``A = I + N C`` with N spanning ``ker(R_attr)`` satisfy
``R_attr A = R_attr`` exactly, so every attribute prediction is bit-identical.
Whatever *does* change under them -- kNN distance, neighbour identity, a
blindspot score -- is a quantity attribute supervision never determined, and
that no amount of attribute annotation could pin down.

For a 768-dim DINOv2 feature with 312 CUB attributes, ``ker`` has dimension at
least 456: the unconstrained space is larger than the constrained one.
"""
import torch

from concept_audit.core.latent_model import LatentModel
from concept_audit.core.observables import LinearObservable


class AttributeReadoutModel(LatentModel):
    """A least-squares linear map from representation to attributes.

    The readout is a *fixed map*: it is what attribute supervision sees, so
    admissibility means ``R A = R``.
    """

    family = "frozen_representation"
    variant = "attribute_readout"

    def __init__(self, matrix, bias, ridge=1.0, concept_names=None):
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix).double())    # (K, d)
        self.register_buffer("bias", torch.as_tensor(bias).double())        # (K,)
        self.ridge = ridge
        self.concept_names = concept_names

    @classmethod
    def fit(cls, z, concepts, ridge=1.0, concept_names=None):
        """Ridge regression ``z -> c``; ridge keeps R well-conditioned when d >> n."""
        z = torch.as_tensor(z).double()
        concepts = torch.as_tensor(concepts).double()
        mean = z.mean(0)
        centered = z - mean
        gram = centered.T @ centered + ridge * torch.eye(z.shape[1], dtype=z.dtype)
        weight = torch.linalg.solve(gram, centered.T @ concepts)            # (d, K)
        return cls(weight.T, concepts.mean(0) - mean @ weight, ridge, concept_names)

    def latent_dim(self):
        return int(self.matrix.shape[1])

    def encode(self, z):
        return torch.as_tensor(z).double()

    def predict_attributes(self, z):
        return torch.as_tensor(z).double() @ self.matrix.T + self.bias

    def observables(self, c):
        return {"attribute_readout": self.predict_attributes(c)}

    def observable_maps(self):
        return [LinearObservable(self.matrix, name="attribute_readout", fixed_map=True)]

    def compensate(self, a):
        """R is unchanged by definition; only the latent moves."""
        return self

    @property
    def rank(self):
        return int(torch.linalg.matrix_rank(self.matrix))

    @property
    def nullity(self):
        """Representation directions no attribute constrains."""
        return self.latent_dim() - self.rank

    def r2(self, z, concepts):
        concepts = torch.as_tensor(concepts).double()
        resid = ((concepts - self.predict_attributes(z)) ** 2).sum()
        total = ((concepts - concepts.mean(0)) ** 2).sum()
        return float(1.0 - resid / total) if total > 0 else float("nan")

    def save(self, path):
        torch.save({"R_attribute": self.matrix, "bias": self.bias,
                    "rank_R": self.rank, "nullity_R": self.nullity,
                    "ridge": self.ridge, "concept_names": self.concept_names}, path)
        return path

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(blob["R_attribute"], blob["bias"], blob.get("ridge", 1.0),
                   blob.get("concept_names"))

from abc import ABC, abstractmethod
import torch
from torch import nn
from concept_audit.core.latent_model import LatentModel
from concept_audit.core.observables import from_readout
from concept_audit.interventions import replace_block, set_readout


class ConceptModel(LatentModel, ABC):
    """Flat c: (batch, latent_dim); R(c): scores; h(c): task logits.

    Exact linear-head auditing additionally requires `readout.matrix` and `head`.
    Concept blocks specify diagnostic/substitution coordinates, independently of R.
    """
    def __init__(self, readout, head, blocks=None):
        super().__init__()
        self.readout = readout
        self.head = head
        blocks = blocks if blocks is not None else [(j,) for j in range(readout.num_concepts)]
        self.blocks = tuple(tuple(b) for b in blocks)
        if len(self.blocks) != readout.num_concepts or any(
            not b or len(set(b)) != len(b) or any(i < 0 or i >= readout.latent_dim for i in b)
            for b in self.blocks
        ):
            raise ValueError("Provide one valid coordinate block per supervised concept")

    @abstractmethod
    def encode(self, z):
        pass

    def concept_readout(self, c):
        return self.readout(c)

    def predict_from_concepts(self, c):
        return self.head(c)

    def intervene(self, c, concept_id, value):
        """Set a supervised concept score with minimum-norm latent displacement."""
        return set_readout(c, self.readout, concept_id, value)

    def substitute(self, c, concept_id, value):
        """Replace a declared coordinate block with a donor block (not a label)."""
        if not 0 <= concept_id < len(self.blocks):
            raise IndexError(concept_id)
        return replace_block(c, self.blocks[concept_id], value)

    def substitute_donor(self, c, concept_id, donor):
        if not 0 <= concept_id < len(self.blocks):
            raise IndexError(concept_id)
        return self.substitute(c, concept_id, donor[:, list(self.blocks[concept_id])])

    def forward(self, z):
        c = self.encode(z)
        return self.concept_readout(c), self.predict_from_concepts(c)

    # -- LatentModel interface --------------------------------------------------
    #
    # Supervision *is* R, so R is a fixed map: admissibility means R A = R, not
    # merely that some value happens to survive.

    family = "cbm"
    variant = "readout"

    def latent_dim(self):
        return int(self.readout.latent_dim)

    def observable_maps(self):
        return [from_readout(self.readout, name="readout")]

    def observables(self, c):
        return {"readout": self.concept_readout(c)}

    def predictions(self, c):
        return self.predict_from_concepts(c)

    def compensate(self, a):
        """The model that *is* c' = A c, via W' = W A^{-1}. R is unchanged.

        Delegates to ReparameterizedModel, which owns the validation and the
        frozen snapshot; returns None when A is inadmissible or too
        ill-conditioned, which is the answer admissible expects.
        """
        from concept_audit.transforms.equivalence import ReparameterizedModel

        try:
            return ReparameterizedModel(self, a)
        except (ValueError, TypeError) as exc:
            del exc
            return None

    @property
    def readout_rank(self):
        return int(torch.linalg.matrix_rank(self.readout.matrix))

    @property
    def unconstrained_dim(self):
        """dim ker(R): the free directions supervision never sees."""
        return int(self.readout.latent_dim) - self.readout_rank

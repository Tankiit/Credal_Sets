from abc import ABC, abstractmethod
from torch import nn
from concept_audit.interventions import replace_block, set_readout


class ConceptModel(nn.Module, ABC):
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

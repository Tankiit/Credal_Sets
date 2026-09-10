from torch import nn
from .base import ConceptModel


class PyCAdapter(ConceptModel):
    """PyC low-level encoder with native R, linear h, and intervention conventions."""
    def __init__(self, feature_dim, num_classes, readout, blocks=None):
        try:
            from torch_concepts.nn import LinearEmbeddingToConcept
        except ImportError as exc:
            raise ImportError("PyC backend requires: pip install '.[concepts]' (including its dependencies)") from exc
        super().__init__(readout, nn.Linear(readout.latent_dim, num_classes), blocks)
        self.encoder = LinearEmbeddingToConcept(in_embeddings=feature_dim, out_concepts=readout.latent_dim)

    def encode(self, z):
        return self.encoder(embeddings=z)

    def substitute(self, c, concept_id, value):
        from concept_audit.interventions.pyc_adapter import replace_block_pyc
        if not 0 <= concept_id < len(self.blocks):
            raise IndexError(concept_id)
        return replace_block_pyc(c, self.blocks[concept_id], value)

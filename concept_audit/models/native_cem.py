import torch
from torch import nn
from .base import ConceptModel
from concept_audit.readouts import BlockProjectionReadout


class NativeCEM(ConceptModel):
    """CEM-like mixed embeddings with an explicit linear supervision readout.

    Learned positive/negative embeddings are mixed with feature-dependent gates.
    R supervises the resulting blocks, not the gates. This is an experimental
    non-injective-readout variant, not a reproduction of the standard CEM loss.
    """
    def __init__(self, feature_dim, num_classes, num_concepts, block_size=4):
        projections = torch.zeros(num_concepts, block_size)
        projections[:, 0] = 1
        readout = BlockProjectionReadout(projections)
        blocks = [tuple(range(j*block_size, (j+1)*block_size)) for j in range(num_concepts)]
        super().__init__(readout, nn.Linear(readout.latent_dim, num_classes), blocks)
        self.num_concepts, self.block_size = num_concepts, block_size
        self.embeddings = nn.Linear(feature_dim, 2 * num_concepts * block_size)
        self.gates = nn.Linear(feature_dim, num_concepts)

    def encode(self, z):
        pairs = self.embeddings(z).reshape(-1, self.num_concepts, 2, self.block_size)
        p = self.gates(z).sigmoid().unsqueeze(-1)
        return (p * pairs[:, :, 1] + (1-p) * pairs[:, :, 0]).flatten(1)

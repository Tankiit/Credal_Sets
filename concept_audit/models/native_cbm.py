from torch import nn
from .base import ConceptModel


class NativeCBM(ConceptModel):
    """Continuous/logit CBM; non-injective R yields a supervised latent model."""
    def __init__(self, feature_dim, num_classes, readout, blocks=None):
        super().__init__(readout, nn.Linear(readout.latent_dim, num_classes), blocks)
        self.encoder = nn.Linear(feature_dim, readout.latent_dim)

    def encode(self, z):
        return self.encoder(z)

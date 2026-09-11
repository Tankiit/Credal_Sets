import torch
from concept_audit.readouts.linear_readout import LinearReadout


class IdentityReadout(LinearReadout):
    def __init__(self, latent_dim):
        super().__init__(torch.eye(latent_dim))


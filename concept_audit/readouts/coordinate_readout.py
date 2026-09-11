import torch
from concept_audit.readouts.linear_readout import LinearReadout


class CoordinateReadout(LinearReadout):
    def __init__(self, latent_dim, coordinates):
        coordinates = tuple(coordinates)
        if not coordinates or len(set(coordinates)) != len(coordinates) or any(
            i < 0 or i >= latent_dim for i in coordinates
        ):
            raise ValueError("Coordinates must be unique, nonempty, and in range")
        super().__init__(torch.eye(latent_dim)[list(coordinates)])


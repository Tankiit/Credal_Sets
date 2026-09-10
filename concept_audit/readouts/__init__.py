"""Explicit, fixed linear readouts. Outputs are scores, not probabilities."""
import torch
from torch import nn


class LinearReadout(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        matrix = torch.as_tensor(matrix)
        if not matrix.is_floating_point():
            matrix = matrix.float()
        if matrix.ndim != 2 or min(matrix.shape) == 0 or not torch.isfinite(matrix).all():
            raise ValueError("Readout must be a finite nonempty (concepts, latent_dim) matrix")
        self.register_buffer("matrix", matrix.detach().clone())

    @property
    def latent_dim(self):
        return self.matrix.shape[1]

    @property
    def num_concepts(self):
        return self.matrix.shape[0]

    def forward(self, c):
        return c @ self.matrix.T


class IdentityReadout(LinearReadout):
    def __init__(self, latent_dim):
        super().__init__(torch.eye(latent_dim))


class CoordinateReadout(LinearReadout):
    def __init__(self, latent_dim, coordinates):
        coordinates = tuple(coordinates)
        if not coordinates or len(set(coordinates)) != len(coordinates) or any(
            i < 0 or i >= latent_dim for i in coordinates
        ):
            raise ValueError("Coordinates must be unique, nonempty, and in range")
        super().__init__(torch.eye(latent_dim)[list(coordinates)])


class GroupReadout(LinearReadout):
    """One mean score per group; groups may overlap but cannot repeat indices."""
    def __init__(self, latent_dim, groups):
        matrix = torch.zeros(len(groups), latent_dim)
        for j, group in enumerate(groups):
            if not group or len(set(group)) != len(group) or any(i < 0 or i >= latent_dim for i in group):
                raise ValueError("Each group must contain unique valid coordinates")
            matrix[j, list(group)] = 1 / len(group)
        super().__init__(matrix)


class BlockProjectionReadout(LinearReadout):
    """One explicit projection per disjoint concept block."""
    def __init__(self, projections):
        projections = torch.as_tensor(projections)
        if projections.ndim != 2 or min(projections.shape) == 0:
            raise ValueError("Expected (num_concepts, block_size) projections")
        if not projections.is_floating_point():
            projections = projections.float()
        k, m = projections.shape
        matrix = projections.new_zeros(k, k * m)
        for j in range(k):
            matrix[j, j*m:(j+1)*m] = projections[j]
        super().__init__(matrix)

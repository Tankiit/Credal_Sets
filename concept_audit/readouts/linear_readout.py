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

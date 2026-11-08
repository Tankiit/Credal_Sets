import torch
from concept_audit.readouts.linear_readout import LinearReadout


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

import torch
from concept_audit.readouts.linear_readout import LinearReadout


class GroupReadout(LinearReadout):
    """One mean score per group; groups may overlap but cannot repeat indices."""
    def __init__(self, latent_dim, groups):
        matrix = torch.zeros(len(groups), latent_dim)
        for j, group in enumerate(groups):
            if not group or len(set(group)) != len(group) or any(i < 0 or i >= latent_dim for i in group):
                raise ValueError("Each group must contain unique valid coordinates")
            matrix[j, list(group)] = 1 / len(group)
        super().__init__(matrix)
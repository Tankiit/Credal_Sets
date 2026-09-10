"""Controlled synthetic features; no image dataset or backbone dependencies."""
import torch


def make_data(seed=0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(512, 12, generator=generator)
    concepts = (z[:, :3] > 0).float()
    labels = (concepts[:, 0] + concepts[:, 1] + (z[:, 3] > 0)).long() % 2
    return z, concepts, labels, 384

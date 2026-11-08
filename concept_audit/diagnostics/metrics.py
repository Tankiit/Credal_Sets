"""Transparent baseline diagnostics, not estimators of causal leakage.

Correlations use evaluation rows only. Probes fit training rows only and are
refitted in every audit condition. Constant variables have zero correlation.
"""
import torch


def correlations(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = x.square().sum(0).sqrt()[:, None] * y.square().sum(0).sqrt()[None, :]
    return (x.T @ y) / denominator.clamp_min(torch.finfo(x.dtype).eps)


def block_correlations(state):
    mask = ~state.train_mask
    return [correlations(state.c[mask][:, list(b)], state.concepts[mask]).abs().amax(0) for b in state.model.blocks]


class Alignment:
    name = "own_concept_abs_correlation"

    def compute(self, state):
        return torch.stack([v[j] for j, v in enumerate(block_correlations(state))])


class CrossConceptCorrelation:
    name = "cross_concept_abs_correlation"

    def compute(self, state):
        values = block_correlations(state)
        return torch.stack([torch.cat((v[:j], v[j+1:])).max() if len(v) > 1 else v.new_zeros(()) for j, v in enumerate(values)])


class BlockPurity:
    name = "own_correlation_share"

    def compute(self, state):
        return torch.stack([v[j] / v.sum().clamp_min(torch.finfo(v.dtype).eps) for j, v in enumerate(block_correlations(state))])


class TaskProbe:
    """Held-out task accuracy of a fixed-ridge, one-hot least-squares probe."""
    name = "block_task_probe_accuracy"

    def __init__(self, ridge=1.0):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge

    def compute(self, state):
        train = state.train_mask
        y = torch.nn.functional.one_hot(state.labels, state.model.head.out_features).to(state.c)
        result = []
        for block in state.model.blocks:
            x = state.c[:, list(block)]
            x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            penalty = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) * self.ridge
            penalty[-1, -1] = 0
            w = torch.linalg.solve(x[train].T @ x[train] + penalty, x[train].T @ y[train])
            result.append(((x[~train] @ w).argmax(1) == state.labels[~train]).to(x.dtype).mean())
        return torch.stack(result)


class HeadSensitivity:
    name = "block_head_frobenius_norm"

    def compute(self, state):
        return torch.stack([state.model.head.weight[:, list(b)].norm() for b in state.model.blocks])

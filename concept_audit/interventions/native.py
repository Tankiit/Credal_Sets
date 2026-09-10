"""Intervention values always live in the representation/readout's score space."""
import torch


def replace_block(c, indices, value):
    indices = tuple(indices)
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= c.shape[-1] for i in indices):
        raise ValueError("Invalid concept block")
    replacement = torch.as_tensor(value, dtype=c.dtype, device=c.device)
    if replacement.ndim == 1 and len(indices) == 1 and replacement.shape[0] == c.shape[0]:
        replacement = replacement[:, None]
    result = c.clone()
    result[:, list(indices)] = replacement
    return result


def set_readout(c, readout, concept_id, value, tol=1e-5):
    """Minimum Euclidean norm change setting R_j(c), preserving other R scores.

    This is a declared convention, not an invariant under general basis changes.
    Rank-deficient readouts can make an independently requested value impossible.
    """
    if not 0 <= concept_id < readout.num_concepts:
        raise IndexError(concept_id)
    scores = readout(c)
    target = scores.clone()
    target[:, concept_id] = torch.as_tensor(value, dtype=c.dtype, device=c.device)
    result = c + (target - scores) @ torch.linalg.pinv(readout.matrix).T
    if not torch.allclose(readout(result), target, atol=tol, rtol=tol):
        raise ValueError("Requested concept intervention is infeasible for this readout")
    return result

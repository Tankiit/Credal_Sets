"""PyC supplies replacement values; block selection remains explicit and native."""
import torch
from .native import replace_block


def replace_block_pyc(c, indices, value):
    from torch_concepts.nn import DoIntervention

    # Native validation/broadcasting defines our public semantics.
    target = replace_block(c, indices, value)
    replacement = DoIntervention(target[:, list(indices)])(c[:, list(indices)])
    return replace_block(c, indices, replacement)

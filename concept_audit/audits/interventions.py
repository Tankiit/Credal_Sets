"""Separate structural, informational, and consequence controls.

Structural: remove task-head edges from a block, leaving c and R unchanged.
Informational: permute a block within each split, leaving head parameters fixed.
These operational controls need not preserve R(c) or task predictions.
"""
from copy import deepcopy
from dataclasses import replace
import torch


@torch.no_grad()
def audit_structural(state, registry):
    reports = []
    for block in state.model.blocks:
        model = deepcopy(state.model)
        model.head.weight[:, list(block)] = 0
        changed = replace(state, model=model)
        reports.append(registry.compute(changed))
    return reports


@torch.no_grad()
def audit_informational(state, registry, seed=0):
    generator = torch.Generator(device=state.c.device).manual_seed(seed)
    reports = []
    for j, block in enumerate(state.model.blocks):
        c = state.c.clone()
        for mask in (state.train_mask, ~state.train_mask):
            rows = mask.nonzero().flatten()
            donors = rows[torch.randperm(len(rows), device=rows.device, generator=generator)]
            c[rows[:, None], list(block)] = state.c[donors[:, None], list(block)]
        reports.append(registry.compute(replace(state, c=c)))
    return reports


@torch.no_grad()
def audit_consequence(state, registry, seed=0):
    """Acc(original)-Acc(donor substitution) on held-out rows; negative is allowed.

    Donor blocks are permuted within each split, independently of labels.
    This avoids treating binary labels as latent logits or embedding vectors.
    """
    generator = torch.Generator(device=state.c.device).manual_seed(seed)
    donor = state.c.clone()
    for mask in (state.train_mask, ~state.train_mask):
        rows = mask.nonzero().flatten()
        donor[rows] = state.c[rows[torch.randperm(len(rows), device=rows.device, generator=generator)]]
    mask = ~state.train_mask
    def accuracy(c):
        return (state.model.predict_from_concepts(c[mask]).argmax(1) == state.labels[mask]).float().mean().item()
    base = accuracy(state.c)
    reports = []
    for j, block in enumerate(state.model.blocks):
        c = state.model.substitute(state.c, j, donor[:, list(block)])
        acc = accuracy(c)
        reports.append({"concept_id": j, "accuracy": acc, "accuracy_drop": base-acc,
                        "diagnostics": registry.compute(replace(state, c=c))})
    return {"baseline_accuracy": base, "replacement": "within_split_random_donor", "per_concept": reports}

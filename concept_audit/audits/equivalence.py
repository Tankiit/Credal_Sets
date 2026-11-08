from dataclasses import replace
import torch
from concept_audit.transforms import ReparameterizedModel


@torch.no_grad()
def audit_equivalence(state, a, registry, tol=1e-5):
    transformed = ReparameterizedModel(state.model, a, tol)
    changed = replace(state, c=state.c @ transformed.a.T, model=transformed)
    r_error = (state.model.concept_readout(state.c) - transformed.concept_readout(changed.c)).abs().max().item()
    h_error = (state.model.predict_from_concepts(state.c) - transformed.predict_from_concepts(changed.c)).abs().max().item()
    if not r_error < tol or not h_error < tol:
        raise AssertionError(f"Equivalence failed: readout error={r_error}, logit error={h_error}, tol={tol}")
    return {
        "max_readout_error": r_error, "max_logit_error": h_error,
        "condition_number": torch.linalg.cond(transformed.a).item(),
        "baseline": registry.compute(state), "transformed": registry.compute(changed),
    }

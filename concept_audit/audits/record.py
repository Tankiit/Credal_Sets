"""One output schema for every family, so runs are comparable and plottable."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch


def _jsonable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


@dataclass
class RunRecord:
    """The unified schema. All families conform, which is what later plots need."""

    model_family: str
    model_variant: str
    latent_dim: int
    observables: dict = field(default_factory=dict)
    equivalence: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_report(cls, model, report, transform_type="stabilizer", dataset=None,
                    provenance=None, keep_arrays=False):
        eq = report.equivalence
        observables = {}
        for obs in model.observable_maps():
            if hasattr(obs, "rank"):
                observables[f"{obs.name}_rank"] = obs.rank
                observables[f"{obs.name}_nullity"] = obs.nullity
        if hasattr(model, "readout_rank"):
            observables.setdefault("readout_rank", model.readout_rank)
            observables.setdefault("unconstrained_dim", model.unconstrained_dim)

        diagnostics = {}
        for name, delta in report.deltas.items():
            entry = delta.summary()
            if keep_arrays:
                entry |= {"before": delta.before.tolist(), "after": delta.after.tolist()}
            diagnostics[name] = entry

        return cls(
            model_family=model.family,
            model_variant=model.variant,
            latent_dim=int(model.latent_dim()),
            observables=observables,
            equivalence={"transform_type": transform_type, "admissible": eq.admissible,
                         "condition_number": eq.condition_number,
                         "observable_error": eq.observable_error,
                         "prediction_error": eq.prediction_error,
                         "constraint_error": eq.constraint_error, "tol": eq.tol},
            diagnostics=diagnostics,
            dataset=dict(dataset or {}),
            provenance=dict(provenance or {}),
        )

    def to_dict(self):
        return _jsonable(asdict(self))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

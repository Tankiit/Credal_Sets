"""
Gradient isolation mixin for `VariationalCredalCBM` training.

Wraps each training step with a three-phase gradient probe and writes one
JSON line per step to `grad_iso_log.jsonl`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


EU_LOSS_KEYS_DEFAULT: List[str] = ["ce_loss", "kl_loss", "concept_loss", "task_loss", "credal_kl", "concept_bce"]
AU_LOSS_KEYS_DEFAULT: List[str] = ["aleatoric_loss"]


@dataclass
class GradIsoRecord:
    """One per-step probe record."""
    epoch: int
    step: int
    global_step: int

    # Loss values driving each phase
    loss_eu: float
    loss_au: float

    g_eu_encoder_norm: float
    g_au_encoder_norm: float
    cos_sim_encoder: float

    g_eu_concept_encoder_norm: float
    g_au_concept_encoder_norm: float
    cos_sim_concept_encoder: float

    g_eu_concept_classifier_norm: float
    g_au_concept_classifier_norm: float
    cos_sim_concept_classifier: float

    g_eu_task_classifier_norm: float
    g_au_task_classifier_norm: float
    cos_sim_task_classifier: float

    g_eu_aleatoric_head_norm: float
    g_au_aleatoric_head_norm: float
    cos_sim_aleatoric_head: float

    freeze_encoder: bool = True


class GradientIsolationMixin:
    """Mixin that overrides `train_epoch` to add per-step gradient probes."""

    # -- Configurable at instance level --
    eu_loss_keys: List[str] = EU_LOSS_KEYS_DEFAULT
    au_loss_keys: List[str] = AU_LOSS_KEYS_DEFAULT
    grad_iso_log_every: int = 1  # per-step

    # ----------------------------------------------------------------------
    def _grad_iso_setup(self):
        if getattr(self, "_grad_iso_ready", False):
            return

        self._grad_iso_path: Path = self.save_dir / "grad_iso_log.jsonl"
        self._grad_iso_path.parent.mkdir(parents=True, exist_ok=True)
        self._grad_iso_path.write_text("")   # truncate on fresh run
        self._grad_iso_global_step = 0

        m = self.model
        self._encoder_params: List[torch.nn.Parameter] = []
        if hasattr(m, "encoder"):
            self._encoder_params = [p for p in m.encoder.parameters() if p.requires_grad]

        if hasattr(m, "concept_encoder"):
            self._concept_encoder_params = list(m.concept_encoder.parameters())
        elif hasattr(m, "body") and hasattr(m.body, "concept_net"):
            self._concept_encoder_params = list(m.body.concept_net.parameters())
        else:
            self._concept_encoder_params = []

        if hasattr(m, "concept_classifier"):
            self._concept_classifier_params = list(m.concept_classifier.parameters())
        elif hasattr(m, "body") and hasattr(m.body, "relevance_net"):
            self._concept_classifier_params = list(m.body.relevance_net.parameters())
        elif hasattr(m, "body") and hasattr(m.body, "task_classifier"):
            self._concept_classifier_params = list(m.body.task_classifier.parameters())
        else:
            self._concept_classifier_params = []

        if hasattr(m, "task_classifier"):
            self._task_classifier_params = list(m.task_classifier.parameters())
        elif hasattr(m, "body") and hasattr(m.body, "task_classifier"):
            self._task_classifier_params = list(m.body.task_classifier.parameters())
        elif hasattr(m, "body") and hasattr(m.body, "relevance_net"):
            self._task_classifier_params = list(m.body.relevance_net.parameters())
        else:
            self._task_classifier_params = []
        self._aleatoric_head_params = list(m.aleatoric_head.parameters()) if hasattr(m, "aleatoric_head") else []

        self._grad_iso_ready = True

    # ----------------------------------------------------------------------
    @staticmethod
    def _flat_grad(params) -> torch.Tensor:
        parts = []
        for p in params:
            if p.grad is None:
                parts.append(torch.zeros_like(p).flatten())
            else:
                parts.append(p.grad.detach().flatten().clone())
        return torch.cat(parts) if parts else torch.tensor([])

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        na, nb = a.norm(), b.norm()
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float((a @ b) / (na * nb))

    def _sum_losses(self, outputs: Dict[str, torch.Tensor],
                    keys: List[str]) -> Optional[torch.Tensor]:
        terms = [outputs[k] for k in keys if k in outputs]
        if not terms:
            return None
        return sum(terms)

    # ----------------------------------------------------------------------
    def _probe_step(self, outputs: Dict[str, torch.Tensor],
                    epoch: int, step: int) -> Optional[GradIsoRecord]:
        """Run the two-backward probe. Returns None if skipped."""
        if step % self.grad_iso_log_every != 0:
            return None

        L_eu = self._sum_losses(outputs, self.eu_loss_keys)
        L_au = self._sum_losses(outputs, self.au_loss_keys)
        if L_eu is None or L_au is None:
            return None
        if float(L_eu.detach()) == 0.0 or float(L_au.detach()) == 0.0:
            return None

        m = self.model

        # -- Phase 1: L_eu only --
        m.zero_grad(set_to_none=False)
        L_eu.backward(retain_graph=True)

        g_eu_encoder = self._flat_grad(self._encoder_params)
        g_eu_concept_encoder = self._flat_grad(self._concept_encoder_params)
        g_eu_concept_classifier = self._flat_grad(self._concept_classifier_params)
        g_eu_task_classifier = self._flat_grad(self._task_classifier_params)
        g_eu_aleatoric_head = self._flat_grad(self._aleatoric_head_params)

        # -- Phase 2: L_au only --
        m.zero_grad(set_to_none=False)
        L_au.backward(retain_graph=True)

        g_au_encoder = self._flat_grad(self._encoder_params)
        g_au_concept_encoder = self._flat_grad(self._concept_encoder_params)
        g_au_concept_classifier = self._flat_grad(self._concept_classifier_params)
        g_au_task_classifier = self._flat_grad(self._task_classifier_params)
        g_au_aleatoric_head = self._flat_grad(self._aleatoric_head_params)

        # -- Phase 3 happens in the caller (combined backward + optimizer step)
        m.zero_grad(set_to_none=False)

        rec = GradIsoRecord(
            epoch=epoch, step=step, global_step=self._grad_iso_global_step,
            loss_eu=float(L_eu.detach()),
            loss_au=float(L_au.detach()),
            g_eu_encoder_norm=float(g_eu_encoder.norm()),
            g_au_encoder_norm=float(g_au_encoder.norm()),
            cos_sim_encoder=self._cos_sim(g_eu_encoder, g_au_encoder),
            g_eu_concept_encoder_norm=float(g_eu_concept_encoder.norm()),
            g_au_concept_encoder_norm=float(g_au_concept_encoder.norm()),
            cos_sim_concept_encoder=self._cos_sim(g_eu_concept_encoder, g_au_concept_encoder),
            g_eu_concept_classifier_norm=float(g_eu_concept_classifier.norm()),
            g_au_concept_classifier_norm=float(g_au_concept_classifier.norm()),
            cos_sim_concept_classifier=self._cos_sim(g_eu_concept_classifier, g_au_concept_classifier),
            g_eu_task_classifier_norm=float(g_eu_task_classifier.norm()),
            g_au_task_classifier_norm=float(g_au_task_classifier.norm()),
            cos_sim_task_classifier=self._cos_sim(g_eu_task_classifier, g_au_task_classifier),
            g_eu_aleatoric_head_norm=float(g_eu_aleatoric_head.norm()),
            g_au_aleatoric_head_norm=float(g_au_aleatoric_head.norm()),
            cos_sim_aleatoric_head=self._cos_sim(g_eu_aleatoric_head, g_au_aleatoric_head),
            freeze_encoder=bool(m.config.freeze_encoder),
        )

        return rec

    def _persist_record(self, rec: GradIsoRecord):
        with self._grad_iso_path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()

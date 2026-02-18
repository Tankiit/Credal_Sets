import os
from typing import Dict, Optional, Tuple, List

import torch
from dataclasses import asdict, is_dataclass


def _to_cpu_detached(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _maybe_param_to_tensor(x) -> torch.Tensor:
    # Handles nn.Parameter or Tensor
    if hasattr(x, "data") and isinstance(x.data, torch.Tensor):
        return _to_cpu_detached(x.data)
    return _to_cpu_detached(x)


def assign_quadrants(
    eu: torch.Tensor,
    au: torch.Tensor,
    method: str = "median",
    q: float = 0.5,
    eu_thr: Optional[float] = None,
    au_thr: Optional[float] = None,
) -> torch.Tensor:
    """
    Assign each sample to a quadrant based on epistemic (eu) and aleatoric (au).

    Returns indices in {0,1,2,3} for (low/low, low/high, high/low, high/high).
    Thresholds use medians by default.
    """
    eu = _to_cpu_detached(eu).flatten()
    au = _to_cpu_detached(au).flatten()

    if method == "median":
        eu_thr = torch.median(eu)
        au_thr = torch.median(au)
    elif method == "quantile":
        eu_thr = torch.quantile(eu, q)
        au_thr = torch.quantile(au, q)
    elif method == "fixed":
        if eu_thr is None or au_thr is None:
            raise ValueError("For method='fixed', eu_thr and au_thr must be provided.")
        eu_thr = torch.tensor(float(eu_thr))
        au_thr = torch.tensor(float(au_thr))
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")

    hi_eu = eu > eu_thr
    hi_au = au > au_thr

    # 0: low/low, 1: low/high, 2: high/low, 3: high/high
    return (hi_eu.int() * 2 + hi_au.int()).to(torch.int64)


@torch.no_grad()
def collect_eval_outputs(
    model,
    encoder,
    loader,
    device: str = "cpu",
    include_heads: bool = True,
    compute_quadrants: bool = True,
    quad_method: str = "median",
    quad_q: float = 0.5,
    quad_eu_thr: Optional[float] = None,
    quad_au_thr: Optional[float] = None,
) -> Dict:
    """
    Run a full pass over `loader` and collect evaluation outputs.

    Returns a dict with keys compatible with torch.save.
    """
    model.eval()
    encoder.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_concepts: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_eps: List[torch.Tensor] = []
    all_ale: List[torch.Tensor] = []  # per-sample scalar AU
    all_ale_full: List[torch.Tensor] = []  # per-concept AU [B,K]

    # Optional: per-head predictions [N, B, K] aggregated later
    head_probs_accum: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)

        concept_entropy = batch.get('concept_entropy', None)
        if concept_entropy is not None:
            concept_entropy = concept_entropy.to(device)

        # Encoder features
        features = encoder(input_ids, attention_mask, return_cls_only=True)

        # Full model forward for logits, epsilon, aleatoric
        out = model(features, labels, concept_labels, is_unknown,
                    concept_entropy=concept_entropy)

        logits = out['logits']  # [B, J]
        preds = logits.argmax(dim=1)
        eps_b = out['epsilon']  # [B]

        all_logits.append(logits)
        all_labels.append(labels)
        all_concepts.append(concept_labels)
        all_preds.append(preds)
        all_eps.append(eps_b)

        # Aleatoric per-sample summary: mean across concepts if available
        a_hat = out.get('a_hat', None)
        if isinstance(a_hat, torch.Tensor) and a_hat.numel() > 1:
            all_ale.append(a_hat.mean(dim=1))  # [B]
            all_ale_full.append(a_hat)         # [B,K]

        # Optionally collect per-head probabilities
        if include_heads:
            # Access the ensemble directly to get [N, B, K]
            _, _, all_probs, _ = model.concept_ensemble(features)
            head_probs_accum.append(all_probs)  # [N, B, K]

    # Stack across batches
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    concepts = torch.cat(all_concepts, dim=0)
    preds = torch.cat(all_preds, dim=0)
    epsilon = torch.cat(all_eps, dim=0)
    ale = torch.cat(all_ale, dim=0) if len(all_ale) > 0 else torch.full_like(epsilon, fill_value=torch.nan)
    ale_full = torch.cat(all_ale_full, dim=0) if len(all_ale_full) > 0 else None

    # Aggregate head predictions to shape [N_total, H, K] as requested
    head_predictions = None
    if include_heads and len(head_probs_accum) > 0:
        # head_probs_accum: list of [N, B_i, K]
        head_probs = torch.cat(head_probs_accum, dim=1)  # [N, sum(B), K]
        head_predictions = head_probs.permute(1, 0, 2).contiguous()  # [N_total, H, K]

    result = {
        'logits': _to_cpu_detached(logits),               # [N, J]
        'predictions': _to_cpu_detached(preds),           # [N]
        'true_labels': _to_cpu_detached(labels),          # [N]
        'true_concepts': _to_cpu_detached(concepts),      # [N, K]
        'epsilon': _to_cpu_detached(epsilon),             # [N]
        'aleatoric_mean': _to_cpu_detached(ale),          # [N]
    }

    if head_predictions is not None:
        result['head_predictions'] = _to_cpu_detached(head_predictions)  # [N, H, K]

    if ale_full is not None:
        result['aleatoric_preds'] = _to_cpu_detached(ale_full)          # [N, K]

    if compute_quadrants:
        try:
            quads = assign_quadrants(
                epsilon,
                ale,
                method=quad_method,
                q=quad_q,
                eu_thr=quad_eu_thr,
                au_thr=quad_au_thr,
            )
            result['quadrant_assignments'] = _to_cpu_detached(quads)     # [N]
        except Exception:
            # Be tolerant if AU is NaN everywhere
            pass

    return result


def save_eval_outputs(payload: Dict,
                      model,
                      config,
                      out_path: str) -> str:
    """
    Save evaluation outputs with model parameters and config metadata.

    - Ensures directory exists
    - Moves tensors to CPU and detaches
    - Adds classifier weights/bias and config dict
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Add classifier params if available
    if hasattr(model, 'label_head') and hasattr(model.label_head, 'W'):
        payload['classifier_W'] = _maybe_param_to_tensor(model.label_head.W)  # [J,K]
    if hasattr(model, 'label_head') and hasattr(model.label_head, 'b'):
        payload['classifier_b'] = _maybe_param_to_tensor(model.label_head.b)  # [J]

    # Attach config as dict for portability
    if is_dataclass(config):
        payload['config'] = asdict(config)
    else:
        # Best-effort fallback
        payload['config'] = getattr(config, '__dict__', str(config))

    torch.save(payload, out_path)
    return out_path

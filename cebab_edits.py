"""Validated CEBaB original-to-edit pairs and model-effect evaluation.

CEBaB stores one original review and several counterfactual edits under the same
``info.original_id``.  This module deliberately does *not* infer pairs from row
order: it identifies the original explicitly, keeps the edited aspect, and only
intervenes on that aspect's two binary concept coordinates.  The resulting
``human_effect`` is the observed ordinal star-rating change in the benchmark.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from concept_datasets import load_meta, load_split
from concept_models.training import predict


ASPECTS = ("food", "service", "ambiance", "noise")


@dataclass(frozen=True)
class EditPair:
    """One original review, one aspect-targeted CEBaB edit, and its human effect."""

    original_index: int
    edit_index: int
    original_id: str
    edit_id: str
    aspect: str
    goal: str
    human_effect: int
    original_label: int
    edited_label: int
    original_text: str
    edited_text: str


def load_edit_pairs(split: str = "test") -> list[EditPair]:
    """Load all valid CEBaB original->single-aspect Positive/Negative edits.

    Raises rather than silently returning an empty result if an invariant is
    violated; a silent zero was the failure mode of the previous evaluator.
    """
    records = load_split("cebab", split)
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        info = record.get("info") or {}
        original_id = info.get("original_id")
        if not original_id:
            raise ValueError(f"CEBaB {split} row {index} has no original_id")
        groups[str(original_id)].append((index, record))

    pairs: list[EditPair] = []
    for original_id, group in groups.items():
        originals = [(i, r) for i, r in group if (r.get("info") or {}).get("is_original")]
        if len(originals) != 1:
            raise ValueError(
                f"CEBaB {split} group {original_id!r} has {len(originals)} originals; expected one"
            )
        original_index, original = originals[0]
        for edit_index, edited in group:
            info = edited.get("info") or {}
            aspect, goal = info.get("edit_type"), info.get("edit_goal")
            if edit_index == original_index:
                continue
            # ``unknown`` is not a directional concept intervention, so it is
            # excluded from this targeted Positive/Negative effect analysis.
            if aspect not in ASPECTS or goal not in {"Positive", "Negative"}:
                continue
            pairs.append(EditPair(
                original_index=original_index,
                edit_index=edit_index,
                original_id=original_id,
                edit_id=str(info.get("id")),
                aspect=aspect,
                goal=goal,
                human_effect=int(edited["label"]) - int(original["label"]),
                original_label=int(original["label"]),
                edited_label=int(edited["label"]),
                original_text=original["text"],
                edited_text=edited["text"],
            ))
    if not pairs:
        raise ValueError(f"No valid Positive/Negative CEBaB edit pairs in split={split!r}")
    return pairs


def _aspect_mask_and_values(pairs: list[EditPair], concepts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    meta = load_meta("cebab")
    names = meta["concept_names"]
    mask = torch.zeros((len(pairs), len(names)), dtype=torch.bool)
    values = torch.zeros((len(pairs), len(names)), dtype=concepts.dtype)
    edit_indices = [pair.edit_index for pair in pairs]
    values[:] = concepts[edit_indices]
    for row, pair in enumerate(pairs):
        for suffix in ("pos", "neg"):
            mask[row, names.index(f"{pair.aspect}_{suffix}")] = True
    return mask, values


def expected_rating(logits: np.ndarray) -> np.ndarray:
    """Expected 1--5 star rating under a model's categorical distribution."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs @ np.arange(1, logits.shape[1] + 1, dtype=probs.dtype)


def evaluate_edit_effects(model, features: dict, pairs: list[EditPair]) -> list[dict]:
    """Estimate each pair's targeted concept-intervention effect for ``model``.

    The source feature is always the original text.  Only the two concept slots
    belonging to the edited aspect are overwritten with annotations from the
    corresponding edited review.  This isolates a concept intervention from a
    text-edit effect and makes model-vs-twin differences interpretable.
    """
    if "X" not in features or "C" not in features:
        raise ValueError("CEBaB edit effects require aligned X and C feature tensors")
    n = len(features["X"])
    if any(p.original_index >= n or p.edit_index >= n for p in pairs):
        raise ValueError("CEBaB records and cached features are not aligned")
    original_indices = [pair.original_index for pair in pairs]
    X = features["X"][original_indices]
    mask, values = _aspect_mask_and_values(pairs, features["C"])
    baseline = predict(model, X)["logits"]
    intervened = predict(model, X, values, mask)["logits"]
    base_rating, edited_rating = expected_rating(baseline), expected_rating(intervened)
    base_class, edited_class = baseline.argmax(axis=1), intervened.argmax(axis=1)

    rows = []
    for i, pair in enumerate(pairs):
        rows.append({
            "original_id": pair.original_id,
            "edit_id": pair.edit_id,
            "aspect": pair.aspect,
            "goal": pair.goal,
            "original_index": pair.original_index,
            "edit_index": pair.edit_index,
            "human_effect": pair.human_effect,
            "human_original_label": pair.original_label,
            "human_edited_label": pair.edited_label,
            "model_expected_effect": float(edited_rating[i] - base_rating[i]),
            "model_argmax_effect": int(edited_class[i] - base_class[i]),
            "model_changed_label": bool(edited_class[i] != base_class[i]),
            "original_text": pair.original_text,
            "edited_text": pair.edited_text,
        })
    return rows


def compare_model_and_twin(model, twin, features: dict, pairs: list[EditPair]) -> list[dict]:
    """Return paired model/twin effects plus their difference and human target."""
    original_rows = evaluate_edit_effects(model, features, pairs)
    twin_rows = evaluate_edit_effects(twin, features, pairs)
    rows = []
    for original, transformed in zip(original_rows, twin_rows, strict=True):
        row = dict(original)
        row["twin_expected_effect"] = transformed["model_expected_effect"]
        row["twin_argmax_effect"] = transformed["model_argmax_effect"]
        row["twin_changed_label"] = transformed["model_changed_label"]
        row["twin_minus_model_expected_effect"] = (
            transformed["model_expected_effect"] - original["model_expected_effect"]
        )
        row["twins_disagree_on_label_change"] = (
            transformed["model_changed_label"] != original["model_changed_label"]
        )
        rows.append(row)
    return rows

"""
Preprocess `ttomov/ambigqa_star` into the format MAQADataset expects.

This loader is intentionally lightweight:
- It loads raw Hugging Face splits.
- It extracts a question text field.
- It builds a uniform `p_star` over the available answers.
- It derives entropy and ambiguity level.

If the dataset schema differs, the loader falls back across common field names.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from datasets import load_dataset


_AMB_LEVEL_MAP = {
    "clear": 0,
    0: 0,
    "0": 0,
    "medium": 1,
    1: 1,
    "1": 1,
    "ambiguous": 2,
    2: 2,
    "2": 2,
}


def _get_question_text(ex: dict[str, Any]) -> str:
    for key in ("question", "text", "query", "prompt", "input"):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    question = ex.get("question")
    if isinstance(question, dict):
        for key in ("text", "stem", "title"):
            value = question.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _get_answers(ex: dict[str, Any]) -> list[str]:
    answers = ex.get("answers")
    if answers is None:
        answers = ex.get("valid_answers") or []
    if isinstance(answers, dict):
        answers = answers.get("text", [])
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, (list, tuple)):
        return []
    out = []
    for a in answers:
        if a is None:
            continue
        s = str(a).strip()
        if s:
            out.append(s)
    return out


def load_maqa_direct(hf_name: str = "ttomov/ambigqa_star", split: str = "train") -> list[dict]:
    """
    Load and preprocess one split into the MAQADataset-ready format.
    """
    raw = load_dataset(hf_name, split=split)
    out: list[dict] = []

    for ex in raw:
        text = _get_question_text(ex)
        if not text:
            continue

        answers = _get_answers(ex)
        if not answers:
            continue

        n = len(answers)
        p_star = np.full(n, 1.0 / n, dtype=np.float32)
        entropy = float(np.log(n))

        amb_raw = ex.get("ambiguity_level")
        if amb_raw is None:
            amb_raw = 0 if n == 1 else (2 if n >= 3 else 1)
        amb_level = _AMB_LEVEL_MAP.get(amb_raw, 1)

        dom_idx = int(np.argmax(p_star))
        out.append(
            {
                "text": text,
                "answers": answers,
                "p_star": p_star,
                "entropy": entropy,
                "ambiguity_level": amb_level,
                "dominant_answer_idx": dom_idx,
            }
        )

    print(f"[maqa] preprocessed {len(out)} items from {hf_name}:{split}")
    return out

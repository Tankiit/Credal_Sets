"""Shared helpers: the standard on-disk format every dataset file writes.

Layout produced by each dataset's ``download()``::

    data/<name>/meta.json
    data/<name>/train.jsonl
    data/<name>/val.jsonl
    data/<name>/test.jsonl
    data/<name>/<extra>.jsonl      # optional extra eval splits (e.g. counterfactual pairs)

Each JSONL line is one example::

    {"text": str,
     "label": int,                       # task label, index into meta["class_names"]
     "concepts": [0/1, ...] | null,      # binary concept labels, index into meta["concept_names"]
     "concept_scores": [float, ...],     # optional soft targets in [0, 1] (same order)
     "info": {...}}                      # optional per-example metadata (ids, pair ids, ...)

All concepts are binary so that every model uses the same readout (one sigmoid per
concept). Ternary aspects (Positive / Negative / unknown) become two binary concepts,
``<aspect>_pos`` and ``<aspect>_neg``; "unknown" is both 0. ``meta["concept_groups"]``
records which binary concepts came from the same aspect.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def dataset_dir(name: str) -> Path:
    path = DATA_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def aspect_to_binary(value) -> tuple[int, int]:
    """'Positive' -> (1, 0), 'Negative' -> (0, 1), anything else ('unknown', '', None) -> (0, 0)."""
    value = (value or "").strip().lower()
    return int(value == "positive"), int(value == "negative")


def aspect_concept_names(aspects: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    names, groups = [], {}
    for a in aspects:
        a = a.replace(" ", "_")
        groups[a] = [f"{a}_pos", f"{a}_neg"]
        names += groups[a]
    return names, groups


def write_split(name: str, split: str, examples: list[dict]) -> None:
    path = dataset_dir(name) / f"{split}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  wrote {len(examples):>6} examples -> {path.relative_to(DATA_DIR.parent)}")


def write_meta(name: str, meta: dict, splits: dict[str, list[dict]]) -> None:
    """Write meta.json, adding sizes, label distribution and concept prevalence per split."""
    meta = dict(meta, name=name, splits={})
    for split, examples in splits.items():
        stats = {"size": len(examples),
                 "label_counts": dict(sorted(Counter(ex["label"] for ex in examples).items()))}
        with_c = [ex["concepts"] for ex in examples if ex.get("concepts") is not None]
        if with_c:
            k = len(meta["concept_names"])
            stats["concept_prevalence"] = {
                meta["concept_names"][j]: round(sum(c[j] for c in with_c) / len(with_c), 4)
                for j in range(k)}
        meta["splits"][split] = stats
    path = dataset_dir(name) / "meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote meta -> {path.relative_to(DATA_DIR.parent)}")


def save_dataset(name: str, meta: dict, splits: dict[str, list[dict]]) -> None:
    for split, examples in splits.items():
        write_split(name, split, examples)
    write_meta(name, meta, splits)
    print_summary(name)


def print_summary(name: str) -> None:
    meta = json.loads((DATA_DIR / name / "meta.json").read_text())
    print(f"\n  {name}: {len(meta['class_names'])} classes {meta['class_names']}, "
          f"{len(meta['concept_names'])} concepts")
    for split, s in meta["splits"].items():
        print(f"    {split:<12} n={s['size']:<7} labels={s['label_counts']}")
    prev = meta["splits"].get("train", {}).get("concept_prevalence", {})
    if prev:
        print("    train concept prevalence: "
              + ", ".join(f"{k}={v:.2f}" for k, v in prev.items()))

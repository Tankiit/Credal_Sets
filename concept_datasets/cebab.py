"""CEBaB — restaurant reviews with aspect-level sentiment and counterfactual edits.

Source : https://huggingface.co/datasets/CEBaB/CEBaB  (Abraham et al., NeurIPS 2022)
Task   : review star rating, 5 classes (1..5 stars); reviews with "no majority" are dropped.
Concepts (8 binary): food / service / ambiance / noise, each split into _pos and _neg
         (aspect majority "Positive" / "Negative"; "unknown" or empty -> both 0).
Splits : train = train_inclusive (originals + their counterfactual edits), val = validation,
         test = test.  ``info`` keeps the edit metadata (original_id, edit_goal, ...) so
         counterfactual pairs can be rebuilt later.

    python -m concept_datasets.cebab
"""

from datasets import load_dataset

from ._common import aspect_concept_names, aspect_to_binary, save_dataset

NAME = "cebab"
ASPECTS = ["food", "service", "ambiance", "noise"]
SPLITS = {"train": "train_inclusive", "val": "validation", "test": "test"}


def convert(row) -> dict | None:
    stars = row["review_majority"]
    if stars not in {"1", "2", "3", "4", "5"} or not (row["description"] or "").strip():
        return None
    concepts = []
    for a in ASPECTS:
        concepts += aspect_to_binary(row[f"{a}_aspect_majority"])
    return {
        "text": row["description"].strip(),
        "label": int(stars) - 1,
        "concepts": concepts,
        "info": {k: row[k] for k in ["id", "original_id", "is_original", "edit_goal", "edit_type"]},
    }


def download() -> None:
    print(f"[{NAME}] downloading CEBaB/CEBaB from the HuggingFace Hub")
    splits = {}
    for ours, theirs in SPLITS.items():
        rows = load_dataset("CEBaB/CEBaB", split=theirs)
        splits[ours] = [ex for ex in map(convert, rows) if ex is not None]
    concept_names, groups = aspect_concept_names(ASPECTS)
    meta = {
        "description": "CEBaB restaurant reviews: predict star rating from aspect sentiments.",
        "source": "https://huggingface.co/datasets/CEBaB/CEBaB",
        "class_names": ["1_star", "2_stars", "3_stars", "4_stars", "5_stars"],
        "concept_names": concept_names,
        "concept_groups": groups,
    }
    save_dataset(NAME, meta, splits)


if __name__ == "__main__":
    download()

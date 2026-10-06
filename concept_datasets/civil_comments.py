"""Civil Comments — online comments with crowd-sourced toxicity sub-type scores.

Source : https://huggingface.co/datasets/google/civil_comments  (Borkan et al., 2019)
Task   : toxic vs non-toxic, label = toxicity >= 0.5.
Concepts (6 binary): severe_toxicity, obscene, threat, insult, identity_attack,
         sexual_explicit, each = score >= CONCEPT_THRESHOLD.  The raw scores (fraction of
         annotators) are kept as ``concept_scores`` for soft-target training.
Splits : the full data has ~1.8M comments with ~8% toxic, so we draw a fixed-seed,
         class-balanced subsample of each split (sizes below).

    python -m concept_datasets.civil_comments
"""

import pandas as pd
from datasets import load_dataset

from ._common import save_dataset

NAME = "civil_comments"
CONCEPTS = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
LABEL_THRESHOLD = 0.5
CONCEPT_THRESHOLD = 0.2  # >= 20% of raters; at 0.5 severe_toxicity is never on
SPLITS = {"train": ("train", 40_000), "val": ("validation", 5_000), "test": ("test", 10_000)}
TOXIC_FRACTION = 0.5
SEED = 0


def balanced_sample(df, n: int):
    toxic = df[df.toxicity >= LABEL_THRESHOLD]
    clean = df[df.toxicity < LABEL_THRESHOLD]
    n_toxic = min(int(n * TOXIC_FRACTION), len(toxic))
    both = pd.concat([toxic.sample(n_toxic, random_state=SEED),
                      clean.sample(n - n_toxic, random_state=SEED)])
    return both.sample(frac=1.0, random_state=SEED)


def convert(row) -> dict:
    scores = [round(float(row[c]), 4) for c in CONCEPTS]
    return {
        "text": row["text"].strip(),
        "label": int(row["toxicity"] >= LABEL_THRESHOLD),
        "concepts": [int(s >= CONCEPT_THRESHOLD) for s in scores],
        "concept_scores": scores,
        "info": {"toxicity": round(float(row["toxicity"]), 4)},
    }


def download() -> None:
    print(f"[{NAME}] downloading google/civil_comments (large: ~2M rows, subsampled)")
    splits = {}
    for ours, (theirs, n) in SPLITS.items():
        df = load_dataset("google/civil_comments", split=theirs).to_pandas()
        df = df[df.text.str.strip().str.len() > 0]
        splits[ours] = [convert(r) for _, r in balanced_sample(df, n).iterrows()]
    meta = {
        "description": "Civil Comments: predict toxicity from toxicity sub-type concepts "
                       f"(class-balanced subsample, seed {SEED}).",
        "source": "https://huggingface.co/datasets/google/civil_comments",
        "class_names": ["non_toxic", "toxic"],
        "concept_names": CONCEPTS,
        "concept_groups": {},
        "concept_threshold": CONCEPT_THRESHOLD,
    }
    save_dataset(NAME, meta, splits)


if __name__ == "__main__":
    download()

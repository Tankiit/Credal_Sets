"""IMDB-C + IMDB CAD — movie reviews with aspect concepts, plus counterfactual pairs.

Two public sources are combined:

1. IMDB-C (Tan et al., "Interpreting Pretrained Language Models via Concept Bottlenecks",
   PAKDD 2024; github.com/Zhen-Tan-dmml/CBM_NLP). 2000/500/500 IMDB reviews with 8
   ChatGPT-annotated aspect concepts (acting, storyline, emotional arousal, cinematography,
   soundtrack, directing, background setting, editing; each Positive / Negative / unknown).
   -> train / val / test.  Task = binary sentiment, 16 binary concepts (<aspect>_pos/_neg).

2. CAD (Kaushik et al., "Learning the Difference that Makes a Difference with
   Counterfactually-Augmented Data", ICLR 2020; github.com/acmi-lab/counterfactually-augmented-data).
   2440 original reviews, each paired with a human edit that flips the sentiment.
   -> extra split ``cad_pairs`` (no concept labels; ``concepts`` is null).
   ``info`` = {pair_id, is_original, cad_split}.

Caveat: despite the name, IMDB-C reviews are mostly *not* the CAD reviews (only ~6% of the
IMDB-C train texts appear in CAD). The concept labels exist only for IMDB-C; the CAD pairs
are kept as a counterfactual evaluation set.

    python -m concept_datasets.imdb_cad
"""

import csv
import io
import re
import urllib.request

from ._common import aspect_concept_names, aspect_to_binary, save_dataset

NAME = "imdb_cad"
ASPECTS = ["acting", "storyline", "emotional arousal", "cinematography", "soundtrack",
           "directing", "background setting", "editing"]
IMDBC_URL = "https://raw.githubusercontent.com/Zhen-Tan-dmml/CBM_NLP/main/dataset/imdb/New/IMDB-{}-generated.csv"
IMDBC_SPLITS = {"train": "train", "val": "dev", "test": "test"}
CAD_URL = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/{}"
CAD_SPLITS = ["train", "dev", "test"]
CLASS_NAMES = ["negative", "positive"]


def fetch_rows(url: str, delimiter: str = ",") -> list[dict]:
    with urllib.request.urlopen(url) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8")), delimiter=delimiter))


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", text)).strip()


def convert_imdbc(row) -> dict:
    concepts = []
    for a in ASPECTS:
        concepts += aspect_to_binary(row[a])
    return {"text": clean(row["review"]),
            "label": CLASS_NAMES.index(row["sentiment"].strip().lower()),
            "concepts": concepts}


def load_cad_pairs() -> list[dict]:
    examples = []
    for split in CAD_SPLITS:
        originals = {clean(r["Text"]) for r in fetch_rows(CAD_URL.format(f"orig/{split}.tsv"), "\t")}
        for r in fetch_rows(CAD_URL.format(f"combined/paired/{split}_paired.tsv"), "\t"):
            text = clean(r["Text"])
            examples.append({
                "text": text,
                "label": CLASS_NAMES.index(r["Sentiment"].strip().lower()),
                "concepts": None,
                "info": {"pair_id": f"{split}-{r['batch_id']}", "is_original": text in originals,
                         "cad_split": split},
            })
    return examples


def download() -> None:
    print(f"[{NAME}] downloading IMDB-C (concepts) and CAD (counterfactual pairs) from GitHub")
    splits = {ours: [convert_imdbc(r) for r in fetch_rows(IMDBC_URL.format(theirs))]
              for ours, theirs in IMDBC_SPLITS.items()}
    splits["cad_pairs"] = load_cad_pairs()
    concept_names, groups = aspect_concept_names(ASPECTS)
    meta = {
        "description": "IMDB-C movie reviews: predict sentiment from 8 aspect sentiments; "
                       "plus CAD counterfactual pairs (no concept labels) as split 'cad_pairs'.",
        "source": ["https://github.com/Zhen-Tan-dmml/CBM_NLP",
                   "https://github.com/acmi-lab/counterfactually-augmented-data"],
        "class_names": CLASS_NAMES,
        "concept_names": concept_names,
        "concept_groups": groups,
    }
    save_dataset(NAME, meta, splits)


if __name__ == "__main__":
    download()

"""GoEmotions — Reddit comments annotated with 27 emotions + neutral.

Source : https://huggingface.co/datasets/google-research-datasets/go_emotions  (simplified config)
Concepts (28 binary): the emotion labels themselves (multi-label).
Task   : coarse sentiment, 4 classes, using the official GoEmotions sentiment grouping
         (github.com/google-research/google-research/blob/master/goemotions/data/sentiment_mapping.json):
         positive / negative / ambiguous / neutral.
         Rule per comment: if it has both positive and negative emotions it is dropped;
         otherwise positive > negative > ambiguous > neutral (neutral only if nothing else).
         The task label is therefore a deterministic function of the concepts (a "complete"
         concept set) and is never itself one of the concepts.

    python -m concept_datasets.goemotions
"""

from datasets import load_dataset

from ._common import save_dataset

NAME = "goemotions"
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
SENTIMENT = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
                 "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness",
                 "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
    "neutral": ["neutral"],
}
CLASS_NAMES = ["positive", "negative", "ambiguous", "neutral"]
GROUP_OF = {e: g for g, es in SENTIMENT.items() for e in es}
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def convert(row) -> dict | None:
    emotions = {EMOTIONS[i] for i in row["labels"]}
    groups = {GROUP_OF[e] for e in emotions}
    if not emotions or {"positive", "negative"} <= groups:
        return None
    label = next(g for g in CLASS_NAMES if g in groups)
    return {
        "text": row["text"].strip(),
        "label": CLASS_NAMES.index(label),
        "concepts": [int(e in emotions) for e in EMOTIONS],
        "info": {"id": row["id"]},
    }


def download() -> None:
    print(f"[{NAME}] downloading google-research-datasets/go_emotions (simplified)")
    splits = {}
    for ours, theirs in SPLITS.items():
        rows = load_dataset("google-research-datasets/go_emotions", "simplified", split=theirs)
        splits[ours] = [ex for ex in map(convert, rows) if ex is not None]
    meta = {
        "description": "GoEmotions Reddit comments: predict coarse sentiment from 28 emotions.",
        "source": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
        "class_names": CLASS_NAMES,
        "concept_names": EMOTIONS,
        "concept_groups": {g: es for g, es in SENTIMENT.items()},
    }
    save_dataset(NAME, meta, splits)


if __name__ == "__main__":
    download()

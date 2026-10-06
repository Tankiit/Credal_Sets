"""One module per dataset; each has ``download()`` writing data/<name>/ in a common format.

    from concept_datasets import load_meta, load_split
    meta = load_meta("cebab")
    train = load_split("cebab", "train")     # list of dicts: text, label, concepts, ...
"""

import importlib
import json
import os

# The Hub's Xet download backend fails on some networks; plain HTTP downloads always work.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ._common import DATA_DIR  # noqa: E402

DATASETS = ["cebab", "goemotions", "civil_comments", "imdb_cad"]


def download(name: str) -> None:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    importlib.import_module(f"concept_datasets.{name}").download()


def _require(name: str):
    path = DATA_DIR / name / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python download.py {name}` first")
    return path


def load_meta(name: str) -> dict:
    return json.loads(_require(name).read_text())


def load_split(name: str, split: str) -> list[dict]:
    _require(name)
    with open(DATA_DIR / name / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]

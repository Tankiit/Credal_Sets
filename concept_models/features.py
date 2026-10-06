"""Frozen-encoder text features, cached on disk.

The construction only touches the concept layer and the head, so the text encoder is kept
frozen: each split is embedded once (mean-pooled last hidden state) and cached at
cache/<dataset>/<encoder>/<split>.pt as a dict of tensors:

    X       (n, d) float32   text features
    y       (n,)   int64     task labels
    C       (n, k) float32   binary concept labels      (absent for splits without concepts)
    C_soft  (n, k) float32   soft concept targets        (only if the dataset provides them)
"""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from concept_datasets import load_split

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"


@torch.no_grad()
def embed_texts(texts: list[str], encoder: str = DEFAULT_ENCODER, max_length: int = 384,
                batch_size: int = 64, device: str | None = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
    out = torch.empty(len(texts), model.config.hidden_size)
    for start in tqdm(range(0, len(texts), batch_size), desc="  embedding", leave=False):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()
    return out


def get_features(dataset: str, split: str, encoder: str = DEFAULT_ENCODER,
                 max_length: int = 384) -> dict[str, torch.Tensor]:
    path = CACHE_DIR / dataset / encoder.replace("/", "__") / f"{split}.pt"
    if path.exists():
        return torch.load(path)
    print(f"  embedding {dataset}/{split} with {encoder} (cached afterwards)")
    examples = load_split(dataset, split)
    feats = {"X": embed_texts([ex["text"] for ex in examples], encoder, max_length),
             "y": torch.tensor([ex["label"] for ex in examples])}
    if all(ex.get("concepts") is not None for ex in examples):
        feats["C"] = torch.tensor([ex["concepts"] for ex in examples], dtype=torch.float32)
    if all("concept_scores" in ex for ex in examples):
        feats["C_soft"] = torch.tensor([ex["concept_scores"] for ex in examples], dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats

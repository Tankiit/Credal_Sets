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
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from concept_datasets import load_split

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"


@torch.no_grad()
def embed_texts(texts: list[str], encoder: str = DEFAULT_ENCODER, max_length: int = 384,
                batch_size: int = 64, device: str | None = None,
                progress_path: Path | None = None, save_every_batches: int = 1) -> torch.Tensor:
    """Embed text, optionally checkpointing progress for long CPU-only jobs."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
    out = torch.empty(len(texts), model.config.hidden_size)
    next_start = 0
    mmap = None
    state_path = data_path = None
    if progress_path is not None:
        state_path = progress_path.with_suffix(".json")
        data_path = progress_path.with_suffix(".npy")
        if progress_path.exists():  # Migrate the original torch checkpoint once.
            saved = torch.load(progress_path, map_location="cpu", weights_only=True)
            if tuple(saved["features"].shape) != tuple(out.shape):
                raise ValueError(f"Embedding checkpoint shape mismatch: {progress_path}")
            mmap = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32,
                                              shape=tuple(out.shape))
            mmap[:] = saved["features"].numpy()
            mmap.flush()
            next_start = int(saved["next_start"])
            state_path.write_text(json.dumps({"next_start": next_start}))
            progress_path.unlink()
        elif data_path.exists() and state_path.exists():
            mmap = np.load(data_path, mmap_mode="r+")
            if tuple(mmap.shape) != tuple(out.shape):
                raise ValueError(f"Embedding checkpoint shape mismatch: {data_path}")
            next_start = int(json.loads(state_path.read_text())["next_start"])
        else:
            mmap = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32,
                                              shape=tuple(out.shape))
        out = torch.from_numpy(mmap)
        if next_start:
            print(f"  resuming embeddings at {next_start}/{len(texts)}")
    for batch_number, start in enumerate(tqdm(range(next_start, len(texts), batch_size), desc="  embedding", leave=False),
                                         start=next_start // batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()
        next_start = start + len(idx)
        if progress_path is not None and (next_start == len(texts) or (batch_number + 1) % save_every_batches == 0):
            mmap.flush()
            state_path.write_text(json.dumps({"next_start": next_start}))
    if progress_path is not None:
        result = torch.from_numpy(np.array(mmap, copy=True))
        del out, mmap
        data_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        return result
    return out


def get_features(dataset: str, split: str, encoder: str = DEFAULT_ENCODER,
                 max_length: int = 384, batch_size: int = 64) -> dict[str, torch.Tensor]:
    path = CACHE_DIR / dataset / encoder.replace("/", "__") / f"{split}.pt"
    if path.exists():
        return torch.load(path)
    print(f"  embedding {dataset}/{split} with {encoder} (cached afterwards)")
    examples = load_split(dataset, split)
    progress_path = path.with_suffix(".embedding.partial.pt")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    feats = {"X": embed_texts([ex["text"] for ex in examples], encoder, max_length, batch_size,
                                progress_path=progress_path),
             "y": torch.tensor([ex["label"] for ex in examples])}
    if all(ex.get("concepts") is not None for ex in examples):
        feats["C"] = torch.tensor([ex["concepts"] for ex in examples], dtype=torch.float32)
    if all("concept_scores" in ex for ex in examples):
        feats["C_soft"] = torch.tensor([ex["concept_scores"] for ex in examples], dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats

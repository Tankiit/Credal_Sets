"""Extract frozen global image embeddings for the CIFAR-10 test set.

Backbone-specific loading and preprocessing live in ``backbones.py``. This script
only owns dataset iteration, ordering, caching, and metadata. Downstream analyses
continue to consume the same ``embeddings.npy`` and ``labels.npy`` files.

Examples
--------
Hugging Face DINOv3 (current default):
    python feature_extraction.py --backbone hf \
      --model facebook/dinov3-vitl16-pretrain-lvd1689m

Hugging Face DINOv2:
    python feature_extraction.py --backbone hf --model facebook/dinov2-base

Timm ResNet-50:
    python feature_extraction.py --backbone timm --model resnet50

Timm ConvNeXt-Tiny:
    python feature_extraction.py --backbone timm --model convnext_tiny

Order is preserved throughout: ``CIFAR10(train=False)`` is never shuffled, so
``embeddings.npy[i]`` remains aligned with ``labels.npy[i]`` and CIFAR-10H row i.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from backbones import build_backbone


def _pil_collate(batch):
    """Keep PIL images as a list; let the selected backbone own preprocessing."""
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images-dir",
        default="data/cifar10",
        help="where CIFAR-10 image files live",
    )
    ap.add_argument("--out-dir", default="features")
    ap.add_argument(
        "--backbone",
        choices=("hf", "timm"),
        default="hf",
        help="visual-backbone provider",
    )
    ap.add_argument(
        "--model",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HF model id or timm model name, depending on --backbone",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detected if unset")
    args = ap.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[device] {device}")
    print(f"[load] provider={args.backbone} model={args.model}")

    backbone = build_backbone(args.backbone, args.model, device)

    # Deliberately keep the raw PIL image here. Each backbone applies the
    # preprocessing associated with its own pretrained weights.
    ds = CIFAR10(root=args.images_dir, train=False, download=False, transform=None)
    assert len(ds) == 10000, f"expected 10000 test images, got {len(ds)}"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=_pil_collate,
    )

    all_embeddings = []
    all_labels = []

    t0 = time.time()
    for images, labels in tqdm(loader, desc="extracting"):
        embeddings = backbone.encode_pil(images)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Backbone must return (B, D) embeddings; got {tuple(embeddings.shape)}"
            )
        all_embeddings.append(embeddings.detach().float().cpu().numpy())
        all_labels.append(labels.numpy())

    elapsed = time.time() - t0
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    assert embeddings.shape[0] == 10000 and labels.shape[0] == 10000

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)

    info = backbone.info
    meta = {
        "backbone_provider": info.kind,
        "model": info.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "reported_feature_dim": info.feature_dim,
        "num_images": int(embeddings.shape[0]),
        "device": device,
        "batch_size": args.batch_size,
        "extraction_seconds": elapsed,
        "frozen": info.frozen,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] embeddings {embeddings.shape} -> {out_dir/'embeddings.npy'}")
    print(f"[done] labels {labels.shape} -> {out_dir/'labels.npy'}")
    print(
        f"[time] {elapsed:.1f}s for {embeddings.shape[0]} images "
        f"({elapsed / embeddings.shape[0] * 1000:.1f} ms/image)"
    )


if __name__ == "__main__":
    main()

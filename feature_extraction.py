"""Extract frozen global image embeddings for the CIFAR-10 test set.

This script delegates model loading to ``models/backbones.py`` and dataset loading
to ``dataloaders/`` so the same uncertainty / supervised-LVM pipeline can swap
visual encoders without changing downstream code. The extracted embeddings are
L2-normalized so cosine-space analyses stay well-behaved.

Examples:
    # DINOv2
    python feature_extraction.py --backbone hf --model facebook/dinov2-base

    # Existing DINOv3 setup
    python feature_extraction.py --backbone hf \
      --model facebook/dinov3-vitl16-pretrain-lvd1689m

    # timm backbones
    python feature_extraction.py --backbone timm --model resnet50
    python feature_extraction.py --backbone timm --model convnext_tiny

Order is preserved: ``CIFAR10(train=False)`` is never shuffled, so
``embeddings.npy[i]`` remains aligned with ``labels.npy[i]`` and CIFAR-10H row i.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataloaders import build_cifar10_test_loader
from models import build_backbone


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
        default="facebook/dinov2-base",
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

    loader = build_cifar10_test_loader(
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
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
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.append(labels.numpy())

    elapsed = time.time() - t0
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    assert embeddings.shape[0] == 10000 and labels.shape[0] == 10000
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

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
        "normalized": True,
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

"""
Extract frozen DINOv3 embeddings for the CIFAR-10 test set.

- Model is loaded frozen: eval() mode, all params requires_grad_(False),
  forward pass under torch.inference_mode(). No fine-tuning happens here.
- CIFAR-10's native 32x32 images are resized up to the processor's expected
  input size (224x224 for the standard DINOv3 configs) via the model's own
  AutoImageProcessor, so preprocessing (resize/crop/normalize) matches what
  the model was trained with.
- Saves the CLS/pooled token as the per-image global embedding. Optionally
  also saves the patch-token grid (register + patch tokens) for later dense
  inspection -- this is much bigger, off by default.

Order preserved throughout: torchvision.datasets.CIFAR10(train=False) with
no shuffling, iterated with a plain (non-shuffling) DataLoader, so
embeddings.npy[i] <-> data/cifar10h-probs.npy[i] <-> CIFAR10(train=False)[i].
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


def build_transform(processor):
    """Wrap the HF image processor as a torchvision-style per-PIL-image transform."""
    def _transform(img):
        # returns a dict with 'pixel_values': (1, C, H, W) -> squeeze batch dim
        out = processor(images=img, return_tensors="pt")
        return out["pixel_values"][0]
    return _transform


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images-dir", default="data/cifar10",
        help="where CIFAR-10 image files live (output of 01_download_data.py --images-dir)",
    )
    ap.add_argument("--out-dir", default="features")
    ap.add_argument(
        "--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HF checkpoint id. Gated -- accept the license on the model page "
             "and `huggingface-cli login` first.",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detected if unset")
    ap.add_argument(
        "--save-patch-tokens", action="store_true",
        help="also save the full patch/register token grid per image (large: "
             "10000 x num_tokens x hidden_dim). Off by default.",
    )
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[device] {device}")

    from transformers import AutoImageProcessor, AutoModel

    print(f"[load] {args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen: no gradients, no fine-tuning

    transform = build_transform(processor)
    ds = CIFAR10(root=args.images_dir, train=False, download=False, transform=transform)
    assert len(ds) == 10000, f"expected 10000 test images, got {len(ds)}"
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,  # never shuffle: order must match cifar10h-probs.npy
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    all_embeddings = []
    all_labels = []
    all_patch_tokens = [] if args.save_patch_tokens else None

    t0 = time.time()
    with torch.inference_mode():
        for pixel_values, labels in tqdm(loader, desc="extracting"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)

            # pooler_output is the CLS token passed through the model's
            # pooling head -- the standard global image embedding for DINOv3.
            cls = outputs.pooler_output
            all_embeddings.append(cls.float().cpu().numpy())
            all_labels.append(labels.numpy())

            if args.save_patch_tokens:
                # last_hidden_state: (B, 1 + num_register_tokens + num_patches, D)
                all_patch_tokens.append(outputs.last_hidden_state.float().cpu().numpy())

    elapsed = time.time() - t0
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    assert embeddings.shape[0] == 10000 and labels.shape[0] == 10000

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)

    meta = {
        "model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_images": int(embeddings.shape[0]),
        "device": device,
        "batch_size": args.batch_size,
        "extraction_seconds": elapsed,
        "frozen": True,
        "saved_patch_tokens": bool(args.save_patch_tokens),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_patch_tokens:
        patch_tokens = np.concatenate(all_patch_tokens, axis=0)
        np.save(out_dir / "patch_tokens.npy", patch_tokens)
        print(f"[saved] patch_tokens.npy {patch_tokens.shape}")

    print(f"[done] embeddings {embeddings.shape} -> {out_dir/'embeddings.npy'}")
    print(f"[done] labels {labels.shape} -> {out_dir/'labels.npy'}")
    print(f"[time] {elapsed:.1f}s for {embeddings.shape[0]} images "
          f"({elapsed/embeddings.shape[0]*1000:.1f} ms/image)")


if __name__ == "__main__":
    main()
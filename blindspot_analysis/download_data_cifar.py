"""
Build manifest.csv + classes.txt + soft_labels.csv for CIFAR-10 + CIFAR-10H,
in the same generic format 00_build_manifest_cub.py produces for CUB. This
is the "worked example" for the single-label / point-probability case; see
00_build_manifest_cub.py for the multi-label / credal-interval case.

Downloads CIFAR-10 test images via torchvision, then materializes each one
as an actual PNG file on disk (rather than keeping them in torchvision's
in-memory/batch format) so 01_extract_features.py can treat this dataset
exactly like any other manifest-described dataset, with no CIFAR-specific
code anywhere downstream.

Also fetches cifar10h-probs.npy (skipping the fetch if you already have it
locally, e.g. from `git clone https://github.com/jcpeterson/cifar-10h`) and
converts it from CIFAR-10H's positional format into an id-keyed
soft_labels.csv, since id-keyed joins are far less error-prone than
"trust that nothing reordered the rows" -- see common.py's module docstring.
"""
import argparse
import csv
import urllib.request
from pathlib import Path

import numpy as np

CIFAR10H_BASE = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def fetch_cifar10h_probs(cifar10h_dir: Path) -> Path:
    cifar10h_dir.mkdir(parents=True, exist_ok=True)
    probs_path = cifar10h_dir / "cifar10h-probs.npy"
    if probs_path.exists():
        print(f"[skip] {probs_path} already exists")
    else:
        url = f"{CIFAR10H_BASE}/cifar10h-probs.npy"
        print(f"[download] {url} -> {probs_path}")
        urllib.request.urlretrieve(url, probs_path)
    probs = np.load(probs_path)
    assert probs.shape == (10000, 10), f"unexpected probs shape {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
    return probs_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cifar10h-dir", default="data/cifar-10h/data",
        help="where cifar10h-probs.npy lives (or should be downloaded to)",
    )
    ap.add_argument(
        "--torchvision-cache", default="data/cifar10",
        help="scratch dir for torchvision's own CIFAR-10 download format",
    )
    ap.add_argument("--out-dir", default="dataset_cifar10h", help="manifest + images output dir")
    args = ap.parse_args()

    from torchvision.datasets import CIFAR10

    probs_path = fetch_cifar10h_probs(Path(args.cifar10h_dir))
    probs = np.load(probs_path)

    print(f"[download] CIFAR-10 test set -> {args.torchvision_cache}")
    ds = CIFAR10(root=args.torchvision_cache, train=False, download=True)
    assert len(ds) == 10000, f"expected 10000 test images, got {len(ds)}"
    if list(ds.classes) != CIFAR10_CLASSES:
        print(
            "[warn] torchvision class order differs from CIFAR-10H's documented order:\n"
            f"  torchvision: {ds.classes}\n  cifar10h:    {CIFAR10_CLASSES}"
        )

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    ids = []
    print(f"[write] {len(ds)} PNG files -> {images_dir}")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_path", "label"])
        for i in range(len(ds)):
            img, label = ds[i]
            image_id = f"{i:05d}"
            rel_path = f"images/{image_id}.png"
            img.save(out_dir / rel_path)
            w.writerow([image_id, rel_path, label])
            ids.append(image_id)
    print(f"[saved] {manifest_path}")

    classes_path = out_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write("\n".join(CIFAR10_CLASSES) + "\n")
    print(f"[saved] {classes_path}")

    soft_labels_path = out_dir / "soft_labels.csv"
    with open(soft_labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"prob_{c}" for c in range(10)])
        for image_id, row in zip(ids, probs):
            w.writerow([image_id] + [f"{v:.6f}" for v in row])
    print(f"[saved] {soft_labels_path}")


if __name__ == "__main__":
    main()
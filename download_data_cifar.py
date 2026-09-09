"""
Download CIFAR-10 (test split) and CIFAR-10H human soft labels.

Order guarantee: CIFAR-10H's cifar10h-probs.npy / cifar10h-counts.npy are
released in the same order as the *original, unshuffled* CIFAR-10 test set.
torchvision.datasets.CIFAR10(train=False) preserves that order by
construction (it just reads the batches file sequentially), so
probs[i] <-> dataset[i] as long as nothing shuffles the dataset. Every
downstream script in this project relies on that alignment.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np

CIFAR10H_BASE = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def download_cifar10h(cifar10h_dir: Path):
    """Fetch cifar10h-probs.npy / cifar10h-counts.npy into cifar10h_dir,
    unless they're already there (e.g. you cloned the cifar-10h repo
    yourself -- this looks for exactly cifar10h_dir/cifar10h-probs.npy)."""
    cifar10h_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["cifar10h-probs.npy", "cifar10h-counts.npy"]:
        out_path = cifar10h_dir / fname
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        url = f"{CIFAR10H_BASE}/{fname}"
        print(f"[download] {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)

    probs = np.load(cifar10h_dir / "cifar10h-probs.npy")
    counts = np.load(cifar10h_dir / "cifar10h-counts.npy")
    assert probs.shape == (10000, 10), f"unexpected probs shape {probs.shape}"
    assert counts.shape == (10000, 10), f"unexpected counts shape {counts.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), "probs rows should sum to 1"
    print(f"[ok] cifar10h-probs.npy {probs.shape}, cifar10h-counts.npy {counts.shape}")


def download_cifar10_test(images_dir: Path):
    # Imported lazily so `--only-cifar10h` doesn't require torchvision.
    from torchvision.datasets import CIFAR10

    print(f"[download] CIFAR-10 test set -> {images_dir}")
    ds = CIFAR10(root=str(images_dir), train=False, download=True)
    assert len(ds) == 10000, f"expected 10000 test images, got {len(ds)}"

    # Sanity check the class-name <-> index mapping matches CIFAR-10H's
    # documented mapping (airplane=0 ... truck=9). torchvision infers this
    # from batches.meta, but we check it explicitly since everything below
    # depends on it.
    if list(ds.classes) != CIFAR10_CLASSES:
        print(
            "[warn] torchvision class order differs from CIFAR-10H's documented "
            f"order.\n  torchvision: {ds.classes}\n  cifar10h:    {CIFAR10_CLASSES}\n"
            "  Labels will still be consistent internally (torchvision's own "
            "integer labels), just double check class-name printouts downstream."
        )
    print(f"[ok] CIFAR-10 test set: {len(ds)} images, classes={ds.classes}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cifar10h-dir", default="data/cifar-10h/data",
        help="where cifar10h-probs.npy / cifar10h-counts.npy live (or should "
             "be downloaded to). Defaults to the layout you get from `git clone "
             "https://github.com/jcpeterson/cifar-10h data/cifar-10h` -- i.e. "
             "the repo's own data/ subfolder, nested under a local data/ dir.",
    )
    ap.add_argument(
        "--images-dir", default="data/cifar10",
        help="where CIFAR-10 image files live (or should be downloaded to)",
    )
    ap.add_argument(
        "--only-cifar10h", action="store_true",
        help="skip the CIFAR-10 image download (e.g. if you already have it, "
             "or just want to inspect the human labels)",
    )
    args = ap.parse_args()

    cifar10h_dir = Path(args.cifar10h_dir)
    images_dir = Path(args.images_dir)

    download_cifar10h(cifar10h_dir)
    if not args.only_cifar10h:
        download_cifar10_test(images_dir)

    print("\nDone.")
    print("cifar10h files:", cifar10h_dir)
    for p in sorted(cifar10h_dir.glob("*")):
        if p.is_file():
            print(" ", p)
    if not args.only_cifar10h:
        print("CIFAR-10 images:", images_dir)
        for p in sorted(images_dir.rglob("*"))[:5]:
            if p.is_file():
                print(" ", p)


if __name__ == "__main__":
    main()
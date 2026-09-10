"""
Compare human perceptual uncertainty (CIFAR-10H) against frozen visual
embedding-space structure, with no classifier trained on top of the
features.

Two per-image signals:
  - human_entropy: Shannon entropy (bits) of the CIFAR-10H soft label.
    0 = every annotator agreed; log2(10)=3.32 = maximal disagreement.
  - knn_purity: fraction of an image's k nearest neighbors (cosine
    distance, in the selected backbone's embedding space) that share its ground-truth
    CIFAR-10 label. This needs no trained classifier -- it's a direct,
    unsupervised read of how locally separable the embedding space is
    around that image.

These are cross-tabulated (median-split each axis) into four quadrants;
the two "disagreement" quadrants are the blind spots:
  - model_blind_spot: humans confident (low entropy), but the backbone's local
    neighborhood around the image is label-impure (low purity) -- the
    model's frozen geometry doesn't cleanly separate an image humans find
    obvious.
  - human_blind_spot: the backbone's neighborhood is label-pure (high purity),
    but humans disagreed a lot (high entropy) -- the embedding space
    separates the class cleanly even though the image confused annotators.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def compute_knn_purity(embeddings: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """For each row i, fraction of its k nearest OTHER embeddings (cosine
    distance) that share labels[i]."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")  # +1 for self
    nn.fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    idx = idx[:, 1:]  # drop self (first neighbor is always the point itself)
    neighbor_labels = labels[idx]  # (N, k)
    purity = (neighbor_labels == labels[:, None]).mean(axis=1)
    return purity


def save_example_images(indices, out_dir: Path, images_dir: Path):
    """Best-effort: dump PNGs for the given CIFAR-10 test indices, if the
    dataset is available locally. Never fails the whole analysis if not."""
    try:
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=str(images_dir), train=False, download=False)
    except Exception as e:
        print(f"[warn] could not load CIFAR-10 images for visual export ({e}); "
              f"skipping image export, CSVs are still written.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        img, _ = ds[int(i)]
        img.save(out_dir / f"{int(i):05d}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", default="features")
    ap.add_argument(
        "--cifar10h-dir", default="data/cifar-10h/data",
        help="where cifar10h-probs.npy lives",
    )
    ap.add_argument(
        "--images-dir", default="data/cifar10",
        help="where CIFAR-10 image files live (only needed for --no-images off, "
             "i.e. exporting blind-spot PNGs)",
    )
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--k", type=int, default=10, help="neighbors for kNN purity")
    ap.add_argument("--top-n", type=int, default=30, help="images per blind-spot report")
    ap.add_argument(
        "--no-images", action="store_true",
        help="skip exporting PNGs of blind-spot images",
    )
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(features_dir / "embeddings.npy")
    labels = np.load(features_dir / "labels.npy")
    probs = np.load(Path(args.cifar10h_dir) / "cifar10h-probs.npy")
    assert embeddings.shape[0] == labels.shape[0] == probs.shape[0] == 10000, (
        "size mismatch between embeddings/labels/cifar10h-probs -- did you "
        "prepare CIFAR-10/CIFAR-10H and run feature_extraction.py without shuffling?"
    )

    n = embeddings.shape[0]

    # --- human side ---
    human_entropy = scipy_entropy(probs, base=2, axis=1)  # (N,)
    human_top_choice = probs.argmax(axis=1)
    human_agrees_with_truth = (human_top_choice == labels).astype(int)
    human_top_prob = probs.max(axis=1)

    # --- model side (frozen embedding geometry, no classifier) ---
    knn_purity = compute_knn_purity(embeddings, labels, k=args.k)

    # --- correlation ---
    rho, pval = spearmanr(human_entropy, knn_purity)
    with open(out_dir / "correlation.txt", "w") as f:
        f.write(
            "Spearman correlation between human label entropy and backbone "
            f"kNN (k={args.k}) embedding purity\n"
            f"rho = {rho:.4f}, p = {pval:.3e}, n = {n}\n\n"
            f"Interpretation: {'negative' if rho < 0 else 'positive'} correlation "
            f"means {'higher human disagreement tends to co-occur with less locally-pure backbone embeddings (model and humans tend to agree on what is hard)' if rho < 0 else 'higher human disagreement does NOT track with less pure embeddings -- the two notions of difficulty diverge'}.\n"
        )
    print(open(out_dir / "correlation.txt").read())

    # --- quadrants (rank-based median split) ---
    # knn_purity is discrete (only k+1 possible values) and CIFAR-10H's
    # entropy distribution is heavily skewed toward 0 (most images are easy
    # for humans), so a large fraction of images commonly tie exactly at the
    # raw median value. Splitting on value directly (e.g. `x > median(x)`)
    # can then leave an entire quadrant empty. Splitting on rank instead
    # (ordinal, ties broken by original index) guarantees a ~50/50 split on
    # both axes regardless of how many ties sit at the median.
    entropy_med = np.median(human_entropy)
    purity_med = np.median(knn_purity)
    entropy_rank = rankdata(human_entropy, method="ordinal") / n
    purity_rank = rankdata(knn_purity, method="ordinal") / n
    high_entropy = entropy_rank > 0.5
    high_purity = purity_rank > 0.5

    quadrant = np.empty(n, dtype=object)
    quadrant[~high_entropy & high_purity] = "easy_both_agree"
    quadrant[high_entropy & ~high_purity] = "hard_both_agree"
    quadrant[~high_entropy & ~high_purity] = "model_blind_spot"
    quadrant[high_entropy & high_purity] = "human_blind_spot"

    # --- per-image CSV ---
    csv_path = out_dir / "per_image_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "true_label", "true_class", "human_top_choice", "human_top_class",
            "human_agrees_with_truth", "human_top_prob", "human_entropy_bits",
            "knn_purity", "quadrant",
        ])
        for i in range(n):
            w.writerow([
                i, int(labels[i]), CIFAR10_CLASSES[labels[i]],
                int(human_top_choice[i]), CIFAR10_CLASSES[human_top_choice[i]],
                int(human_agrees_with_truth[i]), f"{human_top_prob[i]:.4f}",
                f"{human_entropy[i]:.4f}", f"{knn_purity[i]:.4f}", quadrant[i],
            ])
    print(f"[saved] {csv_path} ({n} rows)")

    quadrant_names = [
        "easy_both_agree", "hard_both_agree", "model_blind_spot", "human_blind_spot",
    ]

    # --- scatter plot ---
    try:
        import matplotlib.pyplot as plt
        colors = {
            "easy_both_agree": "#4c72b0",
            "hard_both_agree": "#55a868",
            "model_blind_spot": "#c44e52",
            "human_blind_spot": "#8172b2",
        }
        fig, ax = plt.subplots(figsize=(7, 6))
        for q, c in colors.items():
            mask = quadrant == q
            ax.scatter(
                human_entropy[mask], knn_purity[mask],
                s=6, alpha=0.5, c=c, label=f"{q} (n={mask.sum()})",
            )
        ax.axvline(entropy_med, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(purity_med, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Human label entropy (bits)")
        ax.set_ylabel(f"Backbone kNN (k={args.k}) label purity")
        ax.set_title("Human uncertainty vs. frozen backbone embedding purity")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / "quadrant_scatter.png", dpi=150)
        plt.close(fig)
        print(f"[saved] {out_dir/'quadrant_scatter.png'}")
    except ImportError:
        print("[warn] matplotlib not available, skipping scatter plot")

    # --- top blind-spot lists ---
    def write_blind_spot_csv(mask, sort_key, fname, image_subdir):
        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(sort_key[idxs])][: args.top_n]
        path = out_dir / fname
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "index", "true_class", "human_entropy_bits", "knn_purity",
                "human_top_class", "human_top_prob",
            ])
            for i in idxs:
                w.writerow([
                    int(i), CIFAR10_CLASSES[labels[i]], f"{human_entropy[i]:.4f}",
                    f"{knn_purity[i]:.4f}", CIFAR10_CLASSES[human_top_choice[i]],
                    f"{human_top_prob[i]:.4f}",
                ])
        print(f"[saved] {path} (top {len(idxs)})")
        if not args.no_images:
            save_example_images(idxs, out_dir / "blind_spot_images" / image_subdir, Path(args.images_dir))
        return idxs

    # Rank by "extremity" using both axes' ranks so ties on the (discrete)
    # purity metric are broken by how confident humans were, rather than by
    # arbitrary array order.
    # model_blind_spot: most locally-impure embeddings among the most
    # human-confident images -> low purity_rank + low entropy_rank first.
    model_mask = quadrant == "model_blind_spot"
    model_score = purity_rank + entropy_rank
    write_blind_spot_csv(model_mask, model_score, "blind_spots_model.csv", "model")

    # human_blind_spot: most human disagreement among the most locally-pure
    # embeddings -> high entropy_rank + high purity_rank first.
    human_mask = quadrant == "human_blind_spot"
    human_score = -(purity_rank + entropy_rank)
    write_blind_spot_csv(human_mask, human_score, "blind_spots_human.csv", "human")

    # --- summary ---
    summary = {
        "n_images": n,
        "k": args.k,
        "entropy_median": float(entropy_med),
        "purity_median": float(purity_med),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "quadrant_counts": {q: int((quadrant == q).sum()) for q in quadrant_names},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Compare human perceptual/annotation uncertainty against frozen DINOv3 embedding-space structure.
No classifier is trained anywhere -- both the human and model uncertainty signals are computed 
directly from raw annotations / raw embedding geometry.

Two modes, selected with --mode:

  single-label   (e.g. CIFAR-10H: one soft label distribution per image)
    - human_entropy: Shannon entropy (bits) of the soft label. 
      This is a TOTAL uncertainty measure -- a point probability vector can't be decomposed into 
      aleatoric vs epistemic parts (that needs a *set* of distributions, not one), so "entropy" 
      here is deliberately not called AU or EU.
    - model_knn_purity: fraction of an image's k nearest DINOv3-embedding neighbors that share its ground-truth
      label -- a local, aleatoric-flavored read of the embedding geometry (are nearby points even the same class?).
    
    Cross-tabulated (rank-based split) into 4 quadrants; the interesting two are model_blind_spot 
    (humans confident, model geometry locally confused) and human_blind_spot 
    (model geometry locally clean, humans disagreed).

  multi-label    (e.g. CUB attributes: many binary concepts per image, each
                  with a genuine probability INTERVAL from certainty-weighted
                  votes -- see 00_build_manifest_cub.py)
    - human_AU / human_EU / human_TU: per attribute, computed in closed form
      from the [lower, upper] interval (Huellermeier et al. 2022):
          AU = min(lower, 1-upper), EU = upper-lower, TU = min(1-lower, upper)
      then averaged across attributes into one scalar per image. EU here is
      real epistemic uncertainty (interval width = how much the credal set
      itself is uncertain), not just disagreement dressed up as epistemic.
    - model_knn_distance: mean cosine distance to an image's k nearest
      embedding-space neighbors -- density-based, epistemic-flavored (Deep
      Deterministic Uncertainty style): sparse regions are where a frozen
      deterministic model has the least basis for any prediction,
      independent of whether neighbors agree on a label.
    - model_knn_purity: same aleatoric-flavored local-label-agreement proxy
      as single-label mode, computed on the manifest's single-label ground
      truth (e.g. CUB species), reported alongside for context.
    The main comparison is human_EU vs. model_knn_distance -- two
    (different) epistemic-uncertainty proxies, one from data, one from
    model geometry. Cross-tabulated into 4 quadrants the same way.
"""
# Standard library module for building command-line argument parsers.
import argparse
# Standard library module for reading and writing CSV files.
import csv
# Standard library module for reading and writing JSON files.
import json
# Import Path for convenient, cross-platform filesystem path handling.
from pathlib import Path

# Import NumPy for array manipulation.
import numpy as np
# Import SciPy's rank utility and Spearman correlation test.
from scipy.stats import rankdata, spearmanr

# Import the shared helper functions used throughout this script.
from common import (
    binary_credal_metrics,
    class_name,
    knn_label_purity,
    knn_mean_distance,
    load_attribute_credal,
    load_soft_labels,
    read_classes,
    shannon_entropy_bits,
)


def save_example_images(indices_by_id, out_dir: Path, manifest_path, images_base_dir):
    """Export copies of the images for the given ids (as PNGs) into out_dir, for visual inspection of blind-spot examples."""
    # Try to import the dependencies and load the manifest, bailing out gracefully if anything fails.
    try:
        # Import pandas for reading the manifest.
        import pandas as pd
        # Import PIL's Image for opening/saving images.
        from PIL import Image
        # Read the manifest, forcing ids to strings, and index it by id.
        manifest = pd.read_csv(manifest_path, dtype={"id": str}).set_index("id")
    # Catch any error (missing file, missing dependency, etc.) during setup.
    except Exception as e:
        # Warn and skip image export rather than crashing the whole run.
        print(f"[warn] could not load manifest for image export ({e}); skipping")
        return
    # Create the output directory (and parents) if it doesn't already exist.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the optional base directory used for relative image paths.
    base = Path(images_base_dir) if images_base_dir else None
    # Iterate over each requested image id.
    for image_id in indices_by_id:
        # Skip ids that aren't present in the manifest.
        if image_id not in manifest.index:
            continue
        # Look up the raw image path for this id.
        rel = manifest.loc[image_id, "image_path"]
        # Resolve it to an absolute path, using base if the path is relative.
        path = Path(rel) if Path(rel).is_absolute() or base is None else base / rel
        # Try to open, convert, and save the image; warn on failure instead of crashing.
        try:
            Image.open(path).convert("RGB").save(out_dir / f"{image_id}.png")
        except Exception as e:
            print(f"[warn] could not export image {image_id} ({e})")


def rank_quadrants(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str):
    """Rank-based (not value-based) median split on both axes: robust to the
    heavy ties that discrete/skewed uncertainty metrics commonly produce
    (see README for why a plain `x > median(x)` split can degenerate)."""
    # Determine the number of points being ranked.
    n = len(x)
    # Convert x values to fractional ranks in [0, 1].
    x_rank = rankdata(x, method="ordinal") / n
    # Convert y values to fractional ranks in [0, 1].
    y_rank = rankdata(y, method="ordinal") / n
    # Flag points whose x rank is above the median.
    high_x = x_rank > 0.5
    # Flag points whose y rank is above the median.
    high_y = y_rank > 0.5
    # Allocate an empty object array to hold each point's quadrant label.
    quadrant = np.empty(n, dtype=object)
    # Label points with low x and high y.
    quadrant[~high_x & high_y] = f"low_{x_name}_high_{y_name}"
    # Label points with high x and low y.
    quadrant[high_x & ~high_y] = f"high_{x_name}_low_{y_name}"
    # Label points with low x and low y.
    quadrant[~high_x & ~high_y] = f"low_{x_name}_low_{y_name}"
    # Label points with high x and high y.
    quadrant[high_x & high_y] = f"high_{x_name}_high_{y_name}"
    # Return the quadrant labels along with both rank arrays.
    return quadrant, x_rank, y_rank


def write_scatter(path, x, y, quadrant, xlabel, ylabel, title):
    """Render and save a quadrant-colored scatter plot of x vs y, skipping gracefully if matplotlib isn't installed."""
    # Try to import matplotlib, since it's an optional plotting dependency.
    try:
        import matplotlib.pyplot as plt
    # Warn and skip plotting if matplotlib isn't available.
    except ImportError:
        print("[warn] matplotlib not available, skipping scatter plot")
        return
    # Create the figure and axes for the scatter plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    # Define a fixed 4-color palette, one color per quadrant.
    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    # Plot each quadrant's points in its own color with a labeled count.
    for c, q in zip(palette, sorted(set(quadrant))):
        # Build a boolean mask selecting points in this quadrant.
        mask = quadrant == q
        # Scatter this quadrant's points with the assigned color and legend label.
        ax.scatter(x[mask], y[mask], s=6, alpha=0.5, c=c, label=f"{q} (n={mask.sum()})")
    # Set the x-axis label.
    ax.set_xlabel(xlabel)
    # Set the y-axis label.
    ax.set_ylabel(ylabel)
    # Set the plot title.
    ax.set_title(title)
    # Add a legend with small font in the best available location.
    ax.legend(fontsize=8, loc="best")
    # Adjust subplot layout to avoid clipped labels.
    fig.tight_layout()
    # Save the figure to disk at the given path.
    fig.savefig(path, dpi=150)
    # Close the figure to free memory.
    plt.close(fig)
    # Print confirmation that the plot was saved.
    print(f"[saved] {path}")


def run_single_label(args, ids, embeddings, labels, classes):
    """Run the single-label analysis: compare human soft-label entropy against DINOv3 kNN label purity and write all reports."""
    # Load the soft labels and the subset of ids they're actually defined for.
    probs, aligned_ids = load_soft_labels(args.soft_labels, ids)
    # Build a lookup from id to its position in the original ids array.
    id_to_pos = {i: p for p, i in enumerate(ids)}
    # Compute the positions to keep, in aligned_ids order.
    keep = [id_to_pos[i] for i in aligned_ids]
    # Subset embeddings, labels, and ids down to only the aligned entries.
    embeddings, labels, ids = embeddings[keep], labels[keep], np.array(aligned_ids, dtype=object)
    # Record the number of images remaining after alignment.
    n = len(ids)

    # Compute the Shannon entropy (bits) of each image's soft label distribution.
    human_entropy = shannon_entropy_bits(probs, axis=1)
    # Compute each image's most-voted (top) class index.
    human_top_choice = probs.argmax(axis=1)
    # Compute the probability mass on that top class.
    human_top_prob = probs.max(axis=1)
    # Compute the kNN label-purity metric from the embeddings.
    model_purity = knn_label_purity(embeddings, labels, k=args.k)

    # Compute the Spearman correlation between human entropy and model purity.
    rho, pval = spearmanr(human_entropy, model_purity)
    # Split both metrics into rank-based quadrants for cross-tabulation.
    quadrant, ex_rank, ey_rank = rank_quadrants(human_entropy, model_purity, "entropy", "purity")

    # Wrap the output directory string in a Path object.
    out_dir = Path(args.out_dir)
    # Create the output directory (and parents) if it doesn't already exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open the correlation summary file for writing.
    with open(out_dir / "correlation.txt", "w") as f:
        # Write the Spearman correlation result to the file.
        f.write(
            f"Spearman(human_entropy, model_knn_purity): rho={rho:.4f}, p={pval:.3e}, n={n}\n"
        )
    # Print the contents of the correlation file to the console.
    print(open(out_dir / "correlation.txt").read())

    # Build the path for the per-image metrics CSV.
    csv_path = out_dir / "per_image_metrics.csv"
    # Open the per-image metrics CSV for writing.
    with open(csv_path, "w", newline="") as f:
        # Create a CSV writer bound to the open file.
        w = csv.writer(f)
        # Write the header row.
        w.writerow([
            "id", "true_label", "true_class", "human_top_choice", "human_top_class",
            "human_top_prob", "human_entropy_bits", "model_knn_purity", "quadrant",
        ])
        # Write one row per image with all the computed metrics.
        for i in range(n):
            w.writerow([
                ids[i], int(labels[i]), class_name(classes, labels[i]),
                int(human_top_choice[i]), class_name(classes, human_top_choice[i]),
                f"{human_top_prob[i]:.4f}", f"{human_entropy[i]:.4f}",
                f"{model_purity[i]:.4f}", quadrant[i],
            ])
    # Print confirmation that the CSV was saved, with row count.
    print(f"[saved] {csv_path} ({n} rows)")

    # Render and save the quadrant-colored scatter plot of entropy vs purity.
    write_scatter(
        out_dir / "quadrant_scatter.png", human_entropy, model_purity, quadrant,
        "Human label entropy (bits)", f"DINOv3 kNN (k={args.k}) label purity",
        "Human uncertainty vs. frozen DINOv3 embedding purity",
    )

    # Write the top model-blind-spot examples (humans confident, model locally confused).
    _write_blind_spots(
        out_dir, ids, quadrant, ex_rank, ey_rank,
        "low_entropy_low_purity", "model_blind_spot", args.top_n,
        extra_cols={"true_class": [class_name(classes, l) for l in labels],
                    "human_entropy_bits": human_entropy, "model_knn_purity": model_purity},
        manifest_path=args.manifest, images_base_dir=args.images_base_dir, no_images=args.no_images,
    )
    # Write the top human-blind-spot examples (model geometry clean, humans disagreed).
    _write_blind_spots(
        out_dir, ids, quadrant, ex_rank, ey_rank,
        "high_entropy_high_purity", "human_blind_spot", args.top_n,
        extra_cols={"true_class": [class_name(classes, l) for l in labels],
                    "human_entropy_bits": human_entropy, "model_knn_purity": model_purity},
        manifest_path=args.manifest, images_base_dir=args.images_base_dir, no_images=args.no_images,
    )

    # Assemble a summary dictionary describing this run.
    summary = {
        "mode": "single-label", "n_images": n, "k": args.k,
        "spearman_rho": float(rho), "spearman_p": float(pval),
        "quadrant_counts": {q: int((quadrant == q).sum()) for q in set(quadrant)},
    }
    # Open the summary.json file for writing.
    with open(out_dir / "summary.json", "w") as f:
        # Write the summary dictionary as indented JSON.
        json.dump(summary, f, indent=2)
    # Print the summary as formatted JSON to the console.
    print(json.dumps(summary, indent=2))


def run_multi_label(args, ids, embeddings, labels, classes):
    """Run the multi-label analysis: compare human attribute-credal epistemic uncertainty against DINOv3 kNN distance and write all reports."""
    # Read the attribute names, defaulting to an empty list if none given.
    attributes = read_classes(args.attributes) or []
    # Count how many attributes were loaded.
    num_attributes = len(attributes)
    # Abort if no attributes were found, since the analysis needs them.
    if num_attributes == 0:
        raise SystemExit(f"--attributes {args.attributes} is empty or missing")

    # Load the attribute credal lower/upper bounds aligned to the given ids.
    lower, upper, aligned_ids = load_attribute_credal(args.credal_csv, ids, num_attributes)
    # Build a lookup from id to its position in the original ids array.
    id_to_pos = {i: p for p, i in enumerate(ids)}
    # Compute the positions to keep, in aligned_ids order.
    keep = [id_to_pos[i] for i in aligned_ids]
    # Subset embeddings, labels, and ids down to only the aligned entries.
    embeddings, labels, ids = embeddings[keep], labels[keep], np.array(aligned_ids, dtype=object)
    # Record the number of images remaining after alignment.
    n = len(ids)

    # Compute per-attribute aleatoric, epistemic, and total uncertainty from the credal intervals.
    au, eu, tu = binary_credal_metrics(lower, upper)  # each (n, num_attributes)
    # Average aleatoric uncertainty across attributes into one scalar per image.
    human_au = np.nanmean(au, axis=1)
    # Average epistemic uncertainty across attributes into one scalar per image.
    human_eu = np.nanmean(eu, axis=1)
    # Average total uncertainty across attributes into one scalar per image.
    human_tu = np.nanmean(tu, axis=1)

    # Compute the density-based, epistemic-flavored kNN mean distance metric.
    model_distance = knn_mean_distance(embeddings, k=args.k)  # epistemic-flavored
    # Compute the aleatoric-flavored kNN label purity metric on the species label.
    model_purity = knn_label_purity(embeddings, labels, k=args.k)  # aleatoric-flavored, on species label

    # Compute the Spearman correlation between human EU and model distance.
    rho_eu, pval_eu = spearmanr(human_eu, model_distance)
    # Compute the Spearman correlation between human AU and one minus model purity.
    rho_au, pval_au = spearmanr(human_au, 1 - model_purity)
    # Split human EU and model distance into rank-based quadrants for cross-tabulation.
    quadrant, ex_rank, ey_rank = rank_quadrants(human_eu, model_distance, "humanEU", "modelDist")

    # Wrap the output directory string in a Path object.
    out_dir = Path(args.out_dir)
    # Create the output directory (and parents) if it doesn't already exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open the correlation summary file for writing.
    with open(out_dir / "correlation.txt", "w") as f:
        # Write both Spearman correlation results, with explanatory context, to the file.
        f.write(
            "Spearman(human_EU, model_knn_distance) -- two epistemic-uncertainty "
            f"proxies, one from annotation intervals, one from embedding density:\n"
            f"  rho={rho_eu:.4f}, p={pval_eu:.3e}, n={n}\n\n"
            "Spearman(human_AU, 1 - model_knn_purity) -- aleatoric-flavored "
            f"comparison, for context:\n  rho={rho_au:.4f}, p={pval_au:.3e}, n={n}\n"
        )
    # Print the contents of the correlation file to the console.
    print(open(out_dir / "correlation.txt").read())

    # per-attribute top-EU names, for interpretability of blind-spot examples
    # Find, per image, the indices of the top-EU attributes (NaNs treated as lowest).
    top_attr_idx = np.argsort(-np.nan_to_num(eu, nan=-1.0), axis=1)[:, : args.top_attrs]

    # Build the path for the per-image metrics CSV.
    csv_path = out_dir / "per_image_metrics.csv"
    # Open the per-image metrics CSV for writing.
    with open(csv_path, "w", newline="") as f:
        # Create a CSV writer bound to the open file.
        w = csv.writer(f)
        # Write the header row, including dynamic top-attribute columns.
        w.writerow([
            "id", "true_label", "true_class", "human_AU", "human_EU", "human_TU",
            "model_knn_distance", "model_knn_purity", "quadrant",
            *(f"top_eu_attr_{j}" for j in range(args.top_attrs)),
        ])
        # Write one row per image with all the computed metrics.
        for i in range(n):
            # Look up the human-readable names of this image's top-EU attributes.
            top_names = [attributes[a] for a in top_attr_idx[i]]
            # Write the full row for this image.
            w.writerow([
                ids[i], int(labels[i]), class_name(classes, labels[i]),
                f"{human_au[i]:.4f}", f"{human_eu[i]:.4f}", f"{human_tu[i]:.4f}",
                f"{model_distance[i]:.4f}", f"{model_purity[i]:.4f}", quadrant[i],
                *top_names,
            ])
    # Print confirmation that the CSV was saved, with row count.
    print(f"[saved] {csv_path} ({n} rows)")

    # Render and save the quadrant-colored scatter plot of human EU vs model distance.
    write_scatter(
        out_dir / "quadrant_scatter.png", human_eu, model_distance, quadrant,
        "Human epistemic uncertainty (mean interval width across attributes)",
        f"DINOv3 kNN (k={args.k}) mean neighbor distance",
        "Human annotation-interval EU vs. frozen DINOv3 embedding density",
    )

    # Write the top model-epistemic-blind-spot examples (low human EU, high model distance).
    _write_blind_spots(
        out_dir, ids, quadrant, ex_rank, ey_rank,
        "low_humanEU_high_modelDist", "model_epistemic_blind_spot", args.top_n,
        extra_cols={"true_class": [class_name(classes, l) for l in labels],
                    "human_EU": human_eu, "model_knn_distance": model_distance},
        manifest_path=args.manifest, images_base_dir=args.images_base_dir, no_images=args.no_images,
    )
    # Write the top human-epistemic-blind-spot examples (high human EU, low model distance).
    _write_blind_spots(
        out_dir, ids, quadrant, ex_rank, ey_rank,
        "high_humanEU_low_modelDist", "human_epistemic_blind_spot", args.top_n,
        extra_cols={"true_class": [class_name(classes, l) for l in labels],
                    "human_EU": human_eu, "model_knn_distance": model_distance},
        manifest_path=args.manifest, images_base_dir=args.images_base_dir, no_images=args.no_images,
    )

    # Assemble a summary dictionary describing this run.
    summary = {
        "mode": "multi-label", "n_images": n, "num_attributes": num_attributes, "k": args.k,
        "spearman_rho_EU_vs_modelDistance": float(rho_eu), "spearman_p_EU": float(pval_eu),
        "spearman_rho_AU_vs_1minusModelPurity": float(rho_au), "spearman_p_AU": float(pval_au),
        "quadrant_counts": {q: int((quadrant == q).sum()) for q in set(quadrant)},
    }
    # Open the summary.json file for writing.
    with open(out_dir / "summary.json", "w") as f:
        # Write the summary dictionary as indented JSON.
        json.dump(summary, f, indent=2)
    # Print the summary as formatted JSON to the console.
    print(json.dumps(summary, indent=2))


def _write_blind_spots(out_dir, ids, quadrant, x_rank, y_rank, quadrant_key, out_name, top_n,
                        extra_cols, manifest_path, images_base_dir, no_images):
    """Select the most extreme examples in one quadrant, write them (with extra columns) to a CSV, and optionally export their images."""
    # Build a boolean mask selecting points in the requested quadrant.
    mask = quadrant == quadrant_key
    # Get the integer indices of the points in that quadrant.
    idxs = np.where(mask)[0]
    # most extreme first: lowest x_rank + highest y_rank (or vice versa,
    # whichever this quadrant represents) -- rank sum captures "how deep
    # into this quadrant" regardless of which quadrant it is.
    # Compute a signed depth-into-quadrant score for the x-rank component.
    score = np.where(x_rank[idxs] < 0.5, -x_rank[idxs], x_rank[idxs]) + \
            np.where(y_rank[idxs] < 0.5, -y_rank[idxs], y_rank[idxs])
    # Sort indices by descending absolute score, i.e. most extreme first.
    order = np.argsort(-np.abs(score))
    # Keep only the top_n most extreme indices.
    idxs = idxs[order][:top_n]

    # Build the output path for this blind-spot CSV.
    path = out_dir / f"blind_spots_{out_name}.csv"
    # Open the blind-spot CSV for writing.
    with open(path, "w", newline="") as f:
        # Create a CSV writer bound to the open file.
        w = csv.writer(f)
        # Build the header from "id" plus the extra column names.
        cols = ["id"] + list(extra_cols.keys())
        # Write the header row.
        w.writerow(cols)
        # Write one row per selected blind-spot example.
        for i in idxs:
            # Build the row, formatting float array values to 4 decimal places.
            row = [ids[i]] + [
                (f"{v[i]:.4f}" if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v[i])
                for v in extra_cols.values()
            ]
            # Write the row to the CSV.
            w.writerow(row)
    # Print confirmation that the CSV was saved, with example count.
    print(f"[saved] {path} (top {len(idxs)})")

    # Export the corresponding images unless the caller opted out.
    if not no_images:
        save_example_images(
            [ids[i] for i in idxs], out_dir / "blind_spot_images" / out_name,
            manifest_path, images_base_dir,
        )


def main():
    """Parse CLI arguments, load extracted features, and dispatch to the single-label or multi-label blind-spot analysis."""
    # Create the argument parser, using the module docstring as its description.
    ap = argparse.ArgumentParser(description=__doc__)
    # Add the directory containing the extracted feature files.
    ap.add_argument("--features-dir", default="features")
    # Add the manifest path, used for image export and true-class names.
    ap.add_argument("--manifest", default="dataset/manifest.csv",
                     help="used for image export and true-class names; not re-validated against features")
    # Add an optional base directory for resolving relative image paths.
    ap.add_argument("--images-base-dir", default=None)
    # Add an optional classes.txt path for human-readable class names.
    ap.add_argument("--classes", default=None, help="classes.txt (species/category names); optional")
    # Add the output directory where results will be written.
    ap.add_argument("--out-dir", default="results")
    # Add the number of neighbors to use for the kNN metrics.
    ap.add_argument("--k", type=int, default=10, help="neighbors for kNN metrics")
    # Add the number of images to include per blind-spot report.
    ap.add_argument("--top-n", type=int, default=30, help="images per blind-spot report")
    # Add a flag to skip exporting blind-spot PNG images.
    ap.add_argument("--no-images", action="store_true", help="skip exporting blind-spot PNGs")

    # Add the required mode argument selecting single-label or multi-label analysis.
    ap.add_argument("--mode", required=True, choices=["single-label", "multi-label"])
    # Add the optional soft-labels path, required only in single-label mode.
    ap.add_argument("--soft-labels", help="[single-label] soft_labels.csv or .npy")
    # Add the optional credal CSV path, required only in multi-label mode.
    ap.add_argument("--credal-csv", help="[multi-label] attribute_credal.csv")
    # Add the optional attributes.txt path, required only in multi-label mode.
    ap.add_argument("--attributes", help="[multi-label] attributes.txt")
    # Add the number of top-EU attribute names to report per blind-spot image.
    ap.add_argument("--top-attrs", type=int, default=3,
                     help="[multi-label] highest-EU attribute names to report per blind-spot image")
    # Parse the command-line arguments into the args namespace.
    args = ap.parse_args()

    # Wrap the features directory string in a Path object.
    features_dir = Path(args.features_dir)
    # Load the previously extracted embeddings array.
    embeddings = np.load(features_dir / "embeddings.npy")
    # Load the previously extracted labels array.
    labels = np.load(features_dir / "labels.npy")
    # Load the previously extracted ids array, allowing pickled object dtype.
    ids = np.load(features_dir / "ids.npy", allow_pickle=True)
    # Load class names if a classes file was provided, otherwise leave as None.
    classes = read_classes(args.classes) if args.classes else None

    # Dispatch to the single-label analysis if that mode was selected.
    if args.mode == "single-label":
        # Require --soft-labels to be set for single-label mode.
        if not args.soft_labels:
            raise SystemExit("--soft-labels is required for --mode single-label")
        # Run the single-label analysis.
        run_single_label(args, ids, embeddings, labels, classes)
    # Otherwise dispatch to the multi-label analysis.
    else:
        # Require both --credal-csv and --attributes to be set for multi-label mode.
        if not (args.credal_csv and args.attributes):
            raise SystemExit("--credal-csv and --attributes are required for --mode multi-label")
        # Run the multi-label analysis.
        run_multi_label(args, ids, embeddings, labels, classes)


# Only run main() when this script is executed directly, not when imported.
if __name__ == "__main__":
    # Call the main entry point of the script.
    main()
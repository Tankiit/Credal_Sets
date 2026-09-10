"""
Shared utilities for the dataset-agnostic pipeline.

Everything downstream (feature extraction, blind-spot analysis) operates on
one common contract:

  manifest.csv   columns: id, image_path, label
    - id:         a unique string per image (e.g. filename stem). This is
                  the join key used everywhere -- NOT row position -- so
                  reordering, filtering, or resuming never silently misaligns
                  data the way relying on row order would.
    - image_path: path to the image file, absolute or relative to wherever
                  you run the scripts from.
    - label:      integer ground-truth class index (0-indexed, matching the
                  order in classes.txt). May be empty/missing if you don't
                  have ground-truth labels for some or all images -- feature
                  extraction doesn't need it; the blind-spot analysis's kNN
                  purity metric does.

  classes.txt    one class name per line, line i = class index i.
                  Optional -- only used for human-readable output.

  soft_labels.{npy,csv}   per-image annotation distribution (e.g. human
                  vote probabilities/counts across classes).
    - .npy: shape (N, K), row order must exactly match manifest.csv's row
      order (this is how CIFAR-10H ships its labels -- positional, no ids).
    - .csv: columns "id" + "prob_0","prob_1",...,"prob_{K-1}" (or
      "count_0",...,"count_{K-1}", which get L1-normalized into
      probabilities). Joined on "id", so row order doesn't matter and
      missing ids are just dropped with a warning -- use this format for
      your own datasets, it's much less error-prone than the positional
      .npy convention.

Bring-your-own-dataset: build a manifest.csv (see 00_build_manifest_imagefolder.py
for the common case of one subfolder per class) and, if you have
multi-annotator/soft labels, a soft_labels.csv in the format above. Nothing
else in the pipeline is dataset-specific.
"""
# Import Path for convenient, cross-platform filesystem path handling.
from pathlib import Path
# Import typing helpers used in function signatures.
from typing import List, Optional, Sequence

# Import NumPy for array manipulation.
import numpy as np
# Import pandas for reading and manipulating manifest/label tables.
import pandas as pd
# Import PIL's Image for opening image files.
from PIL import Image
# torch is only needed by ManifestImageDataset (used for feature extraction);
# importing it lazily keeps the manifest-building and analysis scripts usable
# in environments without a torch install at all.
# Try to import torch's Dataset base class.
try:
    from torch.utils.data import Dataset
# Fall back to plain object if torch isn't installed.
except Exception:
    Dataset = object


def read_manifest(path) -> pd.DataFrame:
    """Read and validate a manifest.csv, enforcing required columns, unique ids, and a clean integer label column."""
    # Read the CSV, forcing the id column to be read as strings.
    df = pd.read_csv(path, dtype={"id": str})
    # Define the set of columns that must be present.
    required = {"id", "image_path"}
    # Compute which required columns are missing from the dataframe.
    missing = required - set(df.columns)
    # Raise an error if any required column is missing.
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    # Check whether any id value is duplicated.
    if df["id"].duplicated().any():
        # Collect up to five example duplicate ids for the error message.
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        # Raise an error reporting the duplicate ids.
        raise ValueError(f"{path} has duplicate id values, e.g. {dupes}")
    # If there's no label column, add one filled with a sentinel value.
    if "label" not in df.columns:
        df["label"] = -1
    # Fill any missing label values with -1 and cast the column to int.
    df["label"] = df["label"].fillna(-1).astype(int)
    # Return the validated, cleaned dataframe.
    return df


def read_classes(path) -> Optional[List[str]]:
    """Read classes.txt into a list of class names, or return None if the file doesn't exist."""
    # Wrap the path string in a Path object.
    path = Path(path)
    # Return None if the classes file doesn't exist.
    if not path.exists():
        return None
    # Open the classes file for reading.
    with open(path) as f:
        # Return a list of non-empty, stripped lines from the file.
        return [line.strip() for line in f if line.strip()]


def class_name(classes: Optional[List[str]], label: int) -> str:
    """Look up the human-readable class name for a label index, falling back to the stringified index."""
    # If a class list is available and the label is a valid index into it, use the name.
    if classes is not None and 0 <= label < len(classes):
        return classes[label]
    # Otherwise fall back to the label's string representation.
    return str(label)


class ManifestImageDataset(Dataset):
    """Loads images listed in a manifest.csv. Order = manifest row order;
    labels come along for the ride but aren't needed for feature extraction."""

    def __init__(self, manifest: pd.DataFrame, transform=None, base_dir: Optional[Path] = None):
        """Store the manifest (with a fresh index), optional transform, and optional base directory for resolving relative paths."""
        # Store the manifest with a freshly reset, contiguous integer index.
        self.manifest = manifest.reset_index(drop=True)
        # Store the optional image transform to apply on load.
        self.transform = transform
        # Store the base directory as a Path, or None if not provided.
        self.base_dir = Path(base_dir) if base_dir is not None else None

    def __len__(self):
        """Return the number of images in the manifest."""
        # Return the number of rows in the manifest.
        return len(self.manifest)

    def _resolve(self, image_path: str) -> Path:
        """Resolve an image_path from the manifest to an actual filesystem path, honoring base_dir for relative paths."""
        # Wrap the raw image path string in a Path object.
        p = Path(image_path)
        # If the path is already absolute, or there's no base directory, use it as-is.
        if p.is_absolute() or self.base_dir is None:
            return p
        # Otherwise join it onto the base directory.
        return self.base_dir / p

    def __getitem__(self, idx):
        """Load and return the (image, id, label) triple for the given manifest row index."""
        # Fetch the manifest row at the requested index.
        row = self.manifest.iloc[idx]
        # Resolve the row's image path to an actual filesystem path.
        path = self._resolve(row["image_path"])
        # Open the image file and convert it to RGB.
        img = Image.open(path).convert("RGB")
        # Apply the transform if one was provided.
        if self.transform is not None:
            img = self.transform(img)
        # Return the image, its id, and its label as an int.
        return img, row["id"], int(row["label"])


def load_soft_labels(path, ids: Sequence[str], num_classes: Optional[int] = None):
    """Return (probs, aligned_ids): probs is (len(ids), K) in the exact
    order of `ids`, dropping ids that have no soft label (with a warning) so
    downstream code never has to guess at alignment."""
    # Wrap the path string in a Path object.
    path = Path(path)
    # Materialize the ids sequence into a concrete list.
    ids = list(ids)

    # Handle the positional .npy soft-label format.
    if path.suffix == ".npy":
        # Load the array of soft labels from disk.
        arr = np.load(path)
        # Verify the array's row count matches the number of ids.
        if arr.shape[0] != len(ids):
            # Raise an error explaining the positional-alignment mismatch.
            raise ValueError(
                f"{path} has {arr.shape[0]} rows but manifest has {len(ids)} ids. "
                ".npy soft labels are positional (row i = manifest row i) -- if "
                "these counts don't match, the file wasn't built from this exact "
                "manifest. Use a .csv keyed by id instead if order isn't guaranteed."
            )
        # Return the array as floats along with the unchanged ids list.
        return arr.astype(float), ids

    # Handle the id-keyed .csv soft-label format.
    if path.suffix == ".csv":
        # Read the CSV, forcing ids to strings, and index it by id.
        df = pd.read_csv(path, dtype={"id": str}).set_index("id")
        # Check for duplicate ids in the index.
        if df.index.duplicated().any():
            # Raise an error if duplicates were found.
            raise ValueError(f"{path} has duplicate id values in its 'id' column")
        # Collect any probability columns present.
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        # Collect any count columns present.
        count_cols = [c for c in df.columns if c.startswith("count_")]
        # If probability columns exist, sort them numerically and use them directly.
        if prob_cols:
            cols = sorted(prob_cols, key=lambda c: int(c.split("_")[1]))
            values = df[cols].values.astype(float)
        # Otherwise, if count columns exist, sort and L1-normalize them into probabilities.
        elif count_cols:
            # Sort the count columns numerically.
            cols = sorted(count_cols, key=lambda c: int(c.split("_")[1]))
            # Extract the count values as floats.
            values = df[cols].values.astype(float)
            # Compute the per-row sum of counts, keeping dimensions for broadcasting.
            row_sums = values.sum(axis=1, keepdims=True)
            # Avoid division by zero by treating zero-sum rows as summing to one.
            row_sums[row_sums == 0] = 1.0
            # Normalize the counts into probabilities.
            values = values / row_sums
        # If neither prob_ nor count_ columns exist, the file is invalid.
        else:
            raise ValueError(f"{path} needs prob_0..prob_K-1 or count_0..count_K-1 columns")
        # If an expected class count was given, verify it matches the data.
        if num_classes is not None and values.shape[1] != num_classes:
            # Raise an error describing the mismatch.
            raise ValueError(
                f"{path} has {values.shape[1]} class columns but expected {num_classes} "
                "(from classes.txt / manifest labels)"
            )

        # Determine which requested ids have no matching row in the soft-label file.
        missing = [i for i in ids if i not in df.index]
        # Warn about any missing ids before dropping them.
        if missing:
            print(
                f"[warn] {len(missing)}/{len(ids)} ids have no soft label in {path} "
                f"(e.g. {missing[:5]}) -- dropping them from the aligned output"
            )
        # Keep only the ids that do have a soft label.
        aligned_ids = [i for i in ids if i in df.index]
        # Stack the soft-label rows in exactly the aligned id order.
        aligned = np.stack([values[df.index.get_loc(i)] for i in aligned_ids])
        # Return the aligned probability array and the corresponding ids.
        return aligned, aligned_ids

    # Raise an error for any file extension that isn't .npy or .csv.
    raise ValueError(f"unsupported soft-label file type: {path.suffix}")


# ---------------------------------------------------------------------------
# Credal-set / interval-uncertainty utilities (for multi-label / multi-annotator
# data with an explicit lower/upper bound per concept, e.g. CUB attributes
# with certainty-weighted votes -- see 00_build_manifest_cub.py).
# ---------------------------------------------------------------------------

def shannon_entropy_bits(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits, robust to exact-zero probabilities."""
    # Clip probabilities away from exact zero to avoid log(0).
    p = np.clip(probs, 1e-12, 1.0)
    # Compute and return the entropy in bits along the given axis.
    return -(p * np.log2(p)).sum(axis=axis)


def binary_credal_metrics(lower: np.ndarray, upper: np.ndarray):
    """Closed-form aleatoric/epistemic/total uncertainty for a *binary*
    credal set given as a probability interval [lower, upper] on
    P(attribute present), per Huellermeier et al. 2022 (used e.g. in
    CreINNs, arXiv:2401.05043 eq. 11):
        AU = min(lower, 1 - upper)
        EU = upper - lower
        TU = min(1 - lower, upper)
    NaN entries (no evidence at all) pass through as NaN in all three.
    """
    # Convert the lower bound input to a float array.
    lower = np.asarray(lower, dtype=float)
    # Convert the upper bound input to a float array.
    upper = np.asarray(upper, dtype=float)
    # Compute the aleatoric uncertainty component.
    au = np.minimum(lower, 1 - upper)
    # Compute the epistemic uncertainty component as the interval width.
    eu = upper - lower
    # Compute the total uncertainty component.
    tu = np.minimum(1 - lower, upper)
    # Return the three uncertainty arrays.
    return au, eu, tu


def load_attribute_credal(path, ids: Sequence[str], num_attributes: int):
    """Load a wide attribute-credal CSV (id, lower_0..lower_{A-1},
    upper_0..upper_{A-1}) and align to `ids`, same semantics as
    load_soft_labels: joined on id, missing ids dropped with a warning."""
    # Wrap the path string in a Path object.
    path = Path(path)
    # Materialize the ids sequence into a concrete list.
    ids = list(ids)
    # Read the CSV, forcing ids to strings, and index it by id.
    df = pd.read_csv(path, dtype={"id": str}).set_index("id")
    # Check for duplicate ids in the index.
    if df.index.duplicated().any():
        # Raise an error if duplicates were found.
        raise ValueError(f"{path} has duplicate id values")

    # Collect and numerically sort the lower-bound columns.
    lower_cols = sorted(
        [c for c in df.columns if c.startswith("lower_")],
        key=lambda c: int(c.split("_")[1]),
    )
    # Collect and numerically sort the upper-bound columns.
    upper_cols = sorted(
        [c for c in df.columns if c.startswith("upper_")],
        key=lambda c: int(c.split("_")[1]),
    )
    # Verify the number of lower/upper columns matches the expected attribute count.
    if len(lower_cols) != num_attributes or len(upper_cols) != num_attributes:
        # Raise an error describing the mismatch.
        raise ValueError(
            f"{path} has {len(lower_cols)} lower_/{len(upper_cols)} upper_ columns, "
            f"expected {num_attributes}"
        )

    # Determine which requested ids have no matching row in the credal file.
    missing = [i for i in ids if i not in df.index]
    # Warn about any missing ids before dropping them.
    if missing:
        print(
            f"[warn] {len(missing)}/{len(ids)} ids have no attribute credal data in "
            f"{path} (e.g. {missing[:5]}) -- dropping them from the aligned output"
        )
    # Keep only the ids that do have credal data.
    aligned_ids = [i for i in ids if i in df.index]
    # Extract the lower-bound values for the aligned ids.
    lower = df.loc[aligned_ids, lower_cols].values.astype(float)
    # Extract the upper-bound values for the aligned ids.
    upper = df.loc[aligned_ids, upper_cols].values.astype(float)
    # Return the lower bounds, upper bounds, and the aligned id list.
    return lower, upper, aligned_ids


def knn_label_purity(embeddings: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Fraction of each point's k nearest OTHER embeddings (cosine distance)
    that share its label. A local, label-aware, aleatoric-flavored proxy:
    low purity = locally-confused region of the embedding space -- but says
    nothing about whether the model has ever seen enough data near this
    point, which is what knn_mean_distance below is for."""
    # Import scikit-learn's nearest-neighbors class lazily.
    from sklearn.neighbors import NearestNeighbors

    # Build a nearest-neighbors index using cosine distance, fetching k+1 neighbors (to exclude the point itself).
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    # Fit the nearest-neighbors index on the embeddings.
    nn.fit(embeddings)
    # Query the index to get distances and indices of neighbors for every point.
    dist, idx = nn.kneighbors(embeddings)
    # Drop the first neighbor column, which is each point itself.
    idx = idx[:, 1:]
    # Look up the labels of each point's neighbors.
    neighbor_labels = labels[idx]
    # Compute, per point, the fraction of neighbors sharing its own label.
    purity = (neighbor_labels == labels[:, None]).mean(axis=1)
    # Return the purity scores.
    return purity


def knn_mean_distance(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to each point's k nearest OTHER embeddings,
    label-agnostic. A density-based, epistemic-flavored proxy in the style
    of Deep Deterministic Uncertainty / Mahalanobis-OOD: points in a sparse
    region of embedding space (far from everything else) are exactly where
    a frozen, deterministic model has the least basis for any prediction --
    independent of whether nearby points happen to agree on a label."""
    # Import scikit-learn's nearest-neighbors class lazily.
    from sklearn.neighbors import NearestNeighbors

    # Build a nearest-neighbors index using cosine distance, fetching k+1 neighbors (to exclude the point itself).
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    # Fit the nearest-neighbors index on the embeddings.
    nn.fit(embeddings)
    # Query the index to get distances to each point's neighbors.
    dist, _ = nn.kneighbors(embeddings)
    # Return the mean distance to the k nearest neighbors, excluding the point itself.
    return dist[:, 1:].mean(axis=1)
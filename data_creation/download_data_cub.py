"""
Build manifest.csv + classes.txt + attributes.txt + attribute_credal.csv
from a raw CUB-200-2011 download.

Expects the standard extracted layout:
    CUB_200_2011/
        images/<species_folder>/<image>.jpg
        images.txt                       image_id  path
        image_class_labels.txt           image_id  class_id (1-indexed)
        classes.txt                      class_id  class_name (1-indexed)
        attributes.txt                   attribute_id  attribute_name (1-indexed;
                                          this file sometimes ships separately from
                                          the main archive -- if missing, attributes
                                          are just named "0".."311")
        attributes/image_attribute_labels.txt
                                          image_id attribute_id is_present certainty_id time
        attributes/certainties.txt       certainty_id  certainty_name

CUB gives you something CIFAR-10H doesn't: multiple crowd workers vote on
EACH attribute, AND report how sure they were (certainty_id: 1=not visible,
2=guessing, 3=probably, 4=definitely). That extra certainty axis is what
lets us build an actual probability *interval* per (image, attribute) --
i.e. a genuine credal set -- rather than just a single averaged point
probability like CIFAR-10H's cifar10h-probs.npy.

How the interval is built (Imprecise Dirichlet Model, Walley 1996):
  1. Each worker vote is weighted by how certain they were:
         definitely -> 1.0, probably -> 0.67, guessing -> 0.33,
         not visible -> excluded entirely (treated as "no observation",
         not as a vote for absence).
  2. n_present = sum of weights over votes where is_present=1
     n_absent  = sum of weights over votes where is_present=0
     n = n_present + n_absent
  3. lower = n_present / (n + s),  upper = (n_present + s) / (n + s)
     where s (--idm-s, default 1.0) is the IDM's prior "imprecision"
     hyperparameter -- larger s means less evidence collapses a wider
     interval, i.e. more residual epistemic uncertainty for the same
     number of votes.
  4. If n == 0 (every worker said "not visible", or the pair is simply
     absent from image_attribute_labels.txt) there is NO evidence at all,
     which is exactly maximal epistemic uncertainty: we set [lower, upper]
     = [0, 1], the full interval.

This weighting scheme is a reasonable, documented modeling choice, not a
CUB-provided ground truth -- if you want a different certainty -> weight
mapping, edit CERTAINTY_WEIGHT below (or --idm-s) and rerun; nothing else
downstream needs to change since it just reads the resulting CSV.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

CERTAINTY_WEIGHT = {
    "definitely": 1.0,
    "probably": 0.67,
    "guessing": 0.33,
    "not visible": 0.0,  # excluded from n entirely, see module docstring
}
# fallback if certainties.txt isn't present: CUB's certainty_id is always
# 1=not visible, 2=guessing, 3=probably, 4=definitely
CERTAINTY_ID_WEIGHT = {1: 0.0, 2: 0.33, 3: 0.67, 4: 1.0}


def read_id_name_table(path):
    """Parses CUB's ubiquitous `<int_id> <name>` text format (1-indexed ids)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(maxsplit=1)
            out[int(idx)] = name.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cub-dir", required=True, help="path to extracted CUB_200_2011/")
    ap.add_argument("--out-dir", default="dataset_cub")
    ap.add_argument(
        "--idm-s", type=float, default=1.0,
        help="Imprecise Dirichlet Model prior strength: bigger = wider "
             "intervals (more epistemic uncertainty) for the same evidence",
    )
    args = ap.parse_args()

    cub_dir = Path(args.cub_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- images.txt: image_id -> relative path ---
    images_path = cub_dir / "images.txt"
    image_paths = {}
    with open(images_path) as f:
        for line in f:
            idx, rel_path = line.strip().split(maxsplit=1)
            image_paths[int(idx)] = rel_path

    # --- image_class_labels.txt: image_id -> class_id (1-indexed) ---
    labels_1idx = {}
    with open(cub_dir / "image_class_labels.txt") as f:
        for line in f:
            idx, class_id = line.strip().split()
            labels_1idx[int(idx)] = int(class_id)

    # --- classes.txt: class_id -> species name (1-indexed) ---
    class_names = read_id_name_table(cub_dir / "classes.txt")
    num_classes = len(class_names)

    # --- attributes.txt: attribute_id -> name (1-indexed). Optional file;
    # some CUB mirrors ship it outside the main archive. ---
    attr_names_path = cub_dir / "attributes.txt"
    if attr_names_path.exists():
        attr_names = read_id_name_table(attr_names_path)
    else:
        print(f"[warn] {attr_names_path} not found; naming attributes 0..N-1")
        # inferred below once we've scanned image_attribute_labels.txt
        attr_names = None

    # --- certainties.txt: certainty_id -> name (optional; we already have
    # the standard 1..4 mapping hardcoded as a fallback) ---
    certainties_path = cub_dir / "attributes" / "certainties.txt"
    certainty_weight_by_id = dict(CERTAINTY_ID_WEIGHT)
    if certainties_path.exists():
        cert_names = read_id_name_table(certainties_path)
        certainty_weight_by_id = {
            cid: CERTAINTY_WEIGHT.get(name.lower(), CERTAINTY_ID_WEIGHT.get(cid, 0.0))
            for cid, name in cert_names.items()
        }

    # --- image_attribute_labels.txt: image_id attribute_id is_present certainty_id time ---
    attr_labels_path = cub_dir / "attributes" / "image_attribute_labels.txt"
    n_present = defaultdict(float)  # (image_id, attribute_id) -> weighted count
    n_absent = defaultdict(float)
    max_attr_id = 0
    with open(attr_labels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            image_id, attribute_id, is_present, certainty_id = (
                int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
            )
            max_attr_id = max(max_attr_id, attribute_id)
            w = certainty_weight_by_id.get(certainty_id, 0.0)
            if w == 0.0:
                continue  # "not visible" (or unknown certainty) -> no evidence
            key = (image_id, attribute_id)
            if is_present == 1:
                n_present[key] += w
            else:
                n_absent[key] += w

    if attr_names is None:
        attr_names = {i: str(i - 1) for i in range(1, max_attr_id + 1)}
    num_attributes = max(attr_names.keys())

    # --- assemble manifest.csv (id, image_path, label) ---
    manifest_path = out_dir / "manifest.csv"
    ids = []
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_path", "label"])
        for image_id, rel_path in sorted(image_paths.items()):
            image_id_str = str(image_id)
            label_0idx = labels_1idx[image_id] - 1
            w.writerow([image_id_str, str(Path("images") / rel_path), label_0idx])
            ids.append(image_id_str)
    print(f"[saved] {manifest_path} ({len(ids)} images, {num_classes} classes)")

    classes_path = out_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write("\n".join(class_names[i] for i in sorted(class_names)) + "\n")
    print(f"[saved] {classes_path}")

    attributes_path = out_dir / "attributes.txt"
    with open(attributes_path, "w") as f:
        f.write("\n".join(attr_names[i] for i in range(1, num_attributes + 1)) + "\n")
    print(f"[saved] {attributes_path} ({num_attributes} attributes)")

    # --- attribute_credal.csv (id, lower_0..lower_{A-1}, upper_0..upper_{A-1}) ---
    s = args.idm_s
    credal_path = out_dir / "attribute_credal.csv"
    header = (
        ["id"]
        + [f"lower_{a}" for a in range(num_attributes)]
        + [f"upper_{a}" for a in range(num_attributes)]
    )
    with open(credal_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for image_id, _ in sorted(image_paths.items()):
            lowers, uppers = [], []
            for a in range(1, num_attributes + 1):  # a is 1-indexed attribute id
                key = (image_id, a)
                np_ = n_present.get(key, 0.0)
                na_ = n_absent.get(key, 0.0)
                n = np_ + na_
                if n == 0:
                    lo, up = 0.0, 1.0  # no evidence at all -> maximal EU
                else:
                    lo = np_ / (n + s)
                    up = (np_ + s) / (n + s)
                lowers.append(round(lo, 6))
                uppers.append(round(up, 6))
            w.writerow([str(image_id)] + lowers + uppers)
    print(f"[saved] {credal_path} (IDM s={s})")


if __name__ == "__main__":
    main()
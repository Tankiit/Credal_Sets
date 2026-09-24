"""
Build CUB-S's manifest.csv + classes.txt + attributes.txt + attribute_credal.csv
(+ attribute_groups.txt, annotator_counts.csv) and copy the images, in
whatever combination of stages you need.

CUB-S (Collins et al., AIES 2023) is a relabeling of 144 images from the CUB
test set with *individual* soft labels: each annotator gave a probability in
[0, 1] for every one of CUB's 312 binary attributes (organised in 28 concept
groups such as "wing color"). 116 images have 1 annotator, 22 have 2, 6 have 3.

Three independent stages, each toggleable:
  --images / --no-images      copy the per-example .jpg files to <images-dir>
  --manifest / --no-manifest  write manifest.csv (id, image_path, label) + classes.txt
  --credal / --no-credal      write attribute_credal.csv (see below) + attributes.txt

All default to on. If --manifest is skipped, --images/--credal reuse the
id list from an existing manifest.csv in --out-dir (it must already exist
in that case). This lets you e.g. re-run --credal alone with a different
--interval-mode / --idm-s.

Data sources
  * CUB-S files (cub-s_labels.json, raw_cub-s_human_data.csv): read from
    --local-dir if present, otherwise downloaded from the GitHub repo
    (collinskatie/cub-s) into --local-dir. --download forces a re-download.
  * The images themselves are NOT in the CUB-S repo. --cub-dir must point to
    an extracted CUB_200_2011/ (same one build_cub_data.py uses); we take the
    image paths, species labels, class names and attribute names from it.

id    = the CUB image_id (== the keys of cub-s_labels.json == `img_id` in the
        raw csv). Cross-checked against CUB_200_2011/images.txt using the
        filename column of the raw csv; a mismatch aborts (--skip-verify to
        override).
label = CUB species index, 0-indexed (0..199), same as build_cub_data.py.

How the interval is built. Every annotator k gives a soft label p_k in [0, 1]
per attribute (attributes the annotator did not select count as 0, per the
CUB-S README). With K annotators on an image, --interval-mode picks:

  idm     (default) Imprecise Dirichlet Model, same family as build_cub_data.py.
          Each annotator contributes p_k "votes for present" and 1-p_k "votes
          for absent":  n_present = sum_k p_k,  n = K,
              lower = n_present / (n + s),  upper = (n_present + s) / (n + s)
          s = --idm-s (default 1.0). More annotators -> narrower interval.
          NOTE: with a single annotator the width is s/(1+s) = 0.5 for EVERY
          attribute regardless of the value; that's the IDM's honest "one
          opinion isn't much evidence", but it is not attribute-specific.
  minmax  lower = min_k p_k, upper = max_k p_k. Real disagreement between
          annotators only: zero-width for the 116 single-annotator images.
  point   lower = upper = mean_k p_k. Zero-width, no epistemic uncertainty.

Attribute subset: by default all 312 attributes are exported. Koh et al. (CBM)
use a 112-attribute subset; pass --attr-indices-file (whitespace/comma/newline
separated 0-indexed ints into the 312) to restrict to those. Output columns
are re-numbered 0..A-1 in the order given in the file, and attributes.txt /
attribute_groups.txt are written for that same subset.
"""
import argparse
import csv
import json
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

GITHUB_RAW = "https://raw.githubusercontent.com/collinskatie/cub-s/main"
LABELS_FILE = "cub-s_labels.json"
RAW_CSV_FILE = "raw_cub-s_human_data.csv"
NUM_ATTRS = 312


# ----------------------------------------------------------------------------
# CUB-S loading
# ----------------------------------------------------------------------------
def fetch(local_dir: Path, name: str, force: bool, required: bool):
    dest = local_dir / name
    if dest.exists() and not force:
        print(f"[info] using cached {dest}")
        return dest
    url = f"{GITHUB_RAW}/{name}"
    print(f"[info] downloading {url}")
    local_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as e:  # network / 404
        if tmp.exists():
            tmp.unlink()
        if required:
            raise
        print(f"[warn] could not download {name}: {e}")
        return None
    print(f"[saved] {dest}")
    return dest


def load_soft_labels(path: Path):
    """cub-s_labels.json -> {cub_image_id: array (K_annotators, 312)} in [0, 1]."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, val in raw.items():
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:  # tolerate a non-nested single annotator
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != NUM_ATTRS:
            raise ValueError(f"image {key}: expected (K, {NUM_ATTRS}) soft labels, got {arr.shape}")
        out[int(key)] = arr
    top = max(a.max() for a in out.values())
    if top > 1.0 + 1e-9:  # the raw csv is 0-100; the json is currently already 0-1
        print(f"[info] max soft label is {top:g} > 1, assuming 0-100 scale and dividing by 100")
        out = {k: v / 100.0 for k, v in out.items()}
    return out


def read_csv_filenames(path: Path):
    """raw_cub-s_human_data.csv -> {img_id: 'NNN.Species/Name_0001_12345.jpg'}"""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(float(row["img_id"]))] = row["filename"]
    return out


# ----------------------------------------------------------------------------
# CUB_200_2011 metadata (same parsing as build_cub_data.py)
# ----------------------------------------------------------------------------
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


def read_cub_metadata(cub_dir: Path):
    image_paths = read_id_name_table(cub_dir / "images.txt")
    labels_1idx = {}
    with open(cub_dir / "image_class_labels.txt") as f:
        for line in f:
            idx, class_id = line.strip().split()
            labels_1idx[int(idx)] = int(class_id)
    class_names = read_id_name_table(cub_dir / "classes.txt")
    is_train = {}
    split_path = cub_dir / "train_test_split.txt"
    if split_path.exists():
        with open(split_path) as f:
            for line in f:
                idx, flag = line.strip().split()
                is_train[int(idx)] = flag == "1"
    return image_paths, labels_1idx, class_names, is_train


def read_attr_names(cub_dir: Path):
    for p in (cub_dir / "attributes.txt", cub_dir / "attributes" / "attributes.txt"):
        if p.exists():
            names = read_id_name_table(p)
            if len(names) == NUM_ATTRS:
                return [names[i] for i in range(1, NUM_ATTRS + 1)]
            print(f"[warn] {p} has {len(names)} entries, expected {NUM_ATTRS}; ignoring")
    print(f"[warn] attributes.txt not found under {cub_dir}; naming attributes 0..{NUM_ATTRS - 1}")
    return [str(i) for i in range(NUM_ATTRS)]


def load_attr_subset(path):
    text = Path(path).read_text().replace(",", " ")
    idx = [int(t) for t in text.split()]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{path}: duplicate attribute indices")
    bad = [i for i in idx if not 0 <= i < NUM_ATTRS]
    if bad:
        raise ValueError(f"{path}: indices out of range 0..{NUM_ATTRS - 1}: {bad[:5]}")
    return idx


def tail(path_str: str) -> str:
    """'/any/prefix/CUB_200_2011/images/001.X/Y.jpg' -> '001.X/Y.jpg'"""
    return "/".join(path_str.replace("\\", "/").split("/")[-2:])


# ----------------------------------------------------------------------------
# credal intervals
# ----------------------------------------------------------------------------
def credal_interval(arr: np.ndarray, mode: str, s: float):
    """arr: (K, A) soft labels in [0,1] -> (lower, upper), each (A,)."""
    k = arr.shape[0]
    if mode == "point":
        m = arr.mean(axis=0)
        return m, m
    if mode == "minmax":
        return arr.min(axis=0), arr.max(axis=0)
    n_present = arr.sum(axis=0)  # sum_k p_k; n_absent = k - n_present, n = k
    return n_present / (k + s), (n_present + s) / (k + s)


def read_manifest_ids(manifest_path: Path):
    with open(manifest_path, newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", required=True,
                    help="where cub-s_labels.json / raw_cub-s_human_data.csv live -- read if present, "
                         "downloaded from GitHub if missing")
    ap.add_argument("--download", action="store_true",
                    help="force a fresh download even if the files already exist in --local-dir")
    ap.add_argument("--cub-dir", required=True, help="path to extracted CUB_200_2011/")
    ap.add_argument("--out-dir", required=True,
                    help="where manifest.csv / attribute_credal.csv / classes.txt / attributes.txt / images/ live")
    ap.add_argument("--images-dir", default=None, help="defaults to <out-dir>/images")
    ap.add_argument("--skip-existing-images", action="store_true",
                    help="don't re-copy a .jpg that's already on disk")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink images from --cub-dir instead of copying them")
    ap.add_argument("--interval-mode", choices=["idm", "minmax", "point"], default="idm")
    ap.add_argument("--idm-s", type=float, default=1.0,
                    help="IDM prior strength (idm mode only): bigger = wider intervals")
    ap.add_argument("--attr-indices-file", default=None,
                    help="0-indexed attribute subset (e.g. Koh et al.'s 112); default = all 312")
    ap.add_argument("--skip-verify", action="store_true",
                    help="don't abort if json ids / csv filenames disagree with CUB's images.txt")
    ap.add_argument("--images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--credal", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if not (args.images or args.manifest or args.credal):
        ap.error("nothing to do -- at least one of --images/--manifest/--credal must be on")

    cub_dir = Path(args.cub_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else out_dir / "images"
    manifest_path = out_dir / "manifest.csv"

    # ---- CUB-S soft labels ----
    local_dir = Path(args.local_dir)
    labels_path = fetch(local_dir, LABELS_FILE, args.download, required=True)
    csv_path = fetch(local_dir, RAW_CSV_FILE, args.download, required=False)
    soft = load_soft_labels(labels_path)
    n_ann = Counter(a.shape[0] for a in soft.values())
    print(f"[info] {len(soft)} CUB-S images; annotators per image: {dict(sorted(n_ann.items()))}")

    # ---- CUB metadata + sanity checks ----
    image_paths, labels_1idx, class_names, is_train = read_cub_metadata(cub_dir)
    missing = sorted(i for i in soft if i not in image_paths)
    if missing:
        raise SystemExit(f"[error] {len(missing)} CUB-S ids not in {cub_dir}/images.txt, e.g. {missing[:5]}")
    if not args.skip_verify:
        if csv_path is None:
            print("[warn] raw csv unavailable, cannot cross-check ids against filenames")
        else:
            fnames = read_csv_filenames(csv_path)
            bad = [i for i, fn in fnames.items() if i in image_paths and tail(fn) != image_paths[i]]
            if bad:
                i = bad[0]
                raise SystemExit(
                    f"[error] {len(bad)} ids where the csv filename disagrees with images.txt, e.g. "
                    f"id {i}: csv={tail(fnames[i])!r} vs images.txt={image_paths[i]!r}. "
                    f"Check --cub-dir, or pass --skip-verify.")
            print(f"[ok] json ids match csv filenames / images.txt for {len(fnames)} images")
    n_train = sum(1 for i in soft if is_train.get(i, False))
    if n_train:
        print(f"[warn] {n_train} CUB-S images are in CUB's *training* split (expected test only)")

    # ---- id list ----
    if args.manifest:
        ids = [str(i) for i in sorted(soft)]
    else:
        if not manifest_path.exists():
            ap.error(f"--no-manifest given but {manifest_path} doesn't exist yet")
        ids = read_manifest_ids(manifest_path)
        unknown = [i for i in ids if int(i) not in soft]
        if unknown:
            raise SystemExit(f"[error] {len(unknown)} ids in {manifest_path} aren't CUB-S images, e.g. {unknown[:5]}")
        print(f"[info] reusing {len(ids)} ids from existing {manifest_path}")

    # ---- images ----
    if args.images:
        for id_ in tqdm(ids, desc="Exporting images"):
            rel = image_paths[int(id_)]
            src, dst = cub_dir / "images" / rel, images_dir / rel
            if args.skip_existing_images and dst.exists():
                continue
            if not src.exists():
                raise SystemExit(f"[error] missing source image {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        print(f"[saved] {images_dir} ({len(ids)} images)")

    # ---- manifest ----
    if args.manifest:
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "image_path", "label"])
            for id_ in ids:
                rel = image_paths[int(id_)]
                rel_path = os.path.relpath(images_dir / rel, out_dir).replace(os.sep, "/")
                w.writerow([id_, rel_path, labels_1idx[int(id_)] - 1])
        classes_path = out_dir / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("\n".join(class_names[i] for i in sorted(class_names)) + "\n")
        print(f"[saved] {manifest_path} ({len(ids)} images, {len(class_names)} classes)")
        print(f"[saved] {classes_path}")

    # ---- credal ----
    if args.credal:
        subset = load_attr_subset(args.attr_indices_file) if args.attr_indices_file else list(range(NUM_ATTRS))
        num_attrs = len(subset)
        all_names = read_attr_names(cub_dir)
        names = [all_names[i] for i in subset]

        credal_path = out_dir / "attribute_credal.csv"
        header = (
            ["id"]
            + [f"lower_{a}" for a in range(num_attrs)]
            + [f"upper_{a}" for a in range(num_attrs)]
        )
        with open(credal_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for id_ in tqdm(ids, desc="Building attribute_credal.csv"):
                lo, up = credal_interval(soft[int(id_)][:, subset], args.interval_mode, args.idm_s)
                w.writerow([id_] + [round(float(v), 6) for v in lo] + [round(float(v), 6) for v in up])
        mode_note = f"IDM s={args.idm_s}" if args.interval_mode == "idm" else args.interval_mode
        print(f"[saved] {credal_path} ({num_attrs} attributes, mode={mode_note})")

        attributes_path = out_dir / "attributes.txt"
        with open(attributes_path, "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"[saved] {attributes_path}")

        # concept group = the part of the CUB attribute name before '::'
        # (e.g. has_bill_shape::curved -> has_bill_shape); CUB-S has 28 of them.
        if all("::" in n for n in names):
            groups = [n.split("::")[0] for n in names]
            groups_path = out_dir / "attribute_groups.txt"
            with open(groups_path, "w") as f:
                f.write("\n".join(groups) + "\n")
            print(f"[saved] {groups_path} ({len(set(groups))} concept groups)")

        counts_path = out_dir / "annotator_counts.csv"
        with open(counts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "num_annotators"])
            for id_ in ids:
                w.writerow([id_, soft[int(id_)].shape[0]])
        print(f"[saved] {counts_path}")


if __name__ == "__main__":
    main()


"""
README: What to do :
1. Download CUB-S labels from GitHub + do all three stages (needs the extracted CUB_200_2011/ for images):
python -m download_data_cubs \
  --local-dir ignore/data/cubs_local \
  --cub-dir ignore/data/CUB_200_2011 \
  --out-dir ignore/dataset/dataset_cubs

2. Rebuild only the credal file, e.g. with min/max envelope over annotators:
python -m download_data_cubs \
  --local-dir ignore/data/cubs_local \
  --cub-dir ignore/data/CUB_200_2011 \
  --out-dir ignore/dataset/dataset_cubs \
  --no-images --no-manifest \
  --interval-mode minmax

3. Restrict to Koh et al.'s 112 attributes (indices file from ConceptBottleneck/CUB/generate_new_data.py
   or CEM's SELECTED_CONCEPTS), IDM with a stronger prior:
python -m download_data_cubs \
  --local-dir ignore/data/cubs_local \
  --cub-dir ignore/data/CUB_200_2011 \
  --out-dir ignore/dataset/dataset_cubs \
  --attr-indices-file ignore/data/cub_112_indices.txt \
  --idm-s 2.0
"""
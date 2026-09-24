"""
Build Shapes3D's manifest.csv + concepts.txt + attribute_credal.csv, in
whatever combination of stages you need.

Three independent stages, each toggleable:
  --images / --no-images      write per-example .png files to <images-dir>
  --manifest / --no-manifest  write manifest.csv (id, image_path, label)
  --credal / --no-credal      write attribute_credal.csv (see note below)

All default to on. If --manifest is skipped, --images/--credal reuse the
id list from an existing manifest.csv in --out-dir (it must already exist
in that case). This lets you e.g. re-run --credal alone after tweaking
the concept normalization, without re-exporting 480k images.

Data source -- exactly one applies:
  --local-dir already contains a datasets.save_to_disk() copy -> loaded
    with load_from_disk(), no network needed.
  --local-dir is missing, or --download is passed -> fetched with
    load_dataset("randall-lab/shapes3d", ...) and saved to --local-dir for
    next time. Needs datasets<4.0 (this repo uses a loading script;
    trust_remote_code-based scripts are unsupported on datasets>=4.0).

label = shape_idx * NUM_HUE_BINS + object_idx  (joint class)
concepts (4, in order) = floor, wall, scale (normalized 0-1), orientation
  (normalized 0-1). Every interval is [v, v] -- a zero-width point
  estimate, not genuine credal uncertainty like CUB's IDM intervals (see
  build_cub_data.py's docstring for why CUB's intervals have real width).
"""
import argparse
import csv
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

NUM_HUE_BINS = 10
NUM_SHAPES = 4
CONCEPT_NAMES = ["floor_hue", "wall_hue", "scale", "orientation"]


def load_shapes3d(local_dir: Path, download: bool, hub_id: str):
    if local_dir.exists() and not download:
        print(f"[info] loading cached dataset from {local_dir}")
        return load_from_disk(str(local_dir))
    print(f"[info] downloading {hub_id} (requires datasets<4.0)")
    dataset = load_dataset(hub_id, split="train", trust_remote_code=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(local_dir))
    print(f"[saved] {local_dir}")
    return dataset


def read_manifest_ids(manifest_path: Path):
    with open(manifest_path, newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hub-id", default="randall-lab/shapes3d")
    ap.add_argument("--local-dir", required=True,
                     help="save_to_disk() cache path -- read if present, written after a download")
    ap.add_argument("--download", action="store_true",
                     help="force a fresh download even if --local-dir already exists")
    ap.add_argument("--out-dir", required=True,
                     help="where manifest.csv / attribute_credal.csv / concepts.txt / images/ live")
    ap.add_argument("--images-dir", default=None,
                     help="defaults to <out-dir>/images")
    ap.add_argument("--skip-existing-images", action="store_true",
                     help="don't re-save a .png that's already on disk")
    ap.add_argument("--images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--credal", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else out_dir / "images"
    manifest_path = out_dir / "manifest.csv"

    if not (args.images or args.manifest or args.credal):
        ap.error("nothing to do -- at least one of --images/--manifest/--credal must be on")

    dataset = load_shapes3d(Path(args.local_dir), args.download, args.hub_id)

    # id list: fresh range if (re)building the manifest this run, otherwise
    # reused from whatever manifest.csv already exists in --out-dir.
    if args.manifest:
        ids = [str(i) for i in range(len(dataset))]
    else:
        if not manifest_path.exists():
            ap.error(f"--no-manifest given but {manifest_path} doesn't exist yet")
        ids = read_manifest_ids(manifest_path)
        print(f"[info] reusing {len(ids)} ids from existing {manifest_path}")

    if args.images:
        images_dir.mkdir(parents=True, exist_ok=True)
        for id_ in tqdm(ids, desc="Exporting images"):
            img_path = images_dir / f"{int(id_):06d}.png"
            if args.skip_existing_images and img_path.exists():
                continue
            dataset[int(id_)]["image"].save(img_path)
        print(f"[saved] {images_dir} ({len(ids)} images)")

    if args.manifest:
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "image_path", "label"])
            for id_ in ids:
                ex = dataset[int(id_)]
                label = ex["shape_idx"] * NUM_HUE_BINS + ex["object_idx"]
                rel_path = Path(images_dir.name) / f"{int(id_):06d}.png"
                w.writerow([id_, str(rel_path), label])
        print(f"[saved] {manifest_path} ({len(ids)} images, {NUM_SHAPES * NUM_HUE_BINS} joint classes)")

    if args.credal:
        num_concepts = len(CONCEPT_NAMES)
        credal_path = out_dir / "attribute_credal.csv"
        header = (
            ["id"]
            + [f"lower_{a}" for a in range(num_concepts)]
            + [f"upper_{a}" for a in range(num_concepts)]
        )
        with open(credal_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for id_ in tqdm(ids, desc="Building attribute_credal.csv"):
                ex = dataset[int(id_)]
                values = [
                    ex["floor"],
                    ex["wall"],
                    (ex["scale"] - 0.75) / 0.5,
                    (ex["orientation"] + 30) / 60,
                ]
                row = [id_] + [round(v, 6) for v in values] * 2
                w.writerow(row)
        concepts_path = out_dir / "concepts.txt"
        with open(concepts_path, "w") as f:
            f.write("\n".join(CONCEPT_NAMES) + "\n")
        print(f"[saved] {credal_path} ({num_concepts} concepts, zero-width intervals)")
        print(f"[saved] {concepts_path}")


if __name__ == "__main__":
    main()


"""
README: What to do : 
1. Download from the Hub + do all three stages:
python -m download_data_shapes3d \
  --local-dir ignore/data/shapes3d_local \
  --download \
  --out-dir ignore/dataset/dataset_shapes_3d


2. Export images only (no manifest, no credal — uses local cache if already downloaded):
python -m download_data_shapes3d \
  --local-dir ignore/data/shapes3d_local \
  --no-manifest \
  --no-credal \
  --out-dir ignore/dataset/dataset_shapes_3d

3. Build manifest + credal only, reusing images you already have:
python -m download_data_shapes3d \
  --local-dir ignore/data/shapes3d_local \
  --images-dir ignore/dataset/dataset_shapes_3d/shapes3d_images \
  --out-dir ignore/dataset/dataset_shapes_3d
  --no-images
"""
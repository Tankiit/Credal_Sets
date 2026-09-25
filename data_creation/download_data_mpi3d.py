"""
Build MPI3D's manifest.csv + concepts.txt + attribute_credal.csv, in
whatever combination of stages you need. Mirrors download_data_shapes3d.py
exactly -- same toggle flags, same file layout, same zero-width-interval
choice for concepts (see that script's docstring for the rationale).

Source: https://github.com/rr-learning/disentanglement_dataset (Gondal et
al. 2019), mirrored on the Hub by the same group that hosts shapes3d:
randall-lab/mpi3d-real, mpi3d-toy, mpi3d-realistic, mpi3d-complex.

Three independent stages, each toggleable:
  --images / --no-images      write per-example .png files to <images-dir>
  --manifest / --no-manifest  write manifest.csv (id, image_path, label)
  --credal / --no-credal      write attribute_credal.csv

All default to on. If --manifest is skipped, --images/--credal reuse the
id list from an existing manifest.csv in --out-dir (it must already exist).

Data source -- exactly one applies:
  --local-dir already contains a datasets.save_to_disk() copy -> loaded
    with load_from_disk(), no network needed.
  --local-dir is missing, or --download is passed -> fetched with
    load_dataset(<hub-id>, ...) and saved to --local-dir for next time.
    Needs datasets<4.0 (script-based dataset).

7 factors total: object_color, object_shape, object_size, camera_height,
background_color, horizontal_axis (dof1), vertical_axis (dof2).

label = shape_idx * NUM_COLOR_VALS + color_idx  (joint class -- same
  "shape x color" choice as Shapes3D's "shape x object_hue")
concepts (5, in order) = object_size, camera_height, background_color,
  horizontal_axis, vertical_axis -- each normalized to 0-1 by dividing by
  (its own cardinality - 1). Every interval is [v, v]: a zero-width point
  estimate, since these are exact known factors, not uncertain
  annotations (see build_cub_data.py for what a *real* credal interval
  looks like, and download_data_shapes3d.py for why this is a deliberate
  reuse of that CSV format rather than genuine credal uncertainty).

NOTE on color/shape being treated as ordinal 0-1 here: these are
categorical labels (e.g. color 0=white, 1=green, ...), not a true ordinal
scale, so "normalized position in the list" is a modeling convenience,
not a meaningful distance -- exactly the same caveat that applied to
Shapes3D's hue-as-ordinal treatment. Revisit if your concept audit's
correlation/diagnostic machinery assumes concepts are genuinely ordinal.
"""
import argparse
import csv
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# color/shape cardinality varies by variant; everything else is identical
# across all four MPI3D variants per the dataset cards.
VARIANT_COLOR_SHAPE_CARDINALITY = {
    "mpi3d-real": {"color": 6, "shape": 6},
    "mpi3d-toy": {"color": 6, "shape": 6},
    "mpi3d-realistic": {"color": 6, "shape": 6},
    "mpi3d-complex": {"color": 4, "shape": 4},
}
NUM_SIZE_VALS = 2
NUM_HEIGHT_VALS = 3
NUM_BACKGROUND_VALS = 3
NUM_DOF_VALS = 40

CONCEPT_NAMES = ["object_size", "camera_height", "background_color", "horizontal_axis", "vertical_axis"]


def load_mpi3d(local_dir: Path, download: bool, hub_id: str):
    if local_dir.exists() and not download:
        print(f"[info] loading cached dataset from {local_dir}")
        return load_from_disk(str(local_dir))
    print(f"[info] downloading {hub_id} (requires datasets<4.0, trust_remote_code=True)")
    dataset = load_dataset(hub_id, split="train", trust_remote_code=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(local_dir))
    print(f"[saved] {local_dir}")
    return dataset


def read_manifest_ids(manifest_path: Path):
    with open(manifest_path, newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub-id", default="randall-lab/mpi3d-real",
                     help="one of randall-lab/mpi3d-{real,toy,realistic,complex}")
    ap.add_argument("--local-dir", required=True,
                     help="save_to_disk() cache path -- read if present, written after a download")
    ap.add_argument("--download", action="store_true",
                     help="force a fresh download even if --local-dir already exists")
    ap.add_argument("--out-dir", required=True,
                     help="where manifest.csv / attribute_credal.csv / concepts.txt / images/ live")
    ap.add_argument("--images-dir", default=None, help="defaults to <out-dir>/images")
    ap.add_argument("--skip-existing-images", action="store_true",
                     help="don't re-save a .png that's already on disk")
    ap.add_argument("--images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--credal", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    variant = args.hub_id.rsplit("/", 1)[-1]
    if variant not in VARIANT_COLOR_SHAPE_CARDINALITY:
        ap.error(
            f"unrecognized MPI3D variant '{variant}' -- add its color/shape "
            f"cardinality to VARIANT_COLOR_SHAPE_CARDINALITY at the top of this "
            f"script before running (check the dataset's HF card for the values)"
        )
    num_color_vals = VARIANT_COLOR_SHAPE_CARDINALITY[variant]["color"]
    num_shape_vals = VARIANT_COLOR_SHAPE_CARDINALITY[variant]["shape"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else out_dir / "images"
    manifest_path = out_dir / "manifest.csv"

    if not (args.images or args.manifest or args.credal):
        ap.error("nothing to do -- at least one of --images/--manifest/--credal must be on")

    dataset = load_mpi3d(Path(args.local_dir), args.download, args.hub_id)
    print(f"[info] {len(dataset)} images in {variant} "
          f"(color={num_color_vals}, shape={num_shape_vals} -> "
          f"{num_color_vals * num_shape_vals} joint classes)")

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
            img_path = images_dir / f"{int(id_):07d}.png"
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
                label = ex["shape"] * num_color_vals + ex["color"]
                rel_path = Path(images_dir.name) / f"{int(id_):07d}.png"
                w.writerow([id_, str(rel_path), label])
        print(f"[saved] {manifest_path} ({len(ids)} images, "
              f"{num_shape_vals * num_color_vals} joint classes)")

        classes_path = out_dir / "classes.txt"
        with open(classes_path, "w") as f:
            for shape_i in range(num_shape_vals):
                for color_i in range(num_color_vals):
                    f.write(f"shape{shape_i}_color{color_i}\n")
        print(f"[saved] {classes_path} ({num_shape_vals * num_color_vals} classes)")

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
                    ex["size"] / (NUM_SIZE_VALS - 1),
                    ex["height"] / (NUM_HEIGHT_VALS - 1),
                    ex["background"] / (NUM_BACKGROUND_VALS - 1),
                    ex["dof1"] / (NUM_DOF_VALS - 1),
                    ex["dof2"] / (NUM_DOF_VALS - 1),
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
"""
Extract frozen DINOv3 (or any HF vision-transformer) embeddings for every
image in a manifest.csv, and save them aligned with the dataset id.

Frozen throughout: eval(), every parameter requires_grad_(False), forward
pass under torch.inference_mode(). No fine-tuning, no gradient anywhere.

Images are handed to the model's own AutoImageProcessor, so native
resolution doesn't matter -- CIFAR's 32x32 upsampled and CUB's larger
native photos are both resized/normalized exactly the way the checkpoint
expects.

Outputs (all in manifest.csv row order):
    embeddings.npy    (N, D) float32 -- CLS token / pooled global embedding
    labels.npy        (N,)  int64    -- from manifest's `label` column (-1 if absent)
    ids.npy           (N,)  object   -- manifest's `id` column
    meta.json                        -- model name, embedding dim, counts, etc.
    patch_tokens.npy  (N, T, D) float32, only with --save-patch-tokens --
                       the full non-CLS token grid (patches + any register
                       tokens the checkpoint uses), for dense-feature use later.
"""
# Standard library module for building command-line argument parsers.
import argparse
# Standard library module for reading and writing JSON files.
import json
# Standard library module used to measure elapsed wall-clock time.
import time
# Import Path for convenient, cross-platform filesystem path handling.
from pathlib import Path

# Import NumPy for array manipulation and saving .npy files.
import numpy as np
# Import PyTorch for tensors, devices, and inference utilities.
import torch
# Import the HF classes that auto-select the right image processor and model.
from transformers import AutoImageProcessor, AutoModel

# Import the project's dataset class and manifest-reading helper.
from common import ManifestImageDataset, read_manifest


def build_collate_fn(processor):
    """
    Build a DataLoader collate function that preprocesses a batch of images with the given HF processor.
    """
    # Define the actual collate function that will be passed to the DataLoader.
    def collate(batch):
        # Unzip the batch of (image, id, label) tuples into separate sequences.
        images, ids, labels = zip(*batch)
        # Run the HF processor on the images to get normalized pixel tensors.
        pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
        # Return the pixel tensor, the list of ids, and the labels as a tensor.
        return pixel_values, list(ids), torch.tensor(labels, dtype=torch.int64)
    # Return the inner collate function to the caller.
    return collate


def main():
    """
    Parse CLI arguments, run the frozen vision model over the manifest, and save embeddings, labels, ids, and metadata to disk.
    """
    # Create the argument parser, using the module docstring as its description.
    ap = argparse.ArgumentParser(description=__doc__)
    # Add the required path to the manifest CSV file.
    ap.add_argument("--manifest", required=True)
    # Add an optional base directory to resolve relative image paths against.
    ap.add_argument("--images-base-dir", default=None)
    # Add the required HF model id to load the processor and model from.
    ap.add_argument("--model", required=True,
                     help="HF model id, e.g. facebook/dinov3-vitl16-pretrain-lvd1689m")
    # Add the batch size used for the DataLoader, defaulting to 128.
    ap.add_argument("--batch-size", type=int, default=128)
    # Add the number of DataLoader worker processes, defaulting to 4.
    ap.add_argument("--num-workers", type=int, default=4)
    # Add the output directory where results will be written.
    ap.add_argument("--out-dir", default="features")
    # Add an optional device override; auto-detected later if not given.
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detected if omitted")
    # Add a flag to also save the full patch/register token grid.
    ap.add_argument("--save-patch-tokens", action="store_true",
                     help="also save the full patch/register token grid, not just the CLS embedding")
    # Parse the command-line arguments into the args namespace.
    args = ap.parse_args()

    # Wrap the output directory string in a Path object.
    out_dir = Path(args.out_dir)
    # Create the output directory (and parents) if it doesn't already exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the requested device, or auto-detect CUDA vs CPU if none was given.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Read the manifest CSV into an in-memory structure.
    manifest = read_manifest(args.manifest)
    # Print how many images were found in the manifest.
    print(f"[info] {len(manifest)} images in {args.manifest}")

    # Load the image processor matching the requested model checkpoint.
    processor = AutoImageProcessor.from_pretrained(args.model)
    # Load the pretrained vision model itself.
    model = AutoModel.from_pretrained(args.model)
    # Put the model in evaluation mode to disable dropout/batchnorm updates.
    model.eval()
    # Freeze every parameter so no gradients are tracked or computed.
    for p in model.parameters():
        p.requires_grad_(False)
    # Move the model onto the selected device.
    model.to(device)

    # Build the dataset that reads images according to the manifest rows.
    dataset = ManifestImageDataset(manifest, transform=None, base_dir=args.images_base_dir)
    # Build the DataLoader that batches images in manifest order using the custom collate function.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=build_collate_fn(processor),
    )

    # CLS token sits at index 0; some checkpoints (DINOv3 included) also
    # prepend N register tokens after it, before the patch-token grid.
    # Compute how many leading tokens (CLS + any register tokens) to skip for patch tokens.
    n_prefix_tokens = getattr(model.config, "num_register_tokens", 0) + 1

    # Initialize accumulator lists for embeddings, patch tokens, ids, and labels.
    all_embeddings, all_patch_tokens, all_ids, all_labels = [], [], [], []
    # Record the start time to track elapsed processing time.
    t0 = time.time()
    # Initialize a counter of how many images have been processed so far.
    seen = 0
    # Iterate over each batch produced by the DataLoader.
    for pixel_values, ids, labels in loader:
        # Move the batch of pixel values onto the target device.
        pixel_values = pixel_values.to(device)
        # Run the forward pass without tracking gradients, since the model is frozen.
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)

        # Extract the full sequence of hidden states with shape (B, seq_len, D).
        last_hidden = outputs.last_hidden_state  # (B, seq_len, D)
        # Try to get a pooled output if the model provides one.
        pooled = getattr(outputs, "pooler_output", None)
        # Use the pooled output if available, otherwise fall back to the CLS token.
        cls_embed = pooled if pooled is not None else last_hidden[:, 0, :]

        # Move the CLS/pooled embeddings to CPU numpy and append to the accumulator.
        all_embeddings.append(cls_embed.cpu().numpy())
        # Extend the id accumulator with this batch's ids.
        all_ids.extend(ids)
        # Move the labels to numpy and append to the accumulator.
        all_labels.append(labels.numpy())
        # If requested, also accumulate the non-prefix (patch/register) token grid.
        if args.save_patch_tokens:
            all_patch_tokens.append(last_hidden[:, n_prefix_tokens:, :].cpu().numpy())

        # Update the running count of processed images.
        seen += len(ids)
        # Print progress on the same line, including elapsed time.
        print(f"[info] {seen}/{len(dataset)} ({time.time() - t0:.1f}s)", end="\r")
    # Print a newline to finish the progress line.
    print()

    # Concatenate all embedding batches into one array and cast to float32.
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    # Concatenate all label batches into one array and cast to int64.
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)
    # Convert the collected ids list into a NumPy object array.
    ids_arr = np.array(all_ids, dtype=object)

    # Save the embeddings array to disk.
    np.save(out_dir / "embeddings.npy", embeddings)
    # Save the labels array to disk.
    np.save(out_dir / "labels.npy", labels_arr)
    # Save the ids array to disk.
    np.save(out_dir / "ids.npy", ids_arr)
    # Print confirmation of the saved embeddings file and its shape.
    print(f"[saved] {out_dir / 'embeddings.npy'} {embeddings.shape}")
    # Print confirmation of the saved labels file and its shape.
    print(f"[saved] {out_dir / 'labels.npy'} {labels_arr.shape}")
    # Print confirmation of the saved ids file and its shape.
    print(f"[saved] {out_dir / 'ids.npy'} {ids_arr.shape}")

    # Build a metadata dictionary describing this extraction run.
    meta = {
        "model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_images": int(embeddings.shape[0]),
        "batch_size": args.batch_size,
        "device": device,
        "save_patch_tokens": bool(args.save_patch_tokens),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    # If patch tokens were requested, concatenate, save them, and record their shape.
    if args.save_patch_tokens:
        # Concatenate all patch-token batches into one array and cast to float32.
        patch_tokens = np.concatenate(all_patch_tokens, axis=0).astype(np.float32)
        # Save the patch tokens array to disk.
        np.save(out_dir / "patch_tokens.npy", patch_tokens)
        # Record the patch tokens' shape in the metadata dictionary.
        meta["patch_tokens_shape"] = list(patch_tokens.shape)
        # Print confirmation of the saved patch tokens file and its shape.
        print(f"[saved] {out_dir / 'patch_tokens.npy'} {patch_tokens.shape}")

    # Open the meta.json file for writing.
    with open(out_dir / "meta.json", "w") as f:
        # Write the metadata dictionary as indented JSON.
        json.dump(meta, f, indent=2)
    # Print confirmation that the metadata file was saved.
    print(f"[saved] {out_dir / 'meta.json'}")


# Only run main() when this script is executed directly, not when imported.
if __name__ == "__main__":
    # Call the main entry point of the script.
    main()
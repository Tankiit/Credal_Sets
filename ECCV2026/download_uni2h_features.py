#!/usr/bin/env python3
"""
Download UNI2-h Pre-extracted Features from HuggingFace
=======================================================

This script downloads the EXACT features used by Mahmood Lab.
Pre-extracted features from TCGA using UNI2-h (1536-dim).

Features available for:
- TCGA-BRCA (Breast Cancer) - split into IDC and OTHERS
- TCGA-LUAD (Lung Adenocarcinoma)
- TCGA-KIRC (Kidney Renal Clear Cell)
- And 30+ other TCGA cancer types!

Each .h5 file contains:
- features: [1, N_patches, 1536]  # UNI2-h patch embeddings
- coords:   [1, N_patches, 2]     # (x, y) coordinates of patches
"""

import os
import sys
import argparse
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, whoami
import h5py
import numpy as np

# ============================================================================
# STEP 1: AUTHENTICATION
# ============================================================================

def authenticate_huggingface():
    """
    Authenticate with HuggingFace.
    """
    print("="*70)
    print("HUGGINGFACE AUTHENTICATION")
    print("="*70)

    # Check if already logged in
    try:
        user_info = whoami()
        print(f"\n✓ Already logged in as: {user_info['name']}")
        print(f"  Email: {user_info.get('email', 'N/A')}")
        return True

    except Exception as e:
        print(f"\n❌ Not logged in to HuggingFace: {e}")
        print("\nPlease run FIRST:")
        print("  pip install -U 'huggingface_hub[cli]'")
        print("  huggingface-cli login")
        print("\nThen run this script again.")
        return False

# ============================================================================
# STEP 2: DOWNLOAD FEATURES
# ============================================================================

def download_tcga_features(cancer_types, output_dir, subset_size=None):
    """
    Download UNI2-h features for specified TCGA cancer types.

    Args:
        cancer_types: List of cancer types (e.g., ['TCGA-BRCA', 'TCGA-LUAD'])
        output_dir: Where to save features
        subset_size: Optional, only extract first N patients
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DOWNLOADING UNI2-h FEATURES FROM HUGGINGFACE")
    print("="*70)

    for cancer_type in cancer_types:
        print(f"\n{'─'*70}")
        print(f"CANCER TYPE: {cancer_type}")
        print(f"{'─'*70}")

        # Handle special case for BRCA (split into IDC and OTHERS)
        if cancer_type == "TCGA-BRCA":
            tar_files = [
                "TCGA/TCGA-BRCA_IDC.tar.gz",
                "TCGA/TCGA-BRCA_OTHERS.tar.gz"
            ]
        else:
            tar_files = [f"TCGA/{cancer_type}.tar.gz"]

        all_extracted = []

        for tar_filename in tar_files:
            try:
                print(f"\n[1/3] Downloading {tar_filename}...")
                print("      This may take 5-30 minutes depending on size...")

                local_tar = hf_hub_download(
                    repo_id="MahmoodLab/UNI2-h-features",
                    filename=tar_filename,
                    repo_type="dataset",
                    local_dir=str(output_dir / "downloads"),
                    resume_download=True
                )

                print(f"✓ Downloaded to: {local_tar}")

                # Extract features
                extract_dir = output_dir / cancer_type / "features_uni2h"
                extract_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n[2/3] Extracting features...")

                with tarfile.open(local_tar, 'r:gz') as tar:
                    members = tar.getmembers()

                    # Filter .h5 files only
                    h5_members = [m for m in members if m.name.endswith('.h5')]

                    print(f"      Found {len(h5_members)} .h5 files in archive")

                    # Optionally extract only subset
                    if subset_size and len(all_extracted) + len(h5_members) > subset_size:
                        remaining = subset_size - len(all_extracted)
                        h5_members = h5_members[:remaining]
                        print(f"      Extracting first {remaining} files (subset limit)")

                    # Extract with progress
                    for i, member in enumerate(h5_members):
                        tar.extract(member, extract_dir)
                        all_extracted.append(member.name)
                        if (i + 1) % 50 == 0:
                            print(f"      Extracted {i+1}/{len(h5_members)} files...")

                    if subset_size and len(all_extracted) >= subset_size:
                        print(f"      Reached subset limit of {subset_size} files")
                        break

                print(f"✓ Extracted {len(h5_members)} files from {tar_filename}")

            except Exception as e:
                print(f"\n❌ ERROR downloading {tar_filename}: {e}")
                print(f"\nPossible issues:")
                print(f"  1. Access not granted yet - check HuggingFace email")
                print(f"  2. Wrong cancer type name (must be exact: TCGA-BRCA not tcga-brca)")
                print(f"  3. Network error - try again")
                continue

        # Verify extraction
        print(f"\n[3/3] Verifying features...")
        verify_features(extract_dir, num_samples=3)

        print(f"\n✅ SUCCESS: {cancer_type} features ready!")
        print(f"   Total files: {len(all_extracted)}")
        print(f"   Location: {extract_dir}")

# ============================================================================
# STEP 3: VERIFY FEATURES
# ============================================================================

def verify_features(feature_dir, num_samples=3):
    """
    Verify that features were extracted correctly.
    """
    h5_files = list(Path(feature_dir).glob("**/*.h5"))

    if not h5_files:
        print("⚠️  No .h5 files found!")
        return

    print(f"   Found {len(h5_files)} .h5 files")

    # Check first few files
    for h5_file in h5_files[:num_samples]:
        try:
            with h5py.File(h5_file, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]

                # Features shape: [1, N_patches, 1536]
                # We want: [N_patches, 1536]
                if features.ndim == 3:
                    features = features.squeeze(0)
                if coords.ndim == 3:
                    coords = coords.squeeze(0)

                print(f"   ✓ {h5_file.name}")
                print(f"      Features: {features.shape} (patches × dims)")
                print(f"      Coords:   {coords.shape}")

        except Exception as e:
            print(f"   ❌ Error reading {h5_file.name}: {e}")

# ============================================================================
# STEP 4: CONVERT TO PYTORCH FORMAT (Optional)
# ============================================================================

def convert_h5_to_pt(feature_dir, output_dir):
    """
    Convert .h5 files to .pt (PyTorch) format for easier loading.
    """
    import torch

    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print("CONVERTING .h5 → .pt FORMAT")
    print(f"{'─'*70}")

    h5_files = list(feature_dir.glob("**/*.h5"))
    print(f"Converting {len(h5_files)} files...")

    for i, h5_file in enumerate(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]

                # Remove batch dimension if present
                if features.ndim == 3:
                    features = features.squeeze(0)
                if coords.ndim == 3:
                    coords = coords.squeeze(0)

                # Save as PyTorch
                pt_file = output_dir / h5_file.name.replace('.h5', '.pt')
                torch.save({
                    'features': torch.from_numpy(features).float(),
                    'coords': torch.from_numpy(coords).long()
                }, pt_file)

                if (i + 1) % 50 == 0:
                    print(f"  Converted {i+1}/{len(h5_files)} files...")

        except Exception as e:
            print(f"❌ Error converting {h5_file.name}: {e}")

    print(f"✓ Converted {len(h5_files)} files to {output_dir}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download UNI2-h pre-extracted features from HuggingFace"
    )
    parser.add_argument(
        "--cancer_types",
        nargs="+",
        default=["TCGA-BRCA", "TCGA-LUAD"],
        help="TCGA cancer types to download"
    )
    parser.add_argument(
        "--output_dir",
        default="./data/uni2h_features",
        help="Output directory for features"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Only extract first N patients (for testing)"
    )
    parser.add_argument(
        "--convert_to_pt",
        action="store_true",
        help="Convert .h5 to .pt format"
    )

    args = parser.parse_args()

    # Step 1: Authenticate
    if not authenticate_huggingface():
        sys.exit(1)

    # Step 2: Download features
    download_tcga_features(
        cancer_types=args.cancer_types,
        output_dir=args.output_dir,
        subset_size=args.subset_size
    )

    # Step 3: Optional conversion
    if args.convert_to_pt:
        for cancer_type in args.cancer_types:
            feature_dir = Path(args.output_dir) / cancer_type / "features_uni2h"
            output_dir = Path(args.output_dir) / cancer_type / "features_pt"

            if feature_dir.exists():
                convert_h5_to_pt(feature_dir, output_dir)

    # Summary
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)

    print("\nYour features are organized as:")
    print(f"  {args.output_dir}/")
    for cancer_type in args.cancer_types:
        print(f"  ├── {cancer_type}/")
        print(f"  │   ├── features_uni2h/  ← .h5 files here")
        if args.convert_to_pt:
            print(f"  │   └── features_pt/     ← .pt files here")

    print("\nNext steps:")
    print("  1. Get genomic data from MMP repo")
    print("  2. Get survival labels from TCGA-CDR")
    print("  3. Start training your credal model!")

    print("\nExample loading code:")
    print("""
    import h5py

    with h5py.File('TCGA-02-0001.h5', 'r') as f:
        features = f['features'][0]  # [N_patches, 1536]
        coords = f['coords'][0]      # [N_patches, 2]
    """)

if __name__ == "__main__":
    main()

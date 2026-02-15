#!/usr/bin/env python3
"""
Local Preprocessing Script for ECCV 2025 Project
================================================
Downloads TCGA UNI2-h features from HuggingFace and prepares data for cluster.

Usage:
    python local_preprocess.py --cancer_type TCGA-BRCA --subset_size 150
    python local_preprocess.py --cancer_type TCGA-LUAD --subset_size 100

Requirements:
    pip install huggingface_hub h5py pandas torch tqdm scikit-learn

Output:
    preprocessed_data/
    └── eccv2025_data_{cancer_type}.tar.gz  # Ready for cluster upload
"""

import argparse
import json
import pickle
import tarfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import h5py
from huggingface_hub import hf_hub_download, login, list_repo_files
from sklearn.model_selection import StratifiedKFold


def authenticate_hf():
    """Authenticate with HuggingFace."""
    print("\n[HuggingFace Authentication]")
    print("="*60)

    token_file = Path.home() / ".cache" / "huggingface" / "token"

    if token_file.exists():
        print("✓ Existing HF token found")
        return True

    print("\n⚠️  HuggingFace authentication required for UNI features")
    print("\nSteps:")
    print("1. Create account: https://huggingface.co/join")
    print("2. Request access: https://huggingface.co/MahmoodLab/UNI")
    print("3. Generate token: https://huggingface.co/settings/tokens")
    print("   (Select 'Read' access)")
    print("\n4. Paste token below (or run: huggingface-cli login)")

    token = input("\nEnter HF token (or press Enter to skip): ").strip()

    if token:
        try:
            login(token=token)
            print("✓ Authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("⚠️  Skipping authentication - using cached credentials if available")
        return True


def explore_hf_repo(repo_id="MahmoodLab/UNI"):
    """Explore HuggingFace repository structure."""
    print(f"\n[Exploring {repo_id}]")
    print("="*60)

    try:
        files = list_repo_files(repo_id, repo_type="dataset")

        # Group by directory
        dirs = {}
        for f in files:
            parts = Path(f).parts
            if len(parts) > 1:
                dir_name = parts[0]
                dirs.setdefault(dir_name, []).append(f)

        print(f"Found {len(files)} files in repository")
        print("\nDirectory structure:")
        for dir_name, files_in_dir in sorted(dirs.items())[:5]:
            print(f"  {dir_name}/  ({len(files_in_dir)} files)")
            for f in files_in_dir[:3]:
                print(f"    - {Path(f).name}")
            if len(files_in_dir) > 3:
                print(f"    ... and {len(files_in_dir) - 3} more")

        return files

    except Exception as e:
        print(f"❌ Failed to explore repository: {e}")
        return None


def download_tcga_features(cancer_type, data_dir, subset_size=None):
    """Download TCGA UNI features from HuggingFace.

    The UNI dataset stores features as individual .pt files per patient.
    """
    print(f"\n[Downloading {cancer_type} Features]")
    print("="*60)

    repo_id = "MahmoodLab/UNI"
    features_dir = data_dir / cancer_type / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    try:
        # List all files in repo
        print("Fetching file list from HuggingFace...")
        all_files = list_repo_files(repo_id, repo_type="dataset")

        # Filter for this cancer type
        # Files are typically: TCGA-features/TCGA-BRCA/TCGA-02-0001.pt
        cancer_code = cancer_type.split('-')[1]  # BRCA from TCGA-BRCA

        feature_files = [
            f for f in all_files
            if f.startswith('TCGA-features/')
            and f'/{cancer_type}/' in f
            and f.endswith('.pt')
        ]

        if not feature_files:
            print(f"❌ No feature files found for {cancer_type}")
            print("\nAvailable cancer types:")
            cancer_dirs = set([
                f.split('/')[1] for f in all_files
                if f.startswith('TCGA-features/') and '/' in f
            ])
            for ct in sorted(cancer_dirs):
                print(f"  - {ct}")
            return None

        print(f"Found {len(feature_files)} patient feature files")

        # Subset if requested
        if subset_size and len(feature_files) > subset_size:
            feature_files = feature_files[:subset_size]
            print(f"Using subset of {len(feature_files)} patients")

        # Download each patient's features
        patient_ids = []
        feature_paths = {}

        for feature_file in tqdm(feature_files, desc="Downloading"):
            patient_id = Path(feature_file).stem
            local_path = features_dir / f"{patient_id}.pt"

            if not local_path.exists():
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=feature_file,
                    repo_type="dataset",
                    cache_dir=str(data_dir / "hf_cache"),
                    resume_download=True,
                )

                # Copy to organized location
                import shutil
                shutil.copy(downloaded, local_path)

            patient_ids.append(patient_id)
            feature_paths[patient_id] = str(local_path)

        print(f"✓ Downloaded {len(patient_ids)} patient features")

        return {
            'patient_ids': patient_ids,
            'feature_paths': feature_paths,
            'cancer_type': cancer_type,
        }

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have HuggingFace access (huggingface-cli login)")
        print("2. Request access at: https://huggingface.co/MahmoodLab/UNI")
        print("3. Check internet connection")
        return None


def create_synthetic_clinical(patient_ids, output_path):
    """Create synthetic clinical/survival data for testing."""
    print("\n⚠️  Creating synthetic clinical data for testing")

    np.random.seed(42)

    # Create survival data with realistic distributions
    n_patients = len(patient_ids)

    clinical_df = pd.DataFrame({
        'case_id': patient_ids,
        'survival_months': np.random.exponential(scale=30, size=n_patients).clip(0, 120),
        'censorship': np.random.choice([0, 1], n_patients, p=[0.4, 0.6]),  # 40% died
        'age': np.random.normal(60, 12, n_patients).clip(18, 90),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_patients),
    })

    # Add synthetic genomic features
    n_genes = 100
    for i in range(n_genes):
        clinical_df[f'gene_{i}'] = np.random.randn(n_patients)

    clinical_df.to_csv(output_path, index=False)
    print(f"  Created: {output_path}")

    return clinical_df


def load_clinical_data(cancer_type, data_dir, patient_ids):
    """Load or create clinical data."""
    print(f"\n[Loading Clinical Data]")
    print("="*60)

    clinical_file = data_dir / f"{cancer_type}_clinical.csv"

    if clinical_file.exists():
        print(f"✓ Loading existing clinical data: {clinical_file}")
        clinical_df = pd.read_csv(clinical_file)

        # Filter to available patients
        clinical_df = clinical_df[clinical_df['case_id'].isin(patient_ids)]
        print(f"  Matched {len(clinical_df)}/{len(patient_ids)} patients")

        return clinical_df

    print("⚠️  Clinical data not found")
    print("\nFor real TCGA clinical data, download from:")
    print(f"  https://portal.gdc.cancer.gov/projects/{cancer_type}")
    print(f"  Save as: {clinical_file}")
    print("\n  Required columns: case_id, survival_months, censorship")
    print("\nCreating synthetic data for testing...")

    return create_synthetic_clinical(patient_ids, clinical_file)


def process_patient_data(feature_data, clinical_df, output_dir):
    """Process individual patient data into clean tensors."""
    print("\n[Processing Patient Data]")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    processed_patients = []

    for patient_id in tqdm(feature_data['patient_ids'], desc="Processing"):
        patient_file = output_dir / f"{patient_id}.pt"

        if patient_file.exists():
            processed_patients.append(patient_id)
            continue

        try:
            # Load histology features
            feat_path = feature_data['feature_paths'][patient_id]
            hist_data = torch.load(feat_path, map_location='cpu')

            # Handle different possible formats
            if isinstance(hist_data, dict):
                hist_features = hist_data.get('features', hist_data.get('embeddings', None))
            else:
                hist_features = hist_data

            if hist_features is None:
                print(f"\n⚠️  Skipping {patient_id}: could not extract features")
                continue

            # Ensure correct shape [N_patches, feature_dim]
            if hist_features.dim() == 3:
                hist_features = hist_features.squeeze(0)

            # Get clinical data
            patient_row = clinical_df[clinical_df['case_id'] == patient_id]
            if len(patient_row) == 0:
                print(f"\n⚠️  Skipping {patient_id}: no clinical data")
                continue

            patient_row = patient_row.iloc[0]

            # Extract genomic features (all columns except metadata)
            genomic_cols = [c for c in clinical_df.columns
                           if c not in ['case_id', 'survival_months', 'censorship', 'age', 'stage']]

            genomic_values = np.array([patient_row[col] for col in genomic_cols], dtype=np.float32)
            genomic_features = torch.from_numpy(genomic_values)

            survival_time = float(patient_row['survival_months'])
            event = 1 - int(patient_row['censorship'])  # 1=died, 0=censored

            # Package data
            patient_data = {
                'patient_id': patient_id,
                'hist_features': hist_features,       # [N_patches, 1536]
                'genomic_features': genomic_features,  # [N_genes]
                'survival_time': survival_time,
                'event': event,
            }

            # Save
            torch.save(patient_data, patient_file)
            processed_patients.append(patient_id)

        except Exception as e:
            print(f"\n❌ Error processing {patient_id}: {e}")
            continue

    print(f"\n✓ Successfully processed {len(processed_patients)}/{len(feature_data['patient_ids'])} patients")

    return processed_patients


def create_cv_splits(patient_ids, clinical_df, output_dir, n_folds=5):
    """Create stratified K-fold cross-validation splits."""
    print(f"\n[Creating {n_folds}-Fold CV Splits]")
    print("="*60)

    # Filter to processed patients
    df = clinical_df[clinical_df['case_id'].isin(patient_ids)].copy()

    # Create stratification variable (survival quantile + event status)
    df['survival_bin'] = pd.qcut(
        df['survival_months'],
        q=4,
        labels=False,
        duplicates='drop'
    )
    df['strat_label'] = (
        df['survival_bin'].astype(str) + '_' +
        df['censorship'].astype(str)
    )

    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_label'])):
        splits[f'fold_{fold}'] = {
            'train': df.iloc[train_idx]['case_id'].tolist(),
            'val': df.iloc[val_idx]['case_id'].tolist(),
        }

    # Save
    splits_file = Path(output_dir) / 'cv_splits.pkl'
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)

    print(f"✓ Created {len(splits)} folds:")
    for fold_name, split in splits.items():
        print(f"    {fold_name}: {len(split['train'])} train, {len(split['val'])} val")

    return splits


def create_metadata(processed_patients, clinical_df, cancer_type, output_dir):
    """Create metadata file."""
    print("\n[Creating Metadata]")
    print("="*60)

    output_dir = Path(output_dir)

    # Get feature dimensions from first patient
    first_patient = processed_patients[0]
    first_data = torch.load(output_dir / f"{first_patient}.pt")

    genomic_cols = [c for c in clinical_df.columns
                   if c not in ['case_id', 'survival_months', 'censorship', 'age', 'stage']]

    metadata = {
        'n_patients': len(processed_patients),
        'cancer_type': cancer_type,
        'feature_dim': int(first_data['hist_features'].shape[1]),
        'genomic_dim': len(genomic_cols),
        'n_folds': 5,
        'creation_date': pd.Timestamp.now().isoformat(),
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    return metadata


def compress_for_cluster(output_dir, cancer_type):
    """Compress preprocessed data for cluster upload."""
    print("\n[Compressing for Cluster]")
    print("="*60)

    tar_filename = f"eccv2025_data_{cancer_type}.tar.gz"
    tar_path = Path(tar_filename)

    print(f"Creating {tar_filename}...")

    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(output_dir, arcname='preprocessed_data')

    tar_size = tar_path.stat().st_size / (1024**3)

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n📦 Package: {tar_path}")
    print(f"   Size: {tar_size:.2f} GB")
    print(f"\n📤 Upload to cluster:")
    print(f"   scp {tar_path} username@cluster:/scratch/path/")
    print(f"\n🖥️  On cluster, extract:")
    print(f"   tar -xzf {tar_filename}")
    print(f"   # Data will be in preprocessed_data/")

    return tar_path


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess TCGA data for ECCV 2025 project'
    )
    parser.add_argument(
        '--cancer_type',
        default='TCGA-BRCA',
        help='Cancer type (TCGA-BRCA, TCGA-LUAD, TCGA-KIRC)'
    )
    parser.add_argument(
        '--subset_size',
        type=int,
        default=150,
        help='Number of patients to process (default: 150)'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='Data directory (default: data/)'
    )
    parser.add_argument(
        '--output_dir',
        default='preprocessed_data',
        help='Output directory (default: preprocessed_data/)'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip download if data exists'
    )
    args = parser.parse_args()

    print("="*60)
    print("ECCV 2025 - LOCAL PREPROCESSING")
    print("="*60)
    print(f"Cancer Type:  {args.cancer_type}")
    print(f"Subset Size:  {args.subset_size}")
    print(f"Data Dir:     {args.data_dir}")
    print(f"Output Dir:   {args.output_dir}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Authenticate
    if not args.skip_download:
        authenticated = authenticate_hf()
        if not authenticated:
            print("\n⚠️  Continuing without new authentication")

    # Step 2: Download features
    if not args.skip_download:
        feature_data = download_tcga_features(
            args.cancer_type,
            data_dir,
            args.subset_size
        )

        if feature_data is None:
            print("\n❌ Failed to download features")
            return 1
    else:
        print("\n⚠️  Skipping download (using existing data)")
        # Load existing data
        # This would need to be implemented based on your data structure
        print("❌ --skip_download not yet implemented")
        return 1

    # Step 3: Load/create clinical data
    clinical_df = load_clinical_data(
        args.cancer_type,
        data_dir,
        feature_data['patient_ids']
    )

    if clinical_df is None or len(clinical_df) == 0:
        print("\n❌ No clinical data available")
        return 1

    # Step 4: Process patients
    processed_patients = process_patient_data(
        feature_data,
        clinical_df,
        output_dir
    )

    if len(processed_patients) == 0:
        print("\n❌ No patients successfully processed")
        return 1

    # Step 5: Create CV splits
    create_cv_splits(processed_patients, clinical_df, output_dir)

    # Step 6: Create metadata
    create_metadata(processed_patients, clinical_df, args.cancer_type, output_dir)

    # Step 7: Compress
    compress_for_cluster(output_dir, args.cancer_type)

    print("\n✅ Done! Ready for cluster training.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

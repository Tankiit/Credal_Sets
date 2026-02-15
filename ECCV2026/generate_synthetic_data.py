#!/usr/bin/env python3
"""
Generate Synthetic TCGA Data for ECCV 2025 Project
==================================================
Creates synthetic multi-modal cancer data for testing when real data is unavailable.

Usage:
    python generate_synthetic_data.py --cancer_types TCGA-BRCA TCGA-LUAD --subset_size 500

Output:
    preprocessed_data/
    └── eccv2025_data_combined.tar.gz  # Ready for cluster upload
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
from sklearn.model_selection import StratifiedKFold


def create_synthetic_patients(cancer_type, n_patients, seed=42):
    """Create synthetic patient data."""
    print(f"\n[Creating {n_patients} synthetic patients for {cancer_type}]")
    print("="*60)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate patient IDs
    cancer_code = cancer_type.split('-')[1]  # BRCA from TCGA-BRCA
    patient_ids = [f"{cancer_type}-{i:07d}" for i in range(n_patients)]

    patients_data = []

    for patient_id in tqdm(patient_ids, desc=f"Generating {cancer_type}"):
        # Variable number of patches per patient
        n_patches = np.random.randint(50, 300)

        # Generate histology features [N_patches, 1536]
        hist_features = torch.randn(n_patches, 1536)

        # Generate genomic features [100 genes]
        genomic_features = np.random.randn(100).astype(np.float32)

        # Generate survival data with realistic distributions
        # Use different parameters for different cancer types
        if 'BRCA' in cancer_type:
            survival_months = np.clip(np.random.exponential(scale=40), 0, 200)
            censorship_prob = 0.35  # 35% died
        elif 'LUAD' in cancer_type:
            survival_months = np.clip(np.random.exponential(scale=25), 0, 150)
            censorship_prob = 0.50  # 50% died
        elif 'KIRC' in cancer_type:
            survival_months = np.clip(np.random.exponential(scale=50), 0, 250)
            censorship_prob = 0.30  # 30% died
        else:
            survival_months = np.clip(np.random.exponential(scale=30), 0, 120)
            censorship_prob = 0.40  # 40% died

        event = np.random.choice([0, 1], p=[1-censorship_prob, censorship_prob])

        patient_data = {
            'patient_id': patient_id,
            'hist_features': hist_features,
            'genomic_features': torch.from_numpy(genomic_features),
            'survival_time': float(survival_months),
            'event': int(event),
        }

        patients_data.append(patient_data)

    print(f"✓ Created {len(patients_data)} patients")
    return patients_data


def save_patients(patients_data, cancer_type, output_dir):
    """Save patient tensor files."""
    print(f"\n[Saving {cancer_type} Patient Data]")
    print("="*60)

    output_dir = Path(output_dir) / cancer_type
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_ids = []
    for patient_data in tqdm(patients_data, desc="Saving"):
        patient_file = output_dir / f"{patient_data['patient_id']}.pt"
        torch.save(patient_data, patient_file)
        saved_ids.append(patient_data['patient_id'])

    print(f"✓ Saved {len(saved_ids)} patient files to {output_dir}")
    return saved_ids


def create_clinical_dataframe(patients_data, cancer_type):
    """Create clinical dataframe from patient data."""
    print(f"\n[Creating Clinical DataFrame for {cancer_type}]")
    print("="*60)

    clinical_rows = []
    for patient_data in patients_data:
        row = {
            'case_id': patient_data['patient_id'],
            'survival_months': patient_data['survival_time'],
            'censorship': 1 - patient_data['event'],  # Convert back to censorship
            'age': np.clip(np.random.normal(60, 12), 18, 90),
            'stage': np.random.choice(['I', 'II', 'III', 'IV']),
            'cancer_type': cancer_type,
        }

        # Add gene columns
        genomic = patient_data['genomic_features'].numpy()
        for i, gene_val in enumerate(genomic):
            row[f'gene_{i}'] = gene_val

        clinical_rows.append(row)

    clinical_df = pd.DataFrame(clinical_rows)
    print(f"✓ Created clinical dataframe with {len(clinical_df)} rows")

    return clinical_df


def create_combined_cv_splits(all_patient_ids, all_clinical_dfs, output_dir, n_folds=5):
    """Create combined CV splits across cancer types."""
    print(f"\n[Creating Combined {n_folds}-Fold CV Splits]")
    print("="*60)

    # Combine all clinical data
    combined_clinical = pd.concat(all_clinical_dfs, ignore_index=True)

    # Create stratification variable
    combined_clinical['survival_bin'] = pd.qcut(
        combined_clinical['survival_months'],
        q=4,
        labels=False,
        duplicates='drop'
    )
    combined_clinical['strat_label'] = (
        combined_clinical['cancer_type'].astype(str) + '_' +
        combined_clinical['survival_bin'].astype(str) + '_' +
        combined_clinical['censorship'].astype(str)
    )

    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_clinical, combined_clinical['strat_label'])):
        splits[f'fold_{fold}'] = {
            'train': combined_clinical.iloc[train_idx]['case_id'].tolist(),
            'val': combined_clinical.iloc[val_idx]['case_id'].tolist(),
        }

    # Save
    splits_file = Path(output_dir) / 'cv_splits.pkl'
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)

    print(f"✓ Created {len(splits)} folds:")
    for fold_name, split in splits.items():
        # Count patients per cancer type
        train_types = combined_clinical[combined_clinical['case_id'].isin(split['train'])]['cancer_type'].value_counts().to_dict()
        val_types = combined_clinical[combined_clinical['case_id'].isin(split['val'])]['cancer_type'].value_counts().to_dict()

        print(f"    {fold_name}: {len(split['train'])} train {train_types}, {len(split['val'])} val {val_types}")

    return splits, combined_clinical


def create_metadata(all_patients_data, all_clinical_dfs, cancer_types, output_dir):
    """Create metadata file."""
    print("\n[Creating Metadata]")
    print("="*60)

    output_dir = Path(output_dir)

    # Get feature dimensions from first patient
    first_data = all_patients_data[0][0]
    n_genes = len(first_data['genomic_features'])

    # Count patients per cancer type
    patients_per_type = {
        cancer_type: len(patients_data)
        for cancer_type, patients_data in zip(cancer_types, all_patients_data)
    }

    metadata = {
        'n_patients_total': sum(len(p) for p in all_patients_data),
        'cancer_types': cancer_types,
        'patients_per_type': patients_per_type,
        'feature_dim': int(first_data['hist_features'].shape[1]),
        'genomic_dim': n_genes,
        'n_folds': 5,
        'creation_date': pd.Timestamp.now().isoformat(),
        'note': 'Synthetic data for testing',
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    return metadata


def compress_for_cluster(output_dir, name="combined"):
    """Compress preprocessed data for cluster upload."""
    print("\n[Compressing for Cluster]")
    print("="*60)

    tar_filename = f"eccv2025_data_{name}.tar.gz"
    tar_path = Path(tar_filename)

    print(f"Creating {tar_filename}...")

    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(output_dir, arcname='preprocessed_data')

    tar_size = tar_path.stat().st_size / (1024**2)  # MB

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n📦 Package: {tar_path}")
    print(f"   Size: {tar_size:.2f} MB")
    print(f"\n📤 Upload to cluster:")
    print(f"   scp {tar_path} username@cluster:/scratch/path/")
    print(f"\n🖥️  On cluster, extract:")
    print(f"   tar -xzf {tar_filename}")
    print(f"   # Data will be in preprocessed_data/")

    return tar_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic TCGA data for ECCV 2025 project'
    )
    parser.add_argument(
        '--cancer_types',
        nargs='+',
        default=['TCGA-BRCA', 'TCGA-LUAD'],
        help='Cancer types to generate (default: TCGA-BRCA TCGA-LUAD)'
    )
    parser.add_argument(
        '--subset_size',
        type=int,
        default=500,
        help='Number of patients per cancer type (default: 500)'
    )
    parser.add_argument(
        '--output_dir',
        default='preprocessed_data',
        help='Output directory (default: preprocessed_data/)'
    )
    args = parser.parse_args()

    print("="*60)
    print("ECCV 2025 - SYNTHETIC DATA GENERATION")
    print("="*60)
    print(f"Cancer Types: {args.cancer_types}")
    print(f"Patients per type: {args.subset_size}")
    print(f"Total patients: {args.subset_size * len(args.cancer_types)}")
    print(f"Output Dir: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_patients_data = []
    all_patient_ids = []
    all_clinical_dfs = []

    # Generate data for each cancer type
    seed = 42
    for cancer_type in args.cancer_types:
        patients_data = create_synthetic_patients(cancer_type, args.subset_size, seed)
        all_patients_data.append(patients_data)

        # Save patient files
        patient_ids = save_patients(patients_data, cancer_type, output_dir)
        all_patient_ids.extend(patient_ids)

        # Create clinical dataframe
        clinical_df = create_clinical_dataframe(patients_data, cancer_type)
        all_clinical_dfs.append(clinical_df)

        seed += 1  # Different seed for each cancer type

    # Create combined CV splits
    create_combined_cv_splits(all_patient_ids, all_clinical_dfs, output_dir)

    # Create metadata
    create_metadata(all_patients_data, all_clinical_dfs, args.cancer_types, output_dir)

    # Compress
    cancer_suffix = '_'.join([ct.split('-')[1] for ct in args.cancer_types])
    compress_for_cluster(output_dir, cancer_suffix)

    print("\n✅ Done! Ready for cluster training.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

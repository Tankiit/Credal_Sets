#!/usr/bin/env python3
"""
Test/Demo Preprocessing Script for ECCV 2025 Project
====================================================
Creates synthetic data to demonstrate the preprocessing pipeline.
"""

import os
import pickle
import tarfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import json

def create_synthetic_tcga_data(n_patients=50):
    """Create synthetic TCGA-like data for testing"""
    print(f"Creating {n_patients} synthetic patients...")

    np.random.seed(42)

    # Generate synthetic patient IDs
    patient_ids = [f"TCGA-BRCA-{i:07d}" for i in range(n_patients)]

    # Generate synthetic histology features [N_patches, 1536]
    # Each patient has variable number of patches
    hist_features = {}
    feature_paths = {}
    data_dir = Path("data/processed/hist_features")
    data_dir.mkdir(parents=True, exist_ok=True)

    for pid in patient_ids:
        n_patches = np.random.randint(50, 200)
        features = torch.randn(n_patches, 1536)
        hist_features[pid] = features

        # Save to simulated h5 file
        h5_path = data_dir / f"{pid}.h5"
        # For demo, just save as .pt
        torch.save(features, h5_path.with_suffix('.pt'))
        feature_paths[pid] = str(h5_path.with_suffix('.pt'))

    # Generate synthetic genomic data
    n_genes = 100
    genomic_data = []
    for pid in patient_ids:
        row = {'case_id': pid}
        for i in range(n_genes):
            row[f'gene_{i}'] = float(np.random.randn())
        row['survival_months'] = float(np.random.uniform(10, 200))
        row['censorship'] = int(np.random.randint(0, 2))
        genomic_data.append(row)

    genomic_df = pd.DataFrame(genomic_data)

    # Create processed data package
    data = {
        'patient_ids': patient_ids,
        'feature_paths': feature_paths,
        'genomic_df': genomic_df,
    }

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / 'tcga_brca_processed.pkl', 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Created synthetic data with {n_patients} patients")
    return data

def main():
    print("="*60)
    print("DEMO PREPROCESSING FOR ECCV 2025")
    print("="*60)

    # Step 1: Create synthetic data
    print("\n[1/5] Creating Synthetic TCGA Data...")
    data = create_synthetic_tcga_data(n_patients=50)

    # Step 2: Load and verify
    print("\n[2/5] Loading Processed Data...")
    print(f"Loaded {len(data['patient_ids'])} patients")

    # Step 3: Create clean tensor dataset
    print("\n[3/5] Creating Clean Tensor Dataset...")

    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)

    # For each patient, load features and save as single tensor file
    for patient_id in tqdm(data['patient_ids'], desc="Processing patients"):
        patient_file = output_dir / f"{patient_id}.pt"

        if patient_file.exists():
            continue  # Skip if already processed

        # Load histology features (from our synthetic data)
        hist_features = torch.load(data['feature_paths'][patient_id])

        # Get genomic features and labels
        row = data['genomic_df'][data['genomic_df']['case_id'] == patient_id].iloc[0]
        genomic_cols = [c for c in data['genomic_df'].columns
                       if c not in ['case_id', 'survival_months', 'censorship']]

        genomic_values = np.array([row[col] for col in genomic_cols], dtype=np.float32)
        genomic_features = torch.from_numpy(genomic_values)
        survival_time = float(row['survival_months'])
        event = 1 - int(row['censorship'])  # Convert to event indicator

        # Package into single tensor dict
        patient_data = {
            'patient_id': patient_id,
            'hist_features': hist_features,      # [N_patches, 1536]
            'genomic_features': genomic_features, # [N_genes]
            'survival_time': survival_time,
            'event': event,
        }

        # Save as single .pt file
        torch.save(patient_data, patient_file)

    # Step 4: Create split manifests
    print("\n[4/5] Creating CV Split Manifests...")

    # 5-fold stratified splits
    from sklearn.model_selection import StratifiedKFold

    df = data['genomic_df']
    df['survival_bin'] = pd.qcut(df['survival_months'], q=4, labels=False, duplicates='drop')
    df['strat_label'] = df['survival_bin'].astype(str) + '_' + df['censorship'].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_label'])):
        splits[f'fold_{fold}'] = {
            'train': df.iloc[train_idx]['case_id'].tolist(),
            'val': df.iloc[val_idx]['case_id'].tolist(),
        }

    # Save splits
    with open(output_dir / 'cv_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)

    print(f"Created {len(splits)} folds")
    for fold_name, split in splits.items():
        print(f"  {fold_name}: {len(split['train'])} train, {len(split['val'])} val")

    # Step 5: Create metadata
    metadata = {
        'n_patients': len(data['patient_ids']),
        'cancer_types': ['TCGA-BRCA'],
        'feature_dim': 1536,
        'genomic_dim': len(genomic_cols),
        'n_folds': 5,
        'creation_date': pd.Timestamp.now().isoformat(),
        'note': 'This is synthetic demo data',
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 6: Compress for upload
    print("\n[5/5] Compressing for Cluster Upload...")

    tar_path = Path("eccv2025_demo_data.tar.gz")

    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(output_dir, arcname='preprocessed_data')

    # Print summary
    tar_size = tar_path.stat().st_size / (1024**2)  # MB instead of GB

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n📦 Package ready: {tar_path}")
    print(f"   Size: {tar_size:.2f} MB")
    print(f"\n📊 Summary:")
    print(f"   Patients: {metadata['n_patients']}")
    print(f"   Feature dim: {metadata['feature_dim']}")
    print(f"   Genomic dim: {metadata['genomic_dim']}")
    print(f"   Folds: {metadata['n_folds']}")

    # Show a sample patient
    sample_patient = torch.load(output_dir / f"{data['patient_ids'][0]}.pt")
    print(f"\n🔍 Sample patient structure:")
    print(f"   Patient ID: {sample_patient['patient_id']}")
    print(f"   Histology features: {sample_patient['hist_features'].shape}")
    print(f"   Genomic features: {sample_patient['genomic_features'].shape}")
    print(f"   Survival time: {sample_patient['survival_time']:.1f} months")
    print(f"   Event: {sample_patient['event']}")

    print("\n📤 For production use with real TCGA data:")
    print(f"   1. Obtain real TCGA data")
    print(f"   2. Use local_preprocess.py instead")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

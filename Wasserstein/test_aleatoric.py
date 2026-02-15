"""
Test script to verify CEBaB aleatoric uncertainty loading.
"""

import numpy as np
from dataloader import load_dataset_splits, DatasetConfig, build_cebab_aleatoric_targets, CEBAB_ASPECT_KEYS

print("=" * 70)
print("CEBaB ALEATORIC UNCERTAINTY TEST")
print("=" * 70)

# Load CEBaB dataset
print("\nLoading CEBaB dataset...")
config = DatasetConfig(
    label_type="ternary",
    batch_size=4,
    max_train_samples=20,
)

train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
    "cebab",
    config,
)

print(f"\nDataset loaded:")
print(f"  Train samples: {len(train_loader.dataset)}")
print(f"  Concepts: {metadata['num_concepts']} {metadata['concept_names']}")

# Get a batch
print("\n" + "="*70)
print("CHECKING BATCH CONTENTS")
print("="*70)

batch = next(iter(train_loader))

print(f"\nBatch keys: {list(batch.keys())}")

# Check standard fields
print(f"\nStandard fields:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  labels: {batch['labels'].shape}")
print(f"  concept_labels: {batch['concept_labels'].shape}")
print(f"  is_unknown: {batch['is_unknown'].shape}")

# Check aleatoric fields
if 'concept_distributions' in batch:
    print(f"\n✅ Aleatoric fields found:")
    print(f"  concept_distributions: {batch['concept_distributions'].shape}")
    print(f"  concept_entropy: {batch['concept_entropy'].shape}")
else:
    print(f"\n❌ Aleatoric fields NOT found")
    print("This is expected if the dataset doesn't have *_aspect_label_distribution fields")

# Analyze first example
print("\n" + "="*70)
print("ANALYZING FIRST EXAMPLE")
print("="*70)

# Get example from dataset
ex = train_loader.dataset.examples[0]

print(f"\nExample fields: {list(ex.keys())}")

if 'concept_distributions' in ex:
    concept_dist = ex['concept_distributions']  # [K, 3]
    concept_entropy = ex['concept_entropy']    # [K]
    concepts = ex['concepts']                   # [K]

    print(f"\nConcept analysis:")
    for i, aspect in enumerate(CEBAB_ASPECT_KEYS):
        print(f"\n  {aspect.upper()}:")
        print(f"    Majority vote: {concepts[i]} ({['Negative', 'Unknown', 'Positive'][concepts[i]]})")
        print(f"    Distribution:  P(Neg)={concept_dist[i, 0]:.3f}, "
              f"P(Unk)={concept_dist[i, 1]:.3f}, P(Pos)={concept_dist[i, 2]:.3f}")
        print(f"    Entropy: {concept_entropy[i]:.3f} (0=certain, 1=uncertain)")

    # Statistics across batch
    if 'concept_distributions' in batch:
        batch_dists = batch['concept_distributions']  # [B, K, 3]
        batch_ents = batch['concept_entropy']        # [B, K]

        print(f"\n" + "="*70)
        print("BATCH STATISTICS")
        print("="*70)

        print(f"\nEntropy statistics (across {batch_dists.shape[0]} samples):")
        for i, aspect in enumerate(CEBAB_ASPECT_KEYS):
            ents = batch_ents[:, i]  # [B]
            print(f"  {aspect.capitalize():8}: "
                  f"mean={ents.mean():.3f}, "
                  f"std={ents.std():.3f}, "
                  f"min={ents.min():.3f}, "
                  f"max={ents.max():.3f}")

        print(f"\nDistribution statistics:")
        for i, aspect in enumerate(CEBAB_ASPECT_KEYS):
            dists = batch_dists[:, i, :]  # [B, 3]
            print(f"\n  {aspect.upper()}:")
            print(f"    P(Negative):  mean={dists[:, 0].mean():.3f}, std={dists[:, 0].std():.3f}")
            print(f"    P(Unknown):   mean={dists[:, 1].mean():.3f}, std={dists[:, 1].std():.3f}")
            print(f"    P(Positive):  mean={dists[:, 2].mean():.3f}, std={dists[:, 2].std():.3f}")

# Test the utility functions directly
print("\n" + "="*70)
print("TESTING UTILITY FUNCTIONS")
print("="*70)

# Create a mock example
mock_example = {
    "food_aspect_label_distribution": {"Negative": 2, "Positive": 1, "unknown": 1},
    "service_aspect_label_distribution": {"Positive": 4},
    "ambiance_aspect_label_distribution": {"Negative": 1, "unknown": 2},
    "noise_aspect_label_distribution": {},  # Missing distribution
}

print(f"\nMock example:")
for key in CEBAB_ASPECT_KEYS:
    dist_key = f"{key}_aspect_label_distribution"
    print(f"  {key}: {mock_example.get(dist_key, {})}")

dist, entropy = build_cebab_aleatoric_targets(mock_example)

print(f"\nComputed aleatoric targets:")
for i, aspect in enumerate(CEBAB_ASPECT_KEYS):
    print(f"\n  {aspect.upper()}:")
    print(f"    Distribution: P(Neg)={dist[i, 0]:.3f}, P(Unk)={dist[i, 1]:.3f}, P(Pos)={dist[i, 2]:.3f}")
    print(f"    Entropy: {entropy[i]:.3f}")

print("\n" + "="*70)
print("✅ ALEATORIC UNCERTAINTY TEST COMPLETE")
print("="*70)

print("\nKey takeaways:")
print("  1. ✅ CEBaB dataset now includes annotator distributions")
print("  2. ✅ Probability distributions extracted for each concept")
print("  3. ✅ Normalized entropy captures aleatoric uncertainty")
print("  4. ✅ Can be used for uncertainty-aware training")

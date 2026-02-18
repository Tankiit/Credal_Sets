"""
Run theory experiments with trained CEBaB checkpoint.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os

# Import our modules
from credal_sets import CredalDROConfig, CredalDROModule, cebab_joint_config
from dataloader import load_dataset_splits
from encoder import FrozenDistilBERTEncoder
from theroy_experiments import run_all_theory_experiments

print("=" * 70)
print("THEORY EXPERIMENTS WITH CEBaB CHECKPOINT")
print("=" * 70)

# Configuration
CHECKPOINT_PATH = "checkpoints/cebab_best.pt"
OUTPUT_DIR = "results/theory_validation"
DATASET_NAME = "cebab"

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# 1. Load encoder
print("\nLoading encoder...")
encoder = FrozenDistilBERTEncoder(model_name="distilbert-base-uncased", freeze=True)
encoder = encoder.to(device)
encoder.eval()
print("Encoder loaded: distilbert-base-uncased")

# 2. Load dataset
print(f"\nLoading {DATASET_NAME} dataset...")
train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(DATASET_NAME)
print(f"Dataset loaded:")
print(f"  Val samples: {len(val_loader.dataset)}")
print(f"  Test samples: {len(test_loader.dataset)}")
print(f"  Concepts: {metadata['num_concepts']} {metadata['concept_names']}")
print(f"  Classes: {metadata['num_classes']} {metadata['class_names']}")

# 3. Get latent dimensions
print("\nExtracting latents from first batch...")
batch = next(iter(val_loader))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)

with torch.no_grad():
    latents = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_cls_only=True
    )

D = latents.shape[1]
print(f"Latent dim: {D}")

# 4. Create model config
config = cebab_joint_config(
    num_concepts=metadata['num_concepts'],
    num_classes=metadata['num_classes'],
    input_dim=D,
)

# 5. Load checkpoint
print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
# Need weights_only=False since checkpoint contains CredalDROConfig
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'unknown'):.4f}")

# 6. Create model and load weights
model = CredalDROModule(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# 7. Create a custom dataloader that yields (features, labels, concept_labels)
print("\nCreating feature dataloader for theory experiments...")

class FeatureDataLoader:
    """Wrapper that extracts features from input_ids."""
    def __init__(self, dataloader, encoder, device):
        self.dataloader = dataloader
        self.encoder = encoder
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            concept_labels = batch['concept_labels'].to(self.device)
            is_unknown = batch['is_unknown'].to(self.device)

            with torch.no_grad():
                features = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_cls_only=True
                )

            yield {
                'features': features,
                'labels': labels,
                'concept_labels': concept_labels,
                'is_unknown': is_unknown,
            }

    def __len__(self):
        return len(self.dataloader)

feature_loader = FeatureDataLoader(val_loader, encoder, device)

# 8. Run theory experiments
print("\n" + "=" * 70)
print("RUNNING THEORY EXPERIMENTS")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = run_all_theory_experiments(
    model=model,
    data_loader=feature_loader,
    lambda_dro=config.lambda_dro,
    beta=config.beta_width,
    device=device,
    output_dir=OUTPUT_DIR,
    concept_names=metadata['concept_names'],
    pgd_steps=50,
)

# 9. Print summary
print("\n" + "=" * 70)
print("THEORY EXPERIMENTS SUMMARY")
print("=" * 70)

print("\nExperiment 1 - DRO Equivalence:")
exp1 = results['experiment_1_dro_equivalence']
print(f"  Spearman ρ (PGD vs Dual): {exp1['spearman_pgd_dual']:.4f}")
print(f"  R² (PGD vs Dual): {exp1['r_squared_pgd_dual']:.4f}")
print(f"  Slope (PGD vs Dual): {exp1['slope_pgd_dual']:.4f}")
print(f"  Relative gap: {exp1['relative_gap']:.4f}")

print("\nExperiment 2 - Width-Margin Equilibrium:")
exp2 = results['experiment_2_equilibrium']
print(f"  Mean ε·|margin|: {exp2['product_mean']:.6f}")
print(f"  Predicted β/λ_dro: {exp2['predicted_product']:.6f}")
print(f"  Ratio (empirical/theory): {exp2['product_mean'] / exp2['predicted_product']:.4f}")
print(f"  Marginal cost CV: {exp2['marginal_cost_cv']:.4f}")

print("\nExperiment 3 - Margin Coupling:")
exp3 = results['experiment_3_margin_coupling']
print(f"  Spearman ρ (margin, ε): {exp3['spearman_margin_eps']:+.4f}")

print(f"\n✅ Results saved to {OUTPUT_DIR}/")
print("=" * 70)

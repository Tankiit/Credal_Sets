"""
Simple end-to-end training example with ternary concepts.
This demonstrates how to use the ternary concept implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import our modules
from credal_sets import CredalDROConfig, CredalDROModule, DROMode

print("=" * 70)
print("TERNARY CONCEPT TRAINING EXAMPLE")
print("=" * 70)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# 1. Create synthetic data (simulating CEBaB)
print("\n" + "="*70)
print("STEP 1: Create synthetic data")
print("="*70)

B_train, B_val, B_test = 100, 20, 20
D_latent = 128  # latent dimension
K = 4  # 4 concepts: food, service, ambiance, noise
J = 3  # 3 classes: negative, neutral, positive

# Generate random latent features
X_train = torch.randn(B_train, D_latent)
X_val = torch.randn(B_val, D_latent)
X_test = torch.randn(B_test, D_latent)

# Generate labels (0=negative, 1=neutral, 2=positive)
y_train = torch.randint(0, J, (B_train,))
y_val = torch.randint(0, J, (B_val,))
y_test = torch.randint(0, J, (B_test,))

# Generate ternary concept labels (0=negative, 1=unknown, 2=positive)
# Simulate high unknown rate (like CEBaB: ~60% unknown)
c_train = torch.randint(0, 3, (B_train, K))
c_val = torch.randint(0, 3, (B_val, K))
c_test = torch.randint(0, 3, (B_test, K))

# Mark unknown concepts (class 1)
is_unknown_train = (c_train == 1).float()
is_unknown_val = (c_val == 1).float()
is_unknown_test = (c_test == 1).float()

print(f"\nData splits:")
print(f"  Train: {B_train} samples")
print(f"  Val:   {B_val} samples")
print(f"  Test:  {B_test} samples")

print(f"\nConcept distribution (train):")
for c_idx in range(3):
    count = (c_train == c_idx).sum().item()
    print(f"  Class {c_idx} ({['Negative', 'Unknown', 'Positive'][c_idx]}): "
          f"{count} concepts ({count/c_train.numel()*100:.1f}%)")

print(f"\nUnknown rate: {is_unknown_train.mean().item()*100:.1f}%")

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train, c_train, is_unknown_train)
val_dataset = TensorDataset(X_val, y_val, c_val, is_unknown_val)
test_dataset = TensorDataset(X_test, y_test, c_test, is_unknown_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 2. Create model with ternary concepts
print("\n" + "="*70)
print("STEP 2: Create model with ternary concepts")
print("="*70)

config = CredalDROConfig(
    num_concepts=K,
    num_classes=J,
    input_dim=D_latent,
    concept_classes=3,  # TERNARY: neg/unk/pos
    n_heads=5,
    lambda_concept=1.0,
    lambda_dro=0.1,
    beta_width=0.01,
    mode=DROMode.JOINT,
    pgd_steps=10,
    pgd_lr=0.01,
)

model = CredalDROModule(config).to(device)

print(f"\nModel config:")
print(f"  concept_classes: {config.concept_classes} (ternary)")
print(f"  num_concepts: {config.num_concepts}")
print(f"  num_classes: {config.num_classes}")
print(f"  n_heads: {config.n_heads}")
print(f"  mode: {config.mode.value}")

# 3. Training setup
print("\n" + "="*70)
print("STEP 3: Setup training")
print("="*70)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5

print(f"\nOptimizer: Adam (lr=1e-3)")
print(f"Epochs: {num_epochs}")

# 4. Training loop
print("\n" + "="*70)
print("STEP 4: Training")
print("="*70)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0

    for batch_idx, (features, labels, concepts, is_unknown) in enumerate(train_loader):
        # Move to device
        features = features.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)
        is_unknown = is_unknown.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, labels, concepts, is_unknown)
        loss = outputs['loss_total']

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for features, labels, concepts, is_unknown in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)
            is_unknown = is_unknown.to(device)

            outputs = model(features, labels, concepts, is_unknown)
            loss = outputs['loss_total']
            val_loss += loss.item()

            # Accuracy
            preds = outputs['logits'].argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Acc:    {val_acc:.4f}")

# 5. Evaluation
print("\n" + "="*70)
print("STEP 5: Final evaluation")
print("="*70)

model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

all_epsilons = []
all_sigmas = []

with torch.no_grad():
    for features, labels, concepts, is_unknown in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)
        is_unknown = is_unknown.to(device)

        outputs = model(features, labels, concepts, is_unknown)
        test_loss += outputs['loss_total'].item()

        preds = outputs['logits'].argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

        all_epsilons.append(outputs['epsilon'])
        all_sigmas.append(outputs['sigma_sq'])

test_loss /= len(test_loader)
test_acc = test_correct / test_total

# Aggregate metrics
all_epsilons = torch.cat(all_epsilons)
all_sigmas = torch.cat(all_sigmas)

print(f"\nTest Results:")
print(f"  Loss:   {test_loss:.4f}")
print(f"  Acc:    {test_acc:.4f}")
print(f"  ε mean: {all_epsilons.mean().item():.4f}")
print(f"  ε std:  {all_epsilons.std().item():.4f}")
print(f"  σ² mean: {all_sigmas.mean().item():.6f}")

# 6. Analysis
print("\n" + "="*70)
print("STEP 6: Credal set analysis")
print("="*70)

print(f"\nRobustness radius (ε) distribution:")
print(f"  Min:  {all_epsilons.min().item():.4f}")
print(f"  25%:  {all_epsilons.quantile(0.25).item():.4f}")
print(f"  50%:  {all_epsilons.quantile(0.50).item():.4f}")
print(f"  75%:  {all_epsilons.quantile(0.75).item():.4f}")
print(f"  Max:  {all_epsilons.max().item():.4f}")

print(f"\nPer-concept uncertainty (σ²):")
sigma_mean_per_concept = all_sigmas.mean(dim=0)
for k in range(K):
    print(f"  Concept {k} ({['food', 'service', 'ambiance', 'noise'][k]}): "
          f"σ²={sigma_mean_per_concept[k].item():.6f}")

# 7. Demonstrate ternary vs binary
print("\n" + "="*70)
print("STEP 7: Compare ternary vs binary")
print("="*70)

# Binary model
config_binary = CredalDROConfig(
    num_concepts=K,
    num_classes=J,
    input_dim=D_latent,
    concept_classes=2,  # BINARY
    n_heads=5,
    lambda_concept=1.0,
)

model_binary = CredalDROModule(config_binary).to(device)

# Map ternary to binary for comparison
c_train_binary = (c_train >= 2).long()  # 0 if neg/unk, 1 if pos
c_val_binary = (c_val >= 2).long()

# Simple forward pass comparison
model_binary.eval()
with torch.no_grad():
    features = X_train[:8].to(device)
    labels = y_train[:8].to(device)
    concepts_ternary = c_train[:8].to(device)
    concepts_binary = c_train_binary[:8].to(device)
    is_unknown_batch = is_unknown_train[:8].to(device)

    out_ternary = model(features, labels, concepts_ternary, is_unknown_batch)
    out_binary = model_binary(features, labels, concepts_binary, None)

print(f"\nTernary model:")
print(f"  loss_concept: {out_ternary['loss_concept'].item():.4f}")
print(f"  epsilon mean: {out_ternary['epsilon'].mean().item():.4f}")

print(f"\nBinary model:")
print(f"  loss_concept: {out_binary['loss_concept'].item():.4f}")
print(f"  epsilon mean: {out_binary['epsilon'].mean().item():.4f}")

print(f"\n✅ Training complete!")
print(f"\nKey takeaways:")
print(f"  1. Ternary concepts properly modeled with 3-class CrossEntropy")
print(f"  2. Unknown concepts downweighted (reduces impact of uncertain labels)")
print(f"  3. Credal sets capture epistemic uncertainty via ensemble variance")
print(f"  4. Robustness radius ε(x) varies per instance based on σ²")

print("\n" + "="*70)

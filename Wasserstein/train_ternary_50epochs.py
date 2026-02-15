"""
Train ternary concept model with aleatoric head for 50 epochs and plot results.
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import our modules
from credal_sets import CredalDROConfig, CredalDROModule
from dataloader import load_dataset_splits, DatasetConfig
from encoder import FrozenDistilBERTEncoder

print("=" * 70)
print("TERNARY CONCEPT TRAINING - 50 EPOCHS WITH PLOTTING")
print("=" * 70)

# Force CPU to avoid MPS issues
device = "cpu"
print(f"\nDevice: {device}")

# 1. Load encoder
print("\nLoading encoder...")
encoder = FrozenDistilBERTEncoder(model_name="distilbert-base-uncased", freeze=True)
encoder = encoder.to(device)
encoder.eval()

# 2. Load CEBaB dataset
print("\nLoading CEBaB dataset...")
dataset_config = DatasetConfig(
    label_type="ternary",
    batch_size=16,
    max_train_samples=500,  # Use more samples for meaningful training
    max_val_samples=200,
)

train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
    "cebab",
    dataset_config,
)

print(f"\nDataset loaded:")
print(f"  Train samples: {len(train_loader.dataset)}")
print(f"  Val samples: {len(val_loader.dataset)}")
print(f"  Test samples: {len(test_loader.dataset)}")

# 3. Get latent dimensions
batch = next(iter(train_loader))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)

with torch.no_grad():
    latents = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_cls_only=True
    )

B, D = latents.shape
print(f"\nLatent dim: {D}")

# 4. Create model with ternary concepts and aleatoric head
print("\nCreating model...")
config = CredalDROConfig(
    num_concepts=metadata['num_concepts'],
    num_classes=metadata['num_classes'],
    input_dim=D,
    concept_classes=3,  # TERNARY: neg/unk/pos
    n_heads=5,
    lambda_concept=1.0,
    lambda_dro=0.1,
    beta_width=0.01,
    use_aleatoric=True,
    lambda_ale=1.0,
)

model = CredalDROModule(config).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
num_epochs = 50

print(f"\nTraining configuration:")
print(f"  Optimizer: Adam (lr=1e-3)")
print(f"  Epochs: {num_epochs}")
print(f"  Device: {device}")
print(f"  Aleatoric: Enabled (lambda_ale={config.lambda_ale})")

# 5. Training loop
print("\n" + "="*70)
print("TRAINING START")
print("="*70)

history = {
    'train_loss': [],
    'train_concept_loss': [],
    'train_task_loss': [],
    'train_robust_loss': [],
    'train_aleatoric_loss': [],
    'val_loss': [],
    'val_acc': [],
    'val_aleatoric_preds': [],
}

best_val_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = {'total': [], 'concept': [], 'task': [], 'robust': [], 'aleatoric': []}

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for batch in train_pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        concept_entropy = batch.get('concept_entropy', None)
        if concept_entropy is not None:
            concept_entropy = concept_entropy.to(device)

        # Extract features
        with torch.no_grad():
            features = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=True
            )

        # Forward pass
        optimizer.zero_grad()
        output = model(features, labels, concept_labels, is_unknown, concept_entropy=concept_entropy)

        loss_total = output['loss_total']
        loss_concept = output['loss_concept']
        loss_task = output['loss_task']
        loss_robust = output.get('loss_robust', torch.tensor(0.0).to(device))
        loss_ale = output.get('loss_ale', torch.tensor(0.0).to(device))

        loss_total.backward()
        optimizer.step()

        train_losses['total'].append(loss_total.item())
        train_losses['concept'].append(loss_concept.item())
        train_losses['task'].append(loss_task.item())
        train_losses['robust'].append(loss_robust.item() if isinstance(loss_robust, torch.Tensor) else loss_robust)
        train_losses['aleatoric'].append(loss_ale.item() if isinstance(loss_ale, torch.Tensor) else loss_ale)

        train_pbar.set_postfix({
            'loss': f"{loss_total.item():.4f}",
            'ale': f"{loss_ale.item():.4f}",
        })

    # Validation phase
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    val_aleatoric_preds = []

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            is_unknown = batch['is_unknown'].to(device)
            concept_entropy = batch.get('concept_entropy', None)
            if concept_entropy is not None:
                concept_entropy = concept_entropy.to(device)

            features = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=True
            )

            output = model(features, labels, concept_labels, is_unknown, concept_entropy=concept_entropy)

            val_losses.append(output['loss_total'].item())

            preds = output['logits'].argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            # Collect aleatoric predictions
            if 'a_hat' in output and output['a_hat'].numel() > 1:
                val_aleatoric_preds.append(output['a_hat'].mean().item())

    # Compute averages
    train_avg = {k: sum(v)/len(v) for k, v in train_losses.items()}
    val_avg_loss = sum(val_losses) / len(val_losses)
    val_acc = val_correct / val_total
    val_ale_mean = np.mean(val_aleatoric_preds) if val_aleatoric_preds else 0.0

    # Store history
    history['train_loss'].append(train_avg['total'])
    history['train_concept_loss'].append(train_avg['concept'])
    history['train_task_loss'].append(train_avg['task'])
    history['train_robust_loss'].append(train_avg['robust'])
    history['train_aleatoric_loss'].append(train_avg['aleatoric'])
    history['val_loss'].append(val_avg_loss)
    history['val_acc'].append(val_acc)
    history['val_aleatoric_preds'].append(val_ale_mean)

    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch

    # Print epoch summary every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Total: {train_avg['total']:.4f} | Task: {train_avg['task']:.4f} | Concept: {train_avg['concept']:.4f} | Robust: {train_avg['robust']:.4f} | Ale: {train_avg['aleatoric']:.4f}")
        print(f"  Val   - Loss: {val_avg_loss:.4f} | Acc: {val_acc:.4f} | Ale Mean: {val_ale_mean:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch+1})")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# 6. Plot results
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Ternary Concept Training with Aleatoric Head - 50 Epochs', fontsize=16, fontweight='bold')

epochs = range(1, num_epochs + 1)

# Plot 1: Total Loss
axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Task Loss
axes[0, 1].plot(epochs, history['train_task_loss'], label='Task Loss', color='orange', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Task Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Concept Loss
axes[0, 2].plot(epochs, history['train_concept_loss'], label='Concept Loss', color='green', linewidth=2)
axes[0, 2].set_xlabel('Epoch', fontsize=12)
axes[0, 2].set_ylabel('Loss', fontsize=12)
axes[0, 2].set_title('Concept Loss (Ternary CE)', fontsize=14, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Aleatoric Loss
axes[1, 0].plot(epochs, history['train_aleatoric_loss'], label='Aleatoric Loss', color='red', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Loss', fontsize=12)
axes[1, 0].set_title('Aleatoric Loss (MSE)', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Robust Loss
axes[1, 1].plot(epochs, history['train_robust_loss'], label='Robust Loss', color='purple', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Loss', fontsize=12)
axes[1, 1].set_title('DRO Robust Loss', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Validation Accuracy
axes[1, 2].plot(epochs, history['val_acc'], label='Val Accuracy', color='blue', linewidth=2)
axes[1, 2].axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.4f}')
axes[1, 2].set_xlabel('Epoch', fontsize=12)
axes[1, 2].set_ylabel('Accuracy', fontsize=12)
axes[1, 2].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results_50epochs.png', dpi=300, bbox_inches='tight')
print("✅ Saved plot: training_results_50epochs.png")

plt.show()

# 7. Final statistics
print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)

print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
print(f"\nFinal losses:")
print(f"  Train total: {history['train_loss'][-1]:.4f}")
print(f"  Train task: {history['train_task_loss'][-1]:.4f}")
print(f"  Train concept: {history['train_concept_loss'][-1]:.4f}")
print(f"  Train robust: {history['train_robust_loss'][-1]:.4f}")
print(f"  Train aleatoric: {history['train_aleatoric_loss'][-1]:.4f}")
print(f"  Val total: {history['val_loss'][-1]:.4f}")

# Loss improvement
improvement = history['train_loss'][0] - history['train_loss'][-1]
print(f"\nTotal loss improvement: {improvement:.4f} ({history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f})")

# 8. Test evaluation
print("\n" + "="*70)
print("TEST EVALUATION")
print("="*70)

model.eval()
test_correct = 0
test_total = 0
test_losses = []
all_epsilons = []
all_aleatoric = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        concept_entropy = batch.get('concept_entropy', None)
        if concept_entropy is not None:
            concept_entropy = concept_entropy.to(device)

        features = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_cls_only=True
        )

        output = model(features, labels, concept_labels, is_unknown, concept_entropy=concept_entropy)

        test_losses.append(output['loss_total'].item())
        all_epsilons.append(output['epsilon'])

        if 'a_hat' in output and output['a_hat'].numel() > 1:
            all_aleatoric.append(output['a_hat'].mean().item())

        preds = output['logits'].argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
test_loss = sum(test_losses) / len(test_losses)
all_epsilons = torch.cat(all_epsilons)
all_aleatoric_np = np.array(all_aleatoric)

print(f"\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  ε mean: {all_epsilons.mean().item():.4f}")
print(f"  ε std: {all_epsilons.std().item():.4f}")
if len(all_aleatoric_np) > 0:
    print(f"  Aleatoric mean: {all_aleatoric_np.mean():.4f}")
    print(f"  Aleatoric std: {all_aleatoric_np.std():.4f}")

print("\n" + "="*70)
print("✅ TRAINING AND EVALUATION COMPLETE!")
print("="*70)

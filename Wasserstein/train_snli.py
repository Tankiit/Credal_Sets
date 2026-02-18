"""
SNLI training script with ternary concepts.
"""
import torch
from torch.optim import Adam
from tqdm import tqdm
import os

from credal_sets import CredalDROConfig, CredalDROModule, snli_joint_config
from dataloader import load_dataset_splits
from encoder import FrozenDistilBERTEncoder

print("=" * 70)
print("SNLI TRAINING WITH TERNARY CONCEPTS")
print("=" * 70)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load encoder
print("\nLoading encoder...")
encoder = FrozenDistilBERTEncoder(model_name="distilbert-base-uncased", freeze=True)
encoder = encoder.to(device)
encoder.eval()

# Load SNLI dataset
print("\nLoading SNLI dataset...")
train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits("snli")
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
print(f"Concepts: {metadata['num_concepts']} {metadata['concept_names']}")
print(f"Classes: {metadata['num_classes']} {metadata['class_names']}")

# Get latent dim
batch = next(iter(train_loader))
with torch.no_grad():
    latents = encoder(batch['input_ids'].to(device), batch['attention_mask'].to(device), return_cls_only=True)
D = latents.shape[1]
print(f"Latent dim: {D}")

# Create model
config = snli_joint_config(
    num_concepts=metadata['num_concepts'],
    num_classes=metadata['num_classes'],
    input_dim=D,
    warmup_epochs=5,
    dro_ramp_epochs=5,
)
model = CredalDROModule(config).to(device)
print(config.describe())

# Try to resume from epoch 9
resume_path = os.path.join(CHECKPOINT_DIR, "snli_epoch9.pt")
start_epoch = 0
if os.path.exists(resume_path):
    print(f"\nResuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"Resumed from epoch {start_epoch}")

optimizer = Adam(model.parameters(), lr=1e-3)
if start_epoch > 0:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

# Training loop
num_epochs = 50
best_val_acc = 0.0

for epoch in range(start_epoch, num_epochs):
    epoch_weights = config.get_epoch_weights(epoch)
    phase = epoch_weights['phase']
    
    # Train
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        
        with torch.no_grad():
            features = encoder(input_ids, attention_mask, return_cls_only=True)
        
        optimizer.zero_grad()
        output = model(features, labels, concept_labels, is_unknown)
        output['loss_total'].backward()
        optimizer.step()
        train_loss += output['loss_total'].item()
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            is_unknown = batch['is_unknown'].to(device)
            
            features = encoder(input_ids, attention_mask, return_cls_only=True)
            output = model(features, labels, concept_labels, is_unknown)
            preds = output['logits'].argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs} [{phase}] - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
    
    # Save checkpoint
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
    
    if (epoch + 1) % 5 == 0 or is_best:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"snli_epoch{epoch+1}.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "snli_best.pt"))
            print(f"  New best model! Val Acc: {val_acc:.4f}")

print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")

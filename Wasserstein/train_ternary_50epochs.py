"""
Train ternary concept model with aleatoric head for 50 epochs and plot results.
"""

import os
import argparse
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
from utils.metrics import collect_eval_outputs, save_eval_outputs

print("=" * 70)
print("TERNARY CONCEPT TRAINING - 50 EPOCHS WITH PLOTTING")
print("=" * 70)

# CLI args for eval saving and quadrants
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dataset', type=str, default='cebab')
parser.add_argument('--model_name', type=str, default='credal')
parser.add_argument('--encoder_model', type=str, default='distilbert-base-uncased')
parser.add_argument('--seed', type=int, default=-1, help='Set to -1 for auto')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_length', type=int, nargs='?', const=128, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--max_val_samples', type=int, default=None)
parser.add_argument('--max_test_samples', type=int, default=None)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--skip_plots', action='store_true', help='Skip plotting to save time')
parser.add_argument('--eval_outdir', type=str, default='eval_outputs')
parser.add_argument('--eval_template', type=str, default='{dataset}_{model}_{seed}_epoch{epoch}.pt')
parser.add_argument('--save_eval', action='store_true', help='Enable saving eval outputs at end')
parser.add_argument('--quad_method', type=str, default='median', choices=['median','quantile','fixed'])
parser.add_argument('--quad_q', type=float, default=0.5)
parser.add_argument('--quad_eu_thr', type=float, default=None)
parser.add_argument('--quad_au_thr', type=float, default=None)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--save_ckpt_every', type=int, default=10)
parser.add_argument('--dro_mode', type=str, default='joint', choices=['post_hoc','fixed_eps','joint'])
parser.add_argument('--fixed_eps', type=float, default=0.1)
parser.add_argument('--ckpt_save_per_sample', action='store_true', help='Save per-sample mu/sigma/epsilon in checkpoints (can be large)')
parser.add_argument('--ckpt_per_sample_split', type=str, default='val', choices=['train','val','both'], help='Which split to store per-sample tensors for')
parser.add_argument('--ckpt_per_sample_limit', type=int, default=None, help='Limit number of samples saved per checkpoint')

try:
    args, _ = parser.parse_known_args()
except SystemExit:
    # In notebooks or environments that parse argv, just use defaults
    class _A: pass
    args = _A()
    args.dataset = 'cebab'
    args.model_name = 'credal'
    args.encoder_model = 'distilbert-base-uncased'
    args.seed = -1
    args.epochs = 50
    args.batch_size = 16
    args.max_length = 128
    args.num_workers = 0
    args.max_train_samples = None
    args.max_val_samples = None
    args.max_test_samples = None
    args.grad_accum_steps = 1
    args.skip_plots = False
    args.eval_outdir = 'eval_outputs'
    args.eval_template = '{dataset}_{model}_{seed}_epoch{epoch}.pt'
    args.save_eval = False
    args.quad_method = 'median'
    args.quad_q = 0.5
    args.quad_eu_thr = None
    args.quad_au_thr = None
    args.checkpoint_dir = 'checkpoints'
    args.save_ckpt_every = 10
    args.dro_mode = 'joint'
    args.fixed_eps = 0.1
    args.ckpt_save_per_sample = False
    args.ckpt_per_sample_split = 'val'
    args.ckpt_per_sample_limit = None

# Prefer MPS on Apple Silicon, else CUDA, else CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"\nDevice: {device}")
import platform as _pt
if _pt.system() == 'Darwin' and args.num_workers and int(args.num_workers) > 0:
    print("[Info] macOS detected — forcing num_workers=0 to avoid multiprocessing spawn issues.")
    args.num_workers = 0
os.makedirs(args.checkpoint_dir, exist_ok=True)

# 1. Load encoder
print("\nLoading encoder...")
encoder = FrozenDistilBERTEncoder(model_name=args.encoder_model, freeze=True)
encoder = encoder.to(device)
encoder.eval()

# 2. Load CEBaB dataset
print("\nLoading CEBaB dataset...")
dataset_config = DatasetConfig(
    label_type="ternary",
    batch_size=args.batch_size,
    max_length=args.max_length,
    tokenizer_name=args.encoder_model,
    num_workers=args.num_workers,
    max_train_samples=args.max_train_samples,
    max_val_samples=args.max_val_samples,
    max_test_samples=args.max_test_samples,
)

train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
    args.dataset,
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

# Apply DRO mode overrides (A/B/C)
from credal_sets import DROMode
if args.dro_mode == 'post_hoc':
    config.mode = DROMode.POST_HOC
    config.lambda_dro = 0.0
    config.beta_width = 0.0
elif args.dro_mode == 'fixed_eps':
    config.mode = DROMode.FIXED_EPS
    config.fixed_eps = float(args.fixed_eps)
else:
    config.mode = DROMode.JOINT

model = CredalDROModule(config).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
num_epochs = int(args.epochs)
grad_accum = max(1, int(args.grad_accum_steps))

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

    # Train-time epistemic summaries
    tr_mu_sum = None
    tr_sigma_sum = None
    tr_eps_sum = 0.0
    tr_eps_sqsum = 0.0
    tr_count = 0
    tr_mu_list, tr_sigma_list, tr_eps_list = [], [], []

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    optimizer.zero_grad()
    for step, batch in enumerate(train_pbar, start=1):
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
        output = model(features, labels, concept_labels, is_unknown, concept_entropy=concept_entropy)

        loss_total = output['loss_total']
        loss_concept = output['loss_concept']
        loss_task = output['loss_task']
        loss_robust = output.get('loss_robust', torch.tensor(0.0).to(device))
        loss_ale = output.get('loss_ale', torch.tensor(0.0).to(device))

        (loss_total / grad_accum).backward()
        if (step % grad_accum) == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_losses['total'].append(loss_total.item())
        train_losses['concept'].append(loss_concept.item())
        train_losses['task'].append(loss_task.item())
        train_losses['robust'].append(loss_robust.item() if isinstance(loss_robust, torch.Tensor) else loss_robust)
        train_losses['aleatoric'].append(loss_ale.item() if isinstance(loss_ale, torch.Tensor) else loss_ale)

        # Accumulate train-time mu/sigma/epsilon summaries
        with torch.no_grad():
            mu_b = output.get('mu')
            sig_b = output.get('sigma_sq')
            eps_b = output.get('epsilon')
            if isinstance(mu_b, torch.Tensor) and isinstance(sig_b, torch.Tensor) and isinstance(eps_b, torch.Tensor):
                if tr_mu_sum is None:
                    tr_mu_sum = mu_b.detach().sum(dim=0).cpu()
                    tr_sigma_sum = sig_b.detach().sum(dim=0).cpu()
                else:
                    tr_mu_sum += mu_b.detach().sum(dim=0).cpu()
                    tr_sigma_sum += sig_b.detach().sum(dim=0).cpu()
                tr_eps_sum += float(eps_b.detach().sum().item())
                tr_eps_sqsum += float((eps_b.detach()**2).sum().item())
                tr_count += mu_b.shape[0]
                if args.ckpt_save_per_sample and args.ckpt_per_sample_split in ('train','both'):
                    tr_mu_list.append(mu_b.detach().cpu())
                    tr_sigma_list.append(sig_b.detach().cpu())
                    tr_eps_list.append(eps_b.detach().cpu())

        train_pbar.set_postfix({
            'loss': f"{loss_total.item():.4f}",
            'ale': f"{loss_ale.item():.4f}",
        })

    # Flush remaining grads if any
    if (step % grad_accum) != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Validation phase
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    val_aleatoric_preds = []
    # Val-time epistemic summaries
    va_mu_sum = None
    va_sigma_sum = None
    va_eps_sum = 0.0
    va_eps_sqsum = 0.0
    va_count = 0
    va_mu_list, va_sigma_list, va_eps_list = [], [], []

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

            # Accumulate val mu/sigma/epsilon
            mu_b = output.get('mu')
            sig_b = output.get('sigma_sq')
            eps_b = output.get('epsilon')
            if isinstance(mu_b, torch.Tensor) and isinstance(sig_b, torch.Tensor) and isinstance(eps_b, torch.Tensor):
                if va_mu_sum is None:
                    va_mu_sum = mu_b.detach().sum(dim=0).cpu()
                    va_sigma_sum = sig_b.detach().sum(dim=0).cpu()
                else:
                    va_mu_sum += mu_b.detach().sum(dim=0).cpu()
                    va_sigma_sum += sig_b.detach().sum(dim=0).cpu()
                va_eps_sum += float(eps_b.detach().sum().item())
                va_eps_sqsum += float((eps_b.detach()**2).sum().item())
                va_count += mu_b.shape[0]
                if args.ckpt_save_per_sample and args.ckpt_per_sample_split in ('val','both'):
                    va_mu_list.append(mu_b.detach().cpu())
                    va_sigma_list.append(sig_b.detach().cpu())
                    va_eps_list.append(eps_b.detach().cpu())

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

    # Checkpointing
    base_name = f"{args.dataset}_{args.model_name}"
    is_best = (val_acc >= best_val_acc)
    if is_best:
        best_val_acc = val_acc
        best_epoch = epoch
    if (epoch + 1) % int(args.save_ckpt_every) == 0 or is_best:
        # Build metrics summary for checkpoint
        metrics = {}
        if tr_count > 0:
            tr_eps_mean = tr_eps_sum / tr_count
            tr_eps_std = max(tr_eps_sqsum / tr_count - tr_eps_mean**2, 0.0) ** 0.5
            metrics['train'] = {
                'mu_mean': tr_mu_sum / tr_count,
                'sigma_sq_mean': tr_sigma_sum / tr_count,
                'epsilon_mean': tr_eps_mean,
                'epsilon_std': tr_eps_std,
            }
            if args.ckpt_save_per_sample and args.ckpt_per_sample_split in ('train','both') and len(tr_mu_list) > 0:
                tr_mu_cat = torch.cat(tr_mu_list, dim=0)
                tr_sig_cat = torch.cat(tr_sigma_list, dim=0)
                tr_eps_cat = torch.cat(tr_eps_list, dim=0)
                if args.ckpt_per_sample_limit:
                    L = int(args.ckpt_per_sample_limit)
                    tr_mu_cat = tr_mu_cat[:L]
                    tr_sig_cat = tr_sig_cat[:L]
                    tr_eps_cat = tr_eps_cat[:L]
                metrics['train_per_sample'] = {
                    'mu': tr_mu_cat,
                    'sigma_sq': tr_sig_cat,
                    'epsilon': tr_eps_cat,
                }
        if va_count > 0:
            va_eps_mean = va_eps_sum / va_count
            va_eps_std = max(va_eps_sqsum / va_count - va_eps_mean**2, 0.0) ** 0.5
            metrics['val'] = {
                'mu_mean': va_mu_sum / va_count,
                'sigma_sq_mean': va_sigma_sum / va_count,
                'epsilon_mean': va_eps_mean,
                'epsilon_std': va_eps_std,
            }
            if args.ckpt_save_per_sample and args.ckpt_per_sample_split in ('val','both') and len(va_mu_list) > 0:
                va_mu_cat = torch.cat(va_mu_list, dim=0)
                va_sig_cat = torch.cat(va_sigma_list, dim=0)
                va_eps_cat = torch.cat(va_eps_list, dim=0)
                if args.ckpt_per_sample_limit:
                    L = int(args.ckpt_per_sample_limit)
                    va_mu_cat = va_mu_cat[:L]
                    va_sig_cat = va_sig_cat[:L]
                    va_eps_cat = va_eps_cat[:L]
                metrics['val_per_sample'] = {
                    'mu': va_mu_cat,
                    'sigma_sq': va_sig_cat,
                    'epsilon': va_eps_cat,
                }

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
            'metrics': metrics,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"{base_name}_epoch{epoch+1}.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, f"{base_name}_best.pt"))

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# 6. Plot results
if not args.skip_plots:
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

# 9. Save detailed evaluation outputs (optional)
if args.save_eval:
    print("\nSaving detailed evaluation outputs...")
    # seed handling
    if args.seed is not None and args.seed >= 0:
        seed = int(args.seed)
    else:
        seed = int(torch.initial_seed() % (2**32))

    eval_payload = collect_eval_outputs(
        model, encoder, test_loader, device,
        quad_method=args.quad_method,
        quad_q=args.quad_q,
        quad_eu_thr=args.quad_eu_thr,
        quad_au_thr=args.quad_au_thr,
    )

    # Conform to requested keys
    eval_payload.update({
        'eu_per_sample': eval_payload.get('epsilon'),            # [N]
        'au_per_sample': eval_payload.get('aleatoric_mean'),     # [N]
        'seed': seed,
        'epoch': num_epochs,
    })

    os.makedirs(args.eval_outdir, exist_ok=True)
    filename = args.eval_template.format(
        dataset=args.dataset,
        model=args.model_name,
        seed=seed,
        epoch=num_epochs,
    )
    out_path = os.path.join(args.eval_outdir, filename)
    saved_path = save_eval_outputs(eval_payload, model, config, out_path)
    print(f"✅ Saved eval outputs: {saved_path}")

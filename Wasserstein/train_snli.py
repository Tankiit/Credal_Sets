"""
SNLI training script with ternary concepts.
"""
import argparse
import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from credal_sets import CredalDROConfig, CredalDROModule, snli_joint_config
from dataloader import load_dataset_splits
from encoder import FrozenDistilBERTEncoder
from utils.metrics import collect_eval_outputs, save_eval_outputs

print("=" * 70)
print("SNLI TRAINING WITH TERNARY CONCEPTS")
print("=" * 70)

# CLI args for eval saving and quadrants (mirrors train_ternary_50epochs)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dataset', type=str, default='snli')
parser.add_argument('--model_name', type=str, default='credal')
parser.add_argument('--encoder_model', type=str, default='distilbert-base-uncased')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, nargs='?', const=128, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--max_val_samples', type=int, default=None)
parser.add_argument('--max_test_samples', type=int, default=None)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--save_ckpt_every', type=int, default=5)
parser.add_argument('--dro_mode', type=str, default='joint', choices=['post_hoc','fixed_eps','joint'])
parser.add_argument('--fixed_eps', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=-1, help='Set to -1 for auto')
parser.add_argument('--eval_outdir', type=str, default='eval_outputs')
parser.add_argument('--eval_template', type=str, default='{dataset}_{model}_{seed}_epoch{epoch}.pt')
parser.add_argument('--save_eval', action='store_true', help='Enable saving eval outputs at end')
parser.add_argument('--quad_method', type=str, default='median', choices=['median','quantile','fixed'])
parser.add_argument('--quad_q', type=float, default=0.5)
parser.add_argument('--quad_eu_thr', type=float, default=None)
parser.add_argument('--quad_au_thr', type=float, default=None)

try:
    args, _ = parser.parse_known_args()
except SystemExit:
    class _A: pass
    args = _A()
    args.dataset = 'snli'
    args.model_name = 'credal'
    args.seed = -1
    args.encoder_model = 'distilbert-base-uncased'
    args.epochs = 50
    args.batch_size = 32
    args.max_length = 128
    args.num_workers = 2
    args.max_train_samples = None
    args.max_val_samples = None
    args.max_test_samples = None
    args.grad_accum_steps = 1
    args.checkpoint_dir = 'checkpoints'
    args.save_ckpt_every = 5
    args.dro_mode = 'joint'
    args.fixed_eps = 0.1
    args.eval_outdir = 'eval_outputs'
    args.eval_template = '{dataset}_{model}_{seed}_epoch{epoch}.pt'
    args.save_eval = False
    args.quad_method = 'median'
    args.quad_q = 0.5
    args.quad_eu_thr = None
    args.quad_au_thr = None

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

os.makedirs(args.checkpoint_dir, exist_ok=True)
import platform as _pt
if _pt.system() == 'Darwin' and args.num_workers and int(args.num_workers) > 0:
    print("[Info] macOS detected — forcing num_workers=0 to avoid multiprocessing spawn issues.")
    args.num_workers = 0

# Load encoder
print("\nLoading encoder...")
encoder = FrozenDistilBERTEncoder(model_name=args.encoder_model, freeze=True)
encoder = encoder.to(device)
encoder.eval()

# Load SNLI dataset
print("\nLoading SNLI dataset...")
from dataloader import DatasetConfig
ds_config = DatasetConfig(
    tokenizer_name=args.encoder_model,
    batch_size=args.batch_size,
    max_length=args.max_length,
    num_workers=args.num_workers,
    max_train_samples=args.max_train_samples,
    max_val_samples=args.max_val_samples,
    max_test_samples=args.max_test_samples,
)
train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(args.dataset, ds_config)
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
from credal_sets import DROMode
# Override config based on requested DRO mode
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
print(config.describe())

# Try to resume from epoch 9 (optional convenience)
base_name = f"{args.dataset}_{args.model_name}"
resume_path = os.path.join(args.checkpoint_dir, f"{base_name}_epoch9.pt")
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
num_epochs = int(args.epochs)
grad_accum = max(1, int(args.grad_accum_steps))
best_val_acc = 0.0

for epoch in range(start_epoch, num_epochs):
    epoch_weights = config.get_epoch_weights(epoch)
    phase = epoch_weights['phase']
    
    # Train
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"), start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        
        with torch.no_grad():
            features = encoder(input_ids, attention_mask, return_cls_only=True)
        
        output = model(features, labels, concept_labels, is_unknown)
        (output['loss_total'] / grad_accum).backward()
        if (step % grad_accum) == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss += output['loss_total'].item()

    # Flush remaining grads if any
    if (step % grad_accum) != 0:
        optimizer.step()
        optimizer.zero_grad()
    
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
    
    if (epoch + 1) % int(args.save_ckpt_every) == 0 or is_best:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"{base_name}_epoch{epoch+1}.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"{base_name}_best.pt"))
            print(f"  New best model! Val Acc: {val_acc:.4f}")

print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")

# Optional: save detailed evaluation outputs with configurable quadrants
if args.save_eval:
    print("\nSaving detailed evaluation outputs...")
    # seed handling
    if args.seed is not None and args.seed >= 0:
        run_seed = int(args.seed)
    else:
        run_seed = int(torch.initial_seed() % (2**32))

    payload = collect_eval_outputs(
        model, encoder, test_loader, device,
        quad_method=args.quad_method,
        quad_q=args.quad_q,
        quad_eu_thr=args.quad_eu_thr,
        quad_au_thr=args.quad_au_thr,
    )

    payload.update({
        'eu_per_sample': payload.get('epsilon'),
        'au_per_sample': payload.get('aleatoric_mean'),
        'seed': run_seed,
        'epoch': num_epochs,
    })

    os.makedirs(args.eval_outdir, exist_ok=True)
    fname = args.eval_template.format(
        dataset=args.dataset,
        model=args.model_name,
        seed=run_seed,
        epoch=num_epochs,
    )
    out_path = os.path.join(args.eval_outdir, fname)
    saved = save_eval_outputs(payload, model, config, out_path)
    print(f"✅ Saved eval outputs: {saved}")

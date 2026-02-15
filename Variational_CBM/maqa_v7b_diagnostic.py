"""
MAQA V7b Standalone Diagnostic Script
=====================================

Purpose: Debug why ρ(AU, H[p*]) ≈ 0 instead of ≈ 1

This script runs a minimal training loop with extensive diagnostics to identify
exactly where the σ_ale → entropy mapping is breaking down.

Run: python maqa_v7b_diagnostic.py

Author: Tanmoy
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Dict, List
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'encoder_name': 'distilbert-base-uncased',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'max_length': 128,
    'projection_dim': 256,
    'dropout': 0.1,
    'min_sigma': 0.05,
    'max_sigma': 1.5,
    
    # Loss weights
    'lambda_answer': 1.0,
    'lambda_ale_entropy': 2.0,
    'lambda_ale_rank': 1.0,
    'lambda_epi_residual': 1.5,
    'lambda_decorr': 5.0,
}


# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================
@dataclass
class ModelOutput:
    mu: torch.Tensor
    sigma_epi: torch.Tensor
    sigma_ale: torch.Tensor


# =============================================================================
# MODEL V7b - WITH ENTROPY INPUT TO σ_ale
# =============================================================================
class CredalMAQA_V7b_Debug(nn.Module):
    """
    V7b model with extensive debugging.
    
    KEY: σ_ale head receives [h, entropy] as input.
    """
    
    def __init__(self, encoder, hidden_size, num_answers, config):
        super().__init__()
        self.encoder = encoder
        self.num_answers = num_answers
        self.config = config
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        proj_dim = config['projection_dim']
        
        # μ head (answer logits)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(proj_dim, num_answers)
        )
        
        # σ_epi head (from h only)
        self.sigma_epi_head = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(proj_dim, 1)
        )
        
        # σ_ale head (from h + entropy) - THE KEY!
        self.sigma_ale_head = nn.Sequential(
            nn.Linear(hidden_size + 1, proj_dim),  # +1 for entropy
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(proj_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to reasonable starting values."""
        # σ_epi → ~0.2
        nn.init.constant_(self.sigma_epi_head[-1].bias, -1.5)
        # σ_ale → ~0.5 (mean entropy)
        nn.init.constant_(self.sigma_ale_head[-1].bias, -0.4)
    
    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
    
    def forward(self, input_ids, attention_mask, entropy=None):
        h = self.encode(input_ids, attention_mask)
        
        # μ
        mu = self.mu_head(h)
        
        # σ_epi (from h only)
        sigma_epi_raw = self.sigma_epi_head(h)
        sigma_epi = F.softplus(sigma_epi_raw).squeeze(-1)
        sigma_epi = torch.clamp(sigma_epi, self.config['min_sigma'], self.config['max_sigma'])
        
        # σ_ale (from h + entropy)
        if entropy is not None:
            h_with_entropy = torch.cat([h, entropy.unsqueeze(-1)], dim=-1)
        else:
            # Placeholder for inference
            h_with_entropy = torch.cat([h, torch.zeros(h.size(0), 1, device=h.device)], dim=-1)
        
        sigma_ale_raw = self.sigma_ale_head(h_with_entropy)
        sigma_ale = F.softplus(sigma_ale_raw).squeeze(-1)
        sigma_ale = torch.clamp(sigma_ale, self.config['min_sigma'], self.config['max_sigma'])
        
        return ModelOutput(mu=mu, sigma_epi=sigma_epi, sigma_ale=sigma_ale)


# =============================================================================
# SIMPLE DATASET
# =============================================================================
class SimpleMAQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Pad p_star to max_answers
        p_star = item['p_star']
        max_ans = 10
        if len(p_star) < max_ans:
            p_star = p_star + [0.0] * (max_ans - len(p_star))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'p_star': torch.tensor(p_star, dtype=torch.float32),
            'entropy': torch.tensor(item['entropy'], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'p_star': torch.stack([b['p_star'] for b in batch]),
        'entropy': torch.stack([b['entropy'] for b in batch]),
    }


# =============================================================================
# DIAGNOSTIC LOSS
# =============================================================================
class DiagnosticLoss(nn.Module):
    """Loss with detailed diagnostics."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, output: ModelOutput, p_star, entropy):
        """Compute loss with diagnostics."""
        
        sigma_epi = output.sigma_epi
        sigma_ale = output.sigma_ale
        mu = output.mu
        
        # 1. Answer loss
        answer_loss = self._answer_loss(mu, p_star)
        
        # 2. Aleatoric entropy loss (σ_ale should match entropy)
        ale_entropy_loss = F.mse_loss(sigma_ale, entropy)
        
        # 3. Ranking loss
        ale_rank_loss = self._ranking_loss(sigma_ale, entropy)
        
        # 4. Decorrelation loss
        decorr_loss = self._correlation(sigma_epi, sigma_ale) ** 2
        
        # Total
        total = (
            self.config['lambda_answer'] * answer_loss +
            self.config['lambda_ale_entropy'] * ale_entropy_loss +
            self.config['lambda_ale_rank'] * ale_rank_loss +
            self.config['lambda_decorr'] * decorr_loss
        )
        
        # Compute correlations for diagnostics
        with torch.no_grad():
            rho_ale_entropy = self._correlation(sigma_ale, entropy)
            rho_eu_au = self._correlation(sigma_epi, sigma_ale)
        
        return {
            'total': total,
            'answer': answer_loss,
            'ale_entropy': ale_entropy_loss,
            'ale_rank': ale_rank_loss,
            'decorr': decorr_loss,
            'rho_ale_entropy': rho_ale_entropy,
            'rho_eu_au': rho_eu_au,
            # Raw values for analysis
            'sigma_ale_mean': sigma_ale.mean(),
            'sigma_ale_std': sigma_ale.std(),
            'sigma_epi_mean': sigma_epi.mean(),
            'sigma_epi_std': sigma_epi.std(),
            'entropy_mean': entropy.mean(),
            'entropy_std': entropy.std(),
        }
    
    def _answer_loss(self, mu, p_star):
        batch_size = mu.size(0)
        losses = []
        for i in range(batch_size):
            num_ans = (p_star[i] > 0).sum().item()
            if num_ans > 0:
                pred = F.softmax(mu[i, :num_ans], dim=-1)
                kl = F.kl_div(pred.log().clamp(min=-10), p_star[i, :num_ans], reduction='sum')
                losses.append(kl)
            else:
                losses.append(torch.tensor(0.0, device=mu.device))
        return torch.stack(losses).mean()
    
    def _ranking_loss(self, sigma, target, margin=0.1, num_pairs=32):
        batch_size = sigma.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=sigma.device)
        
        losses = []
        for _ in range(num_pairs):
            i, j = torch.randint(0, batch_size, (2,)).tolist()
            if i == j:
                continue
            if target[i] > target[j]:
                loss = F.relu(margin - (sigma[i] - sigma[j]))
            else:
                loss = F.relu(margin - (sigma[j] - sigma[i]))
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=sigma.device)
    
    def _correlation(self, x, y):
        x_c = x - x.mean()
        y_c = y - y.mean()
        return (x_c * y_c).mean() / ((x.std() + 1e-8) * (y.std() + 1e-8))


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================
def diagnose_gradient_flow(model, loss, output, entropy):
    """Check if gradients flow to σ_ale head."""
    
    # Zero grads
    model.zero_grad()
    
    # Compute loss
    ale_loss = F.mse_loss(output.sigma_ale, entropy)
    ale_loss.backward(retain_graph=True)
    
    # Check gradients
    print("\n📊 GRADIENT FLOW DIAGNOSTICS:")
    print("-" * 50)
    
    # σ_ale head
    for name, param in model.sigma_ale_head.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  σ_ale_head.{name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"  σ_ale_head.{name}: NO GRADIENT!")
    
    model.zero_grad()


def diagnose_input_output_relationship(model, batch, device):
    """Check if σ_ale varies with entropy input."""
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    entropy = batch['entropy'].to(device)
    
    # Forward with actual entropy
    output_with_entropy = model(input_ids, attention_mask, entropy=entropy)
    
    # Forward with different entropy values
    entropy_low = torch.zeros_like(entropy)
    entropy_high = torch.ones_like(entropy) * 2.0
    
    output_low = model(input_ids, attention_mask, entropy=entropy_low)
    output_high = model(input_ids, attention_mask, entropy=entropy_high)
    
    print("\n📊 INPUT-OUTPUT RELATIONSHIP DIAGNOSTICS:")
    print("-" * 50)
    print(f"  Entropy input: mean={entropy.mean():.4f}, std={entropy.std():.4f}")
    print(f"  σ_ale (actual entropy): mean={output_with_entropy.sigma_ale.mean():.4f}, std={output_with_entropy.sigma_ale.std():.4f}")
    print(f"  σ_ale (entropy=0):      mean={output_low.sigma_ale.mean():.4f}, std={output_low.sigma_ale.std():.4f}")
    print(f"  σ_ale (entropy=2):      mean={output_high.sigma_ale.mean():.4f}, std={output_high.sigma_ale.std():.4f}")
    
    # Check if σ_ale changes with entropy
    diff_low_high = (output_high.sigma_ale - output_low.sigma_ale).mean().item()
    print(f"\n  Δσ_ale (high - low entropy): {diff_low_high:.4f}")
    
    if abs(diff_low_high) < 0.01:
        print("  ⚠️  WARNING: σ_ale NOT RESPONDING to entropy input!")
        print("     → Check if entropy is actually being concatenated to h")
    else:
        print("  ✓ σ_ale IS responding to entropy input")
    
    # Check correlation
    rho = stats.spearmanr(output_with_entropy.sigma_ale.detach().cpu().numpy(), 
                          entropy.detach().cpu().numpy())[0]
    print(f"\n  ρ(σ_ale, entropy) in this batch: {rho:.4f}")


def diagnose_head_architecture(model):
    """Print σ_ale head architecture."""
    
    print("\n📊 σ_ale HEAD ARCHITECTURE:")
    print("-" * 50)
    
    for name, module in model.sigma_ale_head.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: Linear({module.in_features} → {module.out_features})")
            print(f"       weight shape: {module.weight.shape}")
            if module.bias is not None:
                print(f"       bias: {module.bias[:5].tolist()}...")  # First 5 values
    
    # Check first layer input size
    first_layer = model.sigma_ale_head[0]
    if isinstance(first_layer, nn.Linear):
        expected_input = first_layer.in_features
        print(f"\n  Expected input dim: {expected_input}")
        print(f"  (Should be hidden_size + 1 = 768 + 1 = 769 for DistilBERT)")


# =============================================================================
# TRAINING LOOP WITH DIAGNOSTICS
# =============================================================================
def train_with_diagnostics():
    """Main training loop with extensive diagnostics."""
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print("MAQA V7b DIAGNOSTIC TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load encoder
    print("\nLoading encoder...")
    encoder = AutoModel.from_pretrained(CONFIG['encoder_name'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_name'])
    hidden_size = encoder.config.hidden_size
    print(f"  Encoder: {CONFIG['encoder_name']}")
    print(f"  Hidden size: {hidden_size}")
    
    # Create synthetic data for testing
    print("\nCreating synthetic data...")
    data = create_synthetic_data(n_samples=500)
    print(f"  Train samples: {len(data['train'])}")
    print(f"  Val samples: {len(data['val'])}")
    
    # Check entropy distribution
    train_entropies = [d['entropy'] for d in data['train']]
    print(f"  Entropy range: [{min(train_entropies):.3f}, {max(train_entropies):.3f}]")
    print(f"  Entropy mean: {np.mean(train_entropies):.3f}")
    
    # Create datasets
    train_dataset = SimpleMAQADataset(data['train'], tokenizer, CONFIG['max_length'])
    val_dataset = SimpleMAQADataset(data['val'], tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("\nCreating model...")
    model = CredalMAQA_V7b_Debug(encoder, hidden_size, num_answers=10, config=CONFIG)
    model.to(device)
    
    # Diagnose architecture
    diagnose_head_architecture(model)
    
    # Loss and optimizer
    loss_fn = DiagnosticLoss(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Get a sample batch for diagnostics
    sample_batch = next(iter(train_loader))
    
    # Initial diagnostics
    print("\n" + "="*60)
    print("INITIAL STATE DIAGNOSTICS (Before Training)")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        sample_output = model(
            sample_batch['input_ids'].to(device),
            sample_batch['attention_mask'].to(device),
            entropy=sample_batch['entropy'].to(device)
        )
    
    diagnose_input_output_relationship(model, sample_batch, device)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        epoch_losses = []
        epoch_rho = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            p_star = batch['p_star'].to(device)
            entropy = batch['entropy'].to(device)
            
            optimizer.zero_grad()
            
            # Forward - PASS ENTROPY!
            output = model(input_ids, attention_mask, entropy=entropy)
            
            # Loss
            losses = loss_fn(output, p_star, entropy)
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(losses['total'].item())
            epoch_rho.append(losses['rho_ale_entropy'].item())
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.3f}",
                'ρ(AU,H)': f"{losses['rho_ale_entropy'].item():.3f}",
                'σ_ale': f"{losses['sigma_ale_mean'].item():.3f}±{losses['sigma_ale_std'].item():.3f}"
            })
            
            # Detailed diagnostics every 50 batches in first epoch
            if epoch == 1 and batch_idx == 0:
                print("\n" + "-"*40)
                print("FIRST BATCH DIAGNOSTICS:")
                print("-"*40)
                print(f"  σ_ale: mean={losses['sigma_ale_mean'].item():.4f}, std={losses['sigma_ale_std'].item():.4f}")
                print(f"  σ_epi: mean={losses['sigma_epi_mean'].item():.4f}, std={losses['sigma_epi_std'].item():.4f}")
                print(f"  entropy: mean={losses['entropy_mean'].item():.4f}, std={losses['entropy_std'].item():.4f}")
                print(f"  ρ(σ_ale, entropy): {losses['rho_ale_entropy'].item():.4f}")
                
                # Check gradient flow
                diagnose_gradient_flow(model, losses, output, entropy)
        
        # Epoch summary
        mean_loss = np.mean(epoch_losses)
        mean_rho = np.mean(epoch_rho)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {mean_loss:.4f}")
        print(f"  Mean ρ(AU, H): {mean_rho:.4f}")
        
        # Validation
        model.eval()
        val_rhos = []
        val_sigma_ales = []
        val_entropies = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                p_star = batch['p_star'].to(device)
                entropy = batch['entropy'].to(device)
                
                output = model(input_ids, attention_mask, entropy=entropy)
                losses = loss_fn(output, p_star, entropy)
                
                val_rhos.append(losses['rho_ale_entropy'].item())
                val_sigma_ales.extend(output.sigma_ale.cpu().numpy())
                val_entropies.extend(entropy.cpu().numpy())
        
        val_rho = np.mean(val_rhos)
        overall_rho = stats.spearmanr(val_sigma_ales, val_entropies)[0]
        
        print(f"  Val ρ(AU, H) batch mean: {val_rho:.4f}")
        print(f"  Val ρ(AU, H) overall: {overall_rho:.4f}")
        
        # Check if σ_ale is constant
        sigma_ale_var = np.var(val_sigma_ales)
        print(f"  Val σ_ale variance: {sigma_ale_var:.6f}")
        
        if sigma_ale_var < 0.001:
            print("  ⚠️  WARNING: σ_ale has nearly zero variance (constant output)!")
        
        # Detailed diagnostics at end of epoch 1 and final epoch
        if epoch == 1 or epoch == CONFIG['num_epochs']:
            print(f"\n{'='*40}")
            print(f"EPOCH {epoch} DETAILED DIAGNOSTICS")
            print(f"{'='*40}")
            diagnose_input_output_relationship(model, sample_batch, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Final analysis
    print("\nFinal σ_ale vs entropy scatter (first 20 samples):")
    print("-"*40)
    for i in range(min(20, len(val_sigma_ales))):
        print(f"  entropy={val_entropies[i]:.3f} → σ_ale={val_sigma_ales[i]:.3f}")


def create_synthetic_data(n_samples=500):
    """Create synthetic MAQA-like data with known entropy."""
    
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War 2 end?",
        "What is the largest planet?",
        "Who painted the Mona Lisa?",
    ]
    
    data = {'train': [], 'val': []}
    
    for i in range(n_samples):
        # Create varied entropy levels
        if i % 3 == 0:
            # Low entropy (clear answer)
            p_star = [0.9, 0.05, 0.05]
        elif i % 3 == 1:
            # Medium entropy
            p_star = [0.5, 0.3, 0.2]
        else:
            # High entropy (ambiguous)
            p_star = [0.35, 0.35, 0.30]
        
        # Add some noise
        p_star = np.array(p_star) + np.random.uniform(-0.05, 0.05, len(p_star))
        p_star = np.clip(p_star, 0.01, 1.0)
        p_star = p_star / p_star.sum()
        
        entropy = -np.sum(p_star * np.log(p_star + 1e-10))
        
        item = {
            'text': questions[i % len(questions)] + f" (variant {i})",
            'p_star': p_star.tolist(),
            'entropy': float(entropy),
        }
        
        if i < int(n_samples * 0.8):
            data['train'].append(item)
        else:
            data['val'].append(item)
    
    return data


# =============================================================================
# ALTERNATIVE: DIRECT LINEAR MODEL
# =============================================================================
def test_direct_linear_model():
    """
    Test an even simpler model: σ_ale = a * entropy + b
    
    If this works but the neural network doesn't, the issue is in the NN architecture.
    """
    
    print("\n" + "="*60)
    print("TESTING DIRECT LINEAR MODEL")
    print("σ_ale = softplus(a * entropy + b)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Create synthetic data
    entropies = torch.linspace(0.1, 2.0, 100).to(device)
    
    # Learnable parameters
    a = nn.Parameter(torch.tensor(1.0, device=device))
    b = nn.Parameter(torch.tensor(0.0, device=device))
    
    optimizer = torch.optim.Adam([a, b], lr=0.1)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        sigma_ale = F.softplus(a * entropies + b)
        loss = F.mse_loss(sigma_ale, entropies)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            rho = stats.spearmanr(sigma_ale.detach().cpu().numpy(), entropies.cpu().numpy())[0]
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, a={a.item():.3f}, b={b.item():.3f}, ρ={rho:.4f}")
    
    print(f"\nFinal: a={a.item():.3f}, b={b.item():.3f}")
    print("This should converge to ρ ≈ 1.0 if the architecture allows it.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # First, test if a simple linear model works
    test_direct_linear_model()
    
    # Then run full diagnostic training
    train_with_diagnostics()
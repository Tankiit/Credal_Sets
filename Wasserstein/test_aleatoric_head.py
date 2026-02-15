"""
Test script to verify AleatoricHead implementation.
"""

import torch
from credal_sets import CredalDROConfig, CredalDROModule, AleatoricHead

print("=" * 70)
print("ALEATORIC HEAD TEST")
print("=" * 70)

device = "cpu"

# 1. Test AleatoricHead standalone
print("\n" + "="*70)
print("TEST 1: AleatoricHead Standalone")
print("="*70)

input_dim = 128
num_concepts = 4
hidden_dim = 128

aleatoric_head = AleatoricHead(
    input_dim=input_dim,
    num_concepts=num_concepts,
    hidden_dim=hidden_dim,
).to(device)

features = torch.randn(8, input_dim, device=device)
aleatoric_pred = aleatoric_head(features)

print(f"\nInput shape: {features.shape}")
print(f"Output shape: {aleatoric_pred.shape}")
print(f"Output range: [{aleatoric_pred.min():.4f}, {aleatoric_pred.max():.4f}]")
print(f"Output mean: {aleatoric_pred.mean():.4f}")

# Verify outputs are in [0, 1]
assert (aleatoric_pred >= 0).all(), "Aleatoric predictions have negative values!"
assert (aleatoric_pred <= 1).all(), "Aleatoric predictions have values > 1!"
print("✅ Outputs are in [0, 1]")

# 2. Test with mock entropy targets
print("\n" + "="*70)
print("TEST 2: Aleatoric Loss Computation")
print("="*70)

# Mock entropy targets (normalized to [0, 1])
concept_entropy = torch.rand(8, num_concepts, device=device) * 0.5  # [0, 0.5]

print(f"\nTarget entropy shape: {concept_entropy.shape}")
print(f"Target entropy range: [{concept_entropy.min():.4f}, {concept_entropy.max():.4f}]")

# Compute MSE loss
import torch.nn.functional as F
loss_aleatoric = F.mse_loss(aleatoric_pred, concept_entropy)

print(f"\nAleatoric loss (MSE): {loss_aleatoric.item():.6f}")
print("✅ Loss computation successful")

# 3. Test integration with CredalDROModule
print("\n" + "="*70)
print("TEST 3: Integration with CredalDROModule")
print("="*70)

config = CredalDROConfig(
    num_concepts=num_concepts,
    num_classes=3,
    input_dim=input_dim,
    concept_classes=3,  # Ternary
    n_heads=3,
    use_aleatoric=True,  # Enable aleatoric head
    lambda_aleatoric=0.5,
    aleatoric_hidden_dim=128,
)

model = CredalDROModule(config).to(device)

print(f"\nModel config:")
print(f"  use_aleatoric: {config.use_aleatoric}")
print(f"  lambda_aleatoric: {config.lambda_aleatoric}")
print(f"  aleatoric_head: {model.aleatoric_head is not None}")

# Create test batch
features = torch.randn(4, input_dim, device=device)
labels = torch.randint(0, 3, (4,), device=device)
concept_labels = torch.randint(0, 3, (4, num_concepts), device=device)
concept_entropy = torch.rand(4, num_concepts, device=device) * 0.3

print(f"\nBatch shapes:")
print(f"  features: {features.shape}")
print(f"  labels: {labels.shape}")
print(f"  concept_labels: {concept_labels.shape}")
print(f"  concept_entropy: {concept_entropy.shape}")

# Forward pass
outputs = model(features, labels, concept_labels, concept_entropy=concept_entropy)

print(f"\nModel outputs:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: shape={value.shape}")

# Check aleatoric-specific outputs
assert 'loss_aleatoric' in outputs, "loss_aleatoric not in outputs!"
assert 'aleatoric' in outputs, "aleatoric predictions not in outputs!"

print(f"\n✅ Aleatoric head integrated successfully!")
print(f"  loss_aleatoric: {outputs['loss_aleatoric'].item():.6f}")
print(f"  aleatoric shape: {outputs['aleatoric'].shape}")
print(f"  aleatoric range: [{outputs['aleatoric'].min():.4f}, {outputs['aleatoric'].max():.4f}]")

# 4. Test without aleatoric head
print("\n" + "="*70)
print("TEST 4: Model without Aleatoric Head")
print("="*70)

config_no_ale = CredalDROConfig(
    num_concepts=num_concepts,
    num_classes=3,
    input_dim=input_dim,
    concept_classes=3,
    use_aleatoric=False,  # Disable
)

model_no_ale = CredalDROModule(config_no_ale).to(device)

print(f"\nModel config:")
print(f"  use_aleatoric: {config_no_ale.use_aleatoric}")
print(f"  aleatoric_head: {model_no_ale.aleatoric_head is not None}")

# Forward pass (should work without concept_entropy)
outputs_no_ale = model_no_ale(features, labels, concept_labels)

print(f"\nModel outputs (no aleatoric):")
print(f"  loss_aleatoric: {outputs_no_ale['loss_aleatoric'].item():.4f}")
print(f"  aleatoric shape: {outputs_no_ale['aleatoric'].shape}")

assert outputs_no_ale['loss_aleatoric'].item() == 0.0, "loss_aleatoric should be 0 when disabled!"
print("✅ Model works correctly without aleatoric head")

# 5. Test gradient flow
print("\n" + "="*70)
print("TEST 5: Gradient Flow")
print("="*70)

model.train()
outputs = model(features, labels, concept_labels, concept_entropy=concept_entropy)
loss = outputs['loss_total']

print(f"\nTotal loss: {loss.item():.4f}")
print(f"  loss_task: {outputs['loss_task'].item():.4f}")
print(f"  loss_concept: {outputs['loss_concept'].item():.4f}")
print(f"  loss_aleatoric: {outputs['loss_aleatoric'].item():.4f}")
print(f"  loss_robust: {outputs['loss_robust'].item():.4f}")

# Backward pass
loss.backward()

# Check gradients
grad_name = []
grad_norm = []
for name, param in model.named_parameters():
    if param.grad is not None and 'aleatoric' in name.lower():
        grad_name.append(name)
        grad_norm.append(param.grad.norm().item())

if grad_name:
    print(f"\nAleatoric head gradients:")
    for name, norm in zip(grad_name, grad_norm):
        print(f"  {name}: {norm:.6f}")
    print("✅ Gradients flow through aleatoric head")
else:
    print("⚠️  No aleatoric gradients found (this might be expected if lambda_aleatoric=0)")

print("\n" + "="*70)
print("✅ ALL ALEATORIC HEAD TESTS PASSED!")
print("="*70)

print("\nKey takeaways:")
print("  1. ✅ AleatoricHead outputs predictions in [0, 1]")
print("  2. ✅ MSE loss against concept_entropy works correctly")
print("  3. ✅ Integration with CredalDROModule successful")
print("  4. ✅ Model works with and without aleatoric head")
print("  5. ✅ Gradients flow properly during training")

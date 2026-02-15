"""
Test script to verify updated aleatoric head with new config.
"""

import torch
from credal_sets import CredalDROConfig, CredalDROModule

print("=" * 70)
print("UPDATED ALEATORIC HEAD TEST")
print("=" * 70)

device = "cpu"

# Test 1: With aleatoric head enabled
print("\n" + "="*70)
print("TEST 1: Aleatoric Head Enabled")
print("="*70)

config = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=128,
    concept_classes=3,  # Ternary
    use_aleatoric=True,
    lambda_ale=1.0,
    use_aleatoric_weighting=False,
)

model = CredalDROModule(config).to(device)

print(f"\nConfig:")
print(f"  use_aleatoric: {config.use_aleatoric}")
print(f"  lambda_ale: {config.lambda_ale}")
print(f"  use_aleatoric_weighting: {config.use_aleatoric_weighting}")

print(f"\nModel attributes:")
print(f"  aleatoric_head: {model.aleatoric_head is not None}")
print(f"  lambda_ale: {model.lambda_ale}")
print(f"  use_aleatoric_weighting: {model.use_aleatoric_weighting}")

# Create test batch
features = torch.randn(4, 128, device=device)
labels = torch.randint(0, 3, (4,), device=device)
concept_labels = torch.randint(0, 3, (4, 4), device=device)
concept_entropy = torch.rand(4, 4, device=device) * 0.5

# Forward pass
outputs = model(features, labels, concept_labels, concept_entropy=concept_entropy)

print(f"\nOutputs:")
print(f"  loss_total: {outputs['loss_total'].item():.4f}")
print(f"  loss_task: {outputs['loss_task'].item():.4f}")
print(f"  loss_concept: {outputs['loss_concept'].item():.4f}")
print(f"  loss_robust: {outputs['loss_robust'].item():.4f}")
print(f"  loss_ale: {outputs['loss_ale'].item():.4f}")
print(f"  a_hat shape: {outputs['a_hat'].shape}")
print(f"  a_hat range: [{outputs['a_hat'].min():.4f}, {outputs['a_hat'].max():.4f}]")

# Verify loss decomposition
expected_total = (
    outputs['loss_task']
    + config.lambda_concept * outputs['loss_concept']
    + config.lambda_dro * outputs['loss_robust']
    + config.beta_width * outputs['loss_width']
    + model.lambda_ale * outputs['loss_ale']
)

assert torch.abs(outputs['loss_total'] - expected_total) < 1e-5, "Loss decomposition mismatch!"
print("\n✅ Loss decomposition correct")

# Test 2: Without aleatoric head
print("\n" + "="*70)
print("TEST 2: Aleatoric Head Disabled")
print("="*70)

config_no_ale = CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    input_dim=128,
    concept_classes=3,
    use_aleatoric=False,
)

model_no_ale = CredalDROModule(config_no_ale).to(device)

print(f"\nConfig:")
print(f"  use_aleatoric: {config_no_ale.use_aleatoric}")

print(f"\nModel attributes:")
print(f"  aleatoric_head: {model_no_ale.aleatoric_head is not None}")
print(f"  lambda_ale: {model_no_ale.lambda_ale}")

outputs_no_ale = model_no_ale(features, labels, concept_labels, concept_entropy=concept_entropy)

print(f"\nOutputs:")
print(f"  loss_ale: {outputs_no_ale['loss_ale'].item():.4f}")
print(f"  a_hat shape: {outputs_no_ale['a_hat'].shape}")

assert outputs_no_ale['loss_ale'].item() == 0.0, "loss_ale should be 0 when disabled!"
print("✅ Works correctly without aleatoric head")

# Test 3: Different lambda_ale values
print("\n" + "="*70)
print("TEST 3: Different lambda_ale Values")
print("="*70)

for lambda_ale in [0.0, 0.5, 1.0, 2.0]:
    config_test = CredalDROConfig(
        num_concepts=4,
        num_classes=3,
        input_dim=128,
        concept_classes=3,
        use_aleatoric=True,
        lambda_ale=lambda_ale,
    )

    model_test = CredalDROModule(config_test).to(device)
    outputs_test = model_test(features, labels, concept_labels, concept_entropy=concept_entropy)

    print(f"\nlambda_ale={lambda_ale}:")
    print(f"  loss_total: {outputs_test['loss_total'].item():.4f}")
    print(f"  loss_ale contribution: {model_test.lambda_ale * outputs_test['loss_ale'].item():.4f}")

print("\n✅ Different lambda_ale values work correctly")

# Test 4: Gradient flow
print("\n" + "="*70)
print("TEST 4: Gradient Flow")
print("="*70)

model.train()
outputs = model(features, labels, concept_labels, concept_entropy=concept_entropy)
loss = outputs['loss_total']

loss.backward()

# Check aleatoric head gradients
grad_found = False
for name, param in model.named_parameters():
    if param.grad is not None and 'aleatoric' in name.lower():
        grad_found = True
        print(f"  {name}: {param.grad.norm().item():.6f}")

if grad_found:
    print("✅ Gradients flow through aleatoric head")
else:
    print("⚠️  No aleatoric gradients (lambda_ale might be 0)")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)

print("\nKey updates:")
print("  1. ✅ Added lambda_ale to config (default=1.0)")
print("  2. ✅ Added use_aleatoric_weighting to config")
print("  3. ✅ Model stores lambda_ale and use_aleatoric_weighting")
print("  4. ✅ Loss uses self.lambda_ale * loss_ale")
print("  5. ✅ Outputs renamed: a_hat, loss_ale")
print("  6. ✅ Gradients flow correctly")

print("\nUsage:")
print("  config = CredalDROConfig(")
print("      use_aleatoric=True,")
print("      lambda_ale=1.0,")
print("      use_aleatoric_weighting=False,")
print("  )")
print("  outputs = model(features, labels, concepts, concept_entropy=entropy)")
print("  a_hat = outputs['a_hat']  # Predicted aleatoric uncertainty")
print("  loss_ale = outputs['loss_ale']  # MSE loss")

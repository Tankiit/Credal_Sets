"""
In-model diagnostic - add this to your training loop

This provides real-time debugging during training to catch issues early.
"""

import torch


def diagnose_during_training(model, batch, epoch, step):
    """Call this inside your training loop to debug."""

    if step % 100 != 0:  # Only every 100 steps
        return

    print(f"\n{'='*60}")
    print(f"TRAINING DIAGNOSTIC - Epoch {epoch}, Step {step}")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        # Get device from model
        device = next(model.parameters()).device

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        concept_labels = batch.get('concept_labels')
        annotator_entropy = batch.get('annotator_entropy')

        if concept_labels is not None:
            concept_labels = concept_labels.to(device)
        if annotator_entropy is not None:
            annotator_entropy = annotator_entropy.to(device)

        # Check concept_labels format
        print(f"\nData checks:")
        print(f"  concept_labels is None: {concept_labels is None}")
        if concept_labels is not None:
            print(f"  Concept labels shape: {concept_labels.shape}")
            print(f"  Concept labels unique values: {concept_labels.unique().tolist()}")
            print(f"  Expected: [0, 1, 2] where 0=neg, 1=unknown, 2=pos")

        # Check annotator_entropy
        print(f"  annotator_entropy is None: {annotator_entropy is None}")
        if annotator_entropy is not None:
            print(f"  Annotator entropy shape: {annotator_entropy.shape}")
            print(f"  Annotator entropy range: [{annotator_entropy.min():.3f}, {annotator_entropy.max():.3f}]")
            print(f"  Annotator entropy unique values (first 10): {annotator_entropy.unique()[:10].tolist()}")
        else:
            print(f"  ⚠️  annotator_entropy is None - aleatoric supervision won't work!")

        result = model(
            input_ids,
            attention_mask,
            labels=batch.get('labels').to(device) if batch.get('labels') is not None else None,
            concept_labels=concept_labels,
            annotator_entropy=annotator_entropy
        )

        # Check all losses
        print(f"\nLosses computed:")
        loss_keys = [k for k in result.keys() if 'loss' in k.lower()]
        print(f"  All loss keys: {loss_keys}")

        for key in ['concept_bce', 'error_supervision', 'credal_kl', 'aleatoric_loss']:
            if key in result:
                print(f"  {key}: {result[key].item():.6f}")
            else:
                print(f"  {key}: NOT COMPUTED! ⚠️")

        # Check Σ_epi
        sigma_epi = result['sigma_epi']
        print(f"\nΣ_epi statistics:")
        print(f"  Shape: {sigma_epi.shape}")
        print(f"  Mean: {sigma_epi.mean().item():.6f}")
        print(f"  Std: {sigma_epi.std().item():.6f}")
        print(f"  Min: {sigma_epi.min().item():.6f}")
        print(f"  Max: {sigma_epi.max().item():.6f}")

        # Check concept labels mask
        if concept_labels is not None:
            known_mask = (concept_labels != 1)
            num_known = known_mask.sum().item()
            total = known_mask.numel()
            print(f"\nConcept mask:")
            print(f"  Known concepts: {num_known}/{total} ({100*num_known/total:.1f}%)")

            if num_known == 0:
                print("  ⚠️  NO KNOWN CONCEPTS - error supervision won't work!")

        # Check EU
        eu = result.get('epistemic')
        if eu is not None:
            print(f"\nEpistemic Uncertainty:")
            print(f"  Mean: {eu.mean().item():.6f}")
            print(f"  Std: {eu.std().item():.6f}")

        # Check AU
        au = result.get('aleatoric')
        if au is not None:
            print(f"\nAleatoric Uncertainty:")
            print(f"  Mean: {au.mean().item():.6f}")
            print(f"  Std: {au.std().item():.6f}")

        # Check config weights
        print(f"\nConfig weights:")
        if hasattr(model, 'config'):
            print(f"  error_supervision_weight: {model.config.error_supervision_weight}")
            print(f"  kl_weight: {model.config.kl_weight}")
            print(f"  aleatoric_weight: {model.config.aleatoric_weight}")
            print(f"  concept_weight: {model.config.concept_weight}")

    model.train()

    # Check gradients after backward
    # (Call this AFTER loss.backward())


def diagnose_gradients(model):
    """Call this AFTER loss.backward() to check gradients."""

    print(f"\nGradient check:")

    # Check projection gradients
    if hasattr(model, 'projection'):
        proj = model.projection
        for name in ['W_concept', 'W_epi', 'W_ale']:
            W = getattr(proj, name)
            if W.weight.grad is not None:
                grad_norm = W.weight.grad.norm().item()
                print(f"  {name} grad norm: {grad_norm:.6f}")
            else:
                print(f"  {name} grad: None ⚠️")

    # Check σ_net gradients
    if hasattr(model, 'credal_head'):
        sigma_net = model.credal_head.log_sigma_net
        total_grad = 0
        for i, layer in enumerate(sigma_net):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                total_grad += layer.weight.grad.norm().item()
        print(f"  σ_net total grad norm: {total_grad:.6f}")

        if total_grad < 1e-8:
            print("  ⚠️  NO GRADIENTS TO σ_net!")

    # Check aleatoric head gradients
    if hasattr(model, 'aleatoric_head'):
        if hasattr(model.aleatoric_head, 'net'):
            ale_net = model.aleatoric_head.net
        else:
            ale_net = model.aleatoric_head

        total_grad = 0
        for i, layer in enumerate(ale_net):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                total_grad += layer.weight.grad.norm().item()
        print(f"  aleatoric_net total grad norm: {total_grad:.6f}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_training_loop_with_diagnostics():
    """
    Example showing how to integrate diagnostics into your training loop.
    """

    # Pseudocode - adapt to your actual training loop

    """
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):

            # Diagnostic before forward (every 100 steps)
            diagnose_during_training(model, batch, epoch, step)

            # Forward
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                concept_labels=batch.get('concept_labels'),
                annotator_entropy=batch.get('annotator_entropy')
            )
            loss = outputs['loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Diagnostic after backward (every 100 steps)
            if step % 100 == 0:
                diagnose_gradients(model)

            optimizer.step()

            # Also check at end of epoch
            if step == len(dataloader) - 1:
                diagnose_during_training(model, batch, epoch, "FINAL")
    """

    pass


# ============================================================================
# QUICK CHECK FUNCTION
# ============================================================================

def quick_model_check(model, device='cpu'):
    """
    Quick sanity check before starting training.
    Run this once before your training loop.
    """

    print("="*80)
    print("QUICK MODEL CHECK")
    print("="*80)

    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 128))
    dummy_attention_mask = torch.ones(batch_size, 128)
    dummy_labels = torch.randint(0, 5, (batch_size,))
    dummy_concept_labels = torch.randint(0, 3, (batch_size, 4))
    dummy_annotator_entropy = torch.rand(batch_size, 4)

    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids.to(device),
            attention_mask=dummy_attention_mask.to(device),
            labels=dummy_labels.to(device),
            concept_labels=dummy_concept_labels.to(device),
            annotator_entropy=dummy_annotator_entropy.to(device)
        )

    print("\n✓ Forward pass successful")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Check Σ_epi
    sigma_epi = outputs['sigma_epi']
    print(f"\nΣ_epi check:")
    print(f"  Mean: {sigma_epi.mean().item():.4f}")
    print(f"  Std: {sigma_epi.std().item():.4f}")

    if sigma_epi.std().item() < 0.001:
        print("  ⚠️  Σ_epi is nearly constant at initialization")
        print("      This is OK if using small random init")
        print("      Should increase during training!")
    else:
        print("  ✓ Σ_epi varies across inputs (good!)")

    # Check required outputs
    required_keys = ['loss', 'sigma_epi', 'epistemic', 'aleatoric',
                     'concept_probs', 'predictions']

    print(f"\nRequired outputs:")
    for key in required_keys:
        if key in outputs:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} MISSING!")

    print("\n" + "="*80)
    print("Model check complete - ready to train!")
    print("="*80)

    model.train()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'diagnose_during_training',
    'diagnose_gradients',
    'quick_model_check',
    'example_training_loop_with_diagnostics',
]

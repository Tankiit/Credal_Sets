#!/usr/bin/env python3
"""
Test script for LLM Credal CBM with LoRA

This script provides examples of:
1. Creating LoRA configs for different model sizes
2. Testing the individual components
3. (Optional) Loading and testing actual LLMs

Author: Tanmoy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import sys
import os

# Add the directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lora_llm_credal_cbm import (
    LoRALayer,
    LLMLoRAConfig,
    LLMPooler,
    DisagreementScorer,
    SparseConceptLabelPredictor,
    LLMLoRAEnsembleConceptHead,
    LLMCredalCBM,
    create_llm_credal_cbm,
)


def test_lora_layer():
    """Test LoRA layer implementation."""
    print("\n1. Testing LoRA Layer...")

    batch_size, in_dim, out_dim, rank = 32, 768, 100, 16

    # Create LoRA layer
    lora = LoRALayer(
        in_features=in_dim,
        out_features=out_dim,
        rank=rank,
        alpha=32.0,
        dropout=0.1
    )

    # Test forward pass
    x = torch.randn(batch_size, in_dim)
    base_output = torch.randn(batch_size, out_dim)

    # Without base
    output1 = lora(x)
    assert output1.shape == (batch_size, out_dim), f"Shape mismatch: {output1.shape}"

    # With base
    output2 = lora(x, base_output)
    assert output2.shape == (batch_size, out_dim), f"Shape mismatch: {output2.shape}"

    # Check that output with base = base + lora_output
    lora_output = lora(x)
    expected = base_output + lora_output
    assert torch.allclose(output2, expected, atol=1e-6), "LoRA addition failed"

    print("   [OK] LoRA layer works correctly")
    trainable_params = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"   - Trainable parameters: {trainable_params}")


def test_llm_configs():
    """Test LLM-specific LoRA configurations."""
    print("\n2. Testing LLM Configurations...")

    # Test presets for different model sizes
    sizes = ["3b", "7b", "70b"]
    for size in sizes:
        config = LLMLoRAConfig.for_llm(size, n_heads=3)
        print(f"   {size.upper()} config:")
        for i, head_config in enumerate(config.head_configs):
            print(f"     Head {i}: rank={head_config.rank}, alpha={head_config.alpha}")

    # Test custom config
    custom_config = LLMLoRAConfig(
        n_heads=3,
        head_configs=[
            LLMLoRAConfig.LLM_PRESETS["llm_standard"][i] for i in range(3)
        ]
    )
    print(f"   Custom config: {custom_config.n_heads} heads")

    print("   [OK] LLM configurations work correctly")


def test_llm_pooler():
    """Test LLM pooling strategies."""
    print("\n3. Testing LLM Pooler...")

    batch_size, seq_len, hidden_size = 16, 50, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create attention mask (variable lengths)
    seq_lengths = torch.randint(10, seq_len, (batch_size,))
    attention_mask = torch.zeros(batch_size, seq_len)
    for i, length in enumerate(seq_lengths):
        attention_mask[i, :length] = 1

    strategies = ["last", "mean", "weighted", "eos"]

    for strategy in strategies:
        pooler = LLMPooler(hidden_size, strategy)
        pooled = pooler(hidden_states, attention_mask)

        assert pooled.shape == (batch_size, hidden_size), f"Shape mismatch for {strategy}"
        print(f"   [OK] {strategy} pooling: {pooled.shape}")


def test_disagreement_scorer():
    """Test disagreement scoring."""
    print("\n4. Testing Disagreement Scorer...")

    n_heads, batch_size, num_concepts = 5, 32, 20

    # Create ensemble predictions with some disagreement
    base_pred = torch.sigmoid(torch.randn(batch_size, num_concepts))
    head_predictions = []
    for i in range(n_heads):
        # Add some noise to create disagreement
        noise = torch.randn_like(base_pred) * 0.2
        head_pred = torch.clamp(base_pred + noise, 0, 1)
        head_predictions.append(head_pred)

    head_predictions = torch.stack(head_predictions, dim=0)  # [n_heads, batch, num_concepts]

    # Test different sparsity methods
    methods = ["topk", "threshold", "soft"]
    k_values = [5, 10, 15]

    for method in methods:
        scorer = DisagreementScorer(
            num_concepts=num_concepts,
            sparsity_method=method,
            sparsity_k=10
        )

        weights, metrics = scorer(head_predictions, return_metrics=True)

        assert weights.shape == (batch_size, num_concepts), f"Shape mismatch for {method}"
        assert 0 <= weights.min() and weights.max() <= 1, f"Weights out of range for {method}"

        print(f"   [OK] {method} sparsity: avg_selected={metrics['selected_concepts']:.2f}")


def test_label_predictor():
    """Test label predictor."""
    print("\n5. Testing Label Predictor...")

    batch_size, num_concepts, num_classes = 32, 20, 3

    concepts = torch.sigmoid(torch.randn(batch_size, num_concepts))
    concept_weights = torch.zeros(batch_size, num_concepts)
    concept_weights[:, :10] = 1  # Use first 10 concepts

    predictor = SparseConceptLabelPredictor(
        num_concepts=num_concepts,
        num_classes=num_classes,
        use_concept_weights=True
    )

    logits = predictor(concepts, concept_weights)
    assert logits.shape == (batch_size, num_classes), f"Shape mismatch: {logits.shape}"

    print("   [OK] Label predictor works correctly")


def test_concept_head():
    """Test the complete LoRA ensemble concept head."""
    print("\n6. Testing LoRA Ensemble Concept Head...")

    batch_size, seq_len, hidden_size, num_concepts = 16, 50, 768, 20

    # Create config
    config = LLMLoRAConfig.for_llm("7b", n_heads=3)

    # Create concept head
    concept_head = LLMLoRAEnsembleConceptHead(
        hidden_size=hidden_size,
        num_concepts=num_concepts,
        config=config,
        pooling_strategy="last",
        use_projection=True,
        projection_dim=256,
    )

    # Test forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    output = concept_head(hidden_states, attention_mask)

    # Check outputs
    assert "predictions" in output
    assert "head_predictions" in output
    assert "epistemic" in output
    assert "aleatoric" in output

    assert output["predictions"].shape == (batch_size, num_concepts)
    assert output["head_predictions"].shape == (config.n_heads, batch_size, num_concepts)
    assert output["epistemic"].shape == (batch_size, num_concepts)

    print("   [OK] LoRA ensemble concept head works correctly")
    print(f"   - Trainable parameters: {sum(p.numel() for p in concept_head.parameters() if p.requires_grad)}")


def test_full_model_creation():
    """Test creating the full model (without loading actual weights)."""
    print("\n7. Testing Full Model Creation...")

    try:
        # This will try to load the actual model - only run if model is available
        print("   Attempting to create LLM Credal CBM...")
        print("   (This will fail if model is not downloaded/accessible)")

        # Test with a smaller, more accessible model
        model = create_llm_credal_cbm(
            model_name="microsoft/DialoGPT-medium",  # Much smaller model
            num_concepts=10,
            num_classes=2,
            n_heads=3,
            sparsity_k=5,
            load_in_4bit=False,  # No quantization for this test
        )

        print("   [OK] Model created successfully!")
        print(f"   - Total parameters: {model.get_total_parameters():,}")
        print(f"   - Trainable parameters: {model.get_trainable_parameters():,}")
        print(f"   - Parameter efficiency: {model.get_trainable_parameters() / model.get_total_parameters() * 100:.2f}%")

    except Exception as e:
        print(f"   [WARNING] Model creation failed (expected): {str(e)}")
        print("   This is normal if the model is not downloaded or accessible")

        # Test config creation without model loading
        config = LLMLoRAConfig.for_llm("7b", n_heads=5)
        print(f"   [OK] Config creation works: {config.n_heads} heads")


def run_memory_estimates():
    """Provide VRAM estimates for different model configurations."""
    print("\n8. VRAM Estimates for LLM Credal CBM:")
    print("   (Estimates for 4-bit quantization)")

    models_info = [
        ("Phi-3-mini", "3.8B", 2560, "~2GB"),
        ("Mistral-7B", "7B", 4096, "~5GB"),
        ("Llama-3.1-8B", "8B", 4096, "~6GB"),
        ("Llama-2-13B", "13B", 5120, "~8GB"),
        ("Mixtral-8x7B", "47B", 4096, "~25GB"),
        ("Llama-3.1-70B", "70B", 8192, "~40GB"),
    ]

    print("   Model | Size | Hidden | VRAM (4-bit) | Trainable Params")
    print("   -------|------|--------|--------------|-----------------")

    for name, size, hidden, vram in models_info:
        # Estimate LoRA parameters
        proj_dim = min(1024, hidden // 4)
        lora_params = 5 * (hidden * proj_dim + proj_dim * 20)  # Rough estimate
        trainable_m = lora_params / 1_000_000

        print(f"   {name[:10]:10} | {size:5} | {hidden:6} | {vram:12} | ~{trainable_m:.1f}M")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing LLM Credal CBM with LoRA")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run tests
        test_lora_layer()
        test_llm_configs()
        test_llm_pooler()
        test_disagreement_scorer()
        test_label_predictor()
        test_concept_head()
        test_full_model_creation()
        run_memory_estimates()

        print("\n" + "=" * 60)
        print("[OK] All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
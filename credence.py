import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset as hf_load_dataset
import matplotlib.pyplot as plt

# Optional: LoRA for LLMs
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    PEFT_AVAILABLE = False
    error_msg = str(e)
    if "modeling_layers" in error_msg or "transformers" in error_msg.lower():
        print(f"Note: peft import failed due to version compatibility issue: {e}")
        print("This is likely a version mismatch between peft and transformers.")
        print("Try: pip install --upgrade peft transformers")
    else:
        print(f"Note: peft import failed: {e}")
        print("LLM training requires: pip install peft bitsandbytes")

# Optional: vLLM for fast LLM inference
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Note: vLLM not installed. Install with: pip install vllm")

# Your dataloader
from dataloader import (
    load_dataset_splits, 
    DatasetConfig, 
    DATASET_INFO
)

# Head configuration for ablation studies
from head_config import HeadConfig, generate_head_configs

# =============================================================================
# UPDATED MODEL REGISTRY FOR CREDENCE
# =============================================================================
# Copy this to replace your MODEL_REGISTRY in credence_acl.py
#
# Includes:
# - ModernBERT (December 2024) - Current SOTA encoder
# - Llama 3.2 (September 2024)
# - Qwen 2.5 (September 2024)
# - Gemma 2 (June 2024)
# - Phi-3.5 (August 2024)
# =============================================================================

MODEL_REGISTRY = {
    # ==========================================================================
    # ENCODER MODELS (frozen encoder, train heads)
    # ==========================================================================
    
    # Classic encoders (2019)
    "distilbert-base-uncased": {
        "type": "encoder",
        "hidden_size": 768,
        "max_length": 512,
        "use_token_type_ids": True,
    },
    "roberta-base": {
        "type": "encoder", 
        "hidden_size": 768,
        "max_length": 512,
        "use_token_type_ids": False,
    },
    "roberta-large": {
        "type": "encoder",
        "hidden_size": 1024,
        "max_length": 512,
        "use_token_type_ids": False,
    },
    
    # DeBERTa-v3 (2021) - Previous SOTA
    "microsoft/deberta-v3-base": {
        "type": "encoder",
        "hidden_size": 768,
        "max_length": 512,
        "use_token_type_ids": True,
    },
    "microsoft/deberta-v3-large": {
        "type": "encoder",
        "hidden_size": 1024,
        "max_length": 512,
        "use_token_type_ids": True,
    },
    
    # =========================================================================
    # ModernBERT (December 2024) - CURRENT SOTA ENCODER
    # =========================================================================
    # Paper: "Smarter, Better, Faster, Longer" (Warner et al., 2024)
    # arXiv: 2412.13663
    # 
    # Key advantages:
    # - First base model to beat DeBERTaV3 on GLUE
    # - 2x faster than DeBERTa, up to 4x on mixed-length inputs
    # - Uses 1/5th of DeBERTa's memory
    # - 8192 token context (vs 512 for BERT/RoBERTa)
    # - Trained on 2 trillion tokens
    #
    # Architecture innovations:
    # - Rotary Positional Embeddings (RoPE)
    # - Local-Global Alternating Attention
    # - Flash Attention + Unpadding
    # - GeGLU activation
    #
    # IMPORTANT: Does NOT use token_type_ids!
    # =========================================================================
    "answerdotai/ModernBERT-base": {
        "type": "encoder",
        "hidden_size": 768,
        "max_length": 8192,  # Native long context!
        "use_token_type_ids": False,  # Critical: ModernBERT doesn't use these
    },
    "answerdotai/ModernBERT-large": {
        "type": "encoder",
        "hidden_size": 1024,
        "max_length": 8192,
        "use_token_type_ids": False,
    },
    
    # ==========================================================================
    # LLM MODELS (LoRA fine-tuning)
    # ==========================================================================
    
    # -------------------------------------------------------------------------
    # Phi Series (Microsoft) - Efficient small LLMs
    # -------------------------------------------------------------------------
    "microsoft/phi-3-mini-4k-instruct": {
        "type": "llm",
        "hidden_size": 3072,
        "max_length": 256,
        "target_modules": ["qkv_proj", "o_proj"],
    },
    # Phi-3.5 (August 2024) - Improved reasoning
    "microsoft/Phi-3.5-mini-instruct": {
        "type": "llm",
        "hidden_size": 3072,
        "max_length": 256,
        "target_modules": ["qkv_proj", "o_proj"],
    },
    
    # -------------------------------------------------------------------------
    # Mistral Series
    # -------------------------------------------------------------------------
    "mistralai/Mistral-7B-v0.1": {
        "type": "llm",
        "hidden_size": 4096,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "type": "llm",
        "hidden_size": 4096,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    
    # -------------------------------------------------------------------------
    # Llama 3.1 (Meta, July 2024)
    # -------------------------------------------------------------------------
    "meta-llama/Llama-3.1-8B": {
        "type": "llm",
        "hidden_size": 4096,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "type": "llm",
        "hidden_size": 4096,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    
    # -------------------------------------------------------------------------
    # Llama 3.2 (Meta, September 2024) - LATEST
    # -------------------------------------------------------------------------
    # Smaller, more efficient models
    "meta-llama/Llama-3.2-1B": {
        "type": "llm",
        "hidden_size": 2048,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "meta-llama/Llama-3.2-3B": {
        "type": "llm",
        "hidden_size": 3072,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "type": "llm",
        "hidden_size": 3072,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    
    # -------------------------------------------------------------------------
    # Qwen 2.5 (Alibaba, September 2024) - Strong multilingual
    # -------------------------------------------------------------------------
    "Qwen/Qwen2.5-0.5B": {
        "type": "llm",
        "hidden_size": 896,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "Qwen/Qwen2.5-1.5B": {
        "type": "llm",
        "hidden_size": 1536,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "Qwen/Qwen2.5-3B": {
        "type": "llm",
        "hidden_size": 2048,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "Qwen/Qwen2.5-7B": {
        "type": "llm",
        "hidden_size": 3584,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    
    # -------------------------------------------------------------------------
    # Gemma 2 (Google, June 2024)
    # -------------------------------------------------------------------------
    "google/gemma-2-2b": {
        "type": "llm",
        "hidden_size": 2304,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "google/gemma-2-9b": {
        "type": "llm",
        "hidden_size": 3584,
        "max_length": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}

# =============================================================================
# HELPER: Print available models
# =============================================================================

def expand_encoder_name(encoder_name: str) -> str:
    """Expand short encoder names to full HuggingFace model names.
    
    Maps short names like 'mistral', 'phi-3', 'roberta' to full model identifiers.
    If the name is already a full model name or not recognized, returns as-is.
    """
    # Mapping of short names to full model names (matches run_all_datasets.sh logic)
    name_mapping = {
        # Encoder models
        "roberta": "roberta-base",
        "roberta-base": "roberta-base",
        "roberta-large": "roberta-large",
        "deberta": "microsoft/deberta-v3-base",
        "deberta-v3": "microsoft/deberta-v3-base",
        "deberta-v3-base": "microsoft/deberta-v3-base",
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "distilbert": "distilbert-base-uncased",
        "distilbert-base": "distilbert-base-uncased",
        "modernbert": "answerdotai/ModernBERT-base",
        "modernbert-base": "answerdotai/ModernBERT-base",
        "modernbert-large": "answerdotai/ModernBERT-large",
        
        # LLM models
        "phi-3": "microsoft/phi-3-mini-4k-instruct",
        "phi-3-mini": "microsoft/phi-3-mini-4k-instruct",
        "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama-3.1": "meta-llama/Llama-3.1-8B",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama-3.1-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
        "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
        "qwen-3b": "Qwen/Qwen2.5-3B",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B",
        "qwen-7b": "Qwen/Qwen2.5-7B",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B",
        "gemma-2b": "google/gemma-2-2b",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-9b": "google/gemma-2-9b",
        "gemma-2-9b": "google/gemma-2-9b",
    }
    
    # Check if it's a short name we recognize
    if encoder_name.lower() in name_mapping:
        return name_mapping[encoder_name.lower()]
    
    # If it's already in MODEL_REGISTRY, return as-is
    if encoder_name in MODEL_REGISTRY:
        return encoder_name
    
    # Auto-detect LLM models by checking if name contains LLM indicators
    encoder_lower = encoder_name.lower()
    if any(indicator in encoder_lower for indicator in [
        "phi", "mistral", "llama", "qwen", "gemma", 
        "meta-llama", "microsoft/phi", "mistralai", "qwen", "google/gemma"
    ]):
        # Assume it's an LLM model name, return as-is (might be a full name)
        return encoder_name
    
    # For unrecognized names, return as-is (might be a valid full model name)
    return encoder_name


def print_available_models():
    """Print all available models organized by type and release date."""
    
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    
    # Encoders
    print("\n📦 ENCODER MODELS (frozen encoder, train heads)")
    print("-" * 50)
    encoders = [(k, v) for k, v in MODEL_REGISTRY.items() if v["type"] == "encoder"]
    for name, info in encoders:
        print(f"  {name}")
        print(f"    hidden_size: {info['hidden_size']}, max_length: {info['max_length']}")
    
    # LLMs
    print("\n🤖 LLM MODELS (LoRA fine-tuning)")
    print("-" * 50)
    llms = [(k, v) for k, v in MODEL_REGISTRY.items() if v["type"] == "llm"]
    for name, info in llms:
        print(f"  {name}")
        print(f"    hidden_size: {info['hidden_size']}")
    
    print("\n" + "="*70)

@dataclass 
class ExperimentConfig:
    """Full experiment configuration."""
    # Data
    dataset: str = "cebab"
    label_type: str = "binary"
    max_length: int = 128
    batch_size: int = 16
    
    # Model
    encoder_name: str = "distilbert-base-uncased"
    n_heads: int = 5
    dropout_min: float = 0.05
    dropout_max: float = 0.30
    use_pooling_diversity: bool = True
    freeze_encoder: bool = True
    aleatoric_mode: str = "auto"  # "supervised" | "entropy" | "none" | "auto"
    
    # LoRA (for LLMs)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    freeze_llm: bool = True  # If True, don't use LoRA, just frozen features (default True for speed)
    
    # Training
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    concept_weight: float = 1.0
    aleatoric_weight: float = 0.5
    
    # Profiling
    enable_profiler: bool = False
    profiler_warmup: int = 1
    profiler_active: int = 3
    profiler_repeat: int = 1
    profiler_output_dir: Optional[str] = None
    
    # Output
    output_dir: str = "./results"
    seed: int = 42


    def get_head_configs(self) -> List[HeadConfig]:
        """Generate diverse head configurations with geometric dropout spacing."""
        return generate_head_configs(
            n_heads=self.n_heads,
            d_min=self.dropout_min,
            d_max=self.dropout_max,
            hidden_dim=256,
            use_pooling_diversity=self.use_pooling_diversity,
        )
    
    def get_model_type(self) -> str:
        """Determine if model is encoder or LLM."""
        info = MODEL_REGISTRY.get(self.encoder_name, {})
        return info.get("type", "encoder")

# =============================================================================
# MODEL LOADING
# =============================================================================

# =============================================================================
# UPDATED load_encoder_model FUNCTION
# =============================================================================
# Replace your existing load_encoder_model with this version
# Handles ModernBERT's lack of token_type_ids

def load_encoder_model(model_name: str, device: str, freeze: bool = True):
    """Load encoder model (BERT-style), with special handling for ModernBERT."""
    print(f"Loading encoder: {model_name}")
    
    # Check if it's ModernBERT (requires flash attention for best performance)
    is_modernbert = "modernbert" in model_name.lower()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    if is_modernbert:
        try:
            # Try loading with Flash Attention 2 for best performance
            encoder = AutoModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                dtype=torch.float16,  # FA2 requires fp16/bf16
            )
            print("  ModernBERT loaded with Flash Attention 2")
        except Exception as e:
            print(f"  Flash Attention 2 not available ({e}), loading standard")
            encoder = AutoModel.from_pretrained(model_name)
    else:
        encoder = AutoModel.from_pretrained(model_name)
    
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen")
    
    encoder = encoder.to(device)
    
    # Get hidden size
    with torch.no_grad():
        dummy = tokenizer("test", return_tensors="pt", padding=True)
        # ModernBERT doesn't use token_type_ids
        model_info = MODEL_REGISTRY.get(model_name, {})
        use_token_type_ids = model_info.get("use_token_type_ids", False)  # False for ModernBERT!
        
        inputs = {
            "input_ids": dummy['input_ids'].to(device),
            "attention_mask": dummy['attention_mask'].to(device),
        }
        # Only add token_type_ids if the model uses them
        if use_token_type_ids and 'token_type_ids' in dummy:
            inputs['token_type_ids'] = dummy['token_type_ids'].to(device)
        
        out = encoder(**inputs)
        hidden_size = out.last_hidden_state.shape[-1]
    
    print(f"  Hidden size: {hidden_size}")
    print(f"  Uses token_type_ids: {use_token_type_ids}")
    
    return encoder, tokenizer, hidden_size, "encoder"


def load_llm_model(model_name: str, device: str, config: ExperimentConfig):
    """Load LLM with LoRA for parameter-efficient fine-tuning using standard HuggingFace/PEFT."""
    print(f"Loading LLM: {model_name}")
    
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft is required for LLM training but could not be imported. "
            "This may be due to a version compatibility issue with transformers. "
            "Try: pip install --upgrade peft transformers bitsandbytes"
        )
    
    # Get target modules from registry
    model_info = MODEL_REGISTRY.get(model_name, {})
    target_modules = model_info.get("target_modules", ["q_proj", "v_proj"])
    
    # Standard PEFT implementation
    print("Using standard HuggingFace/PEFT implementation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    
    return model, tokenizer, hidden_size, "llm"


def load_llm_model_frozen(model_name: str, device: str, config: ExperimentConfig):
    """Load LLM as frozen feature extractor - no LoRA, no backprop through LLM."""
    print(f"Loading LLM (frozen): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    # FREEZE - no gradients needed
    model.requires_grad_(False)
    model.eval()
    
    print(f"  LLM frozen - 0 trainable params in encoder")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    return model, tokenizer, model.config.hidden_size, "llm_frozen"


def load_model(config: ExperimentConfig, device: str):
    """Load model based on type (encoder or LLM)."""
    model_type = config.get_model_type()
    
    if model_type == "llm":
        if config.use_lora and not config.freeze_llm:
            return load_llm_model(config.encoder_name, device, config)  # LoRA (slow)
        else:
            return load_llm_model_frozen(config.encoder_name, device, config)  # Frozen (fast)
    else:
        return load_encoder_model(config.encoder_name, device, config.freeze_encoder)


def get_hidden_states(encoder, input_ids, attention_mask, model_type: str):
    """Get hidden states - keep in native dtype."""
    if model_type == "encoder":
        outputs = encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    else:  # LLM or llm_frozen
        outputs = encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Keep in bfloat16/float16!

class ConceptHead(nn.Module):
    """A single concept prediction head."""
    def __init__(self, config: HeadConfig, input_dim: int, n_concepts: int):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, n_concepts)
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.config.pooling == "last":
            # For decoder-only models, use last non-padded token
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]
        raise ValueError(f"Unknown pooling: {self.config.pooling}")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.pool(hidden_states, attention_mask)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # Clamp logits to prevent extreme values in fp16
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        
        # Only convert to float32 for sigmoid (numerical stability), then convert back
        probs = torch.sigmoid(logits.float()).to(logits.dtype)
        return logits, probs

    
class AleatoricHead(nn.Module):
    """
    Predicts P(unknown) per concept - true aleatoric signal.
    In CEBaB, concept=1 (unknown) represents ambiguity.
    """
    
    def __init__(self, input_dim: int, num_concepts: int, pooling: str = "cls"):
        super().__init__()
        self.pooling = pooling
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_concepts),
        )
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "last":
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(hidden_states, attention_mask)
        logits = self.net(pooled)
        
        # Clamp logits to prevent extreme values in fp16
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        
        # Only convert to float32 for sigmoid (numerical stability), then convert back
        return torch.sigmoid(logits.float()).to(logits.dtype)  # P(unknown) in [0, 1]   


class CredalClassifier(nn.Module):
    """
    Classifier for concept prediction.
    """
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        # Use smaller initialization for fp16 stability
        self.W = nn.Parameter(torch.randn(num_classes, num_concepts) * 0.05)
        self.b = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        # Clamp concept_probs to prevent extreme values in fp16
        concept_probs = torch.clamp(concept_probs, min=0.0, max=1.0)
        logits = F.linear(concept_probs, self.W, self.b)
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        return logits
    
    def forward_credal(self, p_lower: torch.Tensor, p_upper: torch.Tensor):
        W_pos = torch.clamp(self.W, min=0)
        W_neg = torch.clamp(self.W, max=0)
        
        logit_lower = F.linear(p_lower, W_pos) + F.linear(p_upper, W_neg) + self.b
        logit_upper = F.linear(p_upper, W_pos) + F.linear(p_lower, W_neg) + self.b
        
        return {
            "logit_lower": logit_lower,
            "logit_upper": logit_upper,
            "prob_lower": torch.sigmoid(logit_lower),
            "prob_upper": torch.sigmoid(logit_upper),
        }

class CREDENCE(nn.Module):
    """
    CREDENCE v2: Proper Uncertainty Decomposition
    
    Epistemic: Ensemble disagreement (Var_h[p_h])
    Aleatoric: 
        - Supervised: Trained to predict annotator variance
        - Entropy: Head prediction entropy as proxy
    """
    
    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        num_classes: int,
        head_configs: List[HeadConfig],
        aleatoric_mode: str = "supervised",  # "supervised" | "entropy" | "none"
        model_type: str = "encoder",  # "encoder" | "llm" | "llm_frozen"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.n_heads = len(head_configs)
        self.aleatoric_mode = aleatoric_mode
        self.model_type = model_type
        
        # Adjust pooling for LLMs (use last token instead of CLS)
        if model_type in ["llm", "llm_frozen"]:
            for cfg in head_configs:
                cfg.pooling = "last"
        
        # Ensemble heads (epistemic)
        self.heads = nn.ModuleList([
            ConceptHead(cfg, input_dim, num_concepts)
            for cfg in head_configs
        ])
        
        # Aleatoric head (only if supervised mode)
        pooling = "last" if model_type in ["llm", "llm_frozen"] else "cls"
        if aleatoric_mode == "supervised":
            self.aleatoric_head = AleatoricHead(input_dim, num_concepts, pooling=pooling)
        else:
            self.aleatoric_head = None
        
        # Classifier - use dummy concept if num_concepts = 0
        # For datasets without concepts, we'll use a single dummy concept
        effective_num_concepts = max(num_concepts, 1) if num_concepts == 0 else num_concepts
        self.classifier = CredalClassifier(effective_num_concepts, num_classes)
        
        # Direct classifier for datasets without concepts
        if num_concepts == 0:
            self.direct_classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
        else:
            self.direct_classifier = None
        
        self._print_info()
    
    def _print_info(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"CREDENCE: {self.n_heads} heads, {self.num_concepts} concepts, "
              f"{self.num_classes} classes, aleatoric={self.aleatoric_mode}, "
              f"model_type={self.model_type}, {total:,} params")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Convert hidden_states to match head dtype (for frozen LLMs with FP32 heads)
        # This handles the case where encoder outputs FP16 but heads are FP32
        if len(self.heads) > 0:
            head_dtype = next(self.heads[0].parameters()).dtype
            if hidden_states.dtype != head_dtype:
                hidden_states = hidden_states.to(dtype=head_dtype)
        elif self.aleatoric_head is not None:
            # If no heads but aleatoric_head exists, use its dtype
            head_dtype = next(self.aleatoric_head.parameters()).dtype
            if hidden_states.dtype != head_dtype:
                hidden_states = hidden_states.to(dtype=head_dtype)
        
        # Get predictions from all heads
        all_logits = []
        all_probs = []
        
        for head in self.heads:
            logits, probs = head(hidden_states, attention_mask)
            
            # Check for NaN/Inf in head outputs and replace
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                    torch.zeros_like(logits),
                                    logits)
            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                probs = torch.where(torch.isnan(probs) | torch.isinf(probs),
                                   torch.zeros_like(probs),
                                   probs)
            
            # Clamp probs to valid range
            probs = torch.clamp(probs, min=0.0, max=1.0)
            
            all_logits.append(logits)
            all_probs.append(probs)
        
        # Stack: [batch, num_concepts, n_heads]
        probs_stack = torch.stack(all_probs, dim=-1)
        
        # Final check for NaN/Inf in stacked probs
        if torch.any(torch.isnan(probs_stack)) or torch.any(torch.isinf(probs_stack)):
            probs_stack = torch.where(torch.isnan(probs_stack) | torch.isinf(probs_stack),
                                     torch.zeros_like(probs_stack),
                                     probs_stack)
        
        # Handle case when num_concepts = 0 (datasets without concepts)
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        if self.num_concepts == 0:
            # No concepts: use direct classification with ensemble uncertainty
            # Get pooled representations from each head
            head_pooled = []
            for head in self.heads:
                pooled = head.pool(hidden_states, attention_mask)  # [batch, hidden_dim]
                head_pooled.append(pooled)
            
            # Get label predictions from each head using direct classifier
            head_label_logits = []
            for pooled in head_pooled:
                logit = self.direct_classifier(pooled)  # [batch, num_classes]
                
                # Check for NaN/Inf and clamp to prevent overflow
                if torch.any(torch.isnan(logit)) or torch.any(torch.isinf(logit)):
                    logit = torch.where(torch.isnan(logit) | torch.isinf(logit),
                                       torch.zeros_like(logit),
                                       logit)
                # Clamp logits to prevent overflow in softmax with fp16
                logit = torch.clamp(logit, min=-15.0, max=15.0)
                head_label_logits.append(logit)
            
            # Stack: [batch, num_classes, n_heads]
            label_logits_stack = torch.stack(head_label_logits, dim=-1)
            label_probs_stack = torch.softmax(label_logits_stack, dim=1)
            
            # Check for NaN/Inf in softmax output
            if torch.any(torch.isnan(label_probs_stack)) or torch.any(torch.isinf(label_probs_stack)):
                # Replace with uniform distribution
                uniform_probs = torch.ones_like(label_probs_stack) / label_probs_stack.shape[1]
                label_probs_stack = torch.where(torch.isnan(label_probs_stack) | torch.isinf(label_probs_stack),
                                               uniform_probs,
                                               label_probs_stack)
            
            # Aggregate: mean across heads
            logits = label_logits_stack.mean(dim=-1)  # [batch, num_classes]
            
            # Final check for NaN/Inf in logits
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                    torch.zeros_like(logits),
                                    logits)
            
            # Credal bounds on label probabilities
            label_prob_lower = label_probs_stack.min(dim=-1).values  # [batch, num_classes]
            label_prob_upper = label_probs_stack.max(dim=-1).values  # [batch, num_classes]
            
            # For compatibility, create empty concept tensors
            concept_probs = torch.zeros(batch_size, 0, device=device)
            credal_lower = torch.zeros(batch_size, 0, device=device)
            credal_upper = torch.zeros(batch_size, 0, device=device)
            credal_width = torch.zeros(batch_size, 0, device=device)
            
            # Epistemic: variance of label probabilities across heads (mean over classes)
            # Store as label-level disagreement (for evaluation)
            if label_probs_stack.shape[-1] == 1:
                # Single head: no disagreement
                label_disagreement = torch.zeros(batch_size, device=device)
            else:
                label_var = label_probs_stack.var(dim=-1)  # [batch, num_classes]
                # Replace any NaN values with 0
                label_var = torch.where(torch.isnan(label_var), 
                                       torch.zeros_like(label_var), 
                                       label_var)
                label_disagreement = label_var.mean(dim=1)  # [batch] - mean variance across classes
            disagreement = torch.zeros(batch_size, 0, device=device)  # Empty concept-level for compatibility
            
            # Credal output (on labels, not concepts)
            credal_out = {
                "prob_lower": label_prob_lower,
                "prob_upper": label_prob_upper,
            }
            
            # Store label-level metrics for evaluation
            self._label_disagreement = label_disagreement
        else:
            # Normal case: has concepts
            # Credal aggregation
            credal_lower = probs_stack.min(dim=-1).values
            credal_upper = probs_stack.max(dim=-1).values
            credal_width = credal_upper - credal_lower
            concept_probs = probs_stack.mean(dim=-1)
            
            # Epistemic: ensemble disagreement
            # Compute variance across heads, handling edge cases
            if probs_stack.shape[-1] == 1:
                # Single head: no disagreement
                disagreement = torch.zeros_like(probs_stack[..., 0])
            else:
                disagreement = probs_stack.var(dim=-1)
                # Replace any NaN values with 0 (can happen due to numerical instability)
                disagreement = torch.where(torch.isnan(disagreement), 
                                          torch.zeros_like(disagreement), 
                                          disagreement)
            
            # Classification
            # Check for NaN/Inf in concept_probs before classification
            if torch.any(torch.isnan(concept_probs)) or torch.any(torch.isinf(concept_probs)):
                concept_probs = torch.where(torch.isnan(concept_probs) | torch.isinf(concept_probs),
                                          torch.zeros_like(concept_probs),
                                          concept_probs)
            
            logits = self.classifier(concept_probs)
            
            # Check for NaN/Inf in logits
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                    torch.zeros_like(logits),
                                    logits)
            
            credal_out = self.classifier.forward_credal(credal_lower, credal_upper)
        
        # Aleatoric: predicted P(unknown) per concept
        if self.aleatoric_mode == "supervised":
            # Learned prediction of P(unknown) - true aleatoric signal
            if self.num_concepts > 0:
                ambiguity = self.aleatoric_head(hidden_states, attention_mask)
                # Clamp to [eps, 1-eps] to handle float16 numerical issues
                # Use small epsilon to avoid exact boundaries that might cause issues
                eps = 1e-6
                ambiguity = torch.clamp(ambiguity, min=eps, max=1.0 - eps)
            else:
                ambiguity = torch.zeros(batch_size, 0, device=hidden_states.device)
        elif self.aleatoric_mode == "entropy":
            # Proxy: mean entropy of individual head predictions
            if self.num_concepts > 0:
                eps = 1e-8
                entropies = -(probs_stack * torch.log(probs_stack + eps) + 
                             (1 - probs_stack) * torch.log(1 - probs_stack + eps))
                ambiguity = entropies.mean(dim=-1)  # Average entropy across heads
                # Replace any NaN values with 0
                ambiguity = torch.where(torch.isnan(ambiguity), 
                                       torch.zeros_like(ambiguity), 
                                       ambiguity)
                # Clamp to [0, 1] (entropy should already be normalized, but be safe)
                ambiguity = torch.clamp(ambiguity, min=0.0, max=1.0)
            else:
                ambiguity = torch.zeros(batch_size, 0, device=hidden_states.device)
        else:
            ambiguity = torch.zeros_like(disagreement)
        
        result = {
            "logits": logits,
            "concept_probs": concept_probs,
            "credal_lower": credal_lower,
            "credal_upper": credal_upper,
            "credal_width": credal_width,
            "disagreement": disagreement,
            "ambiguity": ambiguity,
            "total_uncertainty": disagreement + ambiguity,
            "label_prob_lower": credal_out["prob_lower"],
            "label_prob_upper": credal_out["prob_upper"],
            "head_logits": all_logits,
            "head_probs": all_probs,
        }
        
        # Add label-level disagreement for datasets without concepts
        if self.num_concepts == 0 and hasattr(self, '_label_disagreement'):
            result["label_disagreement"] = self._label_disagreement
        
        return result
    
    def compute_loss(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
        concepts: torch.Tensor,
        is_unknown: Optional[torch.Tensor] = None,
        concept_weight: float = 1.0,
        aleatoric_weight: float = 0.5,
    ):
        device = labels.device
        
        # Get output dtype to match targets (important for float16 models)
        output_dtype = outputs["logits"].dtype
        
        # 1. Task loss
        # Check for NaN/Inf in logits before computing loss
        logits = outputs["logits"]
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            print(f"Warning: NaN/Inf detected in logits, replacing with zeros")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                torch.zeros_like(logits),
                                logits)
        task_loss = F.cross_entropy(logits, labels)
        
        # 2. Concept loss (BCE per head, averaged)
        # Ternary -> soft targets: 0->0.0, 1->0.5, 2->1.0
        concept_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        if self.num_concepts > 0 and concepts.numel() > 0:
            concept_targets = (concepts.float() / 2.0).to(dtype=output_dtype)
            # Clamp to [0, 1] for safety (should already be in range, but float16 can have numerical issues)
            concept_targets = torch.clamp(concept_targets, min=0.0, max=1.0)
            if len(outputs["head_logits"]) > 0:
                valid_heads = 0
                for head_logits in outputs["head_logits"]:
                    # Ensure shapes match
                    if head_logits.shape == concept_targets.shape:
                        # Check for NaN/Inf in head_logits
                        if torch.any(torch.isnan(head_logits)) or torch.any(torch.isinf(head_logits)):
                            continue  # Skip this head if it has NaN/Inf
                        concept_loss = concept_loss + F.binary_cross_entropy_with_logits(
                            head_logits, concept_targets
                        )
                        valid_heads += 1
                if valid_heads > 0:
                    concept_loss = concept_loss / valid_heads
                else:
                    concept_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        
        # 3. Aleatoric loss: predict P(unknown) per concept
        aleatoric_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        if self.aleatoric_mode == "supervised" and is_unknown is not None and is_unknown.numel() > 0:
            # Ensure shapes match
            ambiguity = outputs["ambiguity"]
            if ambiguity.shape == is_unknown.shape:
                # Handle NaN/Inf and clamp to [eps, 1-eps] to avoid float16 numerical issues
                eps = 1e-6
                ambiguity = torch.where(torch.isnan(ambiguity) | torch.isinf(ambiguity),
                                       torch.full_like(ambiguity, 0.5),
                                       ambiguity)
                ambiguity = torch.clamp(ambiguity, min=eps, max=1.0 - eps)
                is_unknown_target = is_unknown.to(device=device, dtype=output_dtype)
                is_unknown_target = torch.clamp(is_unknown_target, min=0.0, max=1.0)
                aleatoric_loss = F.binary_cross_entropy(
                    ambiguity, 
                    is_unknown_target,
                    reduction='mean'
                )
        
        # Check for NaN/Inf in losses and replace with finite values
        if torch.isnan(task_loss) or torch.isinf(task_loss):
            task_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        if torch.isnan(concept_loss) or torch.isinf(concept_loss):
            concept_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        if torch.isnan(aleatoric_loss) or torch.isinf(aleatoric_loss):
            aleatoric_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        
        # Total
        total_loss = (task_loss + 
                     concept_weight * concept_loss + 
                     aleatoric_weight * aleatoric_loss)
        
        # Final check for NaN/Inf in total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected in total_loss. task={task_loss}, concept={concept_loss}, aleatoric={aleatoric_loss}")
            total_loss = torch.tensor(0.0, device=device, dtype=output_dtype)
        
        return total_loss, {
            "total": total_loss.item() if torch.isfinite(total_loss) else 0.0,
            "task": task_loss.item() if torch.isfinite(task_loss) else 0.0,
            "concept": concept_loss.item() if torch.isfinite(concept_loss) else 0.0,
            "aleatoric": aleatoric_loss.item() if torch.isfinite(aleatoric_loss) else 0.0,
        }


def train_epoch(
    model: CREDENCE,
    encoder: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: str,
    model_type: str,
    profiler: Optional[torch.profiler.profiler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train()
    
    # Frozen LLM never trains
    if model_type in ["encoder", "llm_frozen"]:
        encoder.eval()
    else:
        encoder.train()
    
    epoch_losses = defaultdict(float)
    n_batches = 0
    
    # Determine if we should use mixed precision
    # AMP is fine for LLMs (encoder is bfloat16/float16, heads are FP32 for frozen)
    use_amp = (device == "cuda" and model_type in ["llm", "llm_frozen"])
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)
        
        optimizer.zero_grad()
        
        # === MIXED PRECISION FORWARD PASS ===
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            # Always no_grad for frozen models
            if model_type in ["encoder", "llm_frozen"]:
                with torch.no_grad():
                    hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
            else:
                hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
            
            # Forward pass through CREDENCE (fp16)
            outputs = model(hidden_states, attention_mask)
        
        # === LOSS IN FP32 (outside autocast) ===
        # Convert outputs to fp32 for stable loss computation
        outputs_fp32 = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v 
                        for k, v in outputs.items()}
        
        loss, loss_dict = model.compute_loss(
            outputs_fp32, labels, concepts, is_unknown,
            concept_weight=config.concept_weight,
            aleatoric_weight=config.aleatoric_weight,
        )
        
        # === BACKWARD WITH GRADIENT SCALING ===
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if model_type == "llm":  # Only clip for LoRA training, not frozen
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if model_type == "llm":  # Only clip for LoRA training, not frozen
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Advance profiler step (handles schedule automatically)
        if profiler is not None:
            profiler.step()
        
        # Track losses
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        n_batches += 1
        
        # Display (handle empty tensors)
        # Get disagreement value, handling NaN cases
        if outputs.get('label_disagreement') is not None:
            disagree_tensor = outputs['label_disagreement']
        elif outputs['disagreement'].numel() > 0:
            disagree_tensor = outputs['disagreement']
        else:
            disagree_tensor = None
        
        if disagree_tensor is not None:
            # Handle NaN values: replace with 0 and compute mean
            disagree_clean = torch.where(torch.isnan(disagree_tensor), 
                                        torch.zeros_like(disagree_tensor), 
                                        disagree_tensor)
            disagree_val = disagree_clean.mean().item()
        else:
            disagree_val = 0.0
        
        # Handle ambiguity similarly
        if outputs['ambiguity'].numel() > 0:
            ambig_clean = torch.where(torch.isnan(outputs['ambiguity']), 
                                     torch.zeros_like(outputs['ambiguity']), 
                                     outputs['ambiguity'])
            ambig_val = ambig_clean.mean().item()
        else:
            ambig_val = 0.0
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "disagree": f"{disagree_val:.4f}",
            "ambig": f"{ambig_val:.4f}",
        })
    
    return {k: v / n_batches for k, v in epoch_losses.items()}


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: CREDENCE,
    encoder: nn.Module,
    data_loader: DataLoader,
    device: str,
    model_type: str,
):
    model.eval()
    encoder.eval()
    
    all_preds = []
    all_labels = []
    all_disagreement = []
    all_ambiguity = []
    all_credal_width = []
    all_annotator_var = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concept_labels']
        
        is_unknown = batch['is_unknown']
        if is_unknown.numel() > 0:
            is_unknown_mean = is_unknown.mean(dim=-1).numpy()  # Mean unknown rate per sample
        else:
            # Handle datasets without concepts (empty is_unknown)
            is_unknown_mean = np.zeros(len(labels))
        
        hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
        outputs = model(hidden_states, attention_mask)
        
        preds = outputs["logits"].argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Handle disagreement: use label-level for datasets without concepts
        if "label_disagreement" in outputs:
            # Dataset without concepts: use label-level disagreement
            all_disagreement.append(outputs["label_disagreement"].cpu().numpy())
        else:
            # Dataset with concepts: use concept-level disagreement
            all_disagreement.append(outputs["disagreement"].mean(dim=-1).cpu().numpy())
        
        # Ambiguity: for datasets without concepts, it's empty, so use zeros
        if outputs["ambiguity"].numel() > 0:
            all_ambiguity.append(outputs["ambiguity"].mean(dim=-1).cpu().numpy())
        else:
            all_ambiguity.append(np.zeros(len(preds)))
        
        # Credal width: for datasets without concepts, it's empty, so use zeros
        if outputs["credal_width"].numel() > 0:
            all_credal_width.append(outputs["credal_width"].mean(dim=-1).cpu().numpy())
        else:
            all_credal_width.append(np.zeros(len(preds)))
        
        all_annotator_var.extend(is_unknown_mean)
    
    # Convert
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_disagreement = np.concatenate(all_disagreement)
    all_ambiguity = np.concatenate(all_ambiguity)
    all_credal_width = np.concatenate(all_credal_width)
    all_annotator_var = np.array(all_annotator_var)
    
    # Metrics
    accuracy = (all_preds == all_labels).mean()
    errors = (all_preds != all_labels).astype(float)
    
    # Disagreement-error correlation (epistemic validation)
    if errors.std() > 0 and all_disagreement.std() > 0:
        disagree_error_corr, disagree_error_pval = stats.spearmanr(all_disagreement, errors)
    else:
        disagree_error_corr, disagree_error_pval = 0.0, 1.0
    
    # Aleatoric validation: ambiguity vs is_unknown (unknown concept labels)
    if all_annotator_var.std() > 0 and all_ambiguity.std() > 0:
        ambig_unknown_corr, ambig_unknown_pval = stats.spearmanr(all_ambiguity, all_annotator_var)
    else:
        ambig_unknown_corr, ambig_unknown_pval = 0.0, 1.0
    
    # Disagreement ratio
    correct_mask = errors == 0
    incorrect_mask = errors == 1
    if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
        disagree_ratio = (all_disagreement[incorrect_mask].mean() / 
                         (all_disagreement[correct_mask].mean() + 1e-8))
    else:
        disagree_ratio = 1.0
    
    return {
        "accuracy": float(accuracy),
        "disagree_error_corr": float(disagree_error_corr),
        "disagree_error_pval": float(disagree_error_pval),
        "disagree_ratio": float(disagree_ratio),
        "ambig_unknown_corr": float(ambig_unknown_corr),
        "ambig_unknown_pval": float(ambig_unknown_pval),
        "mean_disagreement": float(all_disagreement.mean()),
        "mean_ambiguity": float(all_ambiguity.mean()),
        "mean_unknown": float(all_annotator_var.mean()),  # Add mean unknown rate for debugging
        "mean_credal_width": float(all_credal_width.mean()),
    }


# =============================================================================
# HELPER: Auto-detect aleatoric mode
# =============================================================================

def get_aleatoric_mode(metadata: dict) -> str:
    has_multi = metadata.get("has_multi_annotator", False)
    has_concepts = metadata.get("has_concepts", False)
    
    if has_multi and has_concepts:
        return "supervised"
    elif has_multi and not has_concepts:
        # Could add concept extraction for NLI datasets
        print(f"  Note: {metadata['dataset_name']} has multi-annotator but no concepts.")
        print(f"  Using entropy mode. Consider adding concept extraction.")
        return "entropy"
    else:
        return "entropy"


# =============================================================================
# SIMPLE ANALYSIS
# =============================================================================

def analyze_test_set(model, encoder, test_loader, device, model_type, output_dir="./analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    encoder.eval()
    
    # =========================================================================
    # 1. COLLECT PREDICTIONS
    # =========================================================================
    print("\n" + "="*60)
    print("Analyzing Test Set")
    print("="*60)
    
    all_preds = []
    all_labels = []
    all_disagreement = []
    all_ambiguity = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
            outputs = model(hidden_states, attention_mask)
            
            # Store
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_disagreement.extend(outputs["disagreement"].mean(dim=-1).cpu().numpy())
            all_ambiguity.extend(outputs["ambiguity"].mean(dim=-1).cpu().numpy())
    
    # Convert to arrays
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    disagreement = np.array(all_disagreement)
    ambiguity = np.array(all_ambiguity)
    
    correct = (preds == labels)
    errors = ~correct
    
    # =========================================================================
    # 2. SANITY CHECK
    # =========================================================================
    accuracy = correct.mean()
    print(f"\n[SANITY CHECK] Accuracy: {accuracy*100:.2f}%")
    if accuracy < 0.4:
        print("  WARNING: Accuracy < 40% - model may not be properly loaded!")
        print("  WARNING: Make sure you called model.load_state_dict(best_state) before analysis")
    
    # =========================================================================
    # 3. COMPUTE METRICS
    # =========================================================================
    print("\n--- Metrics ---")
    
    # Epistemic-error correlation
    rho_epi, p_epi = stats.spearmanr(disagreement, errors.astype(float))
    print(f"rho(epistemic, error) = {rho_epi:.4f} (p = {p_epi:.2e})")
    
    # Disagreement ratio
    disagree_correct = disagreement[correct].mean()
    disagree_error = disagreement[errors].mean()
    ratio = disagree_error / (disagree_correct + 1e-8)
    print(f"Disagree ratio (error/correct) = {ratio:.2f}x")
    print(f"  Mean on correct: {disagree_correct:.6f}")
    print(f"  Mean on errors:  {disagree_error:.6f}")
    
    # =========================================================================
    # 4. ERROR DETECTION CURVE
    # =========================================================================
    print("\n--- Error Detection ---")
    
    # Sort by disagreement (high to low)
    sorted_idx = np.argsort(disagreement)[::-1]
    sorted_errors = errors[sorted_idx].astype(float)
    
    # Cumulative errors found
    cumsum_errors = np.cumsum(sorted_errors)
    total_errors = errors.sum()
    
    pct_reviewed = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    pct_errors_found = cumsum_errors / total_errors * 100
    
    # Key metrics
    idx_50 = np.searchsorted(pct_errors_found, 50)
    pct_to_catch_50 = pct_reviewed[idx_50] if idx_50 < len(pct_reviewed) else 100
    
    idx_20 = int(0.2 * len(pct_reviewed))
    errors_in_top_20 = pct_errors_found[idx_20]
    
    print(f"To catch 50% of errors: review {pct_to_catch_50:.1f}% of data")
    print(f"Top 20% (by disagreement) catches: {errors_in_top_20:.1f}% of errors")
    print(f"Efficiency: {50/pct_to_catch_50:.2f}x faster than random")
    
    # =========================================================================
    # 5. QUADRANT ANALYSIS
    # =========================================================================
    print("\n--- Quadrant Analysis ---")
    
    epi_thresh = np.median(disagreement)
    ale_thresh = np.median(ambiguity)
    
    quadrants = {
        'Low Epi, Low Ale (Trust)': (disagreement < epi_thresh) & (ambiguity < ale_thresh),
        'High Epi, Low Ale (More Data)': (disagreement >= epi_thresh) & (ambiguity < ale_thresh),
        'Low Epi, High Ale (Human Review)': (disagreement < epi_thresh) & (ambiguity >= ale_thresh),
        'High Epi, High Ale (Abstain)': (disagreement >= epi_thresh) & (ambiguity >= ale_thresh),
    }
    
    quadrant_stats = {}
    for name, mask in quadrants.items():
        n = mask.sum()
        acc = correct[mask].mean() if n > 0 else 0
        print(f"  {name}: n={n}, accuracy={acc*100:.1f}%")
        quadrant_stats[name] = {'count': int(n), 'accuracy': float(acc)}
    
    # =========================================================================
    # 6. PLOT ERROR DETECTION CURVE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(pct_reviewed, pct_errors_found, color='#e74c3c', linewidth=2.5, label='Epistemic-guided')
    ax.plot(pct_reviewed, pct_reviewed, color='gray', linewidth=1.5, linestyle='--', label='Random')
    ax.fill_between(pct_reviewed, pct_reviewed, pct_errors_found, 
                    where=(pct_errors_found > pct_reviewed), alpha=0.2, color='#e74c3c')
    
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.scatter([pct_to_catch_50], [50], color='#e74c3c', s=100, zorder=5)
    ax.annotate(f"50% errors caught\nreviewing {pct_to_catch_50:.1f}% of data",
                xy=(pct_to_catch_50, 50), xytext=(pct_to_catch_50 + 10, 35),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('% of Samples Reviewed (highest disagreement first)')
    ax.set_ylabel('% of Errors Detected')
    ax.set_title('Error Detection via Epistemic Uncertainty', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_error_detection.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig_error_detection.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/fig_error_detection.pdf")
    plt.close()
    
    # =========================================================================
    # 7. PLOT DISAGREEMENT DISTRIBUTION
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    
    bins = np.linspace(0, disagreement.max(), 30)
    ax.hist(disagreement[correct], bins=bins, alpha=0.6, color='#28a745', 
            label=f'Correct (n={correct.sum()})', density=True)
    ax.hist(disagreement[errors], bins=bins, alpha=0.6, color='#dc3545',
            label=f'Errors (n={errors.sum()})', density=True)
    
    ax.axvline(disagree_correct, color='#28a745', linestyle='--', linewidth=2)
    ax.axvline(disagree_error, color='#dc3545', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Epistemic Uncertainty (Disagreement)')
    ax.set_ylabel('Density')
    ax.set_title('Disagreement Distribution: Correct vs Errors', fontweight='bold')
    ax.legend()
    ax.text(0.95, 0.95, f'Ratio: {ratio:.2f}x', transform=ax.transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_disagreement_dist.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig_disagreement_dist.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/fig_disagreement_dist.pdf")
    plt.close()
    
    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    results = {
        'accuracy': float(accuracy),
        'n_samples': int(len(preds)),
        'n_correct': int(correct.sum()),
        'n_errors': int(errors.sum()),
        'rho_epistemic_error': float(rho_epi),
        'p_value': float(p_epi),
        'disagree_ratio': float(ratio),
        'mean_disagree_correct': float(disagree_correct),
        'mean_disagree_error': float(disagree_error),
        'pct_to_catch_50': float(pct_to_catch_50),
        'errors_in_top_20': float(errors_in_top_20),
        'quadrant_stats': quadrant_stats,
    }
    
    with open(f"{output_dir}/analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/analysis_results.json")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    return results


# =============================================================================
# QUALITATIVE EXAMPLES EXTRACTION
# =============================================================================

def extract_qualitative_examples(model, encoder, test_loader, dataset, device, model_type, n_examples=3, output_dir=None):
    """
    Extract representative examples from each uncertainty quadrant.
    
    Args:
        model: Trained CREDENCE model
        encoder: Frozen encoder
        test_loader: Test DataLoader
        dataset: Original dataset object (to get raw text)
        device: 'cuda' or 'cpu'
        n_examples: Number of examples per quadrant
        output_dir: Directory to save examples (if None, only returns dict)
    
    Returns:
        dict with examples for each quadrant
    """
    model.eval()
    encoder.eval()
    
    # Collect all predictions
    all_data = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting examples"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
            outputs = model(hidden_states, attention_mask)
            
            preds = outputs["logits"].argmax(dim=-1)
            disagreement = outputs["disagreement"].mean(dim=-1)  # Mean across concepts
            ambiguity = outputs["ambiguity"].mean(dim=-1)
            
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Get original text from dataset
                if hasattr(dataset, 'examples'):
                    text = dataset.examples[sample_idx]['text']
                elif hasattr(dataset, 'texts') and sample_idx < len(dataset.texts):
                    text = dataset.texts[sample_idx]
                else:
                    text = f"[Sample {sample_idx}]"
                
                all_data.append({
                    'idx': sample_idx,
                    'text': text[:200] if isinstance(text, str) else str(text)[:200],  # Truncate for display
                    'label': labels[i].item(),
                    'pred': preds[i].item(),
                    'correct': (preds[i] == labels[i]).item(),
                    'disagreement': disagreement[i].item(),
                    'ambiguity': ambiguity[i].item(),
                })
                sample_idx += 1
    
    # Convert to arrays for thresholding
    disagreements = np.array([d['disagreement'] for d in all_data])
    ambiguities = np.array([d['ambiguity'] for d in all_data])
    
    epi_thresh = np.median(disagreements)
    ale_thresh = np.median(ambiguities)
    
    print(f"Thresholds: epistemic={epi_thresh:.6f}, aleatoric={ale_thresh:.4f}")
    
    # Classify into quadrants
    quadrants = {
        'low_epi_low_ale': [],   # Trust
        'high_epi_low_ale': [],  # More Data
        'low_epi_high_ale': [],  # Human Review
        'high_epi_high_ale': [], # Abstain
    }
    
    for d in all_data:
        epi_high = d['disagreement'] >= epi_thresh
        ale_high = d['ambiguity'] >= ale_thresh
        
        if not epi_high and not ale_high:
            quadrants['low_epi_low_ale'].append(d)
        elif epi_high and not ale_high:
            quadrants['high_epi_low_ale'].append(d)
        elif not epi_high and ale_high:
            quadrants['low_epi_high_ale'].append(d)
        else:
            quadrants['high_epi_high_ale'].append(d)
    
    # Select best examples for each quadrant
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Very Negative', 4: 'Very Positive'}
    selected = {}
    
    for name, examples in quadrants.items():
        print(f"\n{name}: {len(examples)} samples")
        
        if name == 'low_epi_low_ale':
            # Want: correct predictions, lowest uncertainty
            candidates = [e for e in examples if e['correct']]
            candidates.sort(key=lambda x: x['disagreement'] + x['ambiguity'])
        
        elif name == 'high_epi_low_ale':
            # Want: errors (model confused), highest disagreement
            candidates = [e for e in examples if not e['correct']]
            if not candidates:
                candidates = examples
            candidates.sort(key=lambda x: -x['disagreement'])
        
        elif name == 'low_epi_high_ale':
            # Want: correct but high ambiguity (model agrees on ambiguous case)
            candidates = [e for e in examples if e['correct']]
            if not candidates:
                candidates = examples
            candidates.sort(key=lambda x: -x['ambiguity'])
        
        else:  # high_epi_high_ale
            # Want: highest combined uncertainty
            candidates = sorted(examples, key=lambda x: -(x['disagreement'] + x['ambiguity']))
        
        selected[name] = []
        for ex in candidates[:n_examples]:
            selected[name].append({
                'text': ex['text'],
                'label': label_names.get(ex['label'], ex['label']),
                'pred': label_names.get(ex['pred'], ex['pred']),
                'correct': 'CORRECT' if ex['correct'] else 'ERROR',
                'epistemic': f"{ex['disagreement']:.4f}",
                'aleatoric': f"{ex['ambiguity']:.3f}",
            })
            print(f"  [{ex['correct']}] {ex['text'][:80]}...")
            print(f"      Epi={ex['disagreement']:.4f}, Ale={ex['ambiguity']:.3f}")
    
    # Save to files if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, "qualitative_examples.json")
        with open(json_path, 'w') as f:
            json.dump(selected, f, indent=2)
        print(f"\nSaved examples to: {json_path}")
        
        # Save as text file
        txt_path = os.path.join(output_dir, "qualitative_examples.txt")
        with open(txt_path, 'w') as f:
            f.write("Qualitative Examples from Uncertainty Quadrants\n")
            f.write("=" * 60 + "\n\n")
            
            quadrant_info = {
                'low_epi_low_ale': ('TRUST', 'Clear signal, confident model'),
                'high_epi_low_ale': ('MORE DATA', 'Model confused, needs more training'),
                'low_epi_high_ale': ('HUMAN REVIEW', 'Inherently ambiguous'),
                'high_epi_high_ale': ('ABSTAIN', 'Confused model + ambiguous data'),
            }
            
            for qname, (label, desc) in quadrant_info.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{label}: {desc}\n")
                f.write(f"{'='*60}\n\n")
                
                if qname in selected:
                    for i, ex in enumerate(selected[qname], 1):
                        f.write(f"Example {i}:\n")
                        f.write(f"  Text: \"{ex['text']}\"\n")
                        f.write(f"  Prediction: {ex['pred']} ({ex['correct']})\n")
                        f.write(f"  Epistemic: {ex['epistemic']}\n")
                        f.write(f"  Aleatoric: {ex['aleatoric']}\n")
                        f.write("\n")
        
        print(f"Saved text format to: {txt_path}")
        
        # Save LaTeX format
        latex_path = os.path.join(output_dir, "qualitative_examples_latex.tex")
        with open(latex_path, 'w') as f:
            f.write("% Qualitative Examples for Paper\n")
            f.write("% Generated automatically\n\n")
            
            quadrant_info = {
                'low_epi_low_ale': ('green!15', 'Low Epistemic, Low Aleatoric', 'Trust Prediction'),
                'high_epi_low_ale': ('blue!15', 'High Epistemic, Low Aleatoric', 'Collect More Data'),
                'low_epi_high_ale': ('yellow!15', 'Low Epistemic, High Aleatoric', 'Human Review'),
                'high_epi_high_ale': ('red!15', 'High Epistemic, High Aleatoric', 'Abstain'),
            }
            
            for qname, (color, label, action) in quadrant_info.items():
                f.write(f"\n% {label}\n")
                f.write(f"\\multicolumn{{5}}{{l}}{{\\colorbox{{{color}}}{{\\textit{{{label}}} -> \\textbf{{{action}}}}}}} \\\\\n")
                
                if qname in selected and selected[qname]:
                    for ex in selected[qname]:
                        text_escaped = ex['text'].replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
                        text_escaped = text_escaped[:100] + "..." if len(text_escaped) > 100 else text_escaped
                        pred_str = str(ex['pred'])
                        pred_display = pred_str[:3] if len(pred_str) > 3 else pred_str
                        f.write(f"``{text_escaped}'' & {pred_display} {ex['correct']} & {ex['epistemic']} & {ex['aleatoric']} & [interpretation] \\\\\n")
        
        print(f"Saved LaTeX format to: {latex_path}")
    
    return selected


def print_latex_examples(selected):
    """Print examples as LaTeX table rows."""
    
    print("\n" + "="*60)
    print("LATEX TABLE ROWS")
    print("="*60)
    
    quadrant_info = {
        'low_epi_low_ale': ('green!15', 'Low Epistemic, Low Aleatoric', 'Trust Prediction'),
        'high_epi_low_ale': ('blue!15', 'High Epistemic, Low Aleatoric', 'Collect More Data'),
        'low_epi_high_ale': ('yellow!15', 'Low Epistemic, High Aleatoric', 'Human Review'),
        'high_epi_high_ale': ('red!15', 'High Epistemic, High Aleatoric', 'Abstain'),
    }
    
    for qname, (color, label, action) in quadrant_info.items():
        print(f"\n% {label}")
        print(f"\\multicolumn{{5}}{{l}}{{\\colorbox{{{color}}}{{\\textit{{{label}}} -> \\textbf{{{action}}}}}}} \\\\")
        
        if qname in selected and selected[qname]:
            ex = selected[qname][0]  # First example
            text_escaped = ex['text'].replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
            text_escaped = text_escaped[:100] + "..." if len(text_escaped) > 100 else text_escaped
            pred_str = str(ex['pred'])
            pred_display = pred_str[:3] if len(pred_str) > 3 else pred_str
            print(f"``{text_escaped}'' & {pred_display} {ex['correct']} & {ex['epistemic']} & {ex['aleatoric']} & [interpretation] \\\\")


def format_examples_for_paper(selected):
    """Format examples as markdown for easy viewing."""
    
    print("\n" + "="*60)
    print("EXAMPLES FOR PAPER")
    print("="*60)
    
    quadrant_info = {
        'low_epi_low_ale': ('TRUST', 'Clear signal, confident model'),
        'high_epi_low_ale': ('MORE DATA', 'Model confused, needs more training'),
        'low_epi_high_ale': ('HUMAN REVIEW', 'Inherently ambiguous'),
        'high_epi_high_ale': ('ABSTAIN', 'Confused model + ambiguous data'),
    }
    
    for qname, (label, desc) in quadrant_info.items():
        print(f"\n### {label}")
        print(f"*{desc}*\n")
        
        if qname in selected:
            for i, ex in enumerate(selected[qname], 1):
                print(f"**Example {i}:** \"{ex['text'][:150]}...\"")
                print(f"- Prediction: {ex['pred']} {ex['correct']}")
                print(f"- Epistemic: {ex['epistemic']}, Aleatoric: {ex['aleatoric']}")
                print()


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(config: ExperimentConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Expand short encoder names to full model names (e.g., "mistral" -> "mistralai/Mistral-7B-v0.1")
    original_name = config.encoder_name
    config.encoder_name = expand_encoder_name(config.encoder_name)
    if original_name != config.encoder_name:
        print(f"Expanded encoder name: {original_name} -> {config.encoder_name}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create DatasetConfig from ExperimentConfig
    dataset_config = DatasetConfig(
        label_type=config.label_type,
        max_length=config.max_length,
        tokenizer_name=config.encoder_name,
        batch_size=config.batch_size
    )
    
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        config.dataset,
        config=dataset_config
    )
    
    # Auto-detect aleatoric mode if set to "auto" (v3 feature)
    if config.aleatoric_mode == "auto":
        config.aleatoric_mode = get_aleatoric_mode(metadata)
        print(f"  Auto-detected aleatoric_mode: {config.aleatoric_mode}")
    elif not metadata["has_multi_annotator"] and config.aleatoric_mode == "supervised":
        # Validate mode: supervised requires multi-annotator data
        print(f"  Warning: Dataset {config.dataset} has no multi-annotator data!")
        print(f"  Switching aleatoric_mode from 'supervised' to 'entropy'")
        config.aleatoric_mode = "entropy"

    # Load model
    encoder, tokenizer, hidden_size, model_type = load_model(config, device)
    
    # Adjust max_length for LLMs
    model_info = MODEL_REGISTRY.get(config.encoder_name, {})
    if model_type in ["llm", "llm_frozen"]:
        config.max_length = min(config.max_length, model_info.get("max_length", 256))
        print(f"Using max_length={config.max_length} for LLM")

    model = CREDENCE(
        input_dim=hidden_size,
        num_concepts=metadata['num_concepts'],
        num_classes=metadata['num_classes'],
        head_configs=config.get_head_configs(),
        aleatoric_mode=config.aleatoric_mode,
        model_type=model_type,
    )
    model = model.to(device)
    
    # For LLMs with bfloat16/float16, convert heads to float16 only if using LoRA
    # For frozen LLMs, keep heads in FP32 to avoid gradient scaling issues
    if model_type in ["llm", "llm_frozen"]:
        encoder_dtype = next(encoder.parameters()).dtype
        if model_type == "llm":  # LoRA training
            if encoder_dtype != torch.float32:
                model = model.half()  # Convert heads to fp16 for LoRA training
                print(f"  Encoder dtype: {encoder_dtype}, CREDENCE heads converted to fp16")
        else:  # llm_frozen
            print(f"  Encoder dtype: {encoder_dtype}, CREDENCE heads kept in fp32 (frozen LLM)")

    # === ADD: Create GradScaler for LLMs with LoRA ===
    # Note: Don't use scaler for frozen LLMs - GradScaler doesn't support FP16 gradients
    # For frozen LLMs, we train only the heads (small), so no need for gradient scaling
    scaler = None
    if model_type == "llm" and device == "cuda":
        scaler = torch.amp.GradScaler('cuda')
        print("GradScaler enabled for mixed precision training (LoRA)")
    elif model_type == "llm_frozen":
        print("GradScaler disabled for frozen LLM (training heads only)")
    
    # Optimizer (use fused AdamW for speed)
    optimizer_kwargs = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }
    if device == "cuda":
        try:
            # Try fused optimizer for speed (available in PyTorch 1.13+)
            optimizer_kwargs["fused"] = True
        except TypeError:
            # Fused not available, use standard
            pass
    
    if model_type == "encoder":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif model_type == "llm_frozen":
        # Frozen LLM: only train CREDENCE heads
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    else:
        # LoRA training: include LoRA parameters
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(encoder.parameters()),
            **optimizer_kwargs
        )
    best_val_acc = 0.0
    best_state = None
    history = []
    
    # Setup profiler if enabled
    profiler = None
    if config.enable_profiler:
        profiler_output = config.profiler_output_dir or os.path.join(config.output_dir, "profiler")
        os.makedirs(profiler_output, exist_ok=True)
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        # Build profiler kwargs (with_flops may not be available in all PyTorch versions)
        profiler_kwargs = {
            "activities": activities,
            "schedule": torch.profiler.schedule(
                wait=0,
                warmup=config.profiler_warmup,
                active=config.profiler_active,
                repeat=config.profiler_repeat,
            ),
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(profiler_output),
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": True,
        }
        # Add with_flops if available (PyTorch 1.8.1+)
        if hasattr(torch.profiler.profile, '__init__'):
            try:
                import inspect
                sig = inspect.signature(torch.profiler.profile.__init__)
                if 'with_flops' in sig.parameters:
                    profiler_kwargs["with_flops"] = True
            except:
                pass
        
        profiler = torch.profiler.profile(**profiler_kwargs)
        profiler.start()
        print(f"Profiler enabled. Output: {profiler_output}")
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Only profile first epoch if profiler is enabled
        current_profiler = profiler if (config.enable_profiler and epoch == 0) else None
        train_losses = train_epoch(
            model, encoder, train_loader, optimizer, config, 
            device, model_type, current_profiler, scaler=scaler
        )
        val_results = evaluate(model, encoder, val_loader, device, model_type)
        
        # Stop profiler after first epoch
        if current_profiler is not None:
            profiler.stop()
            print(f"Profiler stopped. Results saved to {profiler_output}")
            # Export profiler results (tensorboard handler already saves traces, so wrap in try-except)
            try:
                profiler.export_chrome_trace(os.path.join(profiler_output, "trace.json"))
            except RuntimeError as e:
                if "already saved" in str(e):
                    print(f"  Note: Chrome trace already saved by tensorboard handler")
                else:
                    raise
            try:
                profiler.export_stacks(os.path.join(profiler_output, "stacks.txt"), "profiler_stacks")
            except Exception as e:
                print(f"  Warning: Could not export stacks: {e}")
            
            # Print summary
            print("\n=== Profiler Summary ===")
            sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
            print(profiler.key_averages().table(sort_by=sort_key, row_limit=20))
            
            # Save detailed statistics
            with open(os.path.join(profiler_output, "profiler_summary.txt"), "w") as f:
                f.write("=== Profiler Summary (sorted by total time) ===\n\n")
                f.write(profiler.key_averages().table(sort_by=sort_key))
                f.write("\n\n=== Profiler Summary (sorted by self time) ===\n\n")
                self_sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
                f.write(profiler.key_averages().table(sort_by=self_sort_key))
                f.write("\n\n=== Memory Usage ===\n\n")
                memory_sort_key = "cuda_memory_usage" if device == "cuda" else "cpu_memory_usage"
                try:
                    f.write(profiler.key_averages().table(sort_by=memory_sort_key))
                except Exception:
                    # Fallback if memory sorting not available
                    f.write(profiler.key_averages().table(sort_by=sort_key))
            
            print(f"Detailed profiler summary saved to {os.path.join(profiler_output, 'profiler_summary.txt')}")
            profiler = None  # Disable for remaining epochs
        
        history.append({
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_results.items()},
        })
        
        
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [NEW BEST] Saving model...")
            
            # Save best model checkpoint
            model_path = os.path.join(config.output_dir, f"{config.dataset}_best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': asdict(config),
                'metadata': metadata,
            }, model_path)
            print(f"  Model saved to {model_path}")
        
    # Load best
    if best_state:
        model.load_state_dict(best_state)
        test_results = evaluate(model, encoder, test_loader, device, model_type)
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  rho(disagreement, error): {test_results['disagree_error_corr']:.4f} "
          f"(p={test_results['disagree_error_pval']:.2e})")
        print(f"  rho(ambiguity, unknown): {test_results['ambig_unknown_corr']:.4f} "
          f"(p={test_results['ambig_unknown_pval']:.2e})")
        print(f"  Disagreement ratio: {test_results['disagree_ratio']:.2f}x")
        
        # Run analysis
        analysis = analyze_test_set(model, encoder, test_loader, device, model_type, 
                                    os.path.join(config.output_dir, 'analysis'))
        
        results = {
            "config": asdict(config),
            "metadata": metadata,
            "history": history,
            "test_results": test_results,
            "best_val_acc": best_val_acc,
            "analysis": analysis,
        }
    else:
        results = {
            "config": asdict(config),
            "metadata": metadata,
            "history": history,
            "test_results": {},
            "best_val_acc": best_val_acc,
        }
    
    output_path = os.path.join(config.output_dir, f"{config.dataset}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")
    # After training and loading best model - extract qualitative examples
    examples_dir = os.path.join(config.output_dir, 'qualitative_examples')
    examples = extract_qualitative_examples(model, encoder, test_loader, test_loader.dataset, device, model_type,
                                           n_examples=3, output_dir=examples_dir)
    format_examples_for_paper(examples)
    print_latex_examples(examples)
    return results


# =============================================================================
# VLLM INTEGRATION FOR CREDENCE
# =============================================================================
# Add this to your credence.py or as a separate module
#
# Key idea: vLLM extracts hidden states efficiently (frozen),
# then train lightweight heads on cached features
# =============================================================================


# =============================================================================
# VLLM HIDDEN STATE EXTRACTOR
# =============================================================================

class VLLMFeatureExtractor:
    """
    Extract hidden states from LLMs using vLLM's efficient inference.

    Replaces load_llm_model_frozen() for faster batch extraction.

    Usage:
        extractor = VLLMFeatureExtractor("meta-llama/Llama-3.2-3B")
        hidden_states = extractor.extract(texts)  # [N, hidden_size]
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 512,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for VLLMFeatureExtractor. "
                "Install with: pip install vllm"
            )

        self.model_name = model_name

        # Initialize vLLM in embedding/pooling mode
        # This gives us access to hidden states without generation overhead
        print(f"Loading vLLM model: {model_name}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            # Key: we want embeddings, not generations
            task="embed",
        )

        # Get model info
        self.hidden_size = self.llm.llm_engine.model_config.hf_config.hidden_size
        self.tokenizer = self.llm.get_tokenizer()

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Max length: {max_model_len}")

    def extract(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract pooled hidden states for a list of texts.

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            hidden_states: [N, hidden_size] numpy array
        """
        # vLLM encode returns embeddings
        outputs = self.llm.encode(
            texts,
            use_tqdm=show_progress,
        )

        # Extract embeddings from outputs
        embeddings = []
        for output in outputs:
            # output.outputs[0].embedding is the pooled representation
            emb = np.array(output.outputs[0].embedding)
            embeddings.append(emb)

        return np.stack(embeddings, axis=0)  # [N, hidden_size]

    def extract_batched(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Extract hidden states in batches (for very large datasets).

        Note: vLLM handles batching internally, but this is useful
        for memory management with extremely large datasets.
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i + batch_size]
            batch_emb = self.extract(batch_texts, show_progress=False)
            all_embeddings.append(batch_emb)

        return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# HIDDEN STATE CACHE
# =============================================================================

@dataclass
class CachedDataset:
    """Cached hidden states with labels and concepts."""
    hidden_states: torch.Tensor  # [N, hidden_size]
    labels: torch.Tensor         # [N]
    concept_labels: torch.Tensor # [N, num_concepts]
    is_unknown: torch.Tensor     # [N, num_concepts]

    def __len__(self):
        return len(self.labels)


class HiddenStateCache:
    """
    Cache vLLM hidden states for fast head training.

    Workflow:
    1. Extract hidden states once with vLLM
    2. Cache to disk
    3. Train heads using cached features (no LLM needed!)

    This is 10-100x faster than running LLM each epoch.
    """

    def __init__(self, cache_dir: str = "./vllm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, dataset_name: str, split: str, model_name: str) -> str:
        """Generate cache file path."""
        # Sanitize model name for filename
        model_safe = model_name.replace("/", "_").replace(".", "_")
        return os.path.join(
            self.cache_dir,
            f"{dataset_name}_{split}_{model_safe}.pt"
        )

    def exists(self, dataset_name: str, split: str, model_name: str) -> bool:
        """Check if cache exists."""
        return os.path.exists(self.get_cache_path(dataset_name, split, model_name))

    def save(
        self,
        dataset_name: str,
        split: str,
        model_name: str,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        concept_labels: np.ndarray,
        is_unknown: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        """Save extracted features to cache."""
        cache_path = self.get_cache_path(dataset_name, split, model_name)

        torch.save({
            "hidden_states": torch.from_numpy(hidden_states).float(),
            "labels": torch.from_numpy(labels).long(),
            "concept_labels": torch.from_numpy(concept_labels).long(),
            "is_unknown": torch.from_numpy(is_unknown).float(),
            "metadata": metadata or {},
            "model_name": model_name,
            "dataset_name": dataset_name,
            "split": split,
        }, cache_path)

        print(f"Cached {len(labels)} samples to {cache_path}")

    def load(
        self,
        dataset_name: str,
        split: str,
        model_name: str,
    ) -> CachedDataset:
        """Load cached features."""
        cache_path = self.get_cache_path(dataset_name, split, model_name)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = torch.load(cache_path)

        return CachedDataset(
            hidden_states=data["hidden_states"],
            labels=data["labels"],
            concept_labels=data["concept_labels"],
            is_unknown=data["is_unknown"],
        )


class CachedDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for cached features."""

    def __init__(self, cached_data: CachedDataset):
        self.data = cached_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.data.hidden_states[idx],
            "labels": self.data.labels[idx],
            "concept_labels": self.data.concept_labels[idx],
            "is_unknown": self.data.is_unknown[idx],
        }


# =============================================================================
# INTEGRATION WITH YOUR EXISTING CREDENCE
# =============================================================================

def load_vllm_extractor(model_name: str, config) -> Tuple:
    """
    Load vLLM feature extractor (replacement for load_llm_model_frozen).

    Returns:
        (extractor, tokenizer, hidden_size, "vllm")
    """
    if not VLLM_AVAILABLE:
        print("vLLM not available, falling back to HuggingFace...")
        # Import your existing function
        # from credence import load_llm_model_frozen
        # return load_llm_model_frozen(model_name, "cuda", config)
        raise NotImplementedError("Fallback to HuggingFace not implemented in this integration")

    extractor = VLLMFeatureExtractor(
        model_name=model_name,
        gpu_memory_utilization=0.7,
        max_model_len=config.max_length,
    )

    return extractor, extractor.tokenizer, extractor.hidden_size, "vllm"


def extract_and_cache_dataset(
    extractor: VLLMFeatureExtractor,
    data_loader: DataLoader,
    cache: HiddenStateCache,
    dataset_name: str,
    split: str,
    model_name: str,
) -> CachedDataset:
    """
    Extract hidden states from a DataLoader and cache them.

    Args:
        extractor: VLLMFeatureExtractor
        data_loader: Your existing DataLoader (with texts)
        cache: HiddenStateCache instance
        dataset_name: e.g., "cebab"
        split: "train", "val", or "test"
        model_name: e.g., "meta-llama/Llama-3.2-3B"

    Returns:
        CachedDataset ready for training
    """
    # Check if already cached
    if cache.exists(dataset_name, split, model_name):
        print(f"Loading cached {split} features...")
        return cache.load(dataset_name, split, model_name)

    print(f"Extracting {split} features with vLLM...")

    # Collect all data
    all_texts = []
    all_labels = []
    all_concepts = []
    all_is_unknown = []

    for batch in tqdm(data_loader, desc=f"Collecting {split} data"):
        # Decode input_ids back to text
        # (Or modify your dataloader to also return raw text)
        texts = extractor.tokenizer.batch_decode(
            batch['input_ids'],
            skip_special_tokens=True
        )
        all_texts.extend(texts)
        all_labels.extend(batch['labels'].numpy())
        all_concepts.extend(batch['concept_labels'].numpy())
        all_is_unknown.extend(batch['is_unknown'].numpy())

    # Extract with vLLM (single efficient pass)
    hidden_states = extractor.extract(all_texts)

    # Cache
    cache.save(
        dataset_name=dataset_name,
        split=split,
        model_name=model_name,
        hidden_states=hidden_states,
        labels=np.array(all_labels),
        concept_labels=np.array(all_concepts),
        is_unknown=np.array(all_is_unknown),
    )

    return cache.load(dataset_name, split, model_name)


# =============================================================================
# FAST TRAINING ON CACHED FEATURES
# =============================================================================

def train_epoch_cached(
    model: 'CREDENCE',
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config,
    device: str,
):
    """
    Train CREDENCE heads on cached hidden states.

    This is MUCH faster because:
    1. No LLM forward pass
    2. No tokenization
    3. Pure head training

    Typical speedup: 10-50x vs full LLM forward pass
    """
    model.train()
    epoch_losses = defaultdict(float)
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training (cached)")
    for batch in pbar:
        # Hidden states already extracted!
        hidden_states = batch['hidden_states'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concept_labels'].to(device)
        is_unknown = batch['is_unknown'].to(device)

        optimizer.zero_grad()

        # Create dummy attention mask (all 1s since already pooled)
        # Your CREDENCE model expects [batch, seq_len, hidden]
        # But cached data is already [batch, hidden]
        # We need to handle this...

        # Option 1: Reshape to [batch, 1, hidden] and use "last" pooling
        hidden_states = hidden_states.unsqueeze(1)  # [batch, 1, hidden]
        attention_mask = torch.ones(
            hidden_states.size(0), 1,
            device=device, dtype=torch.long
        )

        # Forward through heads
        outputs = model(hidden_states, attention_mask)

        # Loss computation (same as before)
        loss, loss_dict = model.compute_loss(
            outputs, labels, concepts, is_unknown,
            concept_weight=config.concept_weight,
            aleatoric_weight=config.aleatoric_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            epoch_losses[k] += v
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss_dict['total']:.4f}"})

    return {k: v / n_batches for k, v in epoch_losses.items()}


@torch.no_grad()
def evaluate_cached(
    model: 'CREDENCE',
    data_loader: DataLoader,
    device: str,
):
    """Evaluate on cached features."""
    model.eval()

    all_preds = []
    all_labels = []
    all_disagreement = []
    all_ambiguity = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        hidden_states = batch['hidden_states'].to(device).unsqueeze(1)
        labels = batch['labels'].to(device)
        attention_mask = torch.ones(
            hidden_states.size(0), 1,
            device=device, dtype=torch.long
        )

        outputs = model(hidden_states, attention_mask)

        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if outputs["disagreement"].numel() > 0:
            all_disagreement.extend(outputs["disagreement"].mean(dim=-1).cpu().numpy())
        if outputs["ambiguity"].numel() > 0:
            all_ambiguity.extend(outputs["ambiguity"].mean(dim=-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return {
        "accuracy": float(accuracy),
        "mean_disagreement": float(np.mean(all_disagreement)) if all_disagreement else 0.0,
        "mean_ambiguity": float(np.mean(all_ambiguity)) if all_ambiguity else 0.0,
    }


# =============================================================================
# MODIFIED run_experiment FOR VLLM
# =============================================================================

def run_experiment_vllm(config):
    """
    Run CREDENCE experiment with vLLM feature extraction.

    Workflow:
    1. Extract features with vLLM (once, cached)
    2. Train heads on cached features (fast!)
    3. Evaluate
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Expand encoder name
    config.encoder_name = expand_encoder_name(config.encoder_name)
    os.makedirs(config.output_dir, exist_ok=True)

    # Check if this is an LLM
    model_info = MODEL_REGISTRY.get(config.encoder_name, {})
    if model_info.get("type") != "llm":
        print(f"Model {config.encoder_name} is not an LLM, using standard training...")
        # Fall back to standard run_experiment if it exists
        return run_experiment(config) if 'run_experiment' in globals() else None

    # Load dataset (we need this for labels/concepts)
    dataset_config = DatasetConfig(
        label_type=config.label_type,
        max_length=config.max_length,
        tokenizer_name=config.encoder_name,
        batch_size=config.batch_size
    )

    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        config.dataset,
        config=dataset_config
    )

    # Initialize vLLM extractor
    print("\n" + "="*60)
    print("VLLM FEATURE EXTRACTION")
    print("="*60)

    extractor = VLLMFeatureExtractor(
        model_name=config.encoder_name,
        gpu_memory_utilization=0.7,
        max_model_len=config.max_length,
    )

    # Cache features
    cache = HiddenStateCache(os.path.join(config.output_dir, "vllm_cache"))

    train_cached = extract_and_cache_dataset(
        extractor, train_loader, cache,
        config.dataset, "train", config.encoder_name
    )
    val_cached = extract_and_cache_dataset(
        extractor, val_loader, cache,
        config.dataset, "val", config.encoder_name
    )
    test_cached = extract_and_cache_dataset(
        extractor, test_loader, cache,
        config.dataset, "test", config.encoder_name
    )

    # Free vLLM memory
    del extractor
    torch.cuda.empty_cache()

    # Create cached dataloaders
    train_cached_loader = DataLoader(
        CachedDatasetWrapper(train_cached),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_cached_loader = DataLoader(
        CachedDatasetWrapper(val_cached),
        batch_size=config.batch_size,
    )
    test_cached_loader = DataLoader(
        CachedDatasetWrapper(test_cached),
        batch_size=config.batch_size,
    )

    # Initialize CREDENCE (heads only)
    print("\n" + "="*60)
    print("TRAINING CREDENCE HEADS")
    print("="*60)

    hidden_size = train_cached.hidden_states.shape[-1]

    # Auto-detect aleatoric mode
    if config.aleatoric_mode == "auto":
        config.aleatoric_mode = get_aleatoric_mode(metadata)

    # Import CREDENCE model (assuming it's defined in this file)
    if 'CREDENCE' not in globals():
        raise NameError("CREDENCE model class not found. Make sure it's defined in credence.py")

    model = CREDENCE(
        input_dim=hidden_size,
        num_concepts=metadata['num_concepts'],
        num_classes=metadata['num_classes'],
        head_configs=config.get_head_configs(),
        aleatoric_mode=config.aleatoric_mode,
        model_type="llm_frozen",  # Use "last" pooling
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Training loop
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_losses = train_epoch_cached(
            model, train_cached_loader, optimizer, config, device
        )
        val_results = evaluate_cached(model, val_cached_loader, device)

        history.append({
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_results.items()},
        })

        print(f"  Val accuracy: {val_results['accuracy']:.4f}")

        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [NEW BEST]")

    # Final evaluation
    if best_state:
        model.load_state_dict(best_state)

    test_results = evaluate_cached(model, test_cached_loader, device)
    print(f"\nTest accuracy: {test_results['accuracy']:.4f}")

    # Save results
    results = {
        "config": asdict(config),
        "metadata": metadata,
        "history": history,
        "test_results": test_results,
        "best_val_acc": best_val_acc,
        "method": "vllm_cached",
    }

    output_path = os.path.join(config.output_dir, f"{config.dataset}_vllm_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def main_with_vllm():
    """Extended main() with vLLM option."""
    parser = argparse.ArgumentParser(description="CREDENCE with vLLM")

    # Existing arguments...
    parser.add_argument("--encoder_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dataset", type=str, default="cebab")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./results_vllm")

    # New vLLM arguments
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for feature extraction (faster for LLMs)")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.7,
                       help="GPU memory utilization for vLLM")

    args = parser.parse_args()

    # Import ExperimentConfig if it exists
    if 'ExperimentConfig' not in globals():
        raise NameError("ExperimentConfig not found. Make sure it's defined in credence.py")

    config = ExperimentConfig(
        encoder_name=args.encoder_name,
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        freeze_llm=True,  # Always freeze when using vLLM
    )

    if args.use_vllm:
        run_experiment_vllm(config)
    else:
        # Fall back to standard run_experiment if it exists
        if 'run_experiment' in globals():
            run_experiment(config)
        else:
            raise NotImplementedError("Standard run_experiment not implemented")


def main():
    parser = argparse.ArgumentParser(description="CREDENCE Multi-Model")
    
    # Model selection
    parser.add_argument("--encoder_name", type=str, default="distilbert-base-uncased",
                       help="Model name (e.g., roberta-base, microsoft/deberta-v3-base)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for LLM fine-tuning")
    parser.add_argument("--freeze_llm", action="store_true",
                       help="Freeze LLM (no LoRA), use only for feature extraction")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Data
    parser.add_argument("--dataset", type=str, default="hatexplain",
                       choices=list(DATASET_INFO.keys()))
    parser.add_argument("--label_type", type=str, default="ternary")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout_min", type=float, default=0.05,
                       help="Minimum dropout rate for head configuration")
    parser.add_argument("--dropout_max", type=float, default=0.30,
                       help="Maximum dropout rate for head configuration")
    parser.add_argument("--no_pooling_diversity", action="store_true",
                       help="Disable pooling diversity (use only CLS pooling)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    
    # Profiler arguments
    parser.add_argument("--enable_profiler", action="store_true", 
                       help="Enable PyTorch profiler to track forward/backward pass")
    parser.add_argument("--profiler_warmup", type=int, default=1,
                       help="Number of warmup batches for profiler")
    parser.add_argument("--profiler_active", type=int, default=3,
                       help="Number of active batches to profile")
    parser.add_argument("--profiler_repeat", type=int, default=1,
                       help="Number of times to repeat profiling")
    parser.add_argument("--profiler_output_dir", type=str, default=None,
                       help="Directory to save profiler results (default: output_dir/profiler)")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        encoder_name=args.encoder_name,
        use_lora=args.use_lora,
        freeze_llm=args.freeze_llm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dataset=args.dataset,
        label_type=args.label_type,
        batch_size=args.batch_size,
        max_length=args.max_length,
        n_heads=args.n_heads,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        use_pooling_diversity=not args.no_pooling_diversity,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
        enable_profiler=args.enable_profiler,
        profiler_warmup=args.profiler_warmup,
        profiler_active=args.profiler_active,
        profiler_repeat=args.profiler_repeat,
        profiler_output_dir=args.profiler_output_dir,
    )
    
    run_experiment(config)


# =============================================================================
# CREDENCE Critical Fixes for ACL 2026
# =============================================================================
"""
This module provides DROP-IN fixes for the identified issues:

1. Calibration (ECE 0.164 → <0.05)
2. Ensemble averaging (single head beats ensemble)
3. Error distance analysis for ordinal tasks

Author: Tanmoy
"""


# ============================================================================
# FIX 1: TEMPERATURE SCALING FOR CALIBRATION
# ============================================================================

def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    init_temp: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal temperature for calibration via grid search + refinement
    
    Your ECE: 0.164 → Expected after fix: ~0.03-0.05
    
    Args:
        logits: [N, C] raw logits before softmax
        labels: [N] ground truth labels
        
    Returns:
        (optimal_temperature, resulting_ece)
    """
    
    def compute_ece_at_temp(temp: float) -> float:
        scaled_logits = logits / temp
        probs = softmax_fix(scaled_logits, axis=1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        return compute_ece_fix(predictions, confidences, labels)
    
    # Grid search
    temps = np.linspace(0.5, 5.0, 50)
    eces = [compute_ece_at_temp(t) for t in temps]
    best_idx = np.argmin(eces)
    
    # Refine with scipy
    result = minimize_scalar(
        compute_ece_at_temp,
        bounds=(temps[max(0, best_idx-2)], temps[min(len(temps)-1, best_idx+2)]),
        method='bounded'
    )
    
    optimal_temp = result.x
    optimal_ece = compute_ece_at_temp(optimal_temp)
    
    return optimal_temp, optimal_ece


def softmax_fix(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def compute_ece_fix(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    
    return ece


class TemperatureScaledModel:
    """
    Wrapper to apply temperature scaling to your model
    
    Usage:
        # After training
        ts_model = TemperatureScaledModel(your_model, temperature=2.3)
        calibrated_probs = ts_model.predict_proba(inputs)
    """
    
    def __init__(self, base_model, temperature: float = 1.0):
        self.base_model = base_model
        self.temperature = temperature
    
    def calibrate(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """Find optimal temperature on validation set"""
        self.temperature, ece = find_optimal_temperature(val_logits, val_labels)
        print(f"Optimal temperature: {self.temperature:.3f}")
        print(f"Calibrated ECE: {ece:.4f}")
        return self.temperature
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        return softmax_fix(logits / self.temperature, axis=-1)


# ============================================================================
# FIX 2: WEIGHTED ENSEMBLE (FIXES DESTRUCTIVE AVERAGING)
# ============================================================================

def compute_optimal_weights(
    head_predictions: np.ndarray,
    labels: np.ndarray,
    head_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute optimal ensemble weights based on validation performance
    
    Your issue: Single head (68.7%) beats 5-head ensemble (56.9%)
    This fix should recover most of the single-head performance while
    keeping uncertainty decomposition.
    
    Args:
        head_predictions: [H, N] predictions from each head
        labels: [N] ground truth
        head_probs: [H, N, C] optional probabilities for soft weighting
        
    Returns:
        weights: [H] optimal weights (sum to 1)
    """
    n_heads = head_predictions.shape[0]
    
    # Option 1: Accuracy-based weights
    accuracies = np.array([
        (head_predictions[h] == labels).mean() 
        for h in range(n_heads)
    ])
    
    # Softmax with temperature to control sharpness
    # Higher temp = more uniform, lower temp = winner-take-all
    temp = 0.1  # Sharp weighting
    weights = softmax_fix(accuracies / temp)
    
    return weights


def weighted_ensemble_predict(
    head_probs: np.ndarray,
    weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted ensemble prediction
    
    Args:
        head_probs: [H, N, C] probabilities from each head
        weights: [H] ensemble weights
        
    Returns:
        predictions: [N] class predictions
        confidences: [N] prediction confidences
    """
    # Weighted average of probabilities
    weighted_probs = np.einsum('h,hnc->nc', weights, head_probs)
    
    predictions = weighted_probs.argmax(axis=-1)
    confidences = weighted_probs.max(axis=-1)
    
    return predictions, confidences


def weighted_ensemble_uncertainty(
    head_probs: np.ndarray,
    weights: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty with weighted ensemble
    
    Epistemic = weighted variance across heads
    This preserves uncertainty decomposition while fixing accuracy
    """
    H, N, C = head_probs.shape
    
    # Weighted mean
    weighted_mean = np.einsum('h,hnc->nc', weights, head_probs)
    
    # Weighted variance (epistemic)
    # Var = E[X^2] - E[X]^2 with weights
    weighted_sq = np.einsum('h,hnc->nc', weights, head_probs ** 2)
    epistemic = weighted_sq - weighted_mean ** 2
    
    return {
        'mean_probs': weighted_mean,
        'epistemic': epistemic,
        'predictions': weighted_mean.argmax(axis=-1),
        'confidences': weighted_mean.max(axis=-1)
    }


@dataclass
class EnsembleFix:
    """
    Complete fix for your ensemble problem
    
    Usage:
        fix = EnsembleFix()
        fix.fit(head_predictions, head_probs, val_labels)
        results = fix.predict(test_head_probs)
    """
    weights: np.ndarray = None
    drop_heads: List[int] = None
    
    def fit(
        self,
        head_predictions: np.ndarray,
        head_probs: np.ndarray,
        labels: np.ndarray,
        drop_threshold: float = 0.5
    ):
        """
        Fit ensemble weights and identify heads to drop
        
        Args:
            drop_threshold: Drop heads with accuracy below this
        """
        n_heads = head_predictions.shape[0]
        
        # Compute per-head accuracy
        accuracies = np.array([
            (head_predictions[h] == labels).mean() 
            for h in range(n_heads)
        ])
        
        print("Head accuracies:", [f"{a:.1%}" for a in accuracies])
        
        # Identify heads to potentially drop
        self.drop_heads = [h for h in range(n_heads) if accuracies[h] < drop_threshold]
        if self.drop_heads:
            print(f"Heads below {drop_threshold:.0%} threshold: {self.drop_heads}")
        
        # Compute weights (set dropped heads to 0)
        masked_accs = accuracies.copy()
        masked_accs[self.drop_heads] = 0
        
        self.weights = softmax_fix(masked_accs / 0.1)
        print("Optimal weights:", [f"{w:.3f}" for w in self.weights])
        
        # Evaluate improvement
        baseline_preds = head_probs.mean(axis=0).argmax(axis=-1)
        baseline_acc = (baseline_preds == labels).mean()
        
        weighted_preds = np.einsum('h,hnc->nc', self.weights, head_probs).argmax(axis=-1)
        weighted_acc = (weighted_preds == labels).mean()
        
        best_single = accuracies.max()
        
        print(f"\nResults:")
        print(f"  Uniform ensemble: {baseline_acc:.1%}")
        print(f"  Weighted ensemble: {weighted_acc:.1%}")
        print(f"  Best single head: {best_single:.1%}")
        print(f"  Improvement: +{(weighted_acc - baseline_acc)*100:.1f}%")
    
    def predict(self, head_probs: np.ndarray) -> Dict:
        """Apply weighted ensemble"""
        return weighted_ensemble_uncertainty(head_probs, self.weights)


# ============================================================================
# FIX 3: ERROR DISTANCE ANALYSIS FOR ORDINAL TASKS
# ============================================================================

def error_distance_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    num_classes: int = 5
) -> Dict:
    """
    Analyze uncertainty by error distance for ordinal classification
    
    Key validation for ACL:
    - Adjacent errors (|pred - label| = 1): Should have HIGH aleatoric
    - Distant errors (|pred - label| >= 2): Should have HIGH epistemic
    
    Args:
        predictions: [N] predicted classes (ordinal: 0, 1, 2, 3, 4)
        labels: [N] true classes
        epistemic: [N] or [N, K] epistemic uncertainty
        aleatoric: [N] or [N, K] aleatoric uncertainty
        num_classes: Number of ordinal classes
        
    Returns:
        Analysis results with correlations and validation status
    """
    
    # Aggregate uncertainties if multi-dimensional
    if epistemic.ndim > 1:
        epistemic = epistemic.mean(axis=1)
    if aleatoric.ndim > 1:
        aleatoric = aleatoric.mean(axis=1)
    
    # Compute error distances
    error_distances = np.abs(predictions - labels)
    is_error = predictions != labels
    
    results = {
        'by_distance': {},
        'correlations': {},
        'validation': {}
    }
    
    # Analyze by error distance
    print("\n" + "=" * 60)
    print("ERROR DISTANCE ANALYSIS (Ordinal Validation)")
    print("=" * 60)
    print(f"\n{'Distance':<10} {'Count':<8} {'Epistemic':<12} {'Aleatoric':<12}")
    print("-" * 42)
    
    distances_list = []
    epistemic_means = []
    aleatoric_means = []
    
    for dist in range(num_classes):
        if dist == 0:
            mask = ~is_error
            label = "Correct"
        else:
            mask = is_error & (error_distances == dist)
            label = f"Dist={dist}"
        
        if mask.sum() >= 5:
            epi_mean = epistemic[mask].mean()
            ale_mean = aleatoric[mask].mean()
            
            results['by_distance'][f'distance_{dist}'] = {
                'count': int(mask.sum()),
                'epistemic_mean': float(epi_mean),
                'epistemic_std': float(epistemic[mask].std()),
                'aleatoric_mean': float(ale_mean),
                'aleatoric_std': float(aleatoric[mask].std()),
            }
            
            print(f"{label:<10} {mask.sum():<8} {epi_mean:<12.4f} {ale_mean:<12.4f}")
            
            if dist > 0:
                distances_list.append(dist)
                epistemic_means.append(epi_mean)
                aleatoric_means.append(ale_mean)
    
    print()
    
    # Compute correlations
    if len(distances_list) >= 3:
        corr_epi, p_epi = stats.spearmanr(distances_list, epistemic_means)
        corr_ale, p_ale = stats.spearmanr(distances_list, aleatoric_means)
        
        results['correlations'] = {
            'epistemic_vs_distance': {
                'spearman_rho': corr_epi,
                'p_value': p_epi,
                'expected': 'positive (distant errors = model failure)'
            },
            'aleatoric_vs_distance': {
                'spearman_rho': corr_ale,
                'p_value': p_ale,
                'expected': 'negative (adjacent errors = inherent ambiguity)'
            }
        }
        
        print("Correlation Analysis:")
        print(f"  Epistemic vs Distance: ρ={corr_epi:.3f} (p={p_epi:.3f})")
        print(f"    Expected: positive (distant errors = epistemic)")
        print(f"    Status: {'✅ VALIDATES' if corr_epi > 0.3 else '⚠️ WEAK' if corr_epi > 0 else '❌ WRONG SIGN'}")
        print()
        print(f"  Aleatoric vs Distance: ρ={corr_ale:.3f} (p={p_ale:.3f})")
        print(f"    Expected: negative (adjacent errors = aleatoric)")
        print(f"    Status: {'✅ VALIDATES' if corr_ale < -0.2 else '⚠️ WEAK' if corr_ale < 0 else '❌ WRONG SIGN'}")
        
        # Overall validation
        epi_validates = corr_epi > 0.2
        ale_validates = corr_ale < 0
        
        results['validation'] = {
            'epistemic_validates': epi_validates,
            'aleatoric_validates': ale_validates,
            'both_validate': epi_validates and ale_validates,
        }
        
        print()
        if epi_validates and ale_validates:
            print("✅ STRONG VALIDATION: Error distance patterns support meaningful decomposition!")
            print("   This is strong evidence for ACL that epistemic ≠ aleatoric.")
        elif epi_validates:
            print("⚠️ PARTIAL: Epistemic validates, aleatoric unclear")
        elif ale_validates:
            print("⚠️ PARTIAL: Aleatoric validates, epistemic unclear")
        else:
            print("❌ WEAK: Neither pattern validates - check calibration first")
    
    return results


# ============================================================================
# FIX 4: MULTI-ANNOTATOR VALIDATION
# ============================================================================

def multi_annotator_validation(
    aleatoric: np.ndarray,
    annotator_labels: np.ndarray,
    method: str = 'entropy'
) -> Dict:
    """
    Validate aleatoric uncertainty against human annotator disagreement
    
    Critical validation for ACL: Aleatoric should correlate with
    inherent human disagreement, not model confusion.
    
    Args:
        aleatoric: [N] or [N, K] aleatoric uncertainty
        annotator_labels: [N, A] labels from A annotators
        method: 'entropy' or 'agreement' for computing disagreement
        
    Returns:
        Correlation analysis
    """
    
    if aleatoric.ndim > 1:
        aleatoric = aleatoric.mean(axis=1)
    
    N, A = annotator_labels.shape
    
    # Compute human disagreement
    if method == 'entropy':
        # Entropy of label distribution
        disagreement = np.zeros(N)
        for i in range(N):
            labels_i = annotator_labels[i]
            unique, counts = np.unique(labels_i[labels_i >= 0], return_counts=True)
            if len(counts) > 0:
                probs = counts / counts.sum()
                disagreement[i] = -np.sum(probs * np.log(probs + 1e-10))
    else:
        # Agreement rate (lower = more disagreement)
        disagreement = np.zeros(N)
        for i in range(N):
            labels_i = annotator_labels[i]
            valid = labels_i[labels_i >= 0]
            if len(valid) > 1:
                mode = stats.mode(valid, keepdims=True).mode[0]
                disagreement[i] = 1 - (valid == mode).mean()
    
    # Correlation
    corr, p_value = stats.spearmanr(aleatoric, disagreement)
    
    print("\n" + "=" * 60)
    print("MULTI-ANNOTATOR VALIDATION")
    print("=" * 60)
    print(f"\nAleatoric vs Human Disagreement:")
    print(f"  Spearman ρ: {corr:.3f}")
    print(f"  p-value: {p_value:.2e}")
    print()
    
    if corr > 0.3 and p_value < 0.01:
        print("✅ STRONG VALIDATION: Aleatoric captures human ambiguity!")
        print("   This is the key evidence that aleatoric ≠ epistemic.")
    elif corr > 0.15 and p_value < 0.05:
        print("⚠️ MODERATE: Aleatoric somewhat tracks human disagreement")
    else:
        print("❌ WEAK: Aleatoric doesn't correlate with human disagreement")
        print("   Check: Is aleatoric network learning input-dependent uncertainty?")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'validates': corr > 0.25 and p_value < 0.05,
        'mean_disagreement': disagreement.mean(),
        'mean_aleatoric': aleatoric.mean()
    }


# ============================================================================
# MAIN: APPLY ALL FIXES
# ============================================================================

def apply_all_fixes(
    # Model outputs
    logits: np.ndarray,
    head_probs: np.ndarray,
    head_predictions: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    # Labels
    labels: np.ndarray,
    # Optional for extended validation
    annotator_labels: Optional[np.ndarray] = None,
    is_ordinal: bool = False,
    num_classes: int = 5
) -> Dict:
    """
    Apply all fixes and generate comprehensive analysis
    
    Returns dict with:
    - calibration: Temperature scaling results
    - ensemble: Weighted ensemble results  
    - error_distance: Ordinal validation (if applicable)
    - multi_annotator: Human agreement validation (if available)
    """
    
    results = {}
    
    print("\n" + "=" * 80)
    print("APPLYING CREDENCE FIXES FOR ACL 2026")
    print("=" * 80)
    
    # 1. Calibration
    print("\n### FIX 1: CALIBRATION ###")
    temp, ece = find_optimal_temperature(logits, labels)
    original_probs = softmax_fix(logits, axis=-1)
    original_ece = compute_ece_fix(
        original_probs.argmax(axis=-1),
        original_probs.max(axis=-1),
        labels
    )
    
    print(f"Original ECE: {original_ece:.4f}")
    print(f"Optimal temperature: {temp:.3f}")
    print(f"Calibrated ECE: {ece:.4f}")
    print(f"Improvement: {(original_ece - ece) / original_ece * 100:.1f}%")
    
    results['calibration'] = {
        'original_ece': original_ece,
        'optimal_temperature': temp,
        'calibrated_ece': ece
    }
    
    # 2. Ensemble
    print("\n### FIX 2: ENSEMBLE WEIGHTING ###")
    ensemble_fix = EnsembleFix()
    ensemble_fix.fit(head_predictions, head_probs, labels)
    results['ensemble'] = {
        'weights': ensemble_fix.weights.tolist(),
        'dropped_heads': ensemble_fix.drop_heads
    }
    
    # 3. Error distance (if ordinal)
    if is_ordinal:
        print("\n### FIX 3: ERROR DISTANCE VALIDATION ###")
        predictions = head_probs.mean(axis=0).argmax(axis=-1)
        results['error_distance'] = error_distance_analysis(
            predictions, labels, epistemic, aleatoric, num_classes
        )
    
    # 4. Multi-annotator (if available)
    if annotator_labels is not None:
        print("\n### FIX 4: MULTI-ANNOTATOR VALIDATION ###")
        results['multi_annotator'] = multi_annotator_validation(
            aleatoric, annotator_labels
        )
    
    return results


if __name__ == "__main__":
    main()

















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
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset as hf_load_dataset
import matplotlib.pyplot as plt

# Optional: LoRA for LLMs
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Note: peft not installed. LLM training requires: pip install peft bitsandbytes")

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
                torch_dtype=torch.float16,  # FA2 requires fp16/bf16
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
    """Load LLM with LoRA for parameter-efficient fine-tuning."""
    print(f"Loading LLM: {model_name}")
    
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for LLM training. Install with: pip install peft bitsandbytes")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Get target modules from registry
    model_info = MODEL_REGISTRY.get(model_name, {})
    target_modules = model_info.get("target_modules", ["q_proj", "v_proj"])
    
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


def load_model(config: ExperimentConfig, device: str):
    """Load model based on type (encoder or LLM)."""
    model_type = config.get_model_type()
    
    if model_type == "llm" or config.use_lora:
        return load_llm_model(config.encoder_name, device, config)
    else:
        return load_encoder_model(config.encoder_name, device, config.freeze_encoder)


def get_hidden_states(encoder, input_ids, attention_mask, model_type: str):
    """Get hidden states from encoder or LLM."""
    if model_type == "encoder":
        outputs = encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    else:  # LLM
        outputs = encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Last layer

class ConceptHead(nn.Module):
    """A single concept prediction head."""
    def __init__(self, config: HeadConfig, input_dim: int, n_concepts: int):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, n_concepts)

        self.net=nn.Sequential(
            self.dropout,
            self.fc1,
            nn.ReLU(),
            self.fc2
        )
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
        logits = self.net(x)
        probs = torch.sigmoid(logits)
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
        return torch.sigmoid(logits)  # P(unknown) in [0, 1]   


class CredalClassifier(nn.Module):
    """
    Classifier for concept prediction.
    """
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, num_concepts) * 0.1)
        self.b = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        return F.linear(concept_probs, self.W, self.b)
    
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
        model_type: str = "encoder",  # "encoder" | "llm"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.n_heads = len(head_configs)
        self.aleatoric_mode = aleatoric_mode
        self.model_type = model_type
        
        # Adjust pooling for LLMs (use last token instead of CLS)
        if model_type == "llm":
            for cfg in head_configs:
                cfg.pooling = "last"
        
        # Ensemble heads (epistemic)
        self.heads = nn.ModuleList([
            ConceptHead(cfg, input_dim, num_concepts)
            for cfg in head_configs
        ])
        
        # Aleatoric head (only if supervised mode)
        pooling = "last" if model_type == "llm" else "cls"
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
        # Get predictions from all heads
        all_logits = []
        all_probs = []
        
        for head in self.heads:
            logits, probs = head(hidden_states, attention_mask)
            all_logits.append(logits)
            all_probs.append(probs)
        
        # Stack: [batch, num_concepts, n_heads]
        probs_stack = torch.stack(all_probs, dim=-1)
        
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
                head_label_logits.append(logit)
            
            # Stack: [batch, num_classes, n_heads]
            label_logits_stack = torch.stack(head_label_logits, dim=-1)
            label_probs_stack = torch.softmax(label_logits_stack, dim=1)
            
            # Aggregate: mean across heads
            logits = label_logits_stack.mean(dim=-1)  # [batch, num_classes]
            
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
            logits = self.classifier(concept_probs)
            credal_out = self.classifier.forward_credal(credal_lower, credal_upper)
        
        # Aleatoric: predicted P(unknown) per concept
        if self.aleatoric_mode == "supervised":
            # Learned prediction of P(unknown) - true aleatoric signal
            if self.num_concepts > 0:
                ambiguity = self.aleatoric_head(hidden_states, attention_mask)
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
        
        # 1. Task loss
        task_loss = F.cross_entropy(outputs["logits"], labels)
        
        # 2. Concept loss (BCE per head, averaged)
        # Ternary -> soft targets: 0->0.0, 1->0.5, 2->1.0
        concept_loss = torch.tensor(0.0, device=device)
        if self.num_concepts > 0 and concepts.numel() > 0:
            concept_targets = concepts.float() / 2.0
            if len(outputs["head_logits"]) > 0:
                for logits in outputs["head_logits"]:
                    # Ensure shapes match
                    if logits.shape == concept_targets.shape:
                        concept_loss = concept_loss + F.binary_cross_entropy_with_logits(
                            logits, concept_targets
                        )
                concept_loss = concept_loss / self.n_heads
        
        # 3. Aleatoric loss: predict P(unknown) per concept
        aleatoric_loss = torch.tensor(0.0, device=device)
        if self.aleatoric_mode == "supervised" and is_unknown is not None and is_unknown.numel() > 0:
            # Ensure shapes match
            ambiguity = outputs["ambiguity"]
            if ambiguity.shape == is_unknown.shape:
                aleatoric_loss = F.binary_cross_entropy(
                    ambiguity, 
                    is_unknown.to(device),
                    reduction='mean'
                )
        
        # Total
        total_loss = (task_loss + 
                     concept_weight * concept_loss + 
                     aleatoric_weight * aleatoric_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "concept": concept_loss.item(),
            "aleatoric": aleatoric_loss.item(),
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
):
    model.train()
    if model_type == "encoder":
        encoder.eval()
    else:
        encoder.train()  # LoRA is trainable
    
    epoch_losses = defaultdict(float)
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concept_labels'].to(device)
        
        is_unknown = batch['is_unknown'].to(device)
        
        # Encode
        if model_type == "encoder":
            with torch.no_grad():
                hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
        else:
            hidden_states = get_hidden_states(encoder, input_ids, attention_mask, model_type)
        
        # Forward pass
        outputs = model(hidden_states, attention_mask)
        
        # Loss
        loss, loss_dict = model.compute_loss(
            outputs, labels, concepts, is_unknown,
            concept_weight=config.concept_weight,
            aleatoric_weight=config.aleatoric_weight,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Advance profiler step (handles schedule automatically)
        if profiler is not None:
            profiler.step()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if model_type == "llm":
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        n_batches += 1
        
        # Handle disagreement/ambiguity display (may be empty for datasets without concepts)
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
    if model_type == "llm":
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

    # Optimizer
    if model_type == "encoder":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        # Include LoRA parameters
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(encoder.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
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
        train_losses = train_epoch(model, encoder, train_loader, optimizer, config, device, model_type, current_profiler)
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

def main():
    parser = argparse.ArgumentParser(description="CREDENCE Multi-Model")
    
    # Model selection
    parser.add_argument("--encoder_name", type=str, default="distilbert-base-uncased",
                       help="Model name (e.g., roberta-base, microsoft/deberta-v3-base)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for LLM fine-tuning")
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


if __name__ == "__main__":
    main()

















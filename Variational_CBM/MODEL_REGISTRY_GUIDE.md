# MODEL_REGISTRY and Multi-Dataset Training Guide

## Overview

The `main_train_hybrid_multi_dataset.py` script now supports:
- **Multiple datasets**: CEBaB, HateXplain, GoEmotions (with extensible architecture)
- **Multiple encoders**: DistilBERT, RoBERTa, DeBERTa-v3, ModernBERT (SOTA), LLMs
- **Quantization**: 4-bit/8-bit via BitsAndBytes for memory-efficient training
- **PEFT/LoRA**: Parameter-efficient fine-tuning for LLMs

## MODEL_REGISTRY

### Encoder Models (frozen encoder, train heads)

| Model | Hidden Size | Max Length | Notes |
|-------|-------------|------------|-------|
| `distilbert-base-uncased` | 768 | 512 | Default encoder |
| `roberta-base` | 768 | 512 | |
| `roberta-large` | 1024 | 512 | |
| `microsoft/deberta-v3-base` | 768 | 512 | Previous SOTA |
| `microsoft/deberta-v3-large` | 1024 | 512 | Previous SOTA |
| `answerdotai/ModernBERT-base` | 768 | 8192 | **Current SOTA** |
| `answerdotai/ModernBERT-large` | 1024 | 8192 | **Current SOTA** |

### LLM Models (LoRA fine-tuning)

| Model | Hidden Size | Target Modules | Notes |
|-------|-------------|----------------|-------|
| `microsoft/phi-3-mini-4k-instruct` | 3072 | qkv_proj, o_proj | Small, fast |
| `microsoft/Phi-3.5-mini-instruct` | 3072 | qkv_proj, o_proj | Latest Phi |
| `mistralai/Mistral-7B-v0.1` | 4096 | q/k/v/o_proj | |
| `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | q/k/v/o_proj | |
| `meta-llama/Llama-3.2-1B` | 2048 | q/k/v/o_proj | |
| `meta-llama/Llama-3.2-3B` | 3072 | q/k/v/o_proj | |
| `meta-llama/Llama-3.2-3B-Instruct` | 3072 | q/k/v/o_proj | |
| `Qwen/Qwen2.5-3B` | 2048 | q/k/v/o_proj | |

### Short Names

For convenience, you can use short names:
- `distilbert` → `distilbert-base-uncased`
- `roberta` → `roberta-base`
- `deberta` → `microsoft/deberta-v3-base`
- `modernbert` → `answerdotai/ModernBERT-base`
- `phi-3` → `microsoft/phi-3-mini-4k-instruct`
- `phi-3.5` → `microsoft/Phi-3.5-mini-instruct`
- `llama-3.2-3b` → `meta-llama/Llama-3.2-3B`

## Datasets

### Supported Datasets

| Dataset | Concepts | Classes | Description |
|---------|----------|---------|-------------|
| `cebab` | 4 | 5 | Restaurant reviews (food, service, ambiance, noise) |
| `hatexplain` | 2 | 3 | Hate speech detection (has_target, is_offensive) |
| `goemotions` | 27 | 28 | Emotion classification (multi-label) |

### Dataset-Specific Configurations

Each dataset has optimized defaults in `DATASET_CONFIGS`:
- Batch size
- Learning rate
- Number of epochs
- Prior sigma (σ₀)
- Error scale

## Usage Examples

### Basic Usage (DistilBERT encoder)

```bash
# Train on HateXplain with default encoder
python main_train_hybrid_multi_dataset.py --dataset hatexplain

# Train on CEBaB
python main_train_hybrid_multi_dataset.py --dataset cebab

# Train on GoEmotions
python main_train_hybrid_multi_dataset.py --dataset goemotions
```

### Using ModernBERT (SOTA Encoder)

```bash
# Base model
python main_train_hybrid_multi_dataset.py --dataset cebab --encoder modernbert

# Large model (more capacity)
python main_train_hybrid_multi_dataset.py --dataset cebab --encoder modernbert-large
```

### Using LLMs with Quantization and LoRA

```bash
# Phi-3 with 4-bit quantization and LoRA
python main_train_hybrid_multi_dataset.py \
    --dataset goemotions \
    --encoder phi-3 \
    --quantization 4bit \
    --use_lora

# Llama 3.2 3B with 8-bit quantization
python main_train_hybrid_multi_dataset.py \
    --dataset cebab \
    --encoder llama-3.2-3b \
    --quantization 8bit \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

# Mistral 7B with custom LoRA parameters
python main_train_hybrid_multi_dataset.py \
    --dataset hatexplain \
    --encoder mistralai/Mistral-7B-v0.1 \
    --quantization 4bit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16
```

### Encoder Fine-Tuning

```bash
# Unfreeze encoder for end-to-end fine-tuning
python main_train_hybrid_multi_dataset.py \
    --dataset hatexplain \
    --encoder deberta \
    --unfreeze_encoder

# Lower learning rate for encoder fine-tuning
python main_train_hybrid_multi_dataset.py \
    --dataset cebab \
    --encoder modernbert \
    --unfreeze_encoder \
    --lr 5e-6
```

### Custom Training Parameters

```bash
# Override default epochs and learning rate
python main_train_hybrid_multi_dataset.py \
    --dataset hatexplain \
    --encoder roberta \
    --num_epochs 20 \
    --lr 5e-5 \
    --batch_size 32
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | choice | `cebab` | Dataset: `cebab`, `hatexplain`, `goemotions` |
| `--encoder` | str | `distilbert` | Encoder model (short name or full HF name) |
| `--num_epochs` | int | (dataset default) | Override default epochs |
| `--lr` | float | (dataset default) | Override learning rate |
| `--batch_size` | int | (dataset default) | Override batch size |
| `--quantization` | choice | `none` | Quantization: `none`, `4bit`, `8bit` |
| `--use_lora` | flag | False | Enable LoRA fine-tuning |
| `--lora_r` | int | 8 | LoRA rank |
| `--lora_alpha` | int | 16 | LoRA alpha |
| `--freeze_encoder` | flag | True | Freeze encoder (default behavior) |
| `--unfreeze_encoder` | flag | False | Unfreeze encoder for fine-tuning |

## Technical Details

### Quantization

When `--quantization 4bit` or `--quantization 8bit` is specified:
- Encoder is loaded via `load_encoder_with_quantization()`
- Uses `BitsAndBytesConfig` from transformers
- 4-bit quantization uses NF4 format (recommended for LLMs)
- Automatic device mapping for multi-GPU setups

**Requirements**:
```bash
pip install bitsandbytes
```

### LoRA Fine-Tuning

When `--use_lora` is specified:
- LoRA is applied to target modules (attention projections)
- Only LLM models get LoRA by default (encoder models skip it)
- Model is prepared for k-bit training if quantized
- Dramatically reduces trainable parameters

**Requirements**:
```bash
pip install peft
```

### Encoder Replacement

The script uses a simple encoder replacement strategy:
1. HybridCredalCBM is created with default encoder
2. If quantization/LoRA is enabled, the encoder is replaced:
   ```python
   model.encoder = quantized_or_lora_encoder
   ```

This works because:
- All encoder models share the same interface (AutoModel)
- The model architecture expects a encoder with forward() method
- Gradients flow correctly through the replaced encoder

## Adding New Datasets

To add a new dataset:

1. **Implement data loader** (e.g., `load_mydataset_direct.py`)
2. **Add to DATASET_CONFIGS**:
   ```python
   'mydataset': {
       'name': 'MyDataset',
       'num_concepts': K,
       'concept_names': [...],
       'num_classes': C,
       'save_dir': './checkpoints/hybrid_credal_mydataset',
       'data_loader': get_mydataset_dataloaders,
       'use_multi_loader': False,
       'loader_kwargs': {...},
       'prior_sigma': 0.5,
       'error_scale': 1.0,
       'learning_rate': 1e-3,
       'num_epochs': 10,
   }
   ```

3. **Update argument parser**:
   ```python
   parser.add_argument('--dataset', choices=[..., 'mydataset'])
   ```

## Adding New Encoders

To add a new encoder to MODEL_REGISTRY:

1. **Add to registry**:
   ```python
   MODEL_REGISTRY = {
       ...
       "org/encoder-name": {
           "type": "encoder",  # or "llm"
           "hidden_size": 768,
           "max_length": 512,
           "use_token_type_ids": False,
           # For LLMs only:
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
       }
   }
   ```

2. **Add short name mapping** (optional):
   ```python
   ENCODER_SHORT_NAMES = {
       ...
       "encoder": "org/encoder-name",
   }
   ```

## Performance Considerations

### Memory Usage

| Configuration | VRAM (approx.) | Speed |
|---------------|----------------|-------|
| DistilBERT (frozen) | 2 GB | Fastest |
| ModernBERT-base (frozen) | 3 GB | Fast |
| DeBERTa-v3-large (frozen) | 4 GB | Medium |
| Phi-3 (4-bit + LoRA) | 4 GB | Medium |
| Mistral-7B (4-bit + LoRA) | 8 GB | Slowest |

### Training Speed

1. **Frozen encoders**: Fastest, only train small heads
2. **Unfrozen encoders**: 2-3x slower, full fine-tuning
3. **LLMs with LoRA**: 3-5x slower than frozen encoders
4. **Quantization**: Slows down training by ~20%, but saves memory

### Recommendations

- **For prototyping**: Use DistilBERT or ModernBERT-base (frozen)
- **For SOTA results**: Use ModernBERT-large (unfrozen)
- **For limited GPU memory**: Use LLM + 4-bit quantization + LoRA
- **For production**: Use DeBERTa-v3-large or ModernBERT (frozen)

## Troubleshooting

### BitsAndBytes Errors

If you see:
```
Warning: bitsandbytes not found, 4-bit quantization unavailable
```

Install bitsandbytes:
```bash
pip install bitsandbytes
```

### PEFT Errors

If you see:
```
Warning: PEFT not found, LoRA fine-tuning unavailable
```

Install peft:
```bash
pip install peft
```

### CUDA Out of Memory

Try:
1. Use smaller batch size: `--batch_size 8`
2. Use 4-bit quantization: `--quantization 4bit`
3. Use smaller encoder: `--encoder phi-3`
4. Use gradient checkpointing (not yet implemented)

### Model Not Training

Check:
1. `freeze_encoder=False` or use `--unfreeze_encoder`
2. Learning rate is appropriate (1e-5 to 1e-3)
3. Dataset is loading correctly
4. Check GPU memory usage: `nvidia-smi`

## Citation

If you use this code, please cite:
```
@article{credal_cbm_2026,
  title={Credal Concept Bottleneck Models},
  author={Tanmoy},
  journal={ICML 2026},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.

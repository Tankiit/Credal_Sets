# Variational Credal CBM

Variational Credal Concept Bottleneck Model with uncertainty decomposition for ICML 2026.

## Features

- **Variational Inference**: Bayesian neural networks with epistemic uncertainty
- **Credal Sets**: Imprecise probability bounds via quantile estimation
- **Uncertainty Decomposition**: Separate epistemic and aleatoric pathways
- **K-class Concepts**: Multi-class concept predictions (e.g., negative/unknown/positive)
- **Multiple Covariance Families**: Mean-field, low-rank, and full covariance
- **Intervention Experiments**: Compare epistemic vs aleatoric targeting
- **Quadrant Routing**: 4-quadrant decision framework
- **Multiple Datasets**: AMIQA, CEBaB support
- **Flexible Encoders**: RoBERTa, DeBERTa, ModernBERT, LLMs, and more

## Installation

```bash
# Install dependencies
pip install torch transformers scipy scikit-learn tqdm numpy
```

## Quick Start

### Basic Usage

```bash
# Train with default settings (DistilBERT on dummy data)
python main.py

# Train on AMIQA with RoBERTa
python main.py --dataset amiqa --encoder roberta

# Train on CEBaB with ModernBERT
python main.py --dataset cebab --encoder modernbert

# Train with experiments
python main.py --dataset amiqa --encoder deberta --run-intervention --run-ablation
```

### Using the Shell Script

```bash
# Make script executable (first time only)
chmod +x run_all_datasets.sh

# Run with specific encoder on all datasets
./run_all_datasets.sh roberta

# Run with specific encoder on specific dataset
./run_all_datasets.sh modernbert amiqa

# Run with additional arguments
./run_all_datasets.sh deberta cebab --run-intervention --run-ablation
```

## Available Datasets

- **`amiqa`**: Multi-aspect sentiment analysis dataset
- **`cebab`**: Restaurant review dataset with multi-dimensional annotations
- **`dummy`**: Dummy dataset for testing (default)

## Available Encoders

### Standard Encoder Models

| Short Name | Full Model Name | Notes |
|------------|-----------------|-------|
| `roberta` | `roberta-base` | RoBERTa base (2019) |
| `roberta-large` | `roberta-large` | RoBERTa large (2019) |
| `deberta` | `microsoft/deberta-v3-base` | DeBERTa-v3 base (2021) |
| `deberta-v3-base` | `microsoft/deberta-v3-base` | DeBERTa-v3 base |
| `deberta-v3-large` | `microsoft/deberta-v3-large` | DeBERTa-v3 large |
| `distilbert` | `distilbert-base-uncased` | DistilBERT (2019) |
| `modernbert` | `answerdotai/ModernBERT-base` | ModernBERT base (2024 - SOTA) |
| `modernbert-base` | `answerdotai/ModernBERT-base` | ModernBERT base |
| `modernbert-large` | `answerdotai/ModernBERT-large` | ModernBERT large |

### LLM Models (Frozen by Default)

| Short Name | Full Model Name | Notes |
|------------|-----------------|-------|
| `phi-3-mini` | `microsoft/Phi-3-mini-4k-instruct` | Phi-3 mini |
| `phi-3.5-mini` | `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 mini |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral 7B |
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 1B |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B |
| `qwen-0.5b` | `Qwen/Qwen0.5-0.5B-Instruct` | Qwen 0.5B |
| `qwen-1.5b` | `Qwen/Qwen1.5-1.5B-Instruct` | Qwen 1.5B |
| `gemma-2b` | `google/gemma-2b-it` | Gemma 2B |

**Note**: LLMs use frozen feature extraction by default. To fine-tune with LoRA, set `--freeze-encoder False`.

You can also use any HuggingFace model by providing the full model name.

## Configuration

### Command-Line Arguments

```bash
python main.py \
    --dataset amiqa \                    # Dataset: amiqa, cebab, or dummy
    --encoder roberta \                   # Encoder (short name or full HF name)
    --num-concepts 4 \                   # Number of concepts
    --concept-classes 3 \                # K classes per concept (default: 3)
    --num-classes 2 \                    # Number of task classes (default: 2)
    --covariance mean_field \            # Covariance: mean_field, low_rank, full
    --epochs 10 \                        # Number of epochs
    --batch-size 16 \                    # Batch size
    --lr 2e-5 \                          # Learning rate
    --device cuda \                      # Device: cuda or cpu
    --save-dir ./checkpoints \           # Save directory
    --data-path ./data \                 # Data directory
    --run-intervention \                 # Run intervention experiments
    --run-ablation                       # Run covariance ablation study
```

### Saving/Loading Configuration

```bash
# Train and save config
python main.py --dataset amiqa --encoder roberta --save-dir ./experiments/amiqa-roberta

# Load from saved config
python main.py --config ./experiments/amiqa-roberta/config.json
```

## Model Architecture

```
Input → Encoder → Hidden
         ↓
┌─────────────────────────────────────┐
│ EPISTEMIC PATHWAY                   │
│ VariationalLinearZC → Concepts      │
│                    → Epistemic      │
│                    → Credal bounds  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ ALEATORIC PATHWAY (Separate!)       │
│ AleatoricHead → Aleatoric          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ CONCEPT SUPERVISION                 │
│ KClassConceptClassifier → K-class   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ TASK CLASSIFIER                     │
│ CredalClassifier → Prediction       │
└─────────────────────────────────────┘
```

## Output Structure

```
experiments/
├── amiqa/
│   └── roberta-base/
│       ├── config.json           # Configuration
│       ├── best_model.pt         # Best checkpoint
│       └── final_model.pt        # Final checkpoint
└── cebab/
    └── deberta-v3-base/
        ├── config.json
        ├── best_model.pt
        └── final_model.pt
```

## Experiments

### 1. Intervention Experiments

Compares epistemic vs aleatoric concept targeting:

```bash
python main.py --dataset amiqa --encoder roberta --run-intervention
```

**Expected Output**:
- Baseline accuracy
- Intervention gains for epistemic, aleatoric, and random strategies
- Results for k=1,2,3,4 concepts

### 2. Covariance Ablation Study

Compares different covariance structures:

```bash
python main.py --dataset cebab --encoder deberta --run-ablation
```

**Covariance Families**:
- `mean_field`: Diagonal covariance (O(n) params)
- `low_rank`: Low-rank + diagonal (O(nk) params)
- `full`: Full covariance (O(n²) params)

### 3. Quadrant Analysis

Automatic 4-quadrant routing during evaluation:

- **TRUST**: Low epistemic, Low aleatoric → Auto-approve
- **REVIEW**: Low epistemic, High aleatoric → Human review
- **DATA**: High epistemic, Low aleatoric → Collect more data
- **ABSTAIN**: High epistemic, High aleatoric → Expert review

## Key Metrics

The model tracks:
- **Accuracy**: Task prediction accuracy
- **Epistemic-Error Correlation**: Should be positive (model uncertain about mistakes)
- **Aleatoric-Unknown Correlation**: Should be positive (model detects ambiguity)
- **Epistemic-Aleatoric Separation**: Should be high (independent uncertainty sources)
- **Credal Coverage**: Fraction of true values in credal intervals
- **Quadrant Performance**: Per-quadrant accuracy

## File Structure

```
.
├── VCBM.py                 # Core model implementation
│   ├── VariationalCredalConfig
│   ├── VariationalLinearZC
│   ├── KClassConceptClassifier
│   ├── AleatoricHead
│   ├── CredalClassifier
│   ├── VariationalCredalCBM
│   ├── QuadrantRouter
│   └── InterventionExperiment
│
├── main.py                 # Training script with dataset loading
│   ├── get_config()
│   ├── load_amiqa()
│   ├── load_cebab()
│   ├── train_model()
│   └── main()
│
├── run_all_datasets.sh    # Multi-dataset runner script
└── README.md              # This file
```

## Citation

```bibtex
@inproceedings{tanmoy2026variational,
  title={Variational Credal Concept Bottleneck Models},
  author={Tanmoy},
  booktitle={ICML},
  year={2026}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

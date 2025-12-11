# Credal Concept Bottleneck Model (Credal CBM) with Token Embeddings

A fast implementation of Credal Concept Bottleneck Models using token embeddings from transformer models. Supports multiple datasets, model architectures, and concept configurations.

## [KEY FEATURES] Key Features

## [USAGE] Usage Examples

### [QUICK START] Quick Start - Run These First!

| Experiment | Command | What It Tests | Time |
|------------|---------|---------------|------|
| **Fast Test** | `python run_token_embeddings.py --encoder bert-tiny --dataset sst2 --n-concepts 5 --train-size 100` | Basic functionality | 2-3 min |
| **Real Concepts** | `python run_token_embeddings.py --encoder bert-tiny --dataset cebab --n-concepts 4 --train-size 100` | Human concepts | 5-8 min |
| **Multi-Class** | `python run_token_embeddings.py --encoder distilbert --dataset ag_news --n-concepts 8 --train-size 200` | Topic classification | 8-12 min |

 # Start with CEBaB (real concepts, reasonable runtime)
  python run_token_embeddings.py --encoder distilbert --dataset cebab --n-concepts 4

  # Then try SST-2 (fastest, good baseline)
  python run_token_embeddings.py --encoder distilbert --dataset sst2 --n-concepts 15


### [WEEKLY PLAN] One Week Experiment Plan

**Goal**: Validate system works across configurations and understand uncertainty behavior.

| Day | Dataset | Model | Command | What It Tests |
|-----|---------|-------|---------|---------------|
| **Day 1** | SST-2 | `bert-tiny` | `python run_token_embeddings.py --encoder bert-tiny --dataset sst2 --n-concepts 5 --train-size 100 --test-size 50` | Basic functionality |
| **Day 1** | SST-2 | `distilbert` | `python run_token_embeddings.py --encoder distilbert --dataset sst2 --n-concepts 10 --train-size 200 --test-size 100` | Model size comparison |
| **Day 2** | CEBaB | `bert-tiny` | `python run_token_embeddings.py --encoder bert-tiny --dataset cebab --n-concepts 4 --train-size 100 --test-size 50` | Real concepts |
| **Day 3** | AG News | `distilbert` | `python run_token_embeddings.py --encoder distilbert --dataset ag_news --n-concepts 12 --max-depth 15 --train-size 300` | Multi-class + depth |
| **Day 4** | IMDB | `bert-tiny` | `python run_token_embeddings.py --encoder bert-tiny --dataset imdb --n-concepts 6 --train-size 100 --test-size 50` | Longer text |
| **Day 5** | HateXplain | `bert-tiny` | `python run_token_embeddings.py --encoder bert-tiny --dataset hatexplain --n-concepts 8 --train-size 100 --test-size 50` | Challenging task |
| **Day 6** | SST-2 | `distilbert` | `python run_token_embeddings.py --encoder distilbert --dataset sst2 --layers [0,3,5] --n-concepts 8 --train-size 200` | Multi-layer extraction |

### [TRACKING] What to Track

**For Each Experiment:**
- **Label Accuracy**: How well final prediction works
- **Mean Epistemic**: Model uncertainty level
- **Concept Accuracy** (CEBaB): Real concept learning
- **Training Time**: Performance benchmark

**Quick Results Check:**
```bash
# After each run, check key metrics
cat results/*_results.json | jq '.label_accuracy, .mean_epistemic, .model_info.model_name'
```

### [SUCCESS] Success Criteria

- [OK] **Day 1-2**: System works with basic datasets
- [OK] **Day 3-4**: Multi-class and longer text handled
- [OK] **Day 5**: Can handle challenging cases
- [OK] **Day 6**: Advanced features working

**Expected Patterns:**
- Larger models → Higher accuracy, more uncertainty
- More concepts → Better features, longer training
- Real concepts (CEBaB) → More interpretable
- Multi-layer extraction → Better performance

### [TROUBLESHOOTING] Quick Troubleshooting

```bash
# System test (1 minute)
python run_token_embeddings.py --encoder bert-tiny --dataset sst2 --train-size 50 --test-size 25 --n-concepts 3

# Check GPU usage
nvidia-smi

# Available models
python -c "from run_token_embeddings import ENCODERS; print(list(ENCODERS.keys())[:10])"
```

**Common Issues:**
- **CUDA out of memory**: Use `--encoder bert-tiny` or reduce `--train-size`
- **Slow embedding**: Reduce `--batch-size`
- **Low accuracy**: Increase `--n-concepts` or `--train-size`

- **Multiple Datasets**: SST-2, CEBaB, AG News, HateXplain, GoEmotions, Civil Comments, SNLI, IMDB
- **Many Models**: BERT, RoBERTa, DeBERTa, Llama, Mistral, Sentence Transformers (20+ encoders)
- **Enhanced Embeddings**: Extract from multiple layers, different pooling strategies (CLS, mean, max, concat)
- **Real & Synthetic Concepts**: Support for human-annotated concepts (CEBaB) and synthetic concepts
- **Uncertainty Quantification**: Credal sets with epistemic uncertainty estimation
- **Fast Training**: GPU-accelerated token embedding extraction

## [CONFIG] Configuration Options

## [DATASETS] Supported Datasets

### [DATASET GUIDE] Which Dataset Should You Play With?

Here's a quick guide to choosing the right dataset for your experiments:

| Dataset | Difficulty | Real Concepts | Multi-Class | Use Case | Recommended For |
|---------|------------|---------------|-------------|----------|------------------|
| **SST-2** | [EASY] Easy | [NO] No | [NO] Binary | Quick testing | Beginners, fast experiments |
| **CEBaB** | [MEDIUM] Medium | [YES] Yes | [MULTI-CLASS] Multi-class | Real concept analysis | Concept bottleneck research |
| **AG News** | [EASY] Easy | [NO] No | [MULTI-CLASS] Multi-class | Topic classification | Multi-class experiments |
| **IMDB** | [EASY] Easy | [NO] No | [NO] Binary | Long text analysis | Longer sequences testing |
| **Civil Comments** | [MEDIUM] Medium | [NO] No | [NO] Binary | Large scale learning | Big data experiments |
| **HateXplain** | [HARD] Hard | [NO] No | [MULTI-CLASS] Multi-class | Social NLP research | Challenging classification |
| **GoEmotions** | [HARD] Hard | [NO] No | [MULTI-LABEL] Multi-label | Complex emotions | Multi-label research |
| **SNLI** | [MEDIUM] Medium | [NO] No | [MULTI-CLASS] Multi-class | NLP reasoning | Logical reasoning tasks |

### [EXPERIMENTS] Fun Experimentation Ideas

#### **For Beginners (Start Here)**
1. **SST-2 + BERT-tiny**: Fastest way to see the system work
   ```bash
   python run_token_embeddings.py --encoder bert-tiny --dataset sst2 --train-size 100
   ```
2. **AG News + DistilBERT**: Multi-class classification with clear topics
   ```bash
   python run_token_embeddings.py --encoder distilbert --dataset ag_news --train-size 500
   ```

#### **For Concept Bottleneck Research**
1. **CEBaB + RoBERTa**: Real human-annotated concepts about restaurants
   ```bash
   python run_token_embeddings.py --encoder roberta-base --dataset cebab --n-concepts 4
   ```
2. **CEBaB + Multiple Layers**: See if earlier transformer layers capture different aspects
   ```bash
   python run_token_embeddings.py --encoder roberta-base --dataset cebab --layers [0,6,11] --n-concepts 4
   ```

#### **For Advanced Research**
1. **HateXplain + Llama-3**: Challenging hate speech detection with large model
   ```bash
   python run_token_embeddings.py --encoder llama-3-8b --dataset hatexplain --n-concepts 15
   ```
2. **GoEmotions + Sentence-BERT**: Multi-label emotion classification with optimized embeddings
   ```bash
   python run_token_embeddings.py --encoder sentence-bert-large --dataset goemotions --n-concepts 20
   ```

#### **For Performance Testing**
1. **Civil Comments + Full Dataset**: Large-scale learning with 100K+ samples
   ```bash
   python run_token_embeddings.py --encoder deberta-v3 --dataset civil_comments --train-size 50000
   ```

#### **For Uncertainty Research**
1. **SNLI + Multiple Layers**: Test how epistemic uncertainty varies across layers
   ```bash
   python run_token_embeddings.py --encoder bert-base --dataset snli --layers [0,6,11,12] --n-concepts 12
   ```

### [CEBAB] CEBaB (Restaurant Reviews) - **[RECOMMENDED] RECOMMENDED FOR CONCEPT RESEARCH**
- **Why it's interesting**: **Only dataset with real human-annotated concepts!** Perfect for testing true concept bottleneck models.
- **Task**: Multi-class sentiment analysis (1-5 star ratings)
- **Real Concepts**: 4 human-annotated aspects per review
  - `food_aspect_majority`: Food quality (Positive/Negative/Unknown)
  - `service_aspect_majority`: Service quality (Positive/Negative/Unknown)
  - `ambiance_aspect_majority`: Ambiance (Positive/Negative/Unknown)
  - `noise_aspect_majority`: Noise level (Positive/Negative/Unknown)
- **What you can test**:
  - How well do models learn human-interpretable concepts?
  - Can uncertainty predict when humans disagree on concepts?
  - Do different transformer layers capture different aspects (food vs service)?
- **Fun experiments**: Compare synthetic vs real concepts, test uncertainty calibration
- **Usage**: `--dataset cebab --n-concepts 4`
- **Challenge**: Some reviews have "unknown" concepts - great for uncertainty modeling!

### [SST2] SST-2 (Stanford Sentiment Treebank) - **[BEGINNER] PERFECT FOR BEGINNERS**
- **Why it's interesting**: Clean, binary sentiment dataset - the "hello world" of text classification!
- **Task**: Binary sentiment classification (movie reviews)
- **Labels**: 0 (Negative), 1 (Positive)
- **Concepts**: Synthetic (automatically created via clustering + PCA)
- **What you can test**:
  - Does the system work at all? (quick validation)
  - How does uncertainty correlate with prediction errors?
  - Can synthetic concepts capture sentiment patterns?
- **Fun experiments**: Test different model sizes, compare layer extraction strategies
- **Usage**: `--dataset sst2`
- **Speed**: Very fast! Complete runs in <5 minutes with BERT-tiny
- **Challenge**: Synthetic concepts may not align with human interpretation

### [AGNEWS] AG News (News Topic Classification) - **[MULTICLASS] GREAT FOR MULTI-CLASS TESTING**
- **Why it's interesting**: Clear, distinct topics - perfect for multi-class experiments!
- **Task**: 4-class topic classification (news headlines)
- **Labels**: 0 (World), 1 (Sports), 2 (Business), 3 (Sci/Tech)
- **Concepts**: Synthetic (but topics are very distinct)
- **What you can test**:
  - How does uncertainty vary between clear vs ambiguous topics?
  - Can synthetic concepts capture topic-specific patterns?
  - Multi-class vs binary classification performance
- **Fun experiments**:
  - Test if business vs world news confuses the model
  - See if sports articles have different uncertainty patterns
  - Compare topic-specific concept learning
- **Usage**: `--dataset ag_news`
- **Special**: Very clear class boundaries - great for understanding model behavior
- **Challenge**: Some topics overlap (e.g., tech business news)

### [HATEXPLAIN] HateXplain (Hate Speech Detection) - **[CHALLENGING] CHALLENGING & SOCIALLY IMPORTANT**
- **Why it's interesting**: Hate speech detection with human explanations - very nuanced classification!
- **Task**: Multi-class toxicity classification with rationales
- **Labels**: 0 (Normal), 1 (Hate), 2 (Offensive)
- **Concepts**: Synthetic (but you can use rationales as real concepts!)
- **Special Features**: Provides human-written rationales for classifications
- **What you can test**:
  - Can uncertainty detect ambiguous hate speech?
  - Do rationales help models learn better concepts?
  - How do models handle context-dependent toxicity?
- **Fun experiments**:
  - Test if uncertainty is higher for borderline cases
  - Compare hate vs offensive vs normal speech uncertainty
  - See if certain words trigger higher uncertainty
- **Usage**: `--dataset hatexplain`
- **Social Impact**: Important for content moderation research
- **Challenge**: Very subjective - humans often disagree on labels!

### [GOEMOTIONS] GoEmotions (Emotion Classification) - **[COMPLEX] MOST COMPLEX & FUN**
- **Why it's interesting**: 28 different emotions, multi-label - humans can feel multiple emotions at once!
- **Task**: Multi-label emotion classification (Reddit comments)
- **Labels**: 28 emotion categories (admiration, amusement, anger, annoyance, etc.)
- **Concepts**: Synthetic (but very rich emotional landscape)
- **What you can test**:
  - Can uncertainty predict emotional complexity?
  - Do models capture co-occurring emotions?
  - How does uncertainty vary for different emotions?
- **Fun experiments**:
  - Test if joyful vs angry comments have different uncertainty
  - See if uncertainty is higher for complex emotional states
  - Compare individual emotion vs overall sentiment prediction
- **Usage**: `--dataset goemotions`
- **Special**: Multi-label - great for testing uncertainty in complex scenarios
- **Challenge**: Very fine-grained emotions, high subjectivity

### [CIVILCOMMENTS] Civil Comments (Toxicity Detection) - **[LARGE] LARGE SCALE & REAL WORLD**
- **Why it's interesting**: Massive dataset (100K+ samples) from real-world online comments!
- **Task**: Binary toxicity classification (civil discourse)
- **Labels**: 0 (Not toxic), 1 (Toxic)
- **Concepts**: Synthetic (but huge amount of data for pattern learning)
- **Special**: Large dataset with demographic annotations
- **What you can test**:
  - How does uncertainty scale with data size?
  - Can large datasets reduce epistemic uncertainty?
  - Real-world performance of uncertainty quantification
- **Fun experiments**:
  - Train on 1K vs 10K vs 50K samples and compare uncertainty
  - Test if certain toxic patterns have higher uncertainty
  - See how model confidence changes with more data
- **Usage**: `--dataset civil_comments`
- **Scale**: Perfect for testing scalability of uncertainty methods
- **Challenge**: Very large dataset, can be slow to process

### [SNLI] SNLI (Natural Language Inference) - **[LOGIC] LOGICAL REASONING TEST**
- **Why it's interesting**: Test if models can understand logical relationships between sentences!
- **Task**: Entailment classification (sentence relationships)
- **Labels**: 0 (Entailment), 1 (Neutral), 2 (Contradiction)
- **Examples**:
  - Entailment: "A cat is sleeping" → "An animal is sleeping"
  - Contradiction: "The cat is black" → "The cat is white"
  - Neutral: "The cat sleeps" → "The cat dreams of fish"
- **Concepts**: Synthetic (but tests logical reasoning capabilities)
- **What you can test**:
  - Can uncertainty detect logical uncertainty?
  - Do models understand different types of logical relationships?
  - How does uncertainty vary for different inference types?
- **Fun experiments**:
  - Test if contradictions have different uncertainty than entailments
  - Compare uncertainty for clear vs ambiguous logical relationships
  - See if uncertainty correlates with inference difficulty
- **Usage**: `--dataset snli`
- **Special**: Tests fundamental NLP reasoning capabilities
- **Challenge**: Requires understanding sentence semantics and logic

### [IMDB] IMDB (Movie Reviews) - **[LONGTEXT] LONGER TEXT & REAL REVIEWS**
- **Why it's interesting**: Full-length movie reviews - much longer than other datasets!
- **Task**: Binary sentiment classification (actual movie reviews)
- **Labels**: 0 (Negative), 1 (Positive)
- **Concepts**: Synthetic (but real-world review language)
- **Special Features**: Longer texts (hundreds of words), real user reviews
- **What you can test**:
  - How does uncertainty scale with text length?
  - Do longer reviews have different uncertainty patterns?
  - Can models capture nuanced opinions in long texts?
- **Fun experiments**:
  - Test if review length correlates with uncertainty
  - Compare uncertainty for very positive vs very negative reviews
  - See if sarcasm and irony increase uncertainty
- **Usage**: `--dataset imdb`
- **Advantage**: Tests model on longer, more natural text
- **Challenge**: Longer processing time, more complex language patterns

## [MODELS] Supported Models

### BERT Family
| Encoder | Model | Parameters | Layers | Hidden Size |
|---------|-------|------------|--------|-------------|
| `bert-tiny` | `prajjwal1/bert-tiny` | 4.4M | 2 | 128 |
| `distilbert` | `distilbert-base-uncased` | 66M | 6 | 768 |
| `bert-base` | `bert-base-uncased` | 110M | 12 | 768 |
| `bert-large` | `bert-large-uncased` | 340M | 24 | 1024 |

### RoBERTa Family
| Encoder | Model | Parameters | Layers | Hidden Size |
|---------|-------|------------|--------|-------------|
| `roberta-base` | `roberta-base` | 125M | 12 | 768 |
| `roberta-large` | `roberta-large` | 355M | 24 | 1024 |

### DeBERTa Family
| Encoder | Model | Parameters | Layers | Hidden Size |
|---------|-------|------------|--------|-------------|
| `deberta-v3` | `microsoft/deberta-v3-base` | 184M | 12 | 768 |
| `deberta-v3-large` | `microsoft/deberta-v3-large` | 434M | 24 | 1024 |

### Llama Family
| Encoder | Model | Parameters | Layers | Hidden Size |
|---------|-------|------------|--------|-------------|
| `llama-2-7b` | `meta-llama/Llama-2-7b-hf` | 7B | 32 | 4096 |
| `llama-2-13b` | `meta-llama/Llama-2-13b-hf` | 13B | 40 | 5120 |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B` | 8B | 32 | 4096 |
| `llama-3-70b` | `meta-llama/Meta-Llama-3-70B` | 70B | 80 | 8192 |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` | 8B | 32 | 4096 |
| `llama-3.1-70b` | `meta-llama/Llama-3.1-70B` | 70B | 80 | 8192 |

### Mistral Family
| Encoder | Model | Parameters | Layers | Hidden Size |
|---------|-------|------------|--------|-------------|
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | 7B | 32 | 4096 |
| `mistral-7b-instruct` | `mistralai/Mistral-7B-Instruct-v0.1` | 7B | 32 | 4096 |
| `mixtral-8x7b` | `mistralai/Mixtral-8x7B-v0.1` | 47B | 32 | 4096 |
| `mixtral-8x7b-instruct` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | 32 | 4096 |

### Other Models
| Encoder | Model | Parameters | Notes |
|---------|-------|------------|-------|
| `gpt-2` | `gpt2` | 124M | Text generation model |
| `gpt-2-medium` | `gpt2-medium` | 355M | Larger GPT-2 |
| `electra-base` | `google/electra-base-discriminator` | 110M | Discriminator-only |
| `t5-base` | `t5-base` | 220M | Text-to-text model |
| `flan-t5-base` | `google/flan-t5-base` | 220M | Instruction-tuned T5 |

### Sentence Transformers (Optimized for Embeddings)
| Encoder | Model | Parameters | Use Case |
|---------|-------|------------|----------|
| `sentence-bert` | `sentence-transformers/all-MiniLM-L6-v2` | 22M | Fast, general purpose |
| `
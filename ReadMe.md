# Replaceable Visual Backbones vs. Human Perceptual Uncertainty on CIFAR-10H

Frozen visual embeddings of the CIFAR-10 test set are compared against the
CIFAR-10H human soft labels to find where model embedding geometry and human
perception disagree ("blind spots"). The feature extractor is now modular:
Hugging Face vision encoders and `timm` backbones share the same interface and
produce the same downstream cache format.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download data

```bash
python download_data_cifar.py
```

This downloads the CIFAR-10 test set and CIFAR-10H soft labels into `./data/`.
Order matters: the scripts never shuffle the test set, so row `i` in the cached
embeddings remains aligned with row `i` in CIFAR-10H.

## 3. Choose a visual backbone

All backbone-specific loading and preprocessing live in `backbones.py`.
`feature_extraction.py` only iterates over images and saves global embeddings.

### Hugging Face DINOv3 (default)

```bash
python feature_extraction.py \
  --backbone hf \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m \
  --out-dir features/dinov3
```

DINOv3 checkpoints may require accepting the model license on Hugging Face and
logging in with `hf auth login`.

### Hugging Face DINOv2

```bash
python feature_extraction.py \
  --backbone hf \
  --model facebook/dinov2-base \
  --out-dir features/dinov2
```

### timm ResNet-50

```bash
python feature_extraction.py \
  --backbone timm \
  --model resnet50 \
  --out-dir features/resnet50
```

### timm ConvNeXt-Tiny

```bash
python feature_extraction.py \
  --backbone timm \
  --model convnext_tiny \
  --out-dir features/convnext_tiny
```

The backbone contract is deliberately small:

```python
backbone = build_backbone(kind, model_name, device)
embeddings = backbone.encode_pil(images)  # (batch, feature_dim)
```

Each backend owns its native pretrained preprocessing. Hugging Face models use
their `AutoImageProcessor`; timm models use the data configuration associated
with the pretrained weights.

Every extraction writes the same files:

```text
embeddings.npy   # (10000, D) frozen global embeddings
labels.npy       # (10000,) CIFAR-10 ground-truth labels
meta.json        # provider, model id, dimension, extraction metadata
```

This stable cache contract means downstream uncertainty analysis does not need
to know which backbone produced the representation.

## 4. Analyze blind spots

Point the analysis at any feature cache:

```bash
python blindspot_analysis.py --features-dir features/dinov2
```

The current analysis compares:

- **Human uncertainty:** Shannon entropy of the CIFAR-10H soft label vector.
- **Model structural confidence:** k-NN label purity in the selected frozen
  embedding space.

Outputs are written to `./results/` by default.

## Backbone design

```text
raw PIL image
     │
     ▼
FrozenVisualBackbone
     │
     ├── HuggingFaceBackbone
     │      ├── DINOv2
     │      ├── DINOv3
     │      └── other AutoModel-compatible vision encoders
     │
     └── TimmBackbone
            ├── ResNet
            ├── ConvNeXt
            ├── ViT
            └── other timm models with num_classes=0
     │
     ▼
(N, D) embeddings.npy
     │
     ▼
unchanged uncertainty / blind-spot analysis
```

To add another backend later, subclass `FrozenVisualBackbone`, implement
`encode_pil()`, `feature_dim`, and `info`, then register it in
`build_backbone()`. No downstream analysis code should need to change.

## Files

```text
download_data_cifar.py   # CIFAR-10 + CIFAR-10H
backbones.py             # common interface + HF/timm implementations
feature_extraction.py    # backend-agnostic feature caching
blindspot_analysis.py    # entropy vs embedding-purity analysis
requirements.txt
```

# Replaceable image backbones for uncertainty and supervised LVMs

This branch extracts frozen, global image embeddings behind one provider-independent
interface. Downstream uncertainty, concept-bottleneck, and supervised-LVM code always
receives a matrix `z` with shape `(num_images, feature_dim)`. The extractor now
produces L2-normalized embeddings, which makes cosine-based nearest-neighbor
analysis more stable.

```python
from models import build_backbone

backbone = build_backbone(
    kind="hf",
    model_name="facebook/dinov2-base",
    device="cuda",
)

z = backbone.encode_pil(images)  # shape: (B, D)
```

## Repository layout

```text
.
├── models/
│   ├── __init__.py
│   └── backbones.py          # shared Hugging Face/timm backbone interface
├── dataloaders/
│   ├── __init__.py
│   └── cifar10.py            # ordered PIL-image CIFAR-10 test loader
├── feature_extraction.py     # CIFAR-10 -> frozen global embeddings
├── blindspot_analysis.py     # CIFAR-10H entropy vs embedding-space purity
├── main.py                   # original exploratory entry point
├── requirements.txt          # Python dependencies, including timm
└── ReadMe.md
```

After downloading data and extracting one or more backbones, generated files have
this layout:

```text
.
├── data/
│   ├── cifar10/              # torchvision CIFAR-10 files
│   └── cifar-10h/
│       └── data/
│           ├── cifar10h-probs.npy   # 10,000 x 10 human soft labels
│           └── cifar10h-counts.npy  # raw human vote counts
├── features/
│   ├── dinov2/
│   │   ├── embeddings.npy    # shape: (10,000, D)
│   │   ├── labels.npy        # shape: (10,000,)
│   │   └── meta.json
│   ├── dinov3/
│   │   └── ...               # same three-file contract
│   └── resnet50/
│       └── ...               # same three-file contract
└── results/                  # blind-spot analysis outputs
```

Generated `data/`, `features/`, and `results/` directories are ignored by Git.

## Setup and data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download CIFAR-10 and CIFAR-10H using your preferred workflow before running the
feature extraction or blind-spot analysis scripts.

The extraction loader is deliberately not shuffled. Consequently,
`embeddings.npy[i]`, `labels.npy[i]`, and CIFAR-10H row `i` stay aligned.
When CUDA is available, the loader also uses pinned memory to speed up transfer
into the backbone.

## Extract features

DINOv2 is the default supervised-LVM baseline:

```bash
python feature_extraction.py \
  --backbone hf \
  --model facebook/dinov2-base \
  --out-dir features/dinov2
```

DINOv3 remains selectable:

```bash
python feature_extraction.py \
  --backbone hf \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m \
  --out-dir features/dinov3
```

The DINOv3 checkpoint is gated. Accept its Hugging Face license and authenticate
with `hf auth login` before running that command.

Any compatible `timm` model uses the same output contract:

```bash
python feature_extraction.py \
  --backbone timm \
  --model resnet50 \
  --out-dir features/resnet50

python feature_extraction.py \
  --backbone timm \
  --model convnext_tiny \
  --out-dir features/convnext_tiny
```

The device is selected automatically (`cuda`, then `mps`, then `cpu`). Override it
with `--device`. Use `--batch-size` and `--num-workers` to tune extraction.

Every run writes exactly:

```text
features/<backbone>/
├── embeddings.npy
├── labels.npy
└── meta.json
```

The `meta.json` file records whether the embeddings were normalized.

Downstream code therefore remains independent of the model provider:

```python
from pathlib import Path
import numpy as np

features_dir = Path("features/dinov2")
embeddings = np.load(features_dir / "embeddings.npy")
labels = np.load(features_dir / "labels.npy")
```

## Analyze CIFAR-10H blind spots

```bash
python blindspot_analysis.py --features-dir features/dinov2
```

Run `python blindspot_analysis.py --help` for all paths and analysis options. The
analysis compares human-label entropy with local class purity in frozen embedding
space; it does not train a classifier.

## Supervised-LVM boundary

```text
PIL images
    |
    v
models/backbones.py ── encode_pil(images) ──> z in R^(B x D)
                                           |
                                           v
                                    ConceptEncoder(z)
                                           |
                                           v
                                           c
                                      /         \
                                     v           v
                         R(c): concept loss   h(c): task prediction
```

Future datasets such as CUB, CUB-S, and Shapes3D should adapt only their dataset
loaders to supply PIL images. They should continue using `encode_pil`; the backbone
and supervised-LVM layers do not need dataset-specific changes.

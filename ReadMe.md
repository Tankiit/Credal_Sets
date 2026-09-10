# Replaceable image backbones for uncertainty and supervised LVMs

Experiment entry points and configs live in separate
[`experiments/synthetic/`](experiments/synthetic/README.md) and
[`experiments/real/`](experiments/real/README.md) folders. Outputs default to
`results/synthetic/` and `results/real/`. The
[ICLR experiment plan](experiments/README.md) records the ordered studies and
decision checkpoint after synthetic step 4.

Training uses tqdm progress bars and TensorBoard logging for both experiment
families; extraction uses tqdm as well. Run `tensorboard --logdir results` to
view losses, accuracy, and audit diagnostics. Use `--log-dir` to override a run's
TensorBoard root.

The native concept-model and audit pipeline is now available in
[`concept_audit/`](concept_audit/README.md), including optional PyC support:

```bash
python -m experiments.synthetic.run --out results/synthetic/concept_audit.json
```

This default is a synthetic smoke experiment. See the package guide for frozen
image features, concept annotations, separate evaluation caches, and audit semantics.

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
│   ├── huggingface.py        # shared HF load_dataset interface
│   └── cifar10.py            # legacy torchvision CIFAR-10 loader
├── feature_extraction.py     # HF CIFAR-10/100/CUB -> frozen embeddings
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

Image extraction uses Hugging Face `datasets.load_dataset` and downloads/cache-manages
CIFAR-10, CIFAR-100, or CUB automatically. Download CIFAR-10H separately for blind-spot
analysis. `--cache-dir` controls the HF cache location; the old extraction
`--images-dir` option is replaced by this HF cache option.

Extraction preserves the selected HF split order. Use the full CIFAR-10 test split
for CIFAR-10H analysis and ensure its row order matches your CIFAR-10H annotations.
The legacy torchvision loader remains available for code that uses local CIFAR files.

## Extract features

Start with the existing local CIFAR-10 copy (no dataset download):

```bash
python feature_extraction.py --dataset cifar10 --data-dir /Users/cril/tanmoy/research/data --split test --out-dir features/real/cifar10/dinov2/test
```

Local Python batches are read with torchvision and exposed through the same HF
dataset interface. The source files are unchanged. Use `--split train` and a
separate output directory for training features.

```python
from dataloaders import load_dataset

dataset = load_dataset("cifar100", split="train", cache_dir="data/huggingface")
image, concepts, label = dataset[0]
```

```bash
python feature_extraction.py --dataset cifar100 --split train --out-dir features/real/cifar100/train
python feature_extraction.py --dataset cub --split test --out-dir features/real/cub/test
```

These HF mirrors have task labels but no concept vectors (`concepts=None`).
Use `--concept-column` or an aligned `--concepts-file` for supervised concept
experiments; details are in the [concept-audit guide](concept_audit/README.md).


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

Image-only runs write the files below. Annotated runs additionally write `concepts.npy`:

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

CIFAR-10, CIFAR-100, and CUB share the HF loader. Other HF repositories can provide
explicit image, label, and concept-column mappings. All use `encode_pil`; the
backbone and supervised-LVM layers do not need dataset-specific changes.

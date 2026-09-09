# DINOv3 vs. Human Perceptual Uncertainty on CIFAR-10H

Frozen DINOv3 (ViT-L/16) embeddings of the CIFAR-10 test set, compared against
the CIFAR-10H human soft labels, to find where the model's embedding
geometry and human perception disagree ("blind spots").

Run this on your own machine with a GPU and internet access — it will **not**
run inside a sandboxed/offline environment, because it needs to:
1. Download CIFAR-10 test images (torchvision → cs.toronto.edu)
2. Download the DINOv3 checkpoint from Hugging Face (gated — see below)

## 0. Get access to DINOv3 weights (one-time)

The checkpoint used here, `facebook/dinov3-vitl16-pretrain-lvd1689m`, is
**gated** on Hugging Face. Before running:
1. Go to https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and
   accept the license (Meta DINOv3 License).
2. Run `huggingface-cli login` (or `hf auth login`) locally with a token
   that has access, or set `HF_TOKEN` in your environment.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs `transformers>=4.56.0` (DINOv3 support landed in that release).

## 2. Download data

```bash
python download_data_cifar.py
```

This downloads:
- CIFAR-10 test set (10,000 images) via `torchvision.datasets.CIFAR10`
- `cifar10h-probs.npy` (10000×10 soft labels, human classification
  probabilities per image) and `cifar10h-counts.npy` (raw vote counts)
  directly from the CIFAR-10H GitHub repo

Both land in `./data/`. **Order matters**: the CIFAR-10H labels are in the
same order as `torchvision`'s default (unshuffled) CIFAR-10 test set, so
`cifar10h-probs.npy[i]` corresponds to `CIFAR10(train=False)[i]`. The
scripts never shuffle this order — don't add `shuffle=True` to any loader
here or the alignment breaks silently.

## 3. Extract frozen DINOv3 features

```bash
python feature_extraction.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --batch-size 128
```

- Loads DINOv3 ViT-L/16 frozen (`requires_grad_(False)`, `eval()` mode,
  `torch.inference_mode()` — no gradients, no fine-tuning).
- Upsamples CIFAR's native 32×32 images to 224×224 (DINOv3's expected input
  size / patch-16 grid) using the model's own `AutoImageProcessor`, so
  normalization matches training.
- Extracts, per image: the **CLS token** (`pooler_output`, 1024-d for
  ViT-L) as the global embedding, and optionally the **patch token grid**
  for later dense-feature inspection (off by default — large, use
  `--save-patch-tokens` to enable).
- Saves everything to `./features/`:
  - `embeddings.npy` — (10000, D) CLS embeddings
  - `labels.npy` — (10000,) ground-truth CIFAR-10 integer labels
  - `meta.json` — model name, dim, extraction settings

This is the slow step (ViT-L over 10k images). On a single modern GPU
expect low-single-digit minutes at batch size 128; CPU will take much
longer — reduce `--batch-size` if you're memory constrained, or switch to
`facebook/dinov3-vits16-pretrain-lvd1689m` for a quick end-to-end smoke
test before committing to ViT-L.

## 4. Analyze blind spots (no classifier trained)

```bash
python blindspot_analysis.py
```

Since the goal is to probe the frozen embedding *geometry* itself rather
than a trained classifier, "model confidence" is estimated in an
unsupervised way:

- **Human uncertainty** per image: Shannon entropy of the CIFAR-10H soft
  label vector (0 = every annotator agreed, log2(10)≈3.32 = uniform
  disagreement across all 10 classes).
- **Model structural confidence** per image: k-NN label purity in DINOv3
  embedding space — for each image, look at its k nearest neighbors (cosine
  distance) among the *other* 9,999 embeddings, using ground-truth CIFAR-10
  labels, and compute the fraction that share the image's true class. High
  purity = the image sits in a locally homogeneous, well-separated region of
  DINOv3's feature space (structurally "easy" for the model, with no
  classifier needed to say so). Low purity = the image's neighbors are
  mixed in true class (structurally "confusable" region).

These two signals are then cross-tabulated into four quadrants:

| | Human confident (low entropy) | Human uncertain (high entropy) |
|---|---|---|
| **Model locally pure (high purity)** | Easy — both agree | *Human blind spot*: DINOv3 geometry cleanly separates the class but humans disagree (e.g. ambiguous photography, mislabeled-looking image) |
| **Model locally impure (low purity)** | *Model blind spot*: humans agree confidently, but the image sits in a confused region of DINOv3's embedding space | Hard — both agree it's ambiguous |

Outputs (in `./results/`):
- `per_image_metrics.csv` — index, true label, human entropy, human top
  choice, agreement with ground truth, k-NN purity, quadrant
- `quadrant_scatter.png` — entropy vs. purity scatter, quadrant-colored
- `correlation.txt` — Spearman correlation between human entropy and model
  purity (+ p-value)
- `blind_spots_model.csv` / `blind_spots_human.csv` — top-N images in each
  "blind spot" quadrant, sorted by how extreme they are, with file paths to
  the images (saved as PNGs in `./results/blind_spot_images/`) so you can
  visually inspect them

Tune `--k` (neighbors, default 10) and `--top-n` (images per report,
default 30) as flags.

## Files

```
download_data.py          # CIFAR-10 test images + CIFAR-10H soft labels
feature_extraction.py     # frozen DINOv3 embeddings -> features/
blindspot_analysis.py     # entropy vs kNN-purity blind spot analysis -> results/
requirements.txt
```
"""Frozen access to the student's native concept checkpoints (concept_audit.training).

Checkpoints store weights, backend, readout matrix and blocks, but not the readout
name; the readout is recovered by matching the saved matrix against the three
layouts train_and_audit can build, and must match exactly.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from concept_audit.models import NativeCBM, NativeCEM
from concept_audit.readouts import IdentityReadout, CoordinateReadout, GroupReadout


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _cbm_layouts(k):
    # Mirrors train_and_audit; keep in sync if new readouts are added there.
    yield 'identity', IdentityReadout(k), None
    yield 'coordinates', CoordinateReadout(2*k, range(k)), [(j, k+j) for j in range(k)]
    blocks = [(2*j, 2*j+1) for j in range(k)]
    yield 'grouped', GroupReadout(2*k, blocks), blocks


def build_model(ckpt):
    """Rebuild (model, readout_name) for a train_and_audit checkpoint dict."""
    matrix = ckpt['readout_matrix'].float()
    k, dim, classes = matrix.shape[0], ckpt['feature_dim'], ckpt['num_classes']
    if ckpt['backend'] == 'cem':
        model, name = NativeCEM(dim, classes, k), 'block_projection'
    elif ckpt['backend'] == 'native':
        matches = [(n, r, b) for n, r, b in _cbm_layouts(k) if r.matrix.shape == matrix.shape and torch.equal(r.matrix, matrix)]
        if len(matches) != 1:
            raise ValueError('Saved readout matrix matches no known CBM readout layout')
        name, readout, blocks = matches[0]
        model = NativeCBM(dim, classes, readout, blocks)
    else:  # 'pyc' needs the optional pyc dependency and is not used by these experiments
        raise NotImplementedError(f"Unsupported backend {ckpt['backend']!r}")
    model.load_state_dict(ckpt['model_state_dict'])  # strict: every tensor present and shaped alike
    if model.blocks != tuple(map(tuple, ckpt['blocks'])) or not torch.equal(model.readout.matrix, matrix):
        raise ValueError('Rebuilt blocks/readout differ from the checkpoint')
    return model, name


def load_student(path, device='cpu', dtype=torch.float64):
    """Load a checkpoint as an eval-mode model with every parameter frozen."""
    ckpt = torch.load(Path(path), map_location='cpu', weights_only=True)
    model, readout_name = build_model(ckpt)
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, readout_name, ckpt


@torch.no_grad()
def encode_features(model, embeddings, device, batch=16384):
    """c = model.encode(z) for every row of a cached (n, feature_dim) embedding array, in the model's dtype."""
    p = next(model.parameters())
    out = torch.empty(len(embeddings), model.latent_dim(), dtype=p.dtype)
    for i in range(0, len(embeddings), batch):
        z = torch.from_numpy(np.array(embeddings[i:i+batch])).to(device=device, dtype=p.dtype)
        out[i:i+batch] = model.encode(z).cpu()
    return out


def cached_latents(model, checkpoint, cache_full_dir, out_path, device):
    """Latent states for the whole full cache, stored with the checkpoint and cache hashes they came from."""
    ckpt_sha = sha256(checkpoint)
    ids_sha = sha256(Path(cache_full_dir)/'ids.npy')
    out_path = Path(out_path)
    if out_path.exists():
        saved = np.load(out_path)
        if str(saved['checkpoint_sha256']) == ckpt_sha and str(saved['ids_sha256']) == ids_sha:
            return torch.from_numpy(saved['latents'])
        raise ValueError(f'{out_path} was computed from a different checkpoint or cache')
    z = np.load(Path(cache_full_dir)/'embeddings.npy', mmap_mode='r')
    c = encode_features(model, z, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, latents=c.numpy(), ids=np.load(Path(cache_full_dir)/'ids.npy', allow_pickle=True).astype(str),
             checkpoint=str(checkpoint), checkpoint_sha256=ckpt_sha, ids_sha256=ids_sha)
    return c


@torch.no_grad()
def readout_and_logits(model, c, device, batch=65536):
    p = next(model.parameters())
    scores, logits = [], []
    for i in range(0, len(c), batch):
        x = c[i:i+batch].to(device=device, dtype=p.dtype)
        scores.append(model.concept_readout(x).cpu())
        logits.append(model.predict_from_concepts(x).cpu())
    return torch.cat(scores), torch.cat(logits)


def decode_levels(scores, level_values):
    """Readout scores are BCE-with-logits logits: sigmoid -> nearest declared level value (index)."""
    v = torch.as_tensor(level_values, dtype=scores.dtype)
    return (scores.sigmoid()[..., None] - v).abs().argmin(-1)


def decode_calibrated(pair_scores, level_scores):
    """Nearest level by (lower, upper) score, against per-level median scores fitted on training images."""
    ref = torch.as_tensor(level_scores, dtype=pair_scores.dtype)
    return torch.cdist(pair_scores, ref).argmin(-1)

"""Inventory student checkpoints and caches; verify Shapes3D ID/factor provenance.

Run from the repository root (paths in the student's reports are relative to it):

    python -m experiments.semantic_edits.inventory --out results/semantic_edits/inventory

Writes inventory.json and inventory.md. A checkpoint is usable only if it rebuilds
strictly and reproduces its reported held-out task accuracy within TOLERANCE.
"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .native_adapter import load_student, sha256

# At most one flipped argmax on the evaluation rows (GPU vs CPU float64 roundoff).
TOLERANCE = 'abs(accuracy - reported) <= 1 / n_eval'
HUES, SHAPES = 10, 4
FACTORS = ('floor', 'wall', 'object', 'scale', 'shape', 'orientation')
FACTOR_LEVELS = {'floor': 10, 'wall': 10, 'object': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
CONCEPT_NAMES = ('floor_hue', 'wall_hue', 'scale', 'orientation')


def load_raw_factors(raw_dir):
    """All raw Shapes3D factor columns (values and indices), no image decoding."""
    from datasets import load_from_disk
    ds = load_from_disk(str(raw_dir))
    if ds._indices is not None:
        raise ValueError('Raw dataset has an index mapping; arrow rows would not be dataset rows')
    cols = ['index'] + [f for f in FACTORS] + [f'{f}_idx' for f in FACTORS]
    # Read the arrow table directly: the datasets numpy formatter imports torchvision video APIs.
    return ds, {c: ds.data.column(c).to_numpy() for c in cols}


def array_summary(path):
    a = np.load(path, mmap_mode=None if path.name == 'ids.npy' else 'r', allow_pickle=path.name == 'ids.npy')
    out = {'shape': list(a.shape), 'dtype': str(a.dtype)}
    if a.dtype.kind == 'f':
        out['nonfinite'] = int(sum((~np.isfinite(a[i:i+50000])).sum() for i in range(0, len(a), 50000)))
    if path.name == 'ids.npy':
        out['unique'] = int(len(set(a.tolist())))
    return out


def inspect_caches(cache_root):
    dirs = sorted(p for p in Path(cache_root).iterdir() if (p/'ids.npy').exists())
    report = {p.name: {f.name: array_summary(f) for f in sorted(p.glob('*.npy'))} for p in dirs}
    for name, arrays in report.items():
        n = {a['shape'][0] for a in arrays.values()}
        arrays['row_aligned'] = len(n) == 1
        arrays['ids_unique'] = arrays['ids.npy']['unique'] == arrays['ids.npy']['shape'][0]
    return report, {p.name.rsplit('_', 1)[-1]: p for p in dirs}


def check_partition(dirs, chunk=20000):
    """train/eval partition full, and every row equals the full-cache row with that ID."""
    full_ids = np.load(dirs['full']/'ids.npy', allow_pickle=True)
    pos = {k: i for i, k in enumerate(full_ids.tolist())}
    ids = {s: np.load(dirs[s]/'ids.npy', allow_pickle=True).tolist() for s in ('train', 'eval')}
    out = {'disjoint': not set(ids['train']) & set(ids['eval']),
           'covers_full': set(ids['train']) | set(ids['eval']) == set(pos),
           'rows': {s: len(v) for s, v in ids.items()}, 'rows_equal_full_by_id': {}}
    for s in ('train', 'eval'):
        rows = np.array([pos[k] for k in ids[s]])
        equal = {}
        for name in ('embeddings.npy', 'concepts.npy', 'labels.npy'):
            full, part = np.load(dirs['full']/name, mmap_mode='r'), np.load(dirs[s]/name, mmap_mode='r')
            order = np.argsort(rows)  # read the full cache sequentially
            equal[name] = all(np.array_equal(full[rows[order[i:i+chunk]]], part[order[i:i+chunk]])
                              for i in range(0, len(rows), chunk))
        out['rows_equal_full_by_id'][s] = equal
    return out


def check_shapes3d(dataset_dir, raw_dir, full_dir, features_dir, image_samples, seed):
    dataset_dir = Path(dataset_dir)
    ds, raw = load_raw_factors(raw_dir)
    n = len(raw['index'])
    out = {'num_rows': n, 'raw_columns': sorted(raw), 'factor_levels': FACTOR_LEVELS,
           'index_is_row_number': bool(np.array_equal(raw['index'], np.arange(n)))}
    grid = np.stack([raw[f'{f}_idx'] for f in FACTORS], 1)
    out['factor_index_ranges_ok'] = all(set(np.unique(raw[f'{f}_idx']).tolist()) == set(range(FACTOR_LEVELS[f])) for f in FACTORS)
    out['full_factorial_unique'] = n == int(np.prod(list(FACTOR_LEVELS.values()))) == len(np.unique(grid, axis=0))
    label = raw['shape_idx'] * HUES + raw['object_idx']
    concepts = np.stack([raw['floor'], raw['wall'], (raw['scale'] - .75) / .5, (raw['orientation'] + 30) / 60], 1)

    import pandas as pd
    manifest = pd.read_csv(dataset_dir/'manifest.csv', dtype={'id': str})
    credal = pd.read_csv(dataset_dir/'attribute_credal.csv', dtype={'id': str})
    out['manifest_ids_are_row_numbers'] = manifest['id'].tolist() == [str(i) for i in range(n)]
    out['manifest_label_matches_raw'] = bool(np.array_equal(manifest['label'].to_numpy(), label))
    out['manifest_paths_follow_id'] = bool((manifest['image_path'] == [f'shapes3d_images/{i:06d}.png' for i in range(n)]).all())
    lower = credal[[f'lower_{j}' for j in range(4)]].to_numpy()
    upper = credal[[f'upper_{j}' for j in range(4)]].to_numpy()
    out['credal_ids_are_row_numbers'] = credal['id'].tolist() == [str(i) for i in range(n)]
    out['credal_zero_width'] = bool(np.array_equal(lower, upper))
    out['credal_matches_raw_max_abs'] = float(np.abs(lower - concepts).max())
    out['concepts_txt'] = (dataset_dir/'concepts.txt').read_text().split()

    ids = np.load(full_dir/'ids.npy', allow_pickle=True)
    rows = np.array([int(i) for i in ids])
    cache_concepts = np.load(full_dir/'concepts.npy')
    out['cache_ids_are_row_numbers'] = bool(np.array_equal(rows, np.arange(n)))
    out['cache_labels_match_raw'] = bool(np.array_equal(np.load(full_dir/'labels.npy')[np.argsort(rows)], label))
    out['cache_concepts_match_raw_max_abs'] = float(np.abs(cache_concepts[np.argsort(rows)] - np.c_[concepts, concepts]).max())
    if features_dir and Path(features_dir).exists():
        f_ids = np.load(Path(features_dir)/'ids.npy', allow_pickle=True)
        out['features_meta'] = json.loads((Path(features_dir)/'meta.json').read_text())
        out['features_ids_match_cache'] = bool(np.array_equal(f_ids.astype(str), ids.astype(str)))
        a, b = np.load(Path(features_dir)/'embeddings.npy', mmap_mode='r'), np.load(full_dir/'embeddings.npy', mmap_mode='r')
        out['features_embeddings_match_cache'] = all(np.array_equal(a[i:i+50000], b[i:i+50000]) for i in range(0, len(a), 50000))

    files = dataset_dir/'shapes3d_images'
    out['image_files_present'] = sum((files/f'{i:06d}.png').exists() for i in range(n))
    sample = np.random.default_rng(seed).choice(n, size=min(image_samples, n), replace=False)
    out['sampled_png_equals_raw_image'] = {'n': len(sample), 'equal': int(sum(
        np.array_equal(np.asarray(Image.open(files/f'{i:06d}.png').convert('RGB')), np.asarray(ds[int(i)]['image'].convert('RGB')))
        for i in sample))}
    return out


@torch.no_grad()
def reproduce(model, eval_dir, reported, batch=8192):
    z = np.load(Path(eval_dir)/'embeddings.npy', mmap_mode='r')
    y = torch.from_numpy(np.load(Path(eval_dir)/'labels.npy'))
    hits = 0
    for i in range(0, len(z), batch):
        c = model.encode(torch.from_numpy(np.asarray(z[i:i+batch], dtype=np.float64)))
        hits += int((model.predict_from_concepts(c).argmax(1) == y[i:i+batch]).sum())
    acc = hits / len(y)
    return {'accuracy': acc, 'reported': reported, 'abs_error': abs(acc - reported), 'n_eval': len(y),
            'tolerance': TOLERANCE, 'reproduced': abs(acc - reported) <= 1 / len(y)}


def relocated_cache(eval_dir, n_expected):
    """Map a legacy cache/<ds>_<split> path to ignore/cache/cache_<ds>/cache_<ds>_<split> if row counts agree."""
    p = Path(eval_dir or '')
    if len(p.parts) != 2 or p.parts[0] != 'cache' or '_' not in p.name:
        return None
    ds = p.name.rsplit('_', 1)[0]
    cand = Path('ignore/cache')/f'cache_{ds}'/f'cache_{p.name}'
    if (cand/'embeddings.npy').exists() and len(np.load(cand/'labels.npy')) == n_expected:
        return cand
    return None


def inspect_checkpoints(results_root):
    out = []
    for pt in sorted(Path(results_root).glob('*/*/audit.pt')):
        report_path = pt.with_suffix('.json')
        entry = {'checkpoint': str(pt), 'sha256': sha256(pt)}
        try:
            model, readout_name, ckpt = load_student(pt)
        except Exception as exc:  # recorded, not fatal: one bad checkpoint must not hide the rest
            out.append({**entry, 'rebuilt': False, 'error': repr(exc)})
            continue
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        k = model.readout.num_concepts
        entry.update(
            rebuilt=True, backend=ckpt['backend'], readout=readout_name, seed=ckpt['seed'],
            feature_dim=ckpt['feature_dim'], num_classes=ckpt['num_classes'], num_concepts=k,
            latent_dim=model.latent_dim(), unconstrained_dim=model.unconstrained_dim, blocks=[list(b) for b in model.blocks],
            parameters=sum(p.numel() for p in model.parameters()), checkpoint_keys=sorted(ckpt),
            report=str(report_path) if report else None, epochs=report.get('epochs'),
            train_cache=report.get('train_cache'), eval_cache=report.get('eval_cache'),
            train_samples=report.get('train_samples'), eval_samples=report.get('eval_samples'),
            not_in_checkpoint=['learning_rate', 'batch_size', 'optimizer', 'split_seed', 'selection_rule', 'code_commit'])
        eval_dir = report.get('eval_cache')  # relative to the repository root, as the student ran it
        remapped = relocated_cache(eval_dir, report.get('eval_samples'))
        if eval_dir and (Path(eval_dir)/'embeddings.npy').exists():
            entry['reproduction'] = reproduce(model, Path(eval_dir), report['consequence']['baseline_accuracy'])
        elif remapped:
            # Older runs used cache/<ds>_<split>; accepted only because size matches and accuracy must reproduce.
            entry['reproduction'] = {**reproduce(model, remapped, report['consequence']['baseline_accuracy']),
                                     'eval_cache_remapped': {'reported': eval_dir, 'used': str(remapped)}}
        else:
            entry['reproduction'] = {'reproduced': False, 'error': f'evaluation cache unavailable: {eval_dir}'}
        out.append(entry)
    return out


def preprocessing(model_id, cache_dir=None):
    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True).to_dict()
    except Exception as exc:
        return {'error': f'processor config not available offline: {exc!r}'}


def markdown(inv):
    lines = ['| checkpoint | backend/readout | seed | latent (free) | params | reported acc | reproduced acc | ok |',
             '|---|---|---|---|---|---|---|---|']
    for c in inv['checkpoints']:
        r = c.get('reproduction', {})
        if not c.get('rebuilt'):
            lines.append(f"| {c['checkpoint']} | rebuild failed | | | | | | no |")
            continue
        acc = f"{r['accuracy']:.6f}" if 'accuracy' in r else r.get('error', '')
        rep = f"{r['reported']:.6f}" if 'reported' in r else ''
        lines.append(f"| {c['checkpoint']} | {c['backend']}/{c['readout']} | {c['seed']} | {c['latent_dim']} ({c['unconstrained_dim']}) "
                     f"| {c['parameters']} | {rep} | {acc} | {'yes' if r.get('reproduced') else 'no'}{' (cache path remapped)' if 'eval_cache_remapped' in r else ''} |")
    lines += ['', '| cache | rows | embedding | concepts | aligned | unique ids |', '|---|---|---|---|---|---|']
    for name, a in inv['caches'].items():
        lines.append(f"| {name} | {a['ids.npy']['shape'][0]} | {a['embeddings.npy']['shape']} {a['embeddings.npy']['dtype']} "
                     f"| {a['concepts.npy']['shape']} | {a['row_aligned']} | {a['ids_unique']} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, default=Path('results/semantic_edits/inventory'))
    p.add_argument('--results-root', type=Path, default=Path('ignore/results_experiments'))
    p.add_argument('--cache-root', type=Path, default=Path('ignore/cache/cache_shapes3d'))
    p.add_argument('--features-dir', type=Path, default=Path('ignore/features/features_shapes3d'))
    p.add_argument('--dataset-dir', type=Path, default=Path('ignore/dataset/dataset_shapes_3d'))
    p.add_argument('--raw-dir', type=Path, default=Path('ignore/data/shapes3d_local'))
    p.add_argument('--image-samples', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hf-cache', help='HF hub cache holding the extraction backbone (default: HF_HOME)')
    args = p.parse_args()
    torch.set_num_threads(8)

    caches, dirs = inspect_caches(args.cache_root)
    shapes3d = check_shapes3d(args.dataset_dir, args.raw_dir, dirs['full'], args.features_dir, args.image_samples, args.seed)
    inv = {
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
        'source_sha256': {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'checkpoints': inspect_checkpoints(args.results_root),
        'caches': caches,
        'partition': check_partition(dirs),
        'shapes3d': shapes3d,
        'task_and_concepts': {
            'task_label': 'shape_idx * 10 + object_idx (40 classes)',
            'concept_order': [f'lower_{n}' for n in CONCEPT_NAMES] + [f'upper_{n}' for n in CONCEPT_NAMES],
            'concept_units': 'targets in [0, 1] (hues as stored, scale and orientation min-max scaled); '
                             'readout outputs are logits trained with BCE-with-logits against these targets',
            'unsupervised_raw_factors': ['object_hue', 'shape'],
        },
        'backbone': {'model': shapes3d.get('features_meta', {}).get('model'),
                     'pooling': 'pooler_output if present else CLS token (concept_audit.data.extraction)',
                     'processor': preprocessing(shapes3d.get('features_meta', {}).get('model', ''), args.hf_cache)},
        'split_usage': {
            'cache_shapes3d_train': 'base-model training (train_and_audit, every epoch)',
            'cache_shapes3d_eval': 'per-epoch monitoring and final audit; no checkpoint selection (last epoch saved)',
            'unseen_by_base_model': 'none: train and eval partition all 480000 images',
        },
        'separate_residual_model': 'none found: checkpoints contain only the concept model; the free latent '
                                   'coordinates (CBM 8-15, CEM non-projected block entries) are its in-model residual',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'inventory.json').write_text(json.dumps(inv, indent=2, default=str) + '\n')
    (args.out/'inventory.md').write_text(markdown(inv))
    print(markdown(inv))


if __name__ == '__main__':
    main()

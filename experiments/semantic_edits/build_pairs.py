"""Build verified single-factor image pairs from raw factor annotations.

Pairs come only from raw factors (never from representation proximity). Contexts
(all raw factors except the edited one) are split *before* pairs are drawn, so all
values of the edited factor in one context stay in one split.

    python -m experiments.semantic_edits.build_pairs --dataset shapes3d --out results/semantic_edits/pairs/shapes3d
    python -m experiments.semantic_edits.build_pairs --dataset mpi3d --out results/semantic_edits/pairs/mpi3d

Writes pairs.jsonl (one record per pair), pairs.npz (same, as arrays), images.npz
(per-image factors / levels / split roles), summary.json (assertions) and montage.png.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SPLITS = ('train', 'val', 'test')

# concept_columns: raw factor -> column j of the supervised concept vector (lower bound; upper is j + K).
DATASETS = {
    'shapes3d': dict(
        raw_dir='ignore/data/shapes3d_local', cache='ignore/cache/cache_shapes3d/cache_shapes3d',
        factors=('floor', 'wall', 'object', 'scale', 'shape', 'orientation'), column='{}_idx',
        concept_columns={'floor': 0, 'wall': 1, 'scale': 2, 'orientation': 3},
        label=lambda f: f['shape'] * 10 + f['object'],
        label_definition='shape_idx * 10 + object_idx',
        edit='floor', combo='shape', holdout=2, holdout_step=3, pairs_per_context=6),
    'mpi3d': dict(
        raw_dir='ignore/data/mpi3d_local', cache='ignore/cache/cache_mpi3d/cache_mpi3d',
        factors=('color', 'shape', 'size', 'height', 'background', 'dof1', 'dof2'), column='{}',
        concept_columns={'size': 0, 'height': 1, 'background': 2, 'dof1': 3, 'dof2': 4},
        label=lambda f: f['shape'] * 4 + f['color'],
        label_definition='shape_idx * 4 + color_idx',
        edit='background', combo='shape', holdout=1, holdout_step=1, pairs_per_context=2),
}


def heldout_combination(edit_level, combo_level, n_levels, holdout, step):
    """(edited value, combo-factor value) combinations never used by operator training/validation.

    Every edited value and every combo value still occurs in training; only the conjunction is held out.
    """
    return (edit_level + step * combo_level) % n_levels < holdout


def make_pairs(factors, edit, combo, *, holdout, holdout_step, pairs_per_context, fractions=(0.6, 0.2, 0.2), seed=0):
    """Context-grouped splits and single-factor pairs for a full-factorial factor table.

    factors: dict name -> int array (row i = image id i). Returns (pairs dict of arrays, image_info dict).
    """
    names = list(factors)
    n = len(factors[edit])
    cards = {f: int(factors[f].max()) + 1 for f in names}
    others = [f for f in names if f != edit]
    context = np.ravel_multi_index([factors[f] for f in others], [cards[f] for f in others])
    L = cards[edit]
    rng = np.random.default_rng(seed)

    # Split contexts first (row-random splitting of endpoints would leak contexts across splits).
    uniq = np.unique(context)
    perm = rng.permutation(uniq)
    cut = np.cumsum([int(round(fr * len(uniq))) for fr in fractions[:-1]])
    context_split = {}
    for s, part in zip(SPLITS, np.split(perm, cut)):
        for c in part:
            context_split[int(c)] = s
    image_split = np.array([context_split[int(c)] for c in context])

    held = heldout_combination(factors[edit], factors[combo], L, holdout, holdout_step)

    # Image ids per (context, edit level): full factorial -> exactly one image.
    order = np.lexsort((factors[edit], context))
    ctx_sorted = context[order]
    if not (np.all(np.bincount(np.searchsorted(uniq, ctx_sorted)) == L)):
        raise ValueError('Not a full factorial design: some context lacks some edited level')
    grid = order.reshape(len(uniq), L)  # grid[k, v] = image id with context uniq[k] and edit level v
    if not np.array_equal(factors[edit][grid], np.broadcast_to(np.arange(L), grid.shape)):
        raise AssertionError('grid construction failed')

    rows = []
    all_ordered = np.array([(a, b) for a in range(L) for b in range(L) if a != b])
    for k, c in enumerate(uniq):
        split = context_split[int(c)]
        ids = grid[k]
        if split == 'test':
            cand = all_ordered
        else:  # operator training/validation never touches a held-out combination at either endpoint
            ok = ~held[ids]
            cand = all_ordered[ok[all_ordered[:, 0]] & ok[all_ordered[:, 1]]]
        take = cand[rng.choice(len(cand), size=min(pairs_per_context, len(cand)), replace=False)]
        for a, b in take:
            rows.append((ids[a], ids[b], k, SPLITS.index(split)))
    rows = np.array(rows, dtype=np.int64)
    pairs = {'source': rows[:, 0], 'target': rows[:, 1], 'context': uniq[rows[:, 2]], 'split': rows[:, 3]}
    pairs['source_level'] = factors[edit][pairs['source']]
    pairs['target_level'] = factors[edit][pairs['target']]
    pairs['target_heldout_combination'] = held[pairs['target']]
    pairs['source_heldout_combination'] = held[pairs['source']]
    return pairs, {'image_split': image_split, 'context': context, 'heldout_combination': held, 'cards': cards}


def validate_pairs(pairs, factors, edit, image_info, hashes=None, labels=None):
    """Automated assertions. Raises on any violation; returns a dict of the checks that passed."""
    s, t = pairs['source'], pairs['target']
    checks = {}
    for f, v in factors.items():
        same = v[s] == v[t]
        if f == edit:
            assert not same.any(), 'a pair does not change the edited factor'
        else:
            assert same.all(), f'a pair changes unrequested factor {f}'
    checks['pairs_differ_only_in_edit_factor'] = True
    split_names = np.array(SPLITS)[pairs['split']]
    ep_split = {}
    for ids in (s, t):
        assert np.array_equal(image_info['image_split'][ids], split_names), 'endpoint split != pair split'
    for name in SPLITS:
        m = pairs['split'] == SPLITS.index(name)
        ep_split[name] = set(np.concatenate([s[m], t[m]]).tolist())
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not ep_split[a] & ep_split[b], f'endpoint ids shared by {a} and {b}'
    checks['endpoint_ids_disjoint_across_splits'] = True
    ctx = {name: set(pairs['context'][pairs['split'] == SPLITS.index(name)].tolist()) for name in SPLITS}
    assert not (ctx['train'] & ctx['val'] or ctx['train'] & ctx['test'] or ctx['val'] & ctx['test'])
    checks['contexts_disjoint_across_splits'] = True
    if hashes is not None:
        hs = {name: {hashes[i] for i in ep_split[name]} for name in SPLITS}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                assert not hs[a] & hs[b], f'identical image bytes in {a} and {b}'
        checks['endpoint_image_hashes_disjoint_across_splits'] = True
    fit = pairs['split'] < 2
    assert not (pairs['source_heldout_combination'][fit] | pairs['target_heldout_combination'][fit]).any()
    checks['no_heldout_combination_in_operator_train_or_val'] = True
    if labels is not None:
        assert np.array_equal(labels[s], labels[t]), 'task label changes although the edit factor is not in it'
        checks['task_label_invariant_for_every_pair'] = True
    return checks


def image_hashes(ds):
    """sha1 of each stored (encoded) image, read chunk by chunk from the arrow table."""
    out = []
    for chunk in ds.data.column('image').chunks:
        for b in chunk.field('bytes').to_pylist():
            out.append(hashlib.sha1(b).hexdigest())
    return np.array(out)


def load_dataset_tables(spec):
    from datasets import load_from_disk
    ds = load_from_disk(spec['raw_dir'])
    if ds._indices is not None:
        raise ValueError('raw dataset has an index mapping')
    t = ds.data
    idx = t.column('index').to_numpy()
    assert np.array_equal(idx, np.arange(len(idx))), 'raw index column is not the row number'
    factors = {f: t.column(spec['column'].format(f)).to_numpy().astype(np.int64) for f in spec['factors']}
    return ds, factors


def check_cache(spec, factors):
    """Cache rows == raw rows; labels follow the declared task; each concept column is a function of its factor."""
    base = spec['cache']
    full_ids = np.load(f'{base}_full/ids.npy', allow_pickle=True).astype(np.int64)
    n = len(full_ids)
    assert np.array_equal(full_ids, np.arange(n)), 'full cache ids are not raw row numbers'
    labels = np.load(f'{base}_full/labels.npy')
    assert np.array_equal(labels, spec['label'](factors)), 'cache labels differ from declared task label'
    concepts = np.load(f'{base}_full/concepts.npy')
    K = len(spec['concept_columns'])
    assert concepts.shape[1] == 2 * K and np.array_equal(concepts[:, :K], concepts[:, K:]), 'lower != upper'
    level_values = {}
    for f, j in spec['concept_columns'].items():
        vals = np.zeros(int(factors[f].max()) + 1, dtype=np.float64)
        for v in range(len(vals)):
            u = np.unique(concepts[factors[f] == v, j])
            assert len(u) == 1, f'concept {f} is not a function of its raw factor'
            vals[v] = u[0]
        assert np.all(np.diff(vals) > 0), f'concept {f} not increasing in its factor index'
        level_values[f] = vals.tolist()
    base_eval = np.zeros(n, bool)
    base_eval[np.load(f'{base}_eval/ids.npy', allow_pickle=True).astype(np.int64)] = True
    base_train = np.zeros(n, bool)
    base_train[np.load(f'{base}_train/ids.npy', allow_pickle=True).astype(np.int64)] = True
    assert not (base_eval & base_train).any() and (base_eval | base_train).all()
    return labels, level_values, base_eval


def montage(ds, pairs, factors, edit, out, n=8, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(pairs['split'] == 2)
    pick = np.concatenate([rng.choice(test[pairs['target_heldout_combination'][test]], n // 2, replace=False),
                           rng.choice(test[~pairs['target_heldout_combination'][test]], n - n // 2, replace=False)])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5.4))
    for col, p in enumerate(pick):
        for row, key in enumerate(('source', 'target')):
            i = int(pairs[key][p])
            axes[row, col].imshow(ds[i]['image'].convert('RGB'))
            axes[row, col].set_title(f'{key} #{i}\n{edit}={factors[edit][i]}', fontsize=7)
            axes[row, col].axis('off')
        axes[1, col].text(0.5, -0.08, 'unseen combo' if pairs['target_heldout_combination'][p] else 'seen combo',
                          transform=axes[1, col].transAxes, ha='center', va='top', fontsize=7)
    fig.suptitle(f'Test pairs: only "{edit}" differs (top: source, bottom: target)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    p.add_argument('--edit', help='edited raw factor (default: dataset default; must be a supervised concept)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    spec = DATASETS[args.dataset]
    edit = args.edit or spec['edit']
    if edit not in spec['concept_columns']:
        raise SystemExit(f'{edit} is not a supervised concept; the native intervention could not request it')
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')

    ds, factors = load_dataset_tables(spec)
    labels, level_values, base_eval = check_cache(spec, factors)
    hashes = image_hashes(ds)
    pairs, info = make_pairs(factors, edit, spec['combo'], holdout=spec['holdout'], holdout_step=spec['holdout_step'],
                             pairs_per_context=spec['pairs_per_context'], seed=args.seed)
    checks = validate_pairs(pairs, factors, edit, info, hashes=hashes, labels=labels)
    dup = len(hashes) - len(np.unique(hashes))
    pairs['both_endpoints_base_eval'] = base_eval[pairs['source']] & base_eval[pairs['target']]
    checks['endpoints_seen_by_base_model'] = ('all images were in the base model train or eval cache; '
                                              'both_endpoints_base_eval marks pairs never used for base-model gradient updates')

    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'pairs.npz', **pairs)
    np.savez_compressed(args.out/'images.npz', **{f'factor_{f}': v for f, v in factors.items()}, label=labels,
                        split=info['image_split'], context=info['context'], heldout_combination=info['heldout_combination'],
                        base_eval=base_eval)
    with open(args.out/'pairs.jsonl', 'w') as fh:
        for i in range(len(pairs['source'])):
            s, t = int(pairs['source'][i]), int(pairs['target'][i])
            fh.write(json.dumps({
                'pair_id': i, 'split': SPLITS[pairs['split'][i]], 'context': int(pairs['context'][i]),
                'source_id': s, 'target_id': t,
                'source_factors': {f: int(factors[f][s]) for f in factors},
                'target_factors': {f: int(factors[f][t]) for f in factors},
                'action': {'factor': edit, 'set_level': int(pairs['target_level'][i]),
                           'set_concept_value': level_values[edit][int(pairs['target_level'][i])]},
                'expected_label_source': int(labels[s]), 'expected_label_target': int(labels[t]), 'label_changes': False,
                'target_heldout_combination': bool(pairs['target_heldout_combination'][i]),
                'source_heldout_combination': bool(pairs['source_heldout_combination'][i]),
                'both_endpoints_base_eval': bool(pairs['both_endpoints_base_eval'][i]),
            }) + '\n')
    split_counts = {}
    for k, name in enumerate(SPLITS):
        m = pairs['split'] == k
        split_counts[name] = {'pairs': int(m.sum()), 'contexts': int(len(np.unique(pairs['context'][m]))),
                              'endpoint_images': int(len(np.unique(np.concatenate([pairs['source'][m], pairs['target'][m]])))),
                              'target_heldout_combination': int(pairs['target_heldout_combination'][m].sum()),
                              'both_endpoints_base_eval': int(pairs['both_endpoints_base_eval'][m].sum())}
    summary = {
        'dataset': args.dataset, 'raw_dir': spec['raw_dir'], 'cache': spec['cache'], 'seed': args.seed,
        'edit_factor': edit, 'edit_levels': info['cards'][edit], 'edit_concept_values': level_values[edit],
        'edit_semantics': f'set raw factor "{edit}" to a requested level; all other raw factors fixed',
        'concept_columns': spec['concept_columns'], 'concept_level_values': level_values,
        'factors': list(spec['factors']), 'factor_cardinalities': info['cards'],
        'task_label': spec['label_definition'], 'task_label_changes': False,
        'context_definition': 'all raw factors except the edited one',
        'heldout_combination': {'combo_factor': spec['combo'], 'rule': f'(edit_level + {spec["holdout_step"]} * '
                                f'{spec["combo"]}_level) mod {info["cards"][edit]} < {spec["holdout"]}'},
        'pairs_per_context': spec['pairs_per_context'], 'splits': split_counts,
        'duplicate_image_bytes_in_dataset': int(dup), 'checks': checks,
    }
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    montage(ds, pairs, factors, edit, args.out/'montage.png', seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

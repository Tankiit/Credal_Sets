"""Frozen final evaluation of edit methods on test pairs, with per-pair exports.

    python -m experiments.semantic_edits.evaluate_native --run-dir results/semantic_edits/runs/shapes3d/cbm_seed0

Everything is loaded frozen from train_native's outputs. Success is judged against the
ground-truth factors of the edited image, read by held-out factor readers (not by the
model's own readout, which native interventions set by construction).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .native_adapter import load_student, sha256, cached_latents, readout_and_logits, decode_calibrated, decode_levels
from .torch_operators import Standardizer, OPERATORS, FactorReaders, apply_operator
from .train_native import load_pairs, SPLIT_ID

PRIVILEGED = ('reference_target', 'donor_block')  # need the target image at inference: references, not methods
COMPARISONS = [('mlp', 'affine'), ('mlp', 'native_calibrated'), ('affine', 'native_calibrated'),
               ('mlp', 'native_logit'), ('native_calibrated', 'identity')]


def strata(pairs, m):
    t, s = pairs['target_heldout_combination'][m], pairs['source_heldout_combination'][m]
    return {'all': np.ones(m.sum(), bool), 'seen_combination': ~t & ~s, 'unseen_combination': t,
            'base_model_heldout_endpoints': pairs['both_endpoints_base_eval'][m]}


def edited_latents(model, cfg, std, ops, c_src, c_tgt, levels, values, dev):
    """Every method's edited latent state for the test pairs (float64, CPU)."""
    dtype = next(model.parameters()).dtype
    j, jK = cfg['edit_concept_columns']
    x = c_src.to(dev, dtype)
    out = {'identity': c_src.clone()}
    logit = torch.logit(torch.as_tensor(np.clip(values, 1e-3, 1 - 1e-3), dtype=dtype, device=dev))
    out['native_logit'] = model.intervene(model.intervene(x, j, logit), jK, logit).cpu()
    cal = torch.as_tensor(cfg['native_calibrated_scores'], dtype=dtype, device=dev)[torch.as_tensor(levels, device=dev)]
    out['native_calibrated'] = model.intervene(model.intervene(x, j, cal[:, 0]), jK, cal[:, 1]).cpu()
    a = torch.nn.functional.one_hot(torch.as_tensor(levels), cfg['edit_levels']).float().to(dev)
    xs = std(c_src.float().to(dev))
    for name, op in ops.items():
        out[name] = std.inverse(apply_operator(op, xs, a)).double().cpu()
    y = c_tgt.to(dev, dtype)
    out['donor_block'] = model.substitute_donor(model.substitute_donor(x, j, y), jK, y).cpu()
    out['reference_target'] = c_tgt.clone()
    return out


def bootstrap_diff(per_context_a, per_context_b, B=2000, seed=0):
    d = per_context_a - per_context_b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return {'diff': float(d.mean()), 'ci95': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            'contexts': int(len(d))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--latents-dir', type=Path, default=Path('results/semantic_edits/latents'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    run, dev = args.run_dir, args.device
    out_dir = run/'evaluation'
    if out_dir.exists():
        raise SystemExit(f'{out_dir} exists; the final evaluation is run once')
    cfg = json.loads((run/'train_config.json').read_text())
    if sha256(cfg['checkpoint']) != cfg['checkpoint_sha256'] or sha256(Path(cfg['pairs_dir'])/'pairs.npz') != cfg['pairs_npz_sha256']:
        raise SystemExit('checkpoint or pairs changed since training')
    summary, pairs, images = load_pairs(cfg['pairs_dir'])
    model, _, _ = load_student(cfg['checkpoint'], device=dev)
    c = cached_latents(model, cfg['checkpoint'], summary['cache'] + '_full', args.latents_dir/f"{cfg['run']}.npz", dev)
    d = model.latent_dim()
    std = Standardizer(torch.zeros(d), torch.ones(d))
    std.load_state_dict(torch.load(run/'standardizer.pt', weights_only=True))
    std = std.to(dev).eval()
    cards = cfg['readers']['cards']
    readers = FactorReaders(d, cards)
    readers.load_state_dict(torch.load(run/'readers.pt', weights_only=True))
    readers = readers.to(dev).eval()
    ops = {}
    for seed in cfg['operator_seeds']:
        for kind in ('affine', 'mlp'):
            op = OPERATORS[kind](d, cfg['edit_levels'])
            op.load_state_dict(torch.load(run/'operators'/f'{kind}_seed{seed}.pt', weights_only=True))
            ops[f'{kind}_seed{seed}'] = op.to(dev).eval()

    m = pairs['split'] == SPLIT_ID['test']
    pid = np.flatnonzero(m)
    s_ids, t_ids = pairs['source'][m], pairs['target'][m]
    edit, factors = summary['edit_factor'], summary['factors']
    levels = pairs['target_level'][m]
    values = np.asarray(summary['edit_concept_values'])[levels]
    truth = {f: images[f'factor_{f}'][t_ids] for f in factors}
    label = images['label'][t_ids]
    assert np.array_equal(label, images['label'][s_ids])
    K = len(summary['concept_columns'])
    st = strata(pairs, m)
    ctx = pairs['context'][m]

    def read(cx):
        r = {f: v.cpu().numpy() for f, v in readers.predict(std(cx.float().to(dev))).items()}
        scores, logits = readout_and_logits(model, cx, dev)
        native = {f: decode_calibrated(scores[:, [jj, jj + K]].double(), cfg['native_calibration'][f]).numpy()
                  for f, jj in summary['concept_columns'].items()}
        sig = decode_levels((scores[:, cfg['edit_concept_columns'][0]] + scores[:, cfg['edit_concept_columns'][1]]) / 2,
                            summary['edit_concept_values']).numpy()
        return r, scores.float().numpy(), logits.float().numpy(), native, sig

    def correct_unaffected(r):
        return np.all([r[f] == truth[f] for f in factors if f != edit], 0)

    src_read = read(c[s_ids])[0]
    src_ok = correct_unaffected(src_read)
    edits = edited_latents(model, cfg, std, ops, c[s_ids], c[t_ids], levels, values, dev)
    x_t = std(c[t_ids].float().to(dev))

    rows, arrays, metrics = [], {}, {}
    for name, ce in edits.items():
        r, scores, logits, native, sig = read(ce)
        task_pred = logits.argmax(1)
        per = {
            'target_success': r[edit] == levels,
            'preservation': correct_unaffected(r),
            'task_correct': task_pred == label,
            'native_target_calibrated': native[edit] == levels,
            'native_target_sigmoid': sig == levels,
            'native_preservation_calibrated': np.all([native[f] == truth[f] for f in summary['concept_columns'] if f != edit], 0),
            'latent_mse_std': (std(ce.float().to(dev)) - x_t).square().mean(1).cpu().numpy(),
        }
        per['joint'] = per['target_success'] & per['preservation'] & per['task_correct']
        for f in factors:
            if f != edit:
                per[f'preserved_{f}'] = r[f] == truth[f]
        arrays[f'{name}/readout_scores'], arrays[f'{name}/task_logits'] = scores, logits
        kind, seed = (name.rsplit('_seed', 1) + [None])[:2] if '_seed' in name else (name, None)
        df = pd.DataFrame({'pair_id': pid, 'source_id': s_ids, 'target_id': t_ids, 'context': ctx,
                           'source_level': pairs['source_level'][m], 'target_level': levels,
                           'unseen_combination': st['unseen_combination'], 'seen_combination': st['seen_combination'],
                           'base_model_heldout_endpoints': st['base_model_heldout_endpoints'],
                           'method': kind, 'operator_seed': seed, 'privileged': kind in PRIVILEGED,
                           'read_target_level': r[edit], 'task_pred': task_pred, 'label': label,
                           'source_unaffected_correct': src_ok, **{k: v for k, v in per.items()}})
        rows.append(df)
        metrics[name] = {}
        for sname, sm in st.items():
            agg = {k: float(np.mean(v[sm])) for k, v in per.items()}
            agg['conditional_retention'] = float(per['preservation'][sm & src_ok].mean()) if (sm & src_ok).any() else None
            agg['pairs'] = int(sm.sum())
            metrics[name][sname] = agg
    per_pair = pd.concat(rows, ignore_index=True)

    # Reference quality: readers on original (source) and edited (target) images.
    ref_quality = {f'{side}_reader_accuracy': {f: float(np.mean(rd[f] == images[f'factor_{f}'][ids])) for f in factors}
                   for side, rd, ids in (('source', src_read, s_ids), ('target', read(c[t_ids])[0], t_ids))}
    ref_quality['source_task_accuracy'] = float((read(c[s_ids])[2].argmax(1) == label).mean())

    # Paired context-bootstrap differences in joint success, per operator seed (same test pairs for all methods).
    diffs = {}
    for sname in ('all', 'unseen_combination'):
        sub = per_pair[per_pair[sname] if sname != 'all' else slice(None)]
        by = sub.groupby(['method', 'operator_seed', 'context'], dropna=False)['joint'].mean()
        diffs[sname] = {}
        for seed in cfg['operator_seeds']:
            def ctx_mean(method):
                key = (method, str(seed)) if method in ('mlp', 'affine') else (method, None)
                s = by.xs(key[0], level='method')
                s = s.xs(key[1], level='operator_seed') if key[1] else s[s.index.get_level_values('operator_seed').isna()].droplevel(0)
                return s.sort_index()
            for a, b in COMPARISONS:
                xa, xb = ctx_mean(a), ctx_mean(b)
                assert xa.index.equals(xb.index)
                diffs[sname].setdefault(f'{a}-{b}', {})[f'seed{seed}'] = bootstrap_diff(xa.values, xb.values, seed=seed)

    out_dir.mkdir()
    per_pair.to_csv(out_dir/'per_pair.csv.gz', index=False)
    np.savez_compressed(out_dir/'readouts_logits.npz', pair_id=pid, **arrays)
    (out_dir/'metrics.json').write_text(json.dumps({
        'run': cfg['run'], 'checkpoint': cfg['checkpoint'], 'checkpoint_sha256': cfg['checkpoint_sha256'],
        'edit_factor': edit, 'test_pairs': int(m.sum()), 'test_contexts': int(len(np.unique(ctx))),
        'privileged_methods': list(PRIVILEGED), 'metrics': metrics, 'reference_quality': ref_quality,
        'paired_bootstrap_joint': diffs,
        'notes': 'joint = target_success & preservation (held-out readers, all other raw factors) & task_correct. '
                 'Bootstrap resamples test contexts; it does not capture training variability (see operator seeds).',
    }, indent=2) + '\n')
    for name in ('identity', 'native_logit', 'native_calibrated', 'affine_seed0', 'mlp_seed0', 'donor_block', 'reference_target'):
        a, u = metrics[name]['all'], metrics[name]['unseen_combination']
        print(f"{cfg['run']:24s} {name:18s} target={a['target_success']:.3f} preserve={a['preservation']:.3f} "
              f"task={a['task_correct']:.3f} joint={a['joint']:.3f} | unseen joint={u['joint']:.3f}")


if __name__ == '__main__':
    main()

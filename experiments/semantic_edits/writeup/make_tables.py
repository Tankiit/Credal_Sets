"""Generate the LaTeX tables of the write-up directly from the result files (no hand-copied numbers).

    python -m experiments.semantic_edits.writeup.make_tables
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.semantic_edits.summarize_native import collect, method_stats, METHODS

ROOT = Path('results/semantic_edits')
OUT = Path(__file__).parent/'tables'
NAMES = {'identity': 'Identity', 'native_logit': 'Native (logit)', 'native_calibrated': 'Native (calibrated)',
         'affine': 'Affine', 'mlp': 'MLP', 'donor_block': 'Donor block$^\\dagger$',
         'reference_target': 'Real edited image$^\\dagger$'}


def run_name(r):
    arch = 'CBM' if r['backend'] == 'native' else 'CEM'
    return f"{'Shapes3D' if r['dataset'] == 'shapes3d' else 'MPI3D'} {arch} s{r['model_seed']}"


def f3(m, s=None):
    return f'{m:.3f}' if s is None else f'{m:.3f}\\,{{\\scriptsize$\\pm${s:.3f}}}'


def write(name, body):
    (OUT/f'{name}.tex').write_text(body + '\n')


def inventory():
    inv = json.loads((ROOT/'inventory/inventory.json').read_text())
    rows = []
    for c in inv['checkpoints']:
        r = c['reproduction']
        p = Path(c['checkpoint']).parts
        ds = p[-2].replace('_dataset', '').replace('cubs', 'CUB-S').replace('cub', 'CUB').replace('shapes3d', 'Shapes3D').replace('mpi3d', 'MPI3D')
        arch = 'CBM' if c['backend'] == 'native' else 'CEM'
        note = '$^\\ast$' if 'eval_cache_remapped' in r else ''
        rows.append(f"{ds} & {arch} ({c['readout'].replace('_', ' ')}) & {c['seed']} & {c['latent_dim']} ({c['unconstrained_dim']}) & "
                    f"{c['parameters']:,} & {r['reported']:.6f} & {r['accuracy']:.6f}{note} & {r['n_eval']:,} & "
                    f"{'yes' if r['reproduced'] else 'no'} \\\\")
    write('inventory', '\n'.join(rows))


def pairs():
    rows = []
    for ds in ('shapes3d', 'mpi3d'):
        s = json.loads((ROOT/f'pairs/{ds}/summary.json').read_text())
        for split in ('train', 'val', 'test'):
            x = s['splits'][split]
            rows.append(f"{'Shapes3D' if ds == 'shapes3d' else 'MPI3D'} & {split} & {x['contexts']:,} & {x['pairs']:,} & "
                        f"{x['endpoint_images']:,} & {x['target_heldout_combination']:,} & {x['both_endpoints_base_eval']:,} \\\\")
        rows.append('\\midrule' if ds == 'shapes3d' else '')
    write('pairs', '\n'.join(rows))


def main_joint(runs):
    rows = []
    for meth in METHODS:
        cells = [f3(*method_stats(r['eval'], meth, 'unseen_combination', 'joint')[:2]) for r in runs]
        rows.append(f'{NAMES[meth]} & ' + ' & '.join(cells) + ' \\\\')
        if meth in ('identity', 'native_calibrated', 'mlp'):
            rows.append('\\midrule')
    write('main_joint_header', ' & '.join(['Method'] + ['\\shortstack{' + run_name(r).replace(' ', '\\\\', 1) + '}' for r in runs]))
    write('main_joint', '\n'.join(rows))


def detail(runs, stratum, name):
    keys = ['target_success', 'conditional_retention', 'preservation', 'task_correct', 'joint', 'native_target_calibrated', 'latent_mse_std']
    rows = []
    for i, r in enumerate(runs):
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{run_name(r)}}}}} \\\\")
        for meth in METHODS:
            rows.append(f'\\quad {NAMES[meth]} & ' + ' & '.join(f3(*method_stats(r['eval'], meth, stratum, k)[:2]) for k in keys) + ' \\\\')
        if i < len(runs) - 1:
            rows.append('\\midrule')
    write(name, '\n'.join(rows))


def native_diag(runs):
    rows = []
    for r in runs:
        ev, cfg = r['eval'], r['cfg']
        e = ev['edit_factor']
        nat = cfg['native_checks']['per_concept'][e]
        rows.append(f"{run_name(r)} & {nat['decoded_level_accuracy_all_images']:.3f} & {nat['calibrated_decoding_accuracy_val_images']:.3f} & "
                    f"{ev['reference_quality']['target_reader_accuracy'][e]:.3f} & "
                    f"{method_stats(ev, 'native_calibrated', 'unseen_combination', 'native_target_calibrated')[0]:.3f} & "
                    f"{method_stats(ev, 'native_calibrated', 'unseen_combination', 'target_success')[0]:.3f} & "
                    f"{method_stats(ev, 'native_logit', 'unseen_combination', 'target_success')[0]:.3f} \\\\")
    write('native_diag', '\n'.join(rows))


def bootstrap(runs):
    rows = []
    for r in runs:
        b = r['eval']['paired_bootstrap_joint']['unseen_combination']
        cells = []
        for pair in ('mlp-affine', 'mlp-native_calibrated', 'affine-native_calibrated'):
            vals = b[pair]
            cells.append(' / '.join(f"{v['diff']:+.3f}" for v in vals.values()))
            lo = min(v['ci95'][0] for v in vals.values())
            hi = max(v['ci95'][1] for v in vals.values())
            cells[-1] += f" {{\\scriptsize[{lo:+.3f}, {hi:+.3f}]}}"
        rows.append(f'{run_name(r)} & ' + ' & '.join(cells) + ' \\\\')
    write('bootstrap', '\n'.join(rows))


def seen_unseen(runs):
    rows = []
    for r in runs:
        ev = r['eval']
        c = []
        for meth in ('affine', 'mlp'):
            s = method_stats(ev, meth, 'seen_combination', 'joint')[0]
            u = method_stats(ev, meth, 'unseen_combination', 'joint')[0]
            c += [f'{s:.3f}', f'{u:.3f}', f'{u - s:+.3f}']
        ret = (method_stats(ev, 'mlp', 'unseen_combination', 'conditional_retention')[0]
               - method_stats(ev, 'affine', 'unseen_combination', 'conditional_retention')[0])
        share = method_stats(ev, 'mlp', 'unseen_combination', 'joint')[0] / method_stats(ev, 'reference_target', 'unseen_combination', 'joint')[0]
        rows.append(f'{run_name(r)} & ' + ' & '.join(c) + f' & {ret:+.3f} & {share:.2f} \\\\')
    write('seen_unseen', '\n'.join(rows))


def reader_quality(runs):
    out = {}
    for ds in ('shapes3d', 'mpi3d'):
        sub = [r for r in runs if r['dataset'] == ds]
        factors = sub[0]['cfg']['readers']['factors']
        rows = [' & '.join(['Run'] + [f.replace('_', '\\_') for f in factors] + ['task']) + ' \\\\', '\\midrule']
        for r in sub:
            q = r['eval']['reference_quality']
            rows.append(f'{run_name(r)} & ' + ' & '.join(f"{q['target_reader_accuracy'][f]:.3f}" for f in factors)
                        + f" & {q['source_task_accuracy']:.3f} \\\\")
        out[ds] = rows
    write('reader_quality_shapes3d', '\n'.join(out['shapes3d']))
    write('reader_quality_mpi3d', '\n'.join(out['mpi3d']))


def per_factor(runs):
    for ds in ('shapes3d', 'mpi3d'):
        rows = []
        sub = [r for r in runs if r['dataset'] == ds]
        factors = None
        for i, r in enumerate(sub):
            d = pd.read_csv(r['dir']/'evaluation'/'per_pair.csv.gz')
            d = d[d.unseen_combination]
            cols = [c for c in d.columns if c.startswith('preserved_')]
            factors = [c.replace('preserved_', '') for c in cols]
            g = d.groupby('method')[cols].mean()
            rows.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textbf{{{run_name(r)}}}}} \\\\")
            for meth in ('identity', 'native_calibrated', 'affine', 'mlp', 'reference_target'):
                rows.append(f'\\quad {NAMES[meth]} & ' + ' & '.join(f'{g.loc[meth, c]:.3f}' for c in cols) + ' \\\\')
            if i < len(sub) - 1:
                rows.append('\\midrule')
        write(f'per_factor_{ds}_header', ' & '.join(['Method'] + factors))
        write(f'per_factor_{ds}', '\n'.join(rows))


def operators(runs):
    rows = []
    for r in runs:
        o, cfg = r['cfg']['operators'], r['cfg']
        def sel(kind):
            return '/'.join(str(o[f'{kind}_seed{s}']['selected_step']) for s in cfg['operator_seeds'])
        rows.append(f"{run_name(r)} & {cfg['latent_dim']} & {o['affine_seed0']['parameters']:,} & {o['mlp_seed0']['parameters']:,} & "
                    f"{o['mlp_seed0']['training_pairs']:,} & {sel('affine')} & {sel('mlp')} & {cfg['readers']['val_mean_accuracy']:.3f} \\\\")
    write('operators', '\n'.join(rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect(ROOT/'runs')
    order = {'shapes3d': 0, 'mpi3d': 1}
    runs.sort(key=lambda r: (order[r['dataset']], r['backend'] != 'native', r['model_seed']))
    inventory()
    pairs()
    main_joint(runs)
    detail(runs, 'unseen_combination', 'detail_unseen')
    detail(runs, 'seen_combination', 'detail_seen')
    native_diag(runs)
    bootstrap(runs)
    seen_unseen(runs)
    reader_quality(runs)
    per_factor(runs)
    operators(runs)
    print('tables written to', OUT)


if __name__ == '__main__':
    main()

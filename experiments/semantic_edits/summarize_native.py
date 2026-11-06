"""Aggregate evaluated runs into the result table, figures and the decision-gate checks.

    python -m experiments.semantic_edits.summarize_native --runs results/semantic_edits/runs --out results/semantic_edits/summary
"""
import argparse
import json
from pathlib import Path
import numpy as np

METHODS = ['identity', 'native_logit', 'native_calibrated', 'affine', 'mlp', 'donor_block', 'reference_target']
LABELS = {'identity': 'identity', 'native_logit': 'native (logit)', 'native_calibrated': 'native (calibrated)',
          'affine': 'affine', 'mlp': 'MLP', 'donor_block': 'donor block*', 'reference_target': 'real edited image*'}
# Reference categorical palette (slots 1-6) + neutral ink for the no-change control; markers carry identity too.
COLORS = {'identity': '#8a8a85', 'native_logit': '#e87ba4', 'native_calibrated': '#eb6834', 'affine': '#1baf7a',
          'mlp': '#2a78d6', 'donor_block': '#eda100', 'reference_target': '#008300'}
MARKERS = {'identity': 'o', 'native_logit': 'v', 'native_calibrated': '^', 'affine': 's', 'mlp': 'D',
           'donor_block': 'P', 'reference_target': '*'}
KEYS = ['target_success', 'preservation', 'conditional_retention', 'task_correct', 'joint', 'native_target_calibrated',
        'native_preservation_calibrated', 'latent_mse_std']


def collect(runs_root):
    runs = []
    for mpath in sorted(Path(runs_root).glob('*/*/evaluation/metrics.json')):
        ev = json.loads(mpath.read_text())
        cfg = json.loads((mpath.parent.parent/'train_config.json').read_text())
        runs.append({'dir': mpath.parent.parent, 'dataset': mpath.parent.parent.parent.name, 'model': mpath.parent.parent.name,
                     'backend': cfg['backend'], 'model_seed': cfg['model_seed'], 'eval': ev, 'cfg': cfg})
    return runs


def method_stats(ev, method, stratum, key):
    """Mean (and sd over operator seeds for learned methods)."""
    names = [n for n in ev['metrics'] if n == method or n.startswith(method + '_seed')]
    vals = np.array([ev['metrics'][n][stratum][key] for n in names], dtype=float)
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else None), len(vals)


def fmt(m, s):
    return f'{m:.3f}' if s is None else f'{m:.3f} ± {s:.3f}'


def table(runs, stratum):
    lines = [f'| run | method | target | preservation | cond. retention | task | **joint** | native target (cal.) | latent MSE |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        for meth in METHODS:
            cells = [fmt(*method_stats(r['eval'], meth, stratum, k)[:2]) for k in KEYS if k != 'native_preservation_calibrated']
            lines.append(f"| {r['dataset']}/{r['model']} | {LABELS[meth]} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def gate(runs):
    """Decision-gate evidence. Each item: the numbers, then a pass/fail reading."""
    out = {}
    # 1. Native failure is not a readout-unit artifact or poor perception?
    items = []
    for r in runs:
        ev, cfg = r['eval'], r['cfg']
        edit = ev['edit_factor']
        nat = cfg['native_checks']['per_concept'][edit]
        items.append({
            'run': f"{r['dataset']}/{r['model']}",
            'native_readout_decodes_edit_factor_sigmoid': nat['decoded_level_accuracy_all_images'],
            'native_readout_decodes_edit_factor_calibrated': nat['calibrated_decoding_accuracy_val_images'],
            'reader_accuracy_edit_factor_on_real_images': ev['reference_quality']['target_reader_accuracy'][edit],
            'native_logit_joint': method_stats(ev, 'native_logit', 'all', 'joint')[0],
            'native_calibrated_joint': method_stats(ev, 'native_calibrated', 'all', 'joint')[0],
            'native_calibrated_native_target': method_stats(ev, 'native_calibrated', 'all', 'native_target_calibrated')[0],
            'reference_joint': method_stats(ev, 'reference_target', 'all', 'joint')[0],
        })
    out['1_native_failure_not_units_or_perception'] = items
    # 2-4. MLP vs affine on held-out combinations, per operator seed, across model seeds; and preservation.
    comp = []
    for r in runs:
        ev = r['eval']
        boots = ev['paired_bootstrap_joint']['unseen_combination']
        row = {'run': f"{r['dataset']}/{r['model']}"}
        for pair in ('mlp-affine', 'mlp-native_calibrated', 'affine-native_calibrated'):
            row[pair] = {s: (round(v['diff'], 4), [round(x, 4) for x in v['ci95']]) for s, v in boots[pair].items()}
        for k in ('joint', 'target_success', 'conditional_retention', 'task_correct'):
            row[f'{k}_mlp_minus_affine_unseen'] = round(method_stats(ev, 'mlp', 'unseen_combination', k)[0]
                                                         - method_stats(ev, 'affine', 'unseen_combination', k)[0], 4)
        row['mlp_joint_seen'], row['mlp_joint_unseen'] = (method_stats(ev, 'mlp', s, 'joint')[0]
                                                          for s in ('seen_combination', 'unseen_combination'))
        row['mlp_share_of_reference_joint_unseen'] = (method_stats(ev, 'mlp', 'unseen_combination', 'joint')[0] /
                                                      max(method_stats(ev, 'reference_target', 'unseen_combination', 'joint')[0], 1e-9))
        comp.append(row)
    out['2_4_learned_vs_baselines_on_unseen_combinations'] = comp
    every_ci_positive = all(lo > 0 for row in comp for (_, (lo, _)) in row['mlp-affine'].items())
    every_seed_positive = all(d > 0 for row in comp for (d, _) in row['mlp-affine'].values())
    out['mlp_beats_affine_joint_unseen_every_run_and_seed'] = every_seed_positive
    out['mlp_beats_affine_joint_unseen_ci_excludes_zero_everywhere'] = every_ci_positive
    out['mlp_preservation_not_worse_than_affine_unseen'] = all(row['conditional_retention_mlp_minus_affine_unseen'] > -0.01
                                                              for row in comp)
    return out


def scatter(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#3d3d3a',
                         'xtick.color': '#6b6a65', 'ytick.color': '#6b6a65'})
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 3.4), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        ax.grid(color='#ecebe6', linewidth=0.6)
        ax.set_axisbelow(True)
        for meth in METHODS:
            x, xs, _ = method_stats(r['eval'], meth, 'unseen_combination', 'target_success')
            y, ys, _ = method_stats(r['eval'], meth, 'unseen_combination', 'conditional_retention')
            ax.errorbar(x, y, xerr=xs or 0, yerr=ys or 0, fmt=MARKERS[meth], ms=8 if meth != 'reference_target' else 11,
                        color=COLORS[meth], mec='white', mew=1.2, ecolor=COLORS[meth], elinewidth=1, label=LABELS[meth])
        ax.set_title(f"{r['dataset']} · {r['backend'].replace('native', 'CBM').upper()} seed {r['model_seed']}",
                     fontsize=8, color='#3d3d3a')
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel('target success (edited factor read correctly)')
    np.atleast_1d(axes)[0].set_ylabel('preservation | initially correct')
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(METHODS), frameon=False, fontsize=8)
    fig.suptitle('Held-out (unseen) combinations, identical test pairs for all methods. '
                 '* = uses the target image (reference, not deployable). Bars: sd over 3 operator seeds.',
                 fontsize=8, color='#3d3d3a')
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def qualitative(run, out, n=4, seed=0):
    """Source/target images + what each method's edited state reads as (intended vs unintended changes)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from datasets import load_from_disk
    ev, cfg = run['eval'], run['cfg']
    pairs_summary = json.loads((Path(cfg['pairs_dir'])/'summary.json').read_text())
    ds = load_from_disk(pairs_summary['raw_dir'])
    pp = pd.read_csv(run['dir']/'evaluation'/'per_pair.csv.gz')
    pp = pp[(pp.unseen_combination) & (pp.operator_seed.isna() | (pp.operator_seed == 0))]
    ref = pp[pp.method == 'reference_target'].set_index('pair_id')
    good = ref.index[ref.joint & ref.source_unaffected_correct]  # pairs the readers perceive correctly
    pick = np.random.default_rng(seed).choice(good, n, replace=False)
    edit = ev['edit_factor']
    others = [f for f in cfg['readers']['factors'] if f != edit]
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.3 * n), gridspec_kw={'width_ratios': [1, 1, 2.6]})
    for row, pid in enumerate(pick):
        sub = pp[pp.pair_id == pid].set_index('method')
        s_id, t_id = int(sub.iloc[0].source_id), int(sub.iloc[0].target_id)
        for col, (iid, title) in enumerate(((s_id, f'source  {edit}={int(sub.iloc[0].source_level)}'),
                                            (t_id, f'target  {edit}={int(sub.iloc[0].target_level)}'))):
            axes[row, col].imshow(ds[iid]['image'].convert('RGB'))
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis('off')
        lines = [f'requested: {edit} -> {int(sub.iloc[0].target_level)}', '']
        for meth in ('identity', 'native_calibrated', 'affine', 'mlp', 'reference_target'):
            r = sub.loc[meth]
            lost = [f for f in others if not r[f'preserved_{f}']]
            lines.append(f"{LABELS[meth]:<20s} reads {edit}={int(r.read_target_level)} "
                         f"{'OK ' if r.target_success else 'MISS'}  unintended: {', '.join(lost) or 'none'}"
                         f"{'' if r.task_correct else '  task WRONG'}")
        axes[row, 2].text(0, 0.5, '\n'.join(lines), family='monospace', fontsize=6.6, va='center')
        axes[row, 2].axis('off')
    fig.suptitle(f"{run['dataset']} {run['model']}: unseen-combination test pairs (operator seed 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs', type=Path, default=Path('results/semantic_edits/runs'))
    p.add_argument('--out', type=Path, default=Path('results/semantic_edits/summary'))
    args = p.parse_args(argv)
    runs = collect(args.runs)
    if not runs:
        raise SystemExit('no evaluated runs found')
    args.out.mkdir(parents=True, exist_ok=True)
    g = gate(runs)
    (args.out/'gate.json').write_text(json.dumps(g, indent=2) + '\n')
    md = ['# Semantic edits: native-model results', '',
          'Means over 3 operator seeds (± sd) for affine/MLP. `*` methods use the target image and are references.',
          'All metrics read by held-out factor readers except "native target (cal.)" (the model\'s own readout).', '',
          '## Unseen combinations (test)', '', table(runs, 'unseen_combination'), '',
          '## Seen combinations (test)', '', table(runs, 'seen_combination'), '',
          '## All test pairs', '', table(runs, 'all'), '',
          '## Base-model held-out endpoints (both images only in the base model\'s eval cache)', '',
          table(runs, 'base_model_heldout_endpoints'), '']
    (args.out/'results.md').write_text('\n'.join(md))
    scatter(runs, args.out/'target_vs_preservation.png')
    first = next(r for r in runs if r['dataset'] == 'shapes3d') if any(r['dataset'] == 'shapes3d' for r in runs) else runs[0]
    qualitative(first, args.out/'qualitative_shapes3d.png')
    mp = [r for r in runs if r['dataset'] == 'mpi3d']
    if mp:
        qualitative(mp[0], args.out/'qualitative_mpi3d.png')
    print(json.dumps({k: v for k, v in g.items() if not k[0].isdigit()}, indent=2))


if __name__ == '__main__':
    main()

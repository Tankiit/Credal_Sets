"""Fit held-out readers and edit operators for one frozen student checkpoint (training data only).

    python -m experiments.semantic_edits.train_native \
        --pairs-dir results/semantic_edits/pairs/shapes3d \
        --checkpoint ignore/results_experiments/cbm_results_seed0/shapes3d_dataset/audit.pt \
        --out results/semantic_edits/runs/shapes3d/cbm_seed0

Uses only operator-train pairs/images for fitting and operator-val pairs/images for
selection. Test pairs are never loaded here. Encoder, readout and task head stay frozen.
"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from .native_adapter import load_student, sha256, cached_latents, readout_and_logits, decode_levels, decode_calibrated
from .torch_operators import Standardizer, fit_operator, fit_readers

SPLIT_ID = {'train': 0, 'val': 1, 'test': 2}


def load_pairs(pairs_dir):
    pairs_dir = Path(pairs_dir)
    summary = json.loads((pairs_dir/'summary.json').read_text())
    pairs = dict(np.load(pairs_dir/'pairs.npz'))
    images = dict(np.load(pairs_dir/'images.npz'))
    return summary, pairs, images


def native_setup(checkpoint, summary, latents_dir, device):
    """Frozen model, all-image latents, and the checks that make native comparisons meaningful."""
    model, readout_name, ckpt = load_student(checkpoint, device=device)
    report = json.loads(Path(checkpoint).with_suffix('.json').read_text())
    tag = f"{summary['dataset']}_{ckpt['backend']}_seed{ckpt['seed']}"
    c = cached_latents(model, checkpoint, summary['cache'] + '_full', Path(latents_dir)/f'{tag}.npz', device)
    return model, readout_name, ckpt, report, c, tag


def verify_native(model, c, images, summary, report, device):
    """(1) ordinary forward pass reproduces the student's reported accuracy; (2) readout units are BCE logits."""
    scores, logits = readout_and_logits(model, c, device)
    eval_rows = images['base_eval']
    acc = float((logits[eval_rows].argmax(1).numpy() == images['label'][eval_rows]).mean())
    reported = report['consequence']['baseline_accuracy']
    n = int(eval_rows.sum())
    assert abs(acc - reported) <= 1 / n, f'forward pass does not reproduce reported accuracy ({acc} vs {reported})'
    K = len(summary['concept_columns'])
    units = {}
    for f, j in summary['concept_columns'].items():
        truth = images[f'factor_{f}']
        dec = decode_levels((scores[:, j] + scores[:, j + K]) / 2, summary['concept_level_values'][f]).numpy()
        units[f] = {'decoded_level_accuracy_all_images': float((dec == truth).mean()),
                    'mean_abs_sigmoid_error': float((scores[:, j].sigmoid().numpy() -
                                                     np.asarray(summary['concept_level_values'][f])[truth]).__abs__().mean())}
    # Requested intervention values must round-trip through the readout's units.
    edit = summary['edit_factor']
    j = summary['concept_columns'][edit]
    probe = c[:64].to(device=device, dtype=next(model.parameters()).dtype)
    for v in summary['edit_concept_values']:
        out = probe
        for col in (j, j + K):
            out = model.intervene(out, col, float(torch.logit(torch.tensor(np.clip(v, 1e-3, 1 - 1e-3)))))
        got = decode_levels((model.concept_readout(out)[:, j] + model.concept_readout(out)[:, j + K]).cpu() / 2,
                            summary['edit_concept_values'])
        assert (got == summary['edit_concept_values'].index(v)).all(), 'logit-unit intervention does not decode to target'
    return {'reproduced_accuracy': acc, 'reported_accuracy': reported, 'n_eval': n, 'tolerance': '1 / n_eval',
            'readout_units': 'logits trained with BCE-with-logits against [0,1] targets; decoded as nearest level of sigmoid(score), averaged over lower/upper columns',
            'per_concept': units, 'logit_intervention_round_trip': True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pairs-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--latents-dir', type=Path, default=Path('results/semantic_edits/latents'))
    p.add_argument('--operator-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--reader-steps', type=int, default=8000)
    p.add_argument('--reader-images', type=int, default=150000, help='max original images used to fit readers')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    dev = args.device
    torch.set_float32_matmul_precision('high')

    summary, pairs, images = load_pairs(args.pairs_dir)
    model, readout_name, ckpt, report, c, tag = native_setup(args.checkpoint, summary, args.latents_dir, dev)
    checks = verify_native(model, c, images, summary, report, dev)
    print(json.dumps({'run': tag, **{k: checks[k] for k in ('reproduced_accuracy', 'reported_accuracy')}}))

    edit, K = summary['edit_factor'], len(summary['concept_columns'])
    j = summary['concept_columns'][edit]
    L = summary['edit_levels']
    factors = summary['factors']
    tr_pairs, va_pairs = pairs['split'] == SPLIT_ID['train'], pairs['split'] == SPLIT_ID['val']

    # Normalization from operator-train endpoints only.
    tr_ids = np.unique(np.concatenate([pairs['source'][tr_pairs], pairs['target'][tr_pairs]]))
    cf = c.float()
    std = Standardizer(cf[tr_ids].mean(0), cf[tr_ids].std(0)).to(dev)

    # Held-out readers: original (unedited) images from operator-train / -val contexts only. They may see
    # held-out (edit value, combo) images from those contexts: readers are measuring instruments, and only the
    # edit operators must never have been fitted on a held-out combination.
    rng = np.random.default_rng(0)
    fit_img = np.flatnonzero(images['split'] == 'train')
    val_img = np.flatnonzero(images['split'] == 'val')
    fit_img = np.sort(rng.choice(fit_img, min(args.reader_images, len(fit_img)), replace=False))
    val_img = np.sort(rng.choice(val_img, min(args.reader_images // 3, len(val_img)), replace=False))
    cards = summary['factor_cardinalities']
    y = {f: torch.from_numpy(images[f'factor_{f}'][fit_img]).to(dev) for f in factors}
    yv = {f: torch.from_numpy(images[f'factor_{f}'][val_img]).to(dev) for f in factors}
    readers, reader_info = fit_readers(std(cf[fit_img].to(dev)), y, std(cf[val_img].to(dev)), yv, cards,
                                       steps=args.reader_steps, device=dev)

    # Calibrated native units: median (lower, upper) readout score per level of every supervised concept,
    # over operator-train images. Used for the calibrated native intervention and for calibrated decoding.
    scores, _ = readout_and_logits(model, c, dev)
    # Like the operators, calibration never uses a held-out combination.
    cal_img = fit_img[~images['heldout_combination'][fit_img]]
    calibration = {}
    for f, jj in summary['concept_columns'].items():
        lv = images[f'factor_{f}'][cal_img]
        calibration[f] = [[float(scores[cal_img[lv == v], jj].median()), float(scores[cal_img[lv == v], jj + K].median())]
                          for v in range(cards[f])]
    calib = calibration[edit]
    for f, jj in summary['concept_columns'].items():
        dec = decode_calibrated(scores[val_img][:, [jj, jj + K]], calibration[f]).numpy()
        checks['per_concept'][f]['calibrated_decoding_accuracy_val_images'] = float((dec == images[f'factor_{f}'][val_img]).mean())

    def tensors(mask):
        s, t = pairs['source'][mask], pairs['target'][mask]
        a = torch.nn.functional.one_hot(torch.from_numpy(pairs['target_level'][mask]), L).float().to(dev)
        return std(cf[s].to(dev)), a, std(cf[t].to(dev))

    train, val = tensors(tr_pairs), tensors(va_pairs)
    args.out.mkdir(parents=True)
    (args.out/'operators').mkdir()
    op_info = {}
    for seed in args.operator_seeds:
        for kind in ('affine', 'mlp'):
            op, info = fit_operator(kind, *train, val, seed=seed, steps=args.steps, device=dev)
            torch.save(op.state_dict(), args.out/'operators'/f'{kind}_seed{seed}.pt')
            op_info[f'{kind}_seed{seed}'] = info
            print(f'{tag} {kind} seed{seed}: val_mse={info["selected_val_mse"]:.5f} at step {info["selected_step"]}')
    torch.save(readers.state_dict(), args.out/'readers.pt')
    torch.save(std.state_dict(), args.out/'standardizer.pt')
    config = {
        'run': tag, 'pairs_dir': str(args.pairs_dir), 'pairs_summary_sha256': sha256(args.pairs_dir/'summary.json'),
        'pairs_npz_sha256': sha256(args.pairs_dir/'pairs.npz'),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'backend': ckpt['backend'], 'readout': readout_name, 'model_seed': ckpt['seed'],
        'latent_dim': model.latent_dim(), 'edit_factor': edit, 'edit_concept_columns': [j, j + K], 'edit_levels': L,
        'native_checks': checks, 'native_calibrated_scores': calib, 'native_calibration': calibration,
        'readers': {**reader_info, 'factors': factors, 'cards': cards, 'fit_images': 'all original images in operator-train contexts (subsampled)'},
        'operators': op_info, 'operator_seeds': args.operator_seeds,
        'supervision': 'operators: paired latent reconstruction only (no readout/preservation loss); '
                       'action = one-hot requested level; no target-image input at inference',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': dev},
        'source_sha256': {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
    }
    (args.out/'train_config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()

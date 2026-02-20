"""
Run training across multiple encoder backbones by invoking an existing trainer
script (ternary or SNLI) with different `--encoder_model` values.
"""

import argparse
import subprocess
import sys
from typing import List, Dict, Any


def _load_config(path: str) -> Dict[str, Any]:
    import json
    try:
        import yaml  # type: ignore
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        with open(path, 'r') as f:
            return json.load(f)


def encoder_to_tag(name: str) -> str:
    return name.replace('/', '-').replace(':', '-').replace(' ', '-')


def mode_suffix(mode: str) -> str:
    return {'post_hoc': 'post', 'fixed_eps': 'fixed', 'joint': 'joint'}.get(mode, mode)


def build_cmd(
    trainer: str,
    encoder_model: str,
    dataset: str,
    epochs: int,
    save_eval: bool,
    quad_method: str,
    quad_q: float,
    quad_eu_thr: float,
    quad_au_thr: float,
    eval_outdir: str,
    eval_template: str,
    seed: int,
    checkpoint_dir: str,
    save_ckpt_every: int,
    model_suffix: str | None,
    extra: List[str],
) -> List[str]:
    script = 'train_ternary_50epochs.py' if trainer == 'ternary' else 'train_snli.py'
    tag = encoder_to_tag(encoder_model)
    if model_suffix:
        tag = f"{tag}-{model_suffix}"

    cmd = [sys.executable, script,
           '--dataset', dataset,
           '--encoder_model', encoder_model,
           '--model_name', tag,
           '--epochs', str(int(epochs)),
           '--eval_outdir', eval_outdir,
           '--eval_template', eval_template]

    if seed is not None:
        cmd += ['--seed', str(seed)]
    if save_eval:
        cmd.append('--save_eval')
    if quad_method:
        cmd += ['--quad_method', quad_method]
        if quad_method == 'quantile':
            cmd += ['--quad_q', str(quad_q)]
        elif quad_method == 'fixed':
            if quad_eu_thr is None or quad_au_thr is None:
                raise ValueError('For fixed quad_method, both --quad_eu_thr and --quad_au_thr are required.')
            cmd += ['--quad_eu_thr', str(quad_eu_thr), '--quad_au_thr', str(quad_au_thr)]
    if checkpoint_dir:
        cmd += ['--checkpoint_dir', checkpoint_dir]
    if save_ckpt_every is not None:
        cmd += ['--save_ckpt_every', str(save_ckpt_every)]
    if extra is None:
        extra = []
    return cmd + extra


def run_one(dataset: str, trainer: str, encoders: List[str], epochs: int,
            quad_methods: List[str], quad_q: float,
            dro_modes: List[str], fixed_eps: float,
            checkpoint_dir: str, save_ckpt_every: int,
            save_eval: bool, eval_outdir: str, eval_template: str,
            seed: int, quad_eu_thr, quad_au_thr, dry_run: bool, stop_on_error: bool,
            extra_common: List[str]):
    for enc in encoders:
        for mode in dro_modes:
            for method in quad_methods:
                if method == 'median':
                    method_suffix = 'median'
                elif method == 'quantile':
                    method_suffix = f"q{str(quad_q).replace('.', '')}"
                else:
                    method_suffix = 'fixedthr'
                suffix = f"{mode_suffix(mode)}-{method_suffix}"

                extra_args = list(extra_common) + ['--dro_mode', mode]
                if mode == 'fixed_eps':
                    extra_args += ['--fixed_eps', str(fixed_eps)]

                cmd = build_cmd(
                    trainer=trainer,
                    encoder_model=enc,
                    dataset=dataset,
                    epochs=epochs,
                    save_eval=save_eval,
                    quad_method=method,
                    quad_q=quad_q,
                    quad_eu_thr=quad_eu_thr,
                    quad_au_thr=quad_au_thr,
                    eval_outdir=eval_outdir,
                    eval_template=eval_template,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                    save_ckpt_every=save_ckpt_every,
                    model_suffix=suffix,
                    extra=extra_args,
                )

                print("\n==> Running:")
                print(' '.join(cmd))
                if dry_run:
                    continue
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"Command failed for encoder {enc} (dataset={dataset}, mode={mode}, method={method}) with code {proc.returncode}")
                    if stop_on_error:
                        sys.exit(proc.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trainer', choices=['ternary', 'snli'], default='ternary')
    p.add_argument('--encoders', nargs='+', required=True)
    p.add_argument('--dataset', type=str, default='cebab')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--save_eval', action='store_true')
    p.add_argument('--quad_method', type=str, default='median', choices=['median', 'quantile', 'fixed'])
    p.add_argument('--quad_methods', nargs='+', choices=['median', 'quantile', 'fixed'], default=None)
    p.add_argument('--quad_q', type=float, default=0.5)
    p.add_argument('--quad_eu_thr', type=float, default=None)
    p.add_argument('--quad_au_thr', type=float, default=None)
    p.add_argument('--eval_outdir', type=str, default='eval_outputs')
    p.add_argument('--eval_template', type=str, default='{dataset}_{model}_{seed}_epoch{epoch}.pt')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--save_ckpt_every', type=int, default=None)
    p.add_argument('--dro_mode', type=str, default='joint', choices=['post_hoc', 'fixed_eps', 'joint'])
    p.add_argument('--dro_modes', nargs='+', choices=['post_hoc', 'fixed_eps', 'joint'], default=None)
    p.add_argument('--fixed_eps', type=float, default=0.1)
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--stop_on_error', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER)

    args = p.parse_args()

    extra = [a for a in args.extra if a != '--']
    quad_methods = args.quad_methods if args.quad_methods else [args.quad_method]

    if args.config:
        cfg = _load_config(args.config)
        runs = cfg.get('runs', [])
        for r in runs:
            dataset = r.get('dataset', args.dataset)
            trainer = r.get('trainer', args.trainer)
            encoders = r.get('encoders', args.encoders)
            epochs = int(r.get('epochs', args.epochs))
            q_methods = r.get('quad_methods', quad_methods)
            dro_modes = r.get('dro_modes', args.dro_modes if args.dro_modes else [args.dro_mode])
            fixed_eps = float(r.get('fixed_eps', args.fixed_eps))
            quad_q = float(r.get('quad_q', args.quad_q))
            save_eval = bool(r.get('save_eval', args.save_eval))
            eval_outdir = r.get('eval_outdir', args.eval_outdir)
            eval_template = r.get('eval_template', args.eval_template)
            seed = int(r.get('seed', args.seed))
            checkpoint_dir = r.get('checkpoint_dir', args.checkpoint_dir)
            save_ckpt_every = r.get('save_ckpt_every', args.save_ckpt_every)
            quad_eu_thr = r.get('quad_eu_thr', args.quad_eu_thr)
            quad_au_thr = r.get('quad_au_thr', args.quad_au_thr)

            extra_common: List[str] = []
            for k in ['batch_size', 'grad_accum_steps', 'max_length', 'num_workers',
                      'max_train_samples', 'max_val_samples', 'max_test_samples']:
                if k in r and r[k] is not None:
                    extra_common += [f'--{k}', str(r[k])] if not isinstance(r[k], bool) else ([f'--{k}'] if r[k] else [])
            if r.get('skip_plots', False):
                extra_common.append('--skip_plots')

            run_one(dataset, trainer, encoders, epochs, q_methods, quad_q,
                    dro_modes, fixed_eps, checkpoint_dir, save_ckpt_every,
                    save_eval, eval_outdir, eval_template, seed,
                    quad_eu_thr, quad_au_thr, args.dry_run, args.stop_on_error,
                    extra_common)
    else:
        dro_modes = args.dro_modes if args.dro_modes else [args.dro_mode]
        run_one(args.dataset, args.trainer, args.encoders, args.epochs,
                quad_methods, args.quad_q, dro_modes, args.fixed_eps,
                args.checkpoint_dir, args.save_ckpt_every,
                args.save_eval, args.eval_outdir, args.eval_template,
                args.seed, args.quad_eu_thr, args.quad_au_thr,
                args.dry_run, args.stop_on_error,
                extra)


if __name__ == '__main__':
    main()

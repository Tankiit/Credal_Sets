"""
Run training across multiple encoder backbones by invoking an existing trainer
script (ternary or SNLI) with different `--encoder_model` values.

Examples
- Ternary, three encoders, save evals with 75th percentile thresholds:
  python run_multi_encoders.py \
    --trainer ternary \
    --encoders distilbert-base-uncased bert-base-uncased roberta-base \
    --dataset cebab \
    --save_eval \
    --quad_method quantile --quad_q 0.75

- SNLI, fixed thresholds, custom outdir:
  python run_multi_encoders.py \
    --trainer snli \
    --encoders distilbert-base-uncased bert-base-uncased \
    --save_eval --quad_method fixed --quad_eu_thr 0.6 --quad_au_thr 0.3 \
    --eval_outdir eval_outputs_snli
"""

import argparse
import subprocess
import sys
import os
from typing import List, Dict, Any

def _load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file describing multiple runs.

    Schema:
    runs:
      - dataset: cebab | goemotions | snli | ...
        trainer: ternary | snli
        encoders: [distilbert-base-uncased, roberta-base]
        epochs: 50
        quad_methods: [median, quantile, fixed]
        quad_q: 0.75
        dro_modes: [post_hoc, fixed_eps, joint]
        fixed_eps: 0.1
        # Optional common flags:
        batch_size: 32
        grad_accum_steps: 2
        max_length: 128
        num_workers: 2
        max_train_samples: null
        max_val_samples: null
        max_test_samples: null
        checkpoint_dir: checkpoints
        save_ckpt_every: 5
        save_eval: true
        eval_outdir: eval_outputs
        eval_template: '{dataset}_{model}_{seed}_epoch{epoch}.pt'
        seed: -1
    """
    import json
    try:
        import yaml  # type: ignore
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback to JSON
        with open(path, 'r') as f:
            return json.load(f)


def encoder_to_tag(name: str) -> str:
    """Sanitize a HF model name to a short tag for filenames."""
    return name.replace('/', '-').replace(':', '-').replace(' ', '-')


def mode_suffix(mode: str) -> str:
    mapping = {
        'post_hoc': 'post',
        'fixed_eps': 'fixed',
        'joint': 'joint',
    }
    return mapping.get(mode, mode)


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
                raise ValueError("For fixed quad_method, both --quad_eu_thr and --quad_au_thr are required.")
            cmd += ['--quad_eu_thr', str(quad_eu_thr), '--quad_au_thr', str(quad_au_thr)]

    # Propagate checkpoint options
    if checkpoint_dir:
        cmd += ['--checkpoint_dir', checkpoint_dir]
    if save_ckpt_every is not None:
        cmd += ['--save_ckpt_every', str(save_ckpt_every)]

    # DRO mode and fixed-eps
    if extra is None:
        extra = []
    # Note: pass-through extras can include --dro_mode/--fixed_eps; we still set defaults below if not provided

    return cmd + extra

    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trainer', choices=['ternary', 'snli'], default='ternary',
                   help='Which trainer to run for each encoder')
    p.add_argument('--encoders', nargs='+', required=True,
                   help='List of HF model names to use as encoders')
    p.add_argument('--dataset', type=str, default='cebab')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=-1)

    # Eval saving options (mirrors training scripts)
    p.add_argument('--save_eval', action='store_true')
    p.add_argument('--quad_method', type=str, default='median', choices=['median', 'quantile', 'fixed'])
    p.add_argument('--quad_methods', nargs='+', choices=['median', 'quantile', 'fixed'], default=None,
                   help='Run multiple thresholding methods in one sweep; overrides --quad_method when set')
    p.add_argument('--quad_q', type=float, default=0.5)
    p.add_argument('--quad_eu_thr', type=float, default=None)
    p.add_argument('--quad_au_thr', type=float, default=None)
    p.add_argument('--eval_outdir', type=str, default='eval_outputs')
    p.add_argument('--eval_template', type=str, default='{dataset}_{model}_{seed}_epoch{epoch}.pt')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--save_ckpt_every', type=int, default=None, help='Override trainer checkpoint frequency')
    p.add_argument('--dro_mode', type=str, default='joint', choices=['post_hoc','fixed_eps','joint'])
    p.add_argument('--dro_modes', nargs='+', choices=['post_hoc','fixed_eps','joint'], default=None,
                   help='Run multiple DRO modes in one sweep; overrides --dro_mode when set')
    p.add_argument('--fixed_eps', type=float, default=0.1,
                   help='ε value for fixed_eps mode')
    p.add_argument('--config', type=str, default=None,
                   help='YAML/JSON config describing multiple dataset/encoder/mode sweeps')

    # Control
    p.add_argument('--dry_run', action='store_true', help='Print commands without running')
    p.add_argument('--stop_on_error', action='store_true', help='Stop loop if a command fails')

    # Capture passthrough extra args at end
    p.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args to pass to underlying trainer')

    args = p.parse_args()

    # Normalize potential leading '--' from remainder
    extra = [a for a in args.extra if a != '--']

    quad_methods = args.quad_methods if args.quad_methods else [args.quad_method]

def run_one(dataset, trainer, encoders, epochs, quad_methods, quad_q,
                dro_modes, fixed_eps, checkpoint_dir, save_ckpt_every,
                save_eval, eval_outdir, eval_template, seed,
                quad_eu_thr, quad_au_thr,
                extra_common):
        for enc in encoders:
            for mode in dro_modes:
                for method in quad_methods:
                    # Suffix the model tag to distinguish outputs per method
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
                    if args.dry_run:
                        continue

                    proc = subprocess.run(cmd)
                    if proc.returncode != 0:
                        msg = f"Command failed for encoder {enc} (dataset={dataset}, mode={mode}, method={method}) with code {proc.returncode}"
                        print(msg)
                        if args.stop_on_error:
                            sys.exit(proc.returncode)

    # If a config file is provided, iterate its runs; else use CLI
    if args.config:
        cfg = _load_config(args.config)
        runs = cfg.get('runs', [])
        for r in runs:
            dataset = r.get('dataset', args.dataset)
            trainer = r.get('trainer', args.trainer)
            encoders = r.get('encoders', args.encoders)
            epochs = int(r.get('epochs', args.epochs))
            q_methods = r.get('quad_methods', args.quad_methods if args.quad_methods else [args.quad_method])
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

            # Build extra args list for trainer pass-through
            extra_common = []
            for k in ['batch_size', 'grad_accum_steps', 'max_length', 'num_workers',
                      'max_train_samples', 'max_val_samples', 'max_test_samples']:
                if k in r and r[k] is not None:
                    extra_common += [f'--{k}', str(r[k])] if not isinstance(r[k], bool) else ([f'--{k}'] if r[k] else [])
            if r.get('skip_plots', False):
                extra_common.append('--skip_plots')

            run_one(dataset, trainer, encoders, epochs, q_methods, quad_q,
                    dro_modes, fixed_eps, checkpoint_dir, save_ckpt_every,
                    save_eval, eval_outdir, eval_template, seed,
                    quad_eu_thr, quad_au_thr,
                    extra_common)
    else:
        dro_modes = args.dro_modes if args.dro_modes else [args.dro_mode]
        run_one(args.dataset, args.trainer, args.encoders, args.epochs,
                quad_methods, args.quad_q, dro_modes, args.fixed_eps,
                args.checkpoint_dir, args.save_ckpt_every,
                args.save_eval, args.eval_outdir, args.eval_template,
                args.seed, args.quad_eu_thr, args.quad_au_thr,
                extra)


if __name__ == '__main__':
    main()

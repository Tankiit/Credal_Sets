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
from typing import List


def encoder_to_tag(name: str) -> str:
    """Sanitize a HF model name to a short tag for filenames."""
    return name.replace('/', '-').replace(':', '-').replace(' ', '-')


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
    extra: List[str],
) -> List[str]:
    script = 'train_ternary_50epochs.py' if trainer == 'ternary' else 'train_snli.py'
    tag = encoder_to_tag(encoder_model)

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

    if extra:
        cmd += extra

    # Propagate checkpoint options
    if checkpoint_dir:
        cmd += ['--checkpoint_dir', checkpoint_dir]
    if save_ckpt_every is not None:
        cmd += ['--save_ckpt_every', str(save_ckpt_every)]

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
    p.add_argument('--quad_q', type=float, default=0.5)
    p.add_argument('--quad_eu_thr', type=float, default=None)
    p.add_argument('--quad_au_thr', type=float, default=None)
    p.add_argument('--eval_outdir', type=str, default='eval_outputs')
    p.add_argument('--eval_template', type=str, default='{dataset}_{model}_{seed}_epoch{epoch}.pt')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--save_ckpt_every', type=int, default=None, help='Override trainer checkpoint frequency')

    # Control
    p.add_argument('--dry_run', action='store_true', help='Print commands without running')
    p.add_argument('--stop_on_error', action='store_true', help='Stop loop if a command fails')

    # Capture passthrough extra args at end
    p.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args to pass to underlying trainer')

    args = p.parse_args()

    # Normalize potential leading '--' from remainder
    extra = [a for a in args.extra if a != '--']

    for enc in args.encoders:
        cmd = build_cmd(
            trainer=args.trainer,
            encoder_model=enc,
            dataset=args.dataset,
            epochs=args.epochs,
            save_eval=args.save_eval,
            quad_method=args.quad_method,
            quad_q=args.quad_q,
            quad_eu_thr=args.quad_eu_thr,
            quad_au_thr=args.quad_au_thr,
            eval_outdir=args.eval_outdir,
            eval_template=args.eval_template,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            save_ckpt_every=args.save_ckpt_every,
            extra=extra,
        )

        print("\n==> Running:")
        print(' '.join(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            msg = f"Command failed for encoder {enc} with code {proc.returncode}"
            print(msg)
            if args.stop_on_error:
                sys.exit(proc.returncode)


if __name__ == '__main__':
    main()

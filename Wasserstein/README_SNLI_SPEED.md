# Speeding Up SNLI on Apple Silicon (MPS)

This note summarizes practical changes to speed up SNLI training when running on Apple Silicon (MPS). It works with the existing scripts in this repo (notably `test_ternary.py`) and avoids CUDA-only stacks.

## Summary
- Dynamic tokenization: Uses the fast Rust tokenizer with batched tokenization and dynamic padding for SNLI/ChaosNLI.
- MPS support: Prefers `mps` → `cuda` → `cpu` with `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops.
- CLI controls: Flags for dataset, batch size, max length, workers, epochs, LR, sample limits, and gradient accumulation.
- Gradient accumulation: Increases effective batch size without raising peak memory.

## How To Run (SNLI)
- Fast debug run:
  - `python test_ternary.py --dataset snli --batch_size 32 --num_workers 2 --max_length 128 --epochs 3 --grad_accum_steps 2 --max_train_samples 20000 --max_val_samples 5000`
- Memory‑limited but larger effective batch:
  - `python test_ternary.py --dataset snli --batch_size 16 --grad_accum_steps 4 --num_workers 2`

## Recommended Flags
- Batch and accumulation: `--batch_size 32` with `--grad_accum_steps 2–4` (effective batch = batch_size × grad_accum_steps)
- Sequence length: `--max_length 64–128` (64 often works well for SNLI)
- Data workers: `--num_workers 2` on macOS; increase only if input pipeline is the bottleneck
- Sampling caps for iteration speed: `--max_train_samples`, `--max_val_samples`, `--max_test_samples`

## Notes
- Precision: Stick to float32 on MPS; mixed precision is not mature on MPS.
- Bigger speed stacks (DeepSpeed, xFormers, bitsandbytes) are CUDA-only; consider NVIDIA GPUs to leverage them.
- Main lever for SNLI speed is the data pipeline: batched tokenization + dynamic padding yields the most impact on MPS.

## What Changed in This Repo
- `dataloader.py`: Added fast, batched tokenization for SNLI/ChaosNLI with dynamic padding; uses `use_fast=True` tokenizer; optional deferred tokenization.
- `test_ternary.py`: New CLI flags, device selection with MPS fallback, and gradient accumulation in the training loop.
- Device selection elsewhere: Prefer `mps` → `cuda` → `cpu` where applicable.

If you want the same CLI/accumulation pattern in other training scripts (e.g., `train_ternary_50epochs.py`), it can be added similarly.

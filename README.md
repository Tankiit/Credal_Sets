# NeurIPS Credal CBM

Clean-slate rewrite of the variational Credal CBM for NeurIPS submission.
- **Primary model**: `HybridCredalCBM` — μ / σ_epi / σ_ale heads, single forward pass.
- **Baseline**: `SingleHeadBaseline` — Mucsányi-style post-hoc EU/AU readout.
- **Datasets**: CEBaB, HateXplain, GoEmotions, SST-2.

`legacy/credence_acl/` contains the ACL-accepted CREDENCE code (frozen, do not modify).
`legacy/maqa_losses/` and `legacy/maqa_loaders/` contain MAQA paper code.
`.migrate-src/` contains source files still being adapted into the new tree.

Training logs can be written to TensorBoard and/or Weights & Biases via
`python -m experiments.train --log_dir <path> --wandb ...`; TensorBoard events
land in `<log_dir>/tensorboard/` and W&B runs default to offline mode.

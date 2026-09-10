# Real-data experiments

Image extraction, real-data configs, and annotated-cache training live here.
Synthetic data generation is not available from these entry points.

Start with the existing local CIFAR-10 data:

```bash
python -m experiments.real.extract --dataset cifar10 --data-dir /Users/cril/tanmoy/research/data --split test --out-dir features/real/cifar10/dinov2/test
```

Use `train` and a separate output directory for the training split. Without a
`--data-dir`, extraction uses the HF Hub. `--concept-column` and `--concepts-file`
provide explicit concept annotations; `--require-concepts` checks their presence.
See the [shared data guide](../../concept_audit/README.md) for annotation semantics.

CIFAR-10 has no supplied concept annotations, so its initial caches support image
pipeline validation only. The next scientific real-data stages are Shapes3D and
CUB, after the synthetic decision checkpoint.

Once annotated caches exist:

```bash
python -m experiments.real.run --config experiments/real/configs/cub_dinov2.json
```

The example config expects `features/real/cub/dinov2/{train,test}` containing
`embeddings.npy`, `labels.npy`, and `concepts.npy`. These data are not bundled or
fabricated. The runner requires both train and evaluation caches; omission is an
error, not a switch to synthetic data. Reports/checkpoints default to
`results/real/`, with `experiment_family=real` recorded in reports.

Later readout variants, CUB-S, compact backbone robustness, and optional Tinker
conditions belong in this family, following the [experiment plan](../README.md).

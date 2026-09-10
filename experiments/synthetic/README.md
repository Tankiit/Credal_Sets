# Synthetic experiments

Synthetic data generation lives here. These runs do not load images or frozen
feature caches. Shared training/audit implementation is in `concept_audit.training`.

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/full.json
python -m experiments.synthetic.run --config experiments/synthetic/configs/partial.json
python -m experiments.synthetic.run --config experiments/synthetic/configs/grouped.json
```

Outputs are JSON reports and `.pt` checkpoints under `results/synthetic/`.
The no-config default is `results/synthetic/audit.json`. Override epochs, seed,
or output using CLI flags. Each report identifies `experiment_family=synthetic`.

The current full readout has 3 latent coordinates and rank 3; partial and grouped
readouts have 6 coordinates, rank 3, and 3 unconstrained directions. These sanity
checks compare different latent widths. A causal comparison of supervision alone
must additionally hold latent width/encoder capacity fixed and specify the
corresponding supervised target construction. Identity readouts admit only A=I.

The first four stages and the scientific decision gate are in the parent
[experiment plan](../README.md). Existing smoke tests do not certify that gate.

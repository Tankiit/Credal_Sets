#!/usr/bin/env python
"""Experiment D: attribute-invisible transforms of a frozen DINOv2 representation.

Real CUB features. Fit R_attr: z -> c, generate A with R_attr A = R_attr, and
ask which representation diagnostics survive.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from concept_audit.audits.diagnostic_invariance import audit_diagnostic_invariance
from concept_audit.audits.record import RunRecord
from concept_audit.diagnostics.geometry import (
    blindspot_score, knn_label_purity, knn_mean_distance, neighborhood_overlap,
)
from concept_audit.families import AttributeReadoutModel
from concept_audit.transforms.equivalence import admissible_transform


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="directory with train.pt/test.pt/metadata.json")
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/real/cub_attribute_identifiability.json")
    args = p.parse_args(argv)

    cache = Path(args.cache)
    meta = json.loads((cache / "metadata.json").read_text())
    train = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
    test = torch.load(cache / "test.pt", map_location="cpu", weights_only=False)
    names = meta.get("source", {}).get("concept_names")

    model = AttributeReadoutModel.fit(train["features"], train["concepts"],
                                      ridge=args.ridge, concept_names=names)
    print(f"backbone={meta['backbone']} d={meta['feature_dim']} K={meta['n_concepts']}")
    print(f"R_attr {tuple(model.matrix.shape)} rank={model.rank} nullity={model.nullity}")
    print(f"attribute R2: train={model.r2(train['features'], train['concepts']):.3f} "
          f"test={model.r2(test['features'], test['concepts']):.3f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test["features"]), min(args.n_eval, len(test["features"])), replace=False)
    z = test["features"][idx].double()
    labels = test["labels"][idx].numpy()
    concepts = test["concepts"][idx].double()

    a = admissible_transform(model.matrix, strength=args.strength, seed=args.seed,
                             condition_limit=1e4)
    moved = z @ a.T
    print(f"\nR_attr A = R_attr         : {float((model.matrix @ a - model.matrix).abs().max()):.2e}")
    print(f"max |R z - R (Az)|        : "
          f"{float((model.predict_attributes(z) - model.predict_attributes(moved)).abs().max()):.2e}")
    print(f"latent actually moved     : {float((z - moved).abs().max()):.3f}  "
          f"cond(A)={float(torch.linalg.cond(a)):.1f}")

    diagnostics = {
        "knn_mean_distance": lambda x: knn_mean_distance(x.numpy(), args.k),
        "knn_label_purity": lambda x: knn_label_purity(x.numpy(), labels, args.k),
        "blindspot_score": lambda x: blindspot_score(
            knn_mean_distance(x.numpy(), args.k), concepts.sum(1).numpy()
        ),
    }
    report = audit_diagnostic_invariance(model, z, a, diagnostics, tol=1e-6)

    print(f"\nequivalence: {report.equivalence}")
    for name, delta in report.deltas.items():
        sp = "n/a" if delta.spearman is None else f"{delta.spearman:+.3f}"
        print(f"  {name:20s} rel_change={delta.relative_change:7.3f} spearman={sp:>7s} "
              f"invariant={delta.invariant}")
    overlap = neighborhood_overlap(z.numpy(), moved.numpy(), args.k)
    print(f"  neighbourhood overlap: mean={overlap.mean():.3f} | "
          f"points keeping all {args.k}: {(overlap == 1.0).mean():.1%}")
    print(f"  MOVED: {report.moved}")

    record = RunRecord.from_report(
        model, report, dataset={"name": "cub", "backbone": meta["backbone"], "split": "test",
                                "n_eval": int(len(idx)), "k": args.k},
        provenance={"cache": str(cache), "cache_sha256": meta.get("sha256"),
                    "strength": args.strength, "ridge": args.ridge, "seed": args.seed},
    )
    record.diagnostics["neighborhood_overlap"] = {
        "mean": float(overlap.mean()), "fraction_unchanged": float((overlap == 1.0).mean())}
    print("\nsaved:", record.save(args.out))


if __name__ == "__main__":
    main()

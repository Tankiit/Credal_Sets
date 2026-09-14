#!/usr/bin/env python
"""Experiment C: how large is each family's admissible class on the SAME real data?

PCA, NMF, SAE and the attribute-readout CBM are fitted to one frozen DINOv2 CUB
representation. For each we report the type and dimension of the admissible
class, verify one admissible transform exactly, and measure which geometry
diagnostics survive it.

Two honest caveats the data forces, both reported in the output:

* **NMF needs X >= 0** and ~half of standardised DINOv2 entries are negative.
  ``--nmf-input relu`` (default) factorises relu(Z); ``shift`` subtracts the
  global minimum instead. These are different datasets and give different
  factorisations -- the choice is recorded, not hidden.
* **PCA's continuous class needs repeated eigenvalues**, which a finite sample
  never produces exactly. ``--pca-rtol`` is the tolerance at which two
  eigenvalues count as equal, and the reported dimension is a function of it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from concept_audit.audits.diagnostic_invariance import audit_diagnostic_invariance
from concept_audit.audits.record import RunRecord
from concept_audit.diagnostics.geometry import (
    knn_label_purity, knn_mean_distance, neighborhood_overlap,
)
from concept_audit.families import AttributeReadoutModel, NMFModel, PCAModel, SAEModel
from concept_audit.transforms.equivalence import admissible_transform


def train_sae(z, n_units, steps=400, lr=1e-3, l1=1e-3, seed=0):
    """A small ReLU SAE, trained for real on the cached features."""
    torch.manual_seed(seed)
    d = z.shape[1]
    encoder = torch.nn.Linear(d, n_units).double()
    decoder = torch.nn.Linear(n_units, d, bias=False).double()
    with torch.no_grad():                       # unit-norm dictionary at init
        decoder.weight /= decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for step in range(steps):
        codes = torch.relu(encoder(z))
        loss = ((decoder(codes) - z) ** 2).mean() + l1 * codes.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sparsity = float((torch.relu(encoder(z)) == 0).double().mean())
    return encoder, decoder, float(loss), sparsity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--k", type=int, default=32, help="latent width for PCA/NMF")
    p.add_argument("--sae-units", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=1200)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--pca-rtol", type=float, default=1e-6)
    p.add_argument("--nmf-input", choices=["relu", "shift"], default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/family_comparison")
    args = p.parse_args(argv)

    cache = Path(args.cache)
    meta = json.loads((cache / "metadata.json").read_text())
    train = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
    test = torch.load(cache / "test.pt", map_location="cpu", weights_only=False)

    z_train = train["features"].double()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test["features"]), min(args.n_eval, len(test["features"])), replace=False)
    z_eval = test["features"][idx].double()
    labels = test["labels"][idx].numpy()
    print(f"real data: {meta['dataset']}/{meta['backbone']} d={meta['feature_dim']} "
          f"train={tuple(z_train.shape)} eval={tuple(z_eval.shape)}\n")

    rows, records = [], {}

    def orthogonality_deviation(a):
        """||A^T A - I||. Zero exactly when A is orthogonal.

        This turns out to be the variable that decides whether distance-based
        diagnostics survive: an orthogonal A preserves every pairwise distance,
        so kNN structure is identifiable no matter how large the class is.
        """
        a = torch.as_tensor(a).double()
        return float((a.T @ a - torch.eye(a.shape[0]).double()).abs().max())

    def evaluate(tag, model, codes_eval, transform, note, class_desc, tol=1e-6):
        diagnostics = {
            "knn_mean_distance": lambda x: knn_mean_distance(x.numpy(), args.neighbors),
            "knn_label_purity": lambda x: knn_label_purity(x.numpy(), labels, args.neighbors),
        }
        report = audit_diagnostic_invariance(model, codes_eval, transform, diagnostics,
                                             tol=tol, require_admissible=False)
        moved_codes = codes_eval @ transform.T
        overlap = float(neighborhood_overlap(codes_eval.numpy(), moved_codes.numpy(),
                                             args.neighbors).mean())
        eq = report.equivalence
        rows.append({
            "family": tag, "admissible_class": class_desc,
            "admissible": eq.admissible, "observable_error": eq.observable_error,
            "worst_constraint": eq.worst_constraint,
            "orthogonality_deviation": orthogonality_deviation(transform),
            "knn_distance_rel": report.deltas["knn_mean_distance"].relative_change,
            "knn_purity_rel": report.deltas["knn_label_purity"].relative_change,
            "overlap": overlap, "note": note,
        })
        records[tag] = RunRecord.from_report(
            model, report, dataset={"name": meta["dataset"], "backbone": meta["backbone"],
                                    "n_eval": int(len(idx))},
            provenance={"cache_sha256": meta.get("sha256"), "note": note,
                        "admissible_class": class_desc}).to_dict()
        records[tag]["diagnostics"]["neighborhood_overlap"] = {"mean": overlap}

    # -- PCA --------------------------------------------------------------------
    pca = PCAModel.fit(z_train, args.k)
    blocks = pca.eigenvalue_blocks(args.pca_rtol)
    cont = pca.admissible_dimension(args.pca_rtol)
    gaps = ((pca.eigenvalues[:-1] - pca.eigenvalues[1:]) / pca.eigenvalues[:-1].clamp(min=1e-12))
    print(f"PCA: {len(blocks)} eigen-blocks at rtol={args.pca_rtol:g}; continuous dim={cont}; "
          f"min relative gap={float(gaps.min()):.2e}")
    codes = pca.encode(z_eval)
    if cont > 0:
        evaluate("pca", pca, codes, pca.block_rotation(args.seed, args.pca_rtol),
                 f"rotation inside a repeated-eigenvalue block (rtol={args.pca_rtol:g})",
                 f"finite (sign/perm) + continuous dim {cont}")
    else:
        signs = torch.ones(args.k).double(); signs[::2] = -1.0
        evaluate("pca", pca, codes, pca.sign_flip(signs),
                 "no repeated eigenvalues at this rtol -> only the finite group",
                 f"finite only: 2^{args.k} sign flips (+ permutations if order unobservable)")

    # -- NMF --------------------------------------------------------------------
    if args.nmf_input == "relu":
        x_train, x_eval = torch.relu(z_train), torch.relu(z_eval)
        nmf_note = "factorised relu(Z); ~50% of standardised entries are negative"
    else:
        shift = z_train.min()
        x_train, x_eval = z_train - shift, z_eval - shift
        nmf_note = f"factorised Z - min (shift={float(shift):.3f}); adds a dense offset"
    from sklearn.decomposition import NMF as SkNMF

    sk = SkNMF(n_components=args.k, random_state=args.seed, init="nndsvda", max_iter=300)
    sk.fit(x_train.numpy())
    codes_nmf = torch.as_tensor(sk.transform(x_eval.numpy())).double()
    nmf = NMFModel(torch.as_tensor(sk.components_).double(), reference_codes=codes_nmf)
    print(f"NMF: {nmf_note}; reconstruction err={sk.reconstruction_err_:.2f}")
    evaluate("nmf", nmf, codes_nmf, nmf.monomial(seed=args.seed), nmf_note,
             f"monomial: {args.k}! permutations x positive diagonal (dim {args.k})")

    # -- SAE --------------------------------------------------------------------
    encoder, decoder, loss, sparsity = train_sae(z_train, args.sae_units, seed=args.seed)
    sae = SAEModel(encoder.weight.T.detach(), encoder.bias.detach(), decoder.weight.T.detach(),
                   reference_inputs=z_eval, penalty="l1", enforce_unit_norm=False)
    codes_sae = sae.encode(z_eval)
    print(f"SAE: trained loss={loss:.4f}, code sparsity={sparsity:.3f}")
    order = torch.randperm(args.sae_units, generator=torch.Generator().manual_seed(args.seed))
    evaluate("sae", sae, codes_sae, sae.permutation(order),
             "L1 penalty: permutation OK, positive rescaling is NOT (it changes the penalty)",
             f"permutation only ({args.sae_units}! elements) under L1")

    # -- CBM (attribute readout) ------------------------------------------------
    cbm = AttributeReadoutModel.fit(z_train, train["concepts"], ridge=1.0)
    a = admissible_transform(cbm.matrix, strength=0.5, seed=args.seed, condition_limit=1e4)
    print(f"CBM: R_attr rank={cbm.rank} nullity={cbm.nullity}")
    evaluate("cbm_attribute_readout", cbm, z_eval, a,
             "stabilizer of the fitted attribute readout",
             f"stabilizer: continuous, dim = nullity x d = {cbm.nullity} x {cbm.latent_dim()}")

    # -- report -----------------------------------------------------------------
    print(f"\n{'family':<22}{'obs_err':<11}{'||A^T A - I||':<15}{'kNN dist':<10}"
          f"{'kNN purity':<12}{'overlap':<9}")
    print("-" * 79)
    for r in rows:
        print(f"{r['family']:<22}{r['observable_error']:<11.2e}"
              f"{r['orthogonality_deviation']:<15.2e}{r['knn_distance_rel']:<10.3f}"
              f"{r['knn_purity_rel']:<12.3f}{r['overlap']:<9.3f}")
    print("\nAn orthogonal A preserves every pairwise distance, so geometry")
    print("diagnostics are identifiable exactly where ||A^T A - I|| = 0.")
    print("\nadmissible class per family:")
    for r in rows:
        print(f"  {r['family']:<22} {r['admissible_class']}")
        print(f"  {'':<22} note: {r['note']}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    for tag, record in records.items():
        (out / f"{tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()

"""
Aggregate snapshot + intervention JSONs into paper-ready tables.

Reads outputs/grad_iso_snapshots/*.json and outputs/interventions/*.json,
produces a LaTeX table for the NeurIPS paper's gradient-isolation + validity
section.

Usage:
    python -m analysis.aggregate_snapshots \
        --grad_iso_dir outputs/grad_iso_snapshots \
        --interventions_dir outputs/interventions \
        --output paper/tables/grad_iso.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Pretty-print dataset name
PRETTY = {
    "cebab": "CEBaB",
    "hatexplain": "HateXplain",
    "goemotions": "GoEmotions",
    "sst2": "SST-2",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grad_iso_dir", type=Path, default=Path("outputs/grad_iso_snapshots"))
    p.add_argument("--interventions_dir", type=Path, default=Path("outputs/interventions"))
    p.add_argument("--output", type=Path, default=Path("paper/tables/grad_iso.tex"))
    args = p.parse_args()

    # Read all JSONs
    grad_iso_by_ds = {}
    for f in sorted(args.grad_iso_dir.glob("*.json")):
        data = json.loads(f.read_text())
        grad_iso_by_ds[data["dataset"]] = data

    interv_by_ds = {}
    if args.interventions_dir.exists():
        for f in sorted(args.interventions_dir.glob("*.json")):
            data = json.loads(f.read_text())
            interv_by_ds[data["dataset"]] = data

    # Build table rows
    rows = []
    for ds in ["cebab", "hatexplain", "goemotions", "sst2"]:
        if ds not in grad_iso_by_ds:
            continue
        g = grad_iso_by_ds[ds]["summary"]
        i = interv_by_ds.get(ds, {})
        rows.append({
            "dataset": PRETTY.get(ds, ds),
            "cos_mu_mean": g["mean_cos_sim_mu_head"],
            "cos_mu_std": g["std_cos_sim_mu_head"],
            "cos_mu_abs_max": g["abs_max_cos_sim_mu_head"],
            "cross_sigma_max": max(g["max_g_kl_on_sigma_ale"], g["max_g_ale_on_sigma_epi"]),
            "rho_eu_au": i.get("rho_epi_ale_agg", float("nan")),
            "rho_epi_err": i.get("rho_epi_error_agg", float("nan")),
            "rho_ale_ent": i.get("rho_ale_entropy_agg", float("nan")),
        })

    # Print plain-text summary
    print(f"{'Dataset':12s} | {'cos(μ) mean±std':>18s} | {'cross-σ max':>11s} | "
          f"{'ρ(EU,AU)':>9s} | {'ρ(σ_epi,err)':>13s} | {'ρ(σ_ale,H)':>11s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['dataset']:12s} | "
              f"{r['cos_mu_mean']:+.3f} ± {r['cos_mu_std']:.3f}    | "
              f"{r['cross_sigma_max']:.1e}   | "
              f"{r['rho_eu_au']:+.3f}   | "
              f"{r['rho_epi_err']:+.3f}        | "
              f"{r['rho_ale_ent']:+.3f}")

    # Emit LaTeX
    tex = r"""\begin{table}[t]
\centering
\caption{Gradient isolation and intervention signatures on all benchmarks.
The cosine similarity of $\nabla \cL_{\text{KL}}$ and $\nabla \cL_{\text{ale}}$
on the shared $\mu$-head is measured over 50 validation batches per dataset.
Cross-head gradient norms (columns "cross-$\sigma$ max") verify the disjoint
tensor construction. Correlation columns validate that the decomposed
uncertainties track their intended signals.}
\label{tab:grad-iso-validity}
\small
\begin{tabular}{lccccc}
\toprule
Dataset & $\overline{\cos(g_{\text{kl}}, g_{\text{ale}})|_{\mu}}$
        & $\max \|{\cdot}\|_{\text{cross-}\sigma}$
        & $\rho(U_{\text{epi}}, U_{\text{ale}})$
        & $\rho(\sigma_{\text{epi}}, \text{err})$
        & $\rho(\sigma_{\text{ale}}, H_{\text{ann}})$ \\
\midrule
"""
    for r in rows:
        tex += (
            f"{r['dataset']} & "
            f"${r['cos_mu_mean']:+.3f} \\pm {r['cos_mu_std']:.3f}$ & "
            f"${r['cross_sigma_max']:.1e}$ & "
            f"${r['rho_eu_au']:+.3f}$ & "
            f"${r['rho_epi_err']:+.3f}$ & "
            f"${r['rho_ale_ent']:+.3f}$ \\\\\n"
        )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex)
    print(f"\n[wrote] {args.output}")


if __name__ == "__main__":
    main()
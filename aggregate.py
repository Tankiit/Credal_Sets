"""Aggregate script for tier0 sweep analysis.

Usage:
    python aggregate.py --jsonl results/tier0.jsonl
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

KEY = ["run", "rot", "mix", "shift", "tseed"]


def load(path: Path) -> pd.DataFrame:
    """Load sweep JSONL results and drop duplicate parameter keys."""
    df = pd.read_json(path, lines=True).drop_duplicates(KEY)
    print(f"Loaded {len(df)} unique evaluation records from {path}")
    if len(df) != 720:
        print(f"Warning: expected 720 records, found {len(df)}")
    return df


def table_agree(df: pd.DataFrame) -> pd.DataFrame:
    """Table for CBMs with head_input == 'logits': behavioural agreement & invisible CKA under mix."""
    sub = df[df["head_input"] == "logits"].copy()
    if sub.empty:
        print("No CBM-logits records found!")
        return pd.DataFrame()

    group_cols = ["dataset", "mix"]
    agg_funcs = {
        "intervened_agree": ["mean", "min", "max"],
        "d_push_centered": ["mean", "min", "max"],
        "cka_invis": ["mean", "min", "max"],
        "run": "count"
    }

    res = sub.groupby(group_cols).agg(agg_funcs)
    res.rename(columns={"run": "n"}, inplace=True)
    res = res.round(4)

    print("\n" + "=" * 90)
    print(" TABLE AGREE (CBM with head_input == 'logits')")
    print("=" * 90)
    print(res.to_string())
    return res


def table_cem_vs_cka(df: pd.DataFrame) -> pd.DataFrame:
    """Table for CEM models: shows intervention blindness (disagree ~0) vs dropping CKA/kNN."""
    sub = df[df["model_kind"] == "cem"].copy()
    if sub.empty:
        print("No CEM records found!")
        return pd.DataFrame()

    sub["disagree"] = 1.0 - sub["intervened_agree"]

    res = sub.groupby(["dataset", "mix"]).agg(
        max_disagree=("disagree", "max"),
        max_d_push_centered=("d_push_centered", "max"),
        mean_cka=("cka", "mean"),
        mean_knn=("knn", "mean"),
        n=("run", "count")
    ).round(4)

    print("\n" + "=" * 90)
    print(" TABLE CEM VS CKA (Intervention Blindness vs. Representation Geometry)")
    print("=" * 90)
    print(res.to_string())
    return res


def leakage_test(df: pd.DataFrame) -> pd.DataFrame:
    """Gauge leakage test: checks whether cond_B, mix, or rel_change_z best explains disagreement y = 1 - intervened_agree."""
    sub = df[(df["model_kind"] == "cbm") & (df["head_input"] == "logits")].copy()
    if sub.empty:
        print("No CBM-logits records for leakage test!")
        return pd.DataFrame()

    sub["y"] = 1.0 - sub["intervened_agree"]

    results = []
    for dataset, group in sub.groupby("dataset"):
        y = group["y"].values

        for var in ["mix", "cond_B", "rel_change_z"]:
            x = group[var].values
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                r_p, _ = pearsonr(x, y)
                r_s, _ = spearmanr(x, y)
                r2 = r_p ** 2
            else:
                r_p, r_s, r2 = 0.0, 0.0, 0.0

            results.append({
                "dataset": dataset,
                "predictor": var,
                "R2": round(r2, 4),
                "pearson_r": round(r_p, 4),
                "spearman_rho": round(r_s, 4)
            })

    res_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print(" LEAKAGE TEST (Predicting Gauge Disagreement y = 1 - intervened_agree)")
    print("=" * 90)
    print(res_df.to_string(index=False))
    return res_df


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to return a cleaned copy of sweep dataframe."""
    return df.copy()


def link_plot(df: pd.DataFrame, out_dir: Path = Path("results")):
    """C10, single panel: does cka_invis forecast intervention validity beyond mix?"""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    d = collapse(df)
    d = d[(d["model_kind"] == "cbm") & (d["head_input"] == "logits") & (d["mix"] > 0)].copy()

    if d.empty:
        print("No matching CBM-logits records for link_plot")
        return None, None, {}

    fig, ax = plt.subplots(figsize=(5, 4))
    markers = {0.5: "o", 1.0: "s", 2.0: "^"}
    for (ds, mix), g in d.groupby(["dataset", "mix"]):
        ax.scatter(g["cka_invis"], g["intervened_agree"], s=22, marker=markers.get(mix, "o"),
                   label=ds if mix == 0.5 else None, alpha=0.8)
    ax.set(xlabel="CKA on invisible block", ylabel="intervention agreement")

    rho_pool, _ = spearmanr(d["cka_invis"], d["intervened_agree"])

    # Within-mix rank correlation: center ranks within each (dataset, mix) cell
    d["r_cka"] = d.groupby(["dataset", "mix"])["cka_invis"].rank()
    d["r_agree"] = d.groupby(["dataset", "mix"])["intervened_agree"].rank()

    d["r_cka_centered"] = d["r_cka"] - d.groupby(["dataset", "mix"])["r_cka"].transform("mean")
    d["r_agree_centered"] = d["r_agree"] - d.groupby(["dataset", "mix"])["r_agree"].transform("mean")

    if np.std(d["r_cka_centered"]) > 1e-8 and np.std(d["r_agree_centered"]) > 1e-8:
        rho_within, _ = spearmanr(d["r_cka_centered"], d["r_agree_centered"])
    else:
        rho_within = 0.0

    # Per-dataset Spearman correlations
    per = {}
    for ds, g in d.groupby("dataset"):
        if len(g) > 1 and np.std(g["cka_invis"]) > 1e-8 and np.std(g["intervened_agree"]) > 1e-8:
            per[ds] = float(spearmanr(g["cka_invis"], g["intervened_agree"])[0])
        else:
            per[ds] = 0.0

    ax.set_title(f"pooled rho={rho_pool:.2f}"
                 + (f", within-mix rho={rho_within:.2f}" if rho_within is not None else ""))
    ax.legend(fontsize=7, title="dataset (marker = mix)")
    fig.tight_layout()
    fig.savefig(out_dir / "link_cka_agree.pdf")
    plt.close(fig)

    pd.Series(per).to_csv(out_dir / "link_per_dataset.csv")
    print(f"\nSaved link plot to {out_dir / 'link_cka_agree.pdf'} and per-dataset correlation CSV to {out_dir / 'link_per_dataset.csv'}")
    return rho_pool, rho_within, per


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep analysis")
    parser.add_argument("--jsonl", type=Path, default=Path("results/tier0.jsonl"), help="Path to tier0.jsonl")
    parser.add_argument("--out_dir", type=Path, default=Path("results"), help="Output results directory")
    args = parser.parse_args()

    df = load(args.jsonl)
    table_agree(df)
    table_cem_vs_cka(df)
    leakage_test(df)
    link_plot(df, args.out_dir)


if __name__ == "__main__":
    main()

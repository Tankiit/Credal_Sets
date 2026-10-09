"""Aggregate the 24-run Modal CBM/CEM sweep into reproducible local tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("results/modal_artifacts")
RUN = re.compile(r"^(cebab|goemotions|civil_comments|imdb_cad)-(cbm|cem)-s([0-2])\.metrics\.json$")
EXPECTED = {(dataset, arch, seed) for dataset in ("cebab", "goemotions", "civil_comments", "imdb_cad")
            for arch in ("cbm", "cem") for seed in range(3)}


def main() -> None:
    rows = []
    for path in sorted(ROOT.glob("*.metrics.json")):
        match = RUN.match(path.name)
        if not match:
            continue
        dataset, architecture, seed = match.groups()
        metrics = json.loads(path.read_text())
        history, test = metrics["history"], metrics["test"]
        rows.append({
            "dataset": dataset, "architecture": architecture, "seed": int(seed),
            "epochs_completed": len(history), "best_epoch": metrics["best_epoch"],
            "test_accuracy": test.get("accuracy", test.get("exact_match")),
            "test_task_loss": test["task_loss"], "test_concept_loss": test["concept_loss"],
            "final_train_loss": history[-1].get("train_loss"),
            "final_val_accuracy": history[-1]["val"].get("accuracy", history[-1]["val"].get("exact_match")),
            "final_val_task_loss": history[-1]["val"]["task_loss"],
        })
    actual = {(r["dataset"], r["architecture"], r["seed"]) for r in rows}
    missing = EXPECTED - actual
    if missing or len(rows) != len(EXPECTED):
        raise RuntimeError(f"Expected {len(EXPECTED)} unique runs; missing={sorted(missing)} rows={len(rows)}")
    per_run = pd.DataFrame(rows).sort_values(["dataset", "architecture", "seed"])
    per_run.to_csv(ROOT / "per_run.csv", index=False)

    metrics = ["test_accuracy", "test_task_loss", "test_concept_loss", "best_epoch",
               "final_train_loss", "final_val_accuracy", "final_val_task_loss"]
    grouped = per_run.groupby(["dataset", "architecture"], as_index=False)
    summary = grouped.agg(n=("seed", "count"), **{f"{metric}_mean": (metric, "mean") for metric in metrics},
                          **{f"{metric}_std": (metric, "std") for metric in metrics})
    summary.to_csv(ROOT / "summary.csv", index=False)

    display = summary[["dataset", "architecture", "n", "test_accuracy_mean", "test_accuracy_std",
                       "test_task_loss_mean", "test_task_loss_std", "best_epoch_mean", "best_epoch_std"]].copy()
    for col in display.columns[3:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    headers = list(display.columns)
    markdown = ["| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    lines = ["# Modal CBM/CEM sweep", "", "24 runs: 4 datasets × 2 architectures × 3 seeds; 50 epochs/run.", "",
             "## Test aggregation", "", *markdown, "",
             "`summary.csv` contains all aggregated metrics; `per_run.csv` contains every seed-level result."]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

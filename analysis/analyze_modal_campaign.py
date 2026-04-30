"""
Analyze Modal Credal CBM campaign outputs.

The script expects a directory produced by `modal volume get`, containing some
combination of:
  - run_index.csv
  - checkpoints/<run_id>/{test_metrics.json,training_history.json,grad_iso_log.jsonl}
  - eval_dumps/<run_id>/test_arrays.npz

It writes CSV/Markdown summaries and lightweight plots without loading model
weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text()
        if not text.strip():
            return None
        return json.loads(text)
    except Exception:
        return None


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _mean(values: list[float]) -> float:
    values = _finite(values)
    return mean(values) if values else math.nan


def _std(values: list[float]) -> float:
    values = _finite(values)
    return pstdev(values) if len(values) > 1 else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _checkpoint_root(root: Path) -> Path:
    candidates = [
        root / "checkpoints",
        root / "checkpoints" / "checkpoints",
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*/test_metrics.json")):
            return candidate
    return root / "checkpoints" / "checkpoints"


def _parse_run_id(run_id: str) -> dict[str, Any]:
    parts = run_id.split("_")
    parsed: dict[str, Any] = {
        "run_id": run_id,
        "model": parts[0] if parts else "",
        "dataset": parts[1] if len(parts) > 1 else "",
    }
    for token in parts:
        if token.startswith("seed"):
            parsed["seed"] = token.removeprefix("seed")
        elif token.startswith("aw"):
            parsed["aleatoric_weight"] = _float(token.removeprefix("aw"))
        elif token.startswith("ap"):
            parsed["aleatoric_prior"] = _float(token.removeprefix("ap"))
        elif token.startswith("kw"):
            parsed["kl_weight"] = _float(token.removeprefix("kw"))
        elif token.startswith("tf"):
            parsed["train_fraction"] = _float(token.removeprefix("tf")) / 100.0

    if "matrix" in parts:
        parsed["tag"] = "matrix"
    elif "data" in parts and "scaling" in parts:
        parsed["tag"] = "data_scaling"
    elif "ale" in parts and "weight" in parts:
        parsed["tag"] = "sweep_ale_weight"
    elif "ale" in parts and "prior" in parts:
        parsed["tag"] = "sweep_ale_prior"
    elif "kl" in parts and "weight" in parts:
        parsed["tag"] = "sweep_kl_weight"
    return parsed


def _best_history(history_path: Path) -> dict[str, Any]:
    data = _json_load(history_path)
    if not data:
        return {}
    history = data.get("history", [])
    best = None
    for item in history:
        val = item.get("val")
        if not val:
            continue
        acc = _float(val.get("task_accuracy", val.get("accuracy")))
        if best is None or acc > best["best_val_task_accuracy"]:
            best = {
                "best_epoch": item.get("epoch"),
                "best_val_task_accuracy": acc,
                "best_val_concept_accuracy": _float(val.get("mean_concept_accuracy")),
                "best_val_rho_eu_au": _float(val.get("rho_eu_au")),
                "best_val_rho_eu_error": _float(val.get("rho_eu_error")),
                "best_val_rho_ale_entropy": _float(val.get("rho_ale_entropy")),
            }
    return best or {}


def _grad_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"grad_iso_steps": 0}

    numeric: dict[str, list[float]] = defaultdict(list)
    steps = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            steps += 1
            for key, value in rec.items():
                if isinstance(value, (int, float)):
                    numeric[key].append(float(value))

    out: dict[str, Any] = {"grad_iso_steps": steps}
    for key in (
        "loss_eu",
        "loss_au",
        "cos_sim_encoder",
        "cos_sim_concept_encoder",
        "cos_sim_concept_classifier",
        "cos_sim_task_classifier",
        "cos_sim_aleatoric_head",
        "g_eu_encoder_norm",
        "g_au_encoder_norm",
        "g_eu_concept_encoder_norm",
        "g_au_concept_encoder_norm",
        "g_eu_concept_classifier_norm",
        "g_au_concept_classifier_norm",
        "g_eu_task_classifier_norm",
        "g_au_task_classifier_norm",
        "g_eu_aleatoric_head_norm",
        "g_au_aleatoric_head_norm",
    ):
        vals = numeric.get(key, [])
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_std"] = _std(vals)
    return out


def write_grad_iso_markdown(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    complete = [r for r in rows if r.get("grad_iso_steps", 0)]
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_tag[str(row.get("tag", ""))].append(row)

    def fmt(value: Any, digits: int = 4) -> str:
        try:
            v = float(value)
        except Exception:
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    lines = [
        "# Gradient Isolation Summary",
        "",
        "This table is the readable companion to `grad_iso_summary.csv`.",
        "Cosine columns are `n/a` when the recorded cosine similarity was undefined,",
        "typically because one of the compared gradient vectors had zero norm.",
        "",
    ]

    for tag in sorted(by_tag):
        tag_rows = sorted(by_tag[tag], key=lambda r: (str(r.get("dataset", "")), str(r.get("run_id", ""))))
        lines += [
            f"## {tag}",
            "",
            "| run_id | dataset | grad steps | loss_eu | loss_au | g_eu(concept_enc) | g_au(ale_head) | cos(concept_enc) | cos(ale_head) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in tag_rows:
            lines.append(
                f"| {row.get('run_id', '')} | {row.get('dataset', '')} | "
                f"{int(row.get('grad_iso_steps', 0))} | "
                f"{fmt(row.get('loss_eu_mean'))} | {fmt(row.get('loss_au_mean'))} | "
                f"{fmt(row.get('g_eu_concept_encoder_norm_mean'))} | {fmt(row.get('g_au_aleatoric_head_norm_mean'))} | "
                f"{fmt(row.get('cos_sim_concept_encoder_mean'))} | {fmt(row.get('cos_sim_aleatoric_head_mean'))} |"
            )
        lines.append("")

    (out_dir / "grad_iso_summary.md").write_text("\n".join(lines) + "\n")


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ckpt_root = _checkpoint_root(root)
    index_rows = {row["run_id"]: row for row in _read_csv(root / "run_index.csv") if row.get("run_id")}

    run_ids = set(index_rows)
    run_ids.update(path.parent.name for path in ckpt_root.glob("*/test_metrics.json"))
    run_ids.update(path.parent.name for path in ckpt_root.glob("*/training_history.json"))

    rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []

    for run_id in sorted(run_ids):
        parsed = _parse_run_id(run_id)
        idx = index_rows.get(run_id, {})
        run_dir = ckpt_root / run_id
        metrics = _json_load(run_dir / "test_metrics.json") or {}

        row = {
            **parsed,
            "exit_code": idx.get("exit_code", ""),
            "elapsed_sec": idx.get("elapsed_sec", ""),
            "has_metrics": bool(metrics),
            "test_task_accuracy": _float(metrics.get("task_accuracy", metrics.get("accuracy"))),
            "test_concept_accuracy": _float(metrics.get("mean_concept_accuracy")),
            "concept_coverage": _float(metrics.get("concept_coverage")),
            "test_loss": _float(metrics.get("loss")),
            "rho_eu_au": _float(metrics.get("rho_eu_au")),
            "rho_eu_error": _float(metrics.get("rho_eu_error")),
            "rho_ale_entropy": _float(metrics.get("rho_ale_entropy")),
            "mean_eu": _float(metrics.get("mean_eu")),
            "mean_au": _float(metrics.get("mean_au")),
            "mean_sigma_epi": _float(metrics.get("mean_sigma_epi")),
            "mean_sigma_ale": _float(metrics.get("mean_sigma_ale")),
        }
        row.update(_best_history(run_dir / "training_history.json"))
        rows.append(row)

        for concept, acc in (metrics.get("concept_accs") or {}).items():
            concept_rows.append({
                "run_id": run_id,
                "model": row.get("model", ""),
                "dataset": row.get("dataset", ""),
                "tag": row.get("tag", ""),
                "concept": concept,
                "accuracy": acc,
            })

        grad = _grad_summary(run_dir / "grad_iso_log.jsonl")
        grad_rows.append({"run_id": run_id, **parsed, **grad})

    return rows, concept_rows, grad_rows


def write_markdown(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    complete = [r for r in rows if r.get("has_metrics")]
    failures = [r for r in rows if str(r.get("exit_code")) not in ("", "0")]
    best_task = sorted(complete, key=lambda r: _float(r["test_task_accuracy"]), reverse=True)[:10]
    best_concept = sorted(complete, key=lambda r: _float(r["test_concept_accuracy"]), reverse=True)[:10]

    lines = [
        "# Modal Campaign Analysis",
        "",
        f"- Runs indexed: {len(rows)}",
        f"- Runs with test metrics: {len(complete)}",
        f"- Indexed failures: {len(failures)}",
        "",
        "## Top Task Accuracy",
        "",
        "| run_id | task_acc | concept_acc | rho_ale_entropy |",
        "|---|---:|---:|---:|",
    ]
    for r in best_task:
        lines.append(
            f"| {r['run_id']} | {r['test_task_accuracy']:.4f} | "
            f"{r['test_concept_accuracy']:.4f} | {r['rho_ale_entropy']:.3f} |"
        )

    lines += [
        "",
        "## Top Concept Accuracy",
        "",
        "| run_id | task_acc | concept_acc | coverage |",
        "|---|---:|---:|---:|",
    ]
    for r in best_concept:
        lines.append(
            f"| {r['run_id']} | {r['test_task_accuracy']:.4f} | "
            f"{r['test_concept_accuracy']:.4f} | {r['concept_coverage']:.3f} |"
        )

    if failures:
        lines += ["", "## Failures", "", "| run_id | exit_code |", "|---|---:|"]
        for r in failures:
            lines.append(f"| {r['run_id']} | {r['exit_code']} |")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_sliced_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[str(row.get("tag", ""))].append(row)

    matrix_fields = [
        "run_id",
        "model",
        "dataset",
        "test_task_accuracy",
        "test_concept_accuracy",
        "rho_eu_au",
        "rho_eu_error",
        "rho_ale_entropy",
    ]
    _write_csv(out_dir / "matrix_results.csv", by_tag.get("matrix", []), matrix_fields)

    sweep_specs = {
        "sweep_ale_weight": "aleatoric_weight",
        "sweep_ale_prior": "aleatoric_prior",
        "sweep_kl_weight": "kl_weight",
        "data_scaling": "train_fraction",
    }
    fields = [
        "run_id",
        "test_task_accuracy",
        "test_concept_accuracy",
        "rho_eu_au",
        "rho_eu_error",
        "rho_ale_entropy",
    ]
    for tag, param in sweep_specs.items():
        data = sorted(by_tag.get(tag, []), key=lambda row: _float(row.get(param)))
        _write_csv(out_dir / f"{tag}.csv", data, [param] + fields)


def write_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    complete = [r for r in rows if r.get("has_metrics")]

    matrix = [r for r in complete if r.get("tag") == "matrix"]
    if matrix:
        labels = [f"{r['model']}\n{r['dataset']}" for r in matrix]
        x = range(len(matrix))
        plt.figure(figsize=(10, 4))
        plt.bar(x, [r["test_task_accuracy"] for r in matrix], label="task")
        plt.bar(x, [r["test_concept_accuracy"] for r in matrix], alpha=0.6, label="concept")
        plt.xticks(list(x), labels, rotation=0)
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "matrix_accuracy.png", dpi=160)
        plt.close()

    for tag, param in {
        "sweep_ale_weight": "aleatoric_weight",
        "sweep_ale_prior": "aleatoric_prior",
        "sweep_kl_weight": "kl_weight",
        "data_scaling": "train_fraction",
    }.items():
        data = sorted(
            [r for r in complete if r.get("tag") == tag],
            key=lambda row: _float(row.get(param)),
        )
        if not data:
            continue
        xs = [_float(r.get(param)) for r in data]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [r["test_task_accuracy"] for r in data], marker="o", label="task")
        plt.plot(xs, [r["test_concept_accuracy"] for r in data], marker="o", label="concept")
        plt.xlabel(param)
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{tag}_accuracy.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/modal_credal_runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/modal_credal_runs/analysis"))
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows, concept_rows, grad_rows = collect(args.root)

    summary_fields = [
        "run_id",
        "model",
        "dataset",
        "tag",
        "seed",
        "exit_code",
        "elapsed_sec",
        "has_metrics",
        "aleatoric_weight",
        "aleatoric_prior",
        "kl_weight",
        "train_fraction",
        "test_task_accuracy",
        "test_concept_accuracy",
        "concept_coverage",
        "test_loss",
        "rho_eu_au",
        "rho_eu_error",
        "rho_ale_entropy",
        "best_epoch",
        "best_val_task_accuracy",
        "best_val_concept_accuracy",
    ]
    _write_csv(args.out / "campaign_summary.csv", rows, summary_fields)
    _write_csv(args.out / "concept_accuracy_long.csv", concept_rows)
    _write_csv(args.out / "grad_iso_summary.csv", grad_rows)
    write_grad_iso_markdown(args.out, grad_rows)
    write_sliced_tables(args.out, rows)
    write_markdown(args.out, rows)
    if not args.no_plots:
        write_plots(args.out, rows)

    complete = sum(1 for row in rows if row.get("has_metrics"))
    print(f"Wrote analysis for {len(rows)} indexed runs ({complete} with metrics) to {args.out}")


if __name__ == "__main__":
    main()

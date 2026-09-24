"""
Analyze one or more audit.json reports produced by experiments.real.run /
experiments.synthetic.run.

Usage:
    python analyze_audit.py path/to/audit.json [more paths...]
    python analyze_audit.py path1 path2 path3 --md-out report.md

With one path: prints a detailed single-run summary (aggregated stats over
all concepts, not just index 0), plus the most/least causally-impactful
concepts by consequence.per_concept[*].accuracy_drop.

With multiple paths: prints the same summary for each, then a compact
side-by-side comparison table of the key scalars across all runs.

--md-out PATH: in addition to the console output, write everything as a
single Markdown report (tables instead of plain text) to PATH -- handy as
a README-style summary of the whole comparison.
"""
import argparse
import json
import statistics as stats
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(values):
    """min/mean/max for a list of floats, or None if empty."""
    if not values:
        return None
    return {"min": min(values), "mean": stats.mean(values), "max": max(values)}


def fmt_agg(a, digits=4):
    if a is None:
        return "n/a"
    return f"min={a['min']:.{digits}f} mean={a['mean']:.{digits}f} max={a['max']:.{digits}f}"


def block_field_values(block_list, field):
    """Pull the aggregatable list of values for `field` out of either shape
    this schema uses: a flat dict of field->list (equivalence.baseline/
    .transformed), or a list of per-concept dicts that each redundantly
    hold the full field list (structural/informational)."""
    if isinstance(block_list, dict):
        return block_list.get(field, [])
    return block_list[0].get(field, []) if block_list else []


FIELDS = [
    "own_concept_abs_correlation",
    "cross_concept_abs_correlation",
    "own_correlation_share",
    "block_task_probe_accuracy",
    "block_head_frobenius_norm",
]


def collect_summary(report, path):
    """Pull every number we want out of one report into a plain dict, used
    by both the console printer and the markdown renderer so the two never
    drift apart."""
    eq = report.get("equivalence", {})
    cons = report.get("consequence", {})
    per_concept = cons.get("per_concept", [])
    drops = [(c["concept_id"], c["accuracy_drop"]) for c in per_concept]
    drop_vals = [d for _, d in drops]

    block_sections = {}
    for label, block_list in (
        ("equivalence.baseline", eq.get("baseline")),
        ("equivalence.transformed", eq.get("transformed")),
        ("structural", report.get("structural")),
        ("informational", report.get("informational")),
    ):
        if not block_list:
            continue
        block_sections[label] = {
            field: agg(block_field_values(block_list, field)) for field in FIELDS
        }

    return {
        "path": str(path),
        "name": Path(path).parent.name or Path(path).name,
        "backend": report.get("backend"),
        "epochs": report.get("epochs"),
        "seed": report.get("seed"),
        "train_samples": report.get("train_samples"),
        "eval_samples": report.get("eval_samples"),
        "n_concepts": len(report.get("blocks", [])),
        "readout_rank": report.get("readout_rank"),
        "unconstrained_dim": report.get("unconstrained_dim"),
        "max_readout_error": eq.get("max_readout_error"),
        "max_logit_error": eq.get("max_logit_error"),
        "condition_number": eq.get("condition_number"),
        "block_sections": block_sections,
        "baseline_accuracy": cons.get("baseline_accuracy"),
        "replacement": cons.get("replacement"),
        "n_drop_concepts": len(drops),
        "drop_agg": agg(drop_vals),
        "top_drops": sorted(drops, key=lambda x: -x[1])[:5],
        "bottom_drops": sorted(drops, key=lambda x: x[1])[:5],
        "tensorboard_log_dir": report.get("tensorboard_log_dir"),
    }


# ---------- console rendering ----------

def print_summary(s):
    print(f"\n{'=' * 70}\n{s['path']}\n{'=' * 70}")
    print(f"backend={s['backend']}  epochs={s['epochs']}  seed={s['seed']}")
    print(f"train_samples={s['train_samples']}  eval_samples={s['eval_samples']}")
    rank_ok = s['readout_rank'] == s['unconstrained_dim']
    print(f"num_concepts={s['n_concepts']}  readout_rank={s['readout_rank']}  "
          f"unconstrained_dim={s['unconstrained_dim']}  "
          f"{'[FULL RANK]' if rank_ok else '[RANK DEFICIT]'}")
    print(f"\nEquivalence check:")
    print(f"  max_readout_error={s['max_readout_error']}  "
          f"max_logit_error={s['max_logit_error']}  "
          f"condition_number={s['condition_number']}")
    for label, fields in s["block_sections"].items():
        print(f"\n  -- {label} --")
        for field, a in fields.items():
            print(f"    {field}: {fmt_agg(a)}")
    print(f"\nConsequence (substitution) test:")
    print(f"  baseline_accuracy={s['baseline_accuracy']}  replacement={s['replacement']}")
    if s['drop_agg']:
        print(f"  accuracy_drop across {s['n_drop_concepts']} concepts: {fmt_agg(s['drop_agg'])}")
        print(f"  most impactful concepts:")
        for cid, d in s['top_drops']:
            print(f"    concept {cid}: drop={d:.4f}")
        print(f"  least impactful / negative-drop concepts:")
        for cid, d in s['bottom_drops']:
            print(f"    concept {cid}: drop={d:.4f}")
    print(f"\ntensorboard_log_dir: {s['tensorboard_log_dir']}")


def print_comparison(summaries):
    print(f"\n{'=' * 70}\nSIDE-BY-SIDE COMPARISON\n{'=' * 70}")
    headers = ["name", "train_samples", "eval_samples", "n_concepts",
               "readout_rank", "unconstrained_dim", "max_logit_error",
               "condition_number", "baseline_accuracy"]
    rows = [{h: s.get(h) for h in headers} for s in summaries]
    widths = {h: max(len(h), max(len(f"{r[h]}") for r in rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(f"{r[h]}".ljust(widths[h]) for h in headers))


# ---------- markdown rendering ----------

def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_markdown(summaries):
    out = ["# Concept-Audit Analysis Report", ""]

    if len(summaries) > 1:
        out.append("## Comparison across runs")
        out.append("")
        headers = ["Run", "Train", "Eval", "#Concepts", "Rank", "Unconstrained dim",
                   "Max logit error", "Condition #", "Baseline accuracy"]
        rows = []
        for s in summaries:
            rank_flag = "full rank" if s["readout_rank"] == s["unconstrained_dim"] else "RANK DEFICIT"
            rows.append([
                s["name"], s["train_samples"], s["eval_samples"], s["n_concepts"],
                f"{s['readout_rank']} ({rank_flag})", s["unconstrained_dim"],
                f"{s['max_logit_error']:.2e}" if s["max_logit_error"] is not None else "n/a",
                f"{s['condition_number']:.4f}" if s["condition_number"] is not None else "n/a",
                f"{s['baseline_accuracy']:.4f}" if s["baseline_accuracy"] is not None else "n/a",
            ])
        out.append(md_table(headers, rows))
        out.append("")

    for s in summaries:
        out.append(f"## {s['name']}")
        out.append("")
        out.append(f"- Source: `{s['path']}`")
        out.append(f"- backend=`{s['backend']}`, epochs=`{s['epochs']}`, seed=`{s['seed']}`")
        out.append(f"- train_samples=`{s['train_samples']}`, eval_samples=`{s['eval_samples']}`")
        rank_ok = s['readout_rank'] == s['unconstrained_dim']
        out.append(f"- num_concepts=`{s['n_concepts']}`, readout_rank=`{s['readout_rank']}`, "
                    f"unconstrained_dim=`{s['unconstrained_dim']}` "
                    f"{'✅ full rank' if rank_ok else '⚠️ RANK DEFICIT'}")
        out.append("")

        out.append("### Equivalence check")
        out.append("")
        out.append(md_table(
            ["max_readout_error", "max_logit_error", "condition_number"],
            [[s["max_readout_error"], s["max_logit_error"], s["condition_number"]]],
        ))
        out.append("")

        for label, fields in s["block_sections"].items():
            out.append(f"### {label} (aggregated over {s['n_concepts']} concepts)")
            out.append("")
            rows = [[field, fmt_agg(a)] for field, a in fields.items()]
            out.append(md_table(["field", "min / mean / max"], rows))
            out.append("")

        out.append("### Consequence (substitution) test")
        out.append("")
        out.append(f"- baseline_accuracy=`{s['baseline_accuracy']}`, replacement=`{s['replacement']}`")
        if s["drop_agg"]:
            out.append(f"- accuracy_drop across {s['n_drop_concepts']} concepts: {fmt_agg(s['drop_agg'])}")
            out.append("")
            out.append("**Most impactful concepts (largest accuracy drop when substituted):**")
            out.append("")
            out.append(md_table(["concept_id", "accuracy_drop"],
                                 [[cid, f"{d:.4f}"] for cid, d in s["top_drops"]]))
            out.append("")
            out.append("**Least impactful / negative-drop concepts:**")
            out.append("")
            out.append(md_table(["concept_id", "accuracy_drop"],
                                 [[cid, f"{d:.4f}"] for cid, d in s["bottom_drops"]]))
        out.append("")
        out.append(f"tensorboard_log_dir: `{s['tensorboard_log_dir']}`")
        out.append("")
        out.append("---")
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more audit.json files")
    ap.add_argument("--md-out", default=None,
                     help="also write a Markdown report to this path")
    args = ap.parse_args()

    summaries = []
    for path in args.paths:
        report = load(path)
        s = collect_summary(report, path)
        summaries.append(s)
        print_summary(s)

    if len(summaries) > 1:
        print_comparison(summaries)

    if args.md_out:
        md = render_markdown(summaries)
        with open(args.md_out, "w") as f:
            f.write(md)
        print(f"\n[saved] Markdown report -> {args.md_out}")


if __name__ == "__main__":
    main()
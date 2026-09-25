"""
Local GPU campaign runner for MAQA / AmbigQA.

Produces a Modal-like artifact layout:
  outputs/qa_campaign/
    checkpoints/<run_id>/
    eval_dumps/<run_id>/
    logs/<run_id>.log
    run_index.csv

Usage:
    export HF_TOKEN=...
    python scripts/run_qa_campaign.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("outputs/qa_campaign")
SEED = 42


def _run_one(cfg: dict) -> dict:
    model = cfg["model"]
    dataset = cfg["dataset"]
    run_tag = cfg.get("run_tag", dataset)
    epochs = cfg.get("epochs", 50)
    seed = cfg.get("seed", SEED)
    batch_size = cfg.get("batch_size", 32)
    max_length = cfg.get("max_length", 128)
    encoder = cfg.get("encoder", "distilbert-base-uncased")
    hf_dataset = cfg.get("hf_dataset", "ttomov/ambigqa_star")

    run_id = f"{dataset}_seed{seed}_{run_tag}"
    save_dir = ROOT / "checkpoints" / run_id
    eval_dir = ROOT / "eval_dumps" / run_id
    log_path = ROOT / "logs" / f"{run_id}.log"
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = env.get("HF_TOKEN", "") or env.get("HUGGINGFACE_TOKEN", "")
    env["HUGGINGFACE_TOKEN"] = env.get("HUGGINGFACE_TOKEN", "") or env.get("HF_TOKEN", "")

    cmd = [
        "python",
        "-u",
        "-m",
        "experiments.train_maqa",
        "--dataset",
        dataset,
        "--hf_dataset",
        hf_dataset,
        "--encoder",
        encoder,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--eval_every",
        str(cfg.get("eval_every", 1)),
        "--num_workers",
        str(cfg.get("num_workers", 4)),
        "--save_dir",
        str(save_dir),
        "--eval_dump_dir",
        str(eval_dir),
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
    ]
    if cfg.get("use_paired", False):
        cmd.append("--use_paired")

    start = time.time()
    with log_path.open("w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=env,
            check=False,
        )
    elapsed = time.time() - start

    metrics_path = eval_dir / "test_metrics.json"
    tail = ""
    if log_path.exists():
        try:
            tail = log_path.read_text()[-2000:]
        except Exception:
            tail = ""

    return {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "exit_code": result.returncode,
        "elapsed_sec": round(elapsed, 1),
        "save_dir": str(save_dir),
        "log_path": str(log_path),
        "eval_dir": str(eval_dir),
        "tag": run_tag,
        "has_metrics": metrics_path.exists(),
        "tail": tail,
    }


def _write_run_index(rows: list[dict]) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "run_index.csv"
    fields = [
        "run_id",
        "model",
        "dataset",
        "seed",
        "exit_code",
        "elapsed_sec",
        "save_dir",
        "log_path",
        "eval_dir",
        "tag",
        "has_metrics",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / f"run_index_{stamp}.json").write_text(json.dumps(rows, indent=2))
    return path


def main() -> int:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("HF_TOKEN or HUGGINGFACE_TOKEN is required for this campaign.")
        return 2

    configs = [
        {
            "model": "maqa",
            "dataset": "maqa",
            "run_tag": "full",
            "hf_dataset": "ttomov/ambigqa_star",
            "epochs": 50,
            "batch_size": 32,
            "max_length": 128,
            "use_paired": False,
        },
        {
            "model": "maqa",
            "dataset": "ambigqa",
            "run_tag": "full",
            "hf_dataset": "ttomov/ambigqa_star",
            "epochs": 50,
            "batch_size": 32,
            "max_length": 128,
            "use_paired": False,
        },
    ]

    rows = []
    for cfg in configs:
        print(f"[launch] {cfg['dataset']}")
        row = _run_one(cfg)
        rows.append(row)
        if row["exit_code"] != 0:
            print(f"[warn] run failed: {row['run_id']}")
            print(row["tail"])
            break

    index_path = _write_run_index(rows)
    print(f"[done] wrote {index_path}")
    return 0 if all(r["exit_code"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

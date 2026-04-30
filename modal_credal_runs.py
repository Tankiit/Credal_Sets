"""
Modal launcher for the Credal CBM/SENN experimental campaign.

Usage:
    modal run modal_credal_runs.py --which matrix
    modal run modal_credal_runs.py --which sweep_ale_weight
    modal run modal_credal_runs.py --which sweep_ale_prior
    modal run modal_credal_runs.py --which sweep_kl_weight
    modal run modal_credal_runs.py --which data_scaling
    modal run modal_credal_runs.py --which all

Pulling results back to local:
    modal volume get credal-runs /persistent/checkpoints ./checkpoints
    modal volume get credal-runs /persistent/eval_dumps ./eval_dumps
    modal volume get credal-runs /persistent/run_index.csv ./
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("credal-cbm-runs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "datasets==2.21.0",
        "scipy",
        "scikit-learn",
        "pandas",
        "wandb",
        "tensorboard",
        "tqdm",
    )
    .add_local_dir(
        ".",
        remote_path="/repo",
        ignore=[
            "checkpoints/",
            "logs/",
            "wandb/",
            ".git/",
            "__pycache__/",
            "*.pyc",
        ],
    )
)

volume = modal.Volume.from_name("credal-runs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-api-key")

SEED = 42


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/persistent": volume},
    secrets=[wandb_secret],
    timeout=4 * 3600,
    retries=0,
)
def run_training(cfg: dict) -> dict:
    """Run one training job from a config dictionary."""
    import os
    import subprocess
    import time

    model = cfg["model"]
    dataset = cfg["dataset"]
    seed = cfg["seed"]
    epochs = cfg.get("epochs", 50)
    aleatoric_weight = cfg.get("aleatoric_weight")
    aleatoric_prior = cfg.get("aleatoric_prior")
    kl_weight = cfg.get("kl_weight")
    train_fraction = cfg.get("train_fraction")
    run_tag = cfg.get("run_tag", "default")
    hf_token = cfg.get("hf_token")

    parts = [model, dataset, f"seed{seed}", run_tag]
    if aleatoric_weight is not None:
        parts.append(f"aw{aleatoric_weight}")
    if aleatoric_prior is not None:
        parts.append(f"ap{aleatoric_prior}")
    if kl_weight is not None:
        parts.append(f"kw{kl_weight}")
    if train_fraction is not None:
        parts.append(f"tf{int(train_fraction * 100)}")
    run_id = "_".join(parts)

    save_dir = f"/persistent/checkpoints/{run_id}"
    log_path = f"/persistent/logs/{run_id}.log"
    eval_dir = f"/persistent/eval_dumps/{run_id}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    env = os.environ.copy()
    env["WANDB_PROJECT"] = "credal-cbm-neurips"
    env["WANDB_GROUP"] = run_tag
    env["WANDB_NAME"] = run_id
    env["WANDB_RUN_ID"] = run_id
    env["WANDB_RESUME"] = "allow"
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_TOKEN"] = hf_token

    if dataset in {"maqa", "ambigqa"}:
        cmd = [
            "python",
            "-u",
            "-m",
            "experiments.train_maqa",
            "--dataset",
            dataset,
            "--hf_dataset",
            cfg.get("hf_dataset", "ttomov/ambigqa_star"),
            "--encoder",
            cfg.get("encoder", "distilbert-base-uncased"),
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
            save_dir,
            "--eval_dump_dir",
            eval_dir,
        ]
    else:
        cmd = [
            "python",
            "-u",
            "-m",
            "experiments.train",
            "--model",
            model,
            "--dataset",
            dataset,
            "--encoder",
            "distilbert-base-uncased",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--grad_iso",
            "--wandb",
            "--wandb_mode",
            "online",
            "--eval_every",
            str(cfg.get("eval_every", 5)),
            "--save_every",
            str(cfg.get("save_every", 0)),
            "--num_workers",
            str(cfg.get("num_workers", 4)),
            "--save_dir",
            save_dir,
            "--eval_dump_dir",
            eval_dir,
        ]
        if aleatoric_weight is not None:
            cmd += ["--aleatoric_weight", str(aleatoric_weight)]
        if aleatoric_prior is not None:
            cmd += ["--aleatoric_prior", str(aleatoric_prior)]
        if kl_weight is not None:
            cmd += ["--kl_weight", str(kl_weight)]
        if train_fraction is not None:
            cmd += ["--train_fraction", str(train_fraction)]

    start = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(
            cmd,
            cwd="/repo",
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start

    volume.commit()

    try:
        with open(log_path) as f:
            tail = f.read()[-2000:]
    except Exception:
        tail = ""

    return {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "exit_code": result.returncode,
        "elapsed_sec": round(elapsed, 1),
        "save_dir": save_dir,
        "log_path": log_path,
        "eval_dir": eval_dir,
        "tag": run_tag,
        "tail": tail,
    }


@app.function(image=image, volumes={"/persistent": volume})
def write_run_index(results: list[dict], which: str) -> str:
    """Persist the campaign index in the Modal volume."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = Path(f"/persistent/run_index_{which}_{ts}.json")
    csv_path = Path("/persistent/run_index.csv")

    json_path.write_text(json.dumps(results, indent=2))

    keys = [
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
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in keys})

    volume.commit()
    return str(csv_path)


def cfg_matrix() -> list[dict]:
    return [
        {"model": model, "dataset": dataset, "seed": SEED, "run_tag": "matrix"}
        for model in ("cbm", "senn")
        for dataset in ("cebab", "hatexplain", "goemotions")
    ]


def cfg_sweep_ale_weight() -> list[dict]:
    return [
        {
            "model": "cbm",
            "dataset": "cebab",
            "seed": SEED,
            "aleatoric_weight": weight,
            "run_tag": "sweep_ale_weight",
        }
        for weight in (0.5, 1.0, 2.0, 4.0, 8.0)
    ]


def cfg_sweep_ale_prior() -> list[dict]:
    return [
        {
            "model": "cbm",
            "dataset": "cebab",
            "seed": SEED,
            "aleatoric_prior": prior,
            "run_tag": "sweep_ale_prior",
        }
        for prior in (0.02, 0.05, 0.10, 0.20, 0.30)
    ]


def cfg_sweep_kl_weight() -> list[dict]:
    return [
        {
            "model": "cbm",
            "dataset": "cebab",
            "seed": SEED,
            "kl_weight": weight,
            "run_tag": "sweep_kl_weight",
        }
        for weight in (0.001, 0.01, 0.1, 1.0)
    ]


def cfg_data_scaling() -> list[dict]:
    return [
        {
            "model": "cbm",
            "dataset": "cebab",
            "seed": SEED,
            "train_fraction": fraction,
            "run_tag": "data_scaling",
        }
        for fraction in (0.25, 0.50, 0.75, 1.00)
    ]


def cfg_qa(which: str) -> list[dict]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return [
        {
            "model": "maqa",
            "dataset": which,
            "seed": SEED,
            "hf_dataset": "ttomov/ambigqa_star",
            "run_tag": which,
            **({"hf_token": hf_token} if hf_token else {}),
        }
    ]


@app.local_entrypoint()
def main(which: str = "matrix"):
    """Fan out the requested workload across A100-40GB workers in parallel."""
    table = {
        "matrix": cfg_matrix(),
        "sweep_ale_weight": cfg_sweep_ale_weight(),
        "sweep_ale_prior": cfg_sweep_ale_prior(),
        "sweep_kl_weight": cfg_sweep_kl_weight(),
        "data_scaling": cfg_data_scaling(),
        "maqa": cfg_qa("maqa"),
        "ambigqa": cfg_qa("ambigqa"),
        "all": (
            cfg_matrix()
            + cfg_sweep_ale_weight()
            + cfg_sweep_ale_prior()
            + cfg_sweep_kl_weight()
            + cfg_data_scaling()
            + cfg_qa("maqa")
            + cfg_qa("ambigqa")
        ),
    }
    if which not in table:
        raise SystemExit(f"unknown workload: {which}; pick from {list(table)}")

    configs = table[which]
    print(f"[launch] {which}: {len(configs)} run(s)")
    for cfg in configs:
        print("   ", cfg)

    results = list(run_training.map(configs))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_path = Path(f"results_{which}_{ts}.json")
    local_path.write_text(json.dumps(results, indent=2))
    remote_index = write_run_index.remote(results, which)

    print(f"\n[done] wrote {local_path} and {remote_index} ({len(results)} runs)")
    n_fail = sum(1 for result in results if result["exit_code"] != 0)
    if n_fail:
        print(f"[warn] {n_fail}/{len(results)} run(s) failed")
        for result in results:
            if result["exit_code"] != 0:
                print(f"\n--- {result['run_id']} tail ---")
                print(result["tail"])

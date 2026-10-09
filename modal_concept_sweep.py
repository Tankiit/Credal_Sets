"""Modal T4 sweep for the four supported concept datasets.

Run all 24 jobs (4 datasets × CBM/CEM × seeds 0,1,2):

    modal run modal_concept_sweep.py --epochs 50

Artifacts persist in the ``credal-concept-sweep`` Modal Volume:
``prepared/`` holds frozen encoder features, ``runs/`` checkpoints/metrics, and
``tensorboard/`` event files.  TensorBoard can consume the downloaded
``tensorboard/`` directory directly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "credal-concept-sweep"
DATASETS = ("cebab", "goemotions", "civil_comments", "imdb_cad")
ARCHITECTURES = ("cbm", "residual-cbm", "cem")
SEEDS = (0, 1, 2)
REMOTE_ROOT = "/root/credal-sets"

app = modal.App(APP_NAME)
results = modal.Volume.from_name("credal-concept-sweep", create_if_missing=True)
hf_cache = modal.Volume.from_name("credal-concept-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3", "transformers>=4.44,<5", "datasets>=3,<5",
        "huggingface_hub", "numpy>=1.26,<3", "pandas>=2.2", "scikit-learn>=1.4",
        "tensorboard>=2.18", "pytorch_concepts==1.0.0a5", "tqdm", "sentencepiece", "protobuf",
    )
    .add_local_dir("concept_models", remote_path=f"{REMOTE_ROOT}/concept_models")
    .add_local_dir("concept_datasets", remote_path=f"{REMOTE_ROOT}/concept_datasets")
    .add_local_file("train_concept_models.py", remote_path=f"{REMOTE_ROOT}/train_concept_models.py")
)

COMMON = dict(
    image=image,
    gpu="T4",  # smallest readily available NVIDIA GPU class for this workload
    timeout=60 * 60,
    volumes={"/artifacts": results, "/root/.cache/huggingface": hf_cache},
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)


@app.function(**COMMON)
def prepare_dataset(dataset: str, embed_batch_size: int = 128) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    out = f"/artifacts/prepared/{dataset}"
    schema = Path(out) / "schema.json"
    if not all((Path(out) / f"{split}.pt").exists() for split in ("train", "val", "test")):
        run([sys.executable, "train_concept_models.py", "prepare", "--dataset", dataset,
             "--out", out, "--embed-batch-size", str(embed_batch_size)])
        results.commit()
    return {"dataset": dataset, "data": out, "prepared": schema.exists()}


@app.function(**COMMON)
def train_one(dataset: str, architecture: str, seed: int, epochs: int = 50,
              learning_rate: float = 1e-3, backend: str = "torch", task_loss: str = "cross_entropy") -> dict:
    if dataset not in DATASETS or architecture not in ARCHITECTURES or seed not in SEEDS or backend not in {"torch", "torch-concepts"} or task_loss not in {"cross_entropy", "mse"}:
        raise ValueError("Unsupported dataset, architecture, seed, or backend")
    data = f"/artifacts/prepared/{dataset}"
    if not all((Path(data) / f"{split}.pt").exists() for split in ("train", "val", "test")):
        raise FileNotFoundError(f"Prepared data missing for {dataset}; run prepare_dataset first.")
    name = f"{dataset}-{architecture}-{backend}-{task_loss}-s{seed}"
    out = f"/artifacts/runs/{name}"
    metrics = Path(out) / "metrics.json"
    if not metrics.exists():
        run([
            sys.executable, "train_concept_models.py", "train", "--data", data,
            "--arch", architecture, "--out", out, "--epochs", str(epochs),
            "--patience", str(epochs), "--lr", str(learning_rate), "--batch-size", "256",
            "--seed", str(seed), "--backend", backend, "--task-loss", task_loss, "--device", "cuda",
            "--tensorboard-dir", f"/artifacts/tensorboard/{name}",
        ])
        results.commit()
    return {"run": name, "metrics": str(metrics), "tensorboard": f"/artifacts/tensorboard/{name}"}


@app.function(**COMMON)
def cebab_cbm_mse_all(epochs: int = 50, learning_rate: float = 1e-3) -> list[dict]:
    """Run every MSE seed in one durable remote task (safe to rerun)."""
    data = "/artifacts/prepared/cebab"
    if not all((Path(data) / f"{split}.pt").exists() for split in ("train", "val", "test")):
        raise FileNotFoundError("Prepared CEBaB data is missing")
    completed = []
    for seed in SEEDS:
        name = f"cebab-cbm-torch-mse-s{seed}"
        out = Path(f"/artifacts/runs/{name}")
        metrics = out / "metrics.json"
        if not metrics.exists():
            run([
                sys.executable, "train_concept_models.py", "train", "--data", data,
                "--arch", "cbm", "--out", str(out), "--epochs", str(epochs),
                "--patience", str(epochs), "--lr", str(learning_rate), "--batch-size", "256",
                "--seed", str(seed), "--backend", "torch", "--task-loss", "mse", "--device", "cuda",
                "--tensorboard-dir", f"/artifacts/tensorboard/{name}",
            ])
            results.commit()
        completed.append({"run": name, "metrics": str(metrics)})
    return completed


@app.function(**COMMON)
def torch_concepts_all(epochs: int = 50, learning_rate: float = 1e-3) -> list[dict]:
    """Complete the Torch Concepts CBM/residual-CBM grid in one durable task."""
    completed = []
    for dataset in DATASETS:
        data = Path(f"/artifacts/prepared/{dataset}")
        if not all((data / f"{split}.pt").exists() for split in ("train", "val", "test")):
            raise FileNotFoundError(f"Prepared data missing for {dataset}")
        for architecture in ("cbm", "residual-cbm"):
            for seed in SEEDS:
                name = f"{dataset}-{architecture}-torch-concepts-s{seed}"
                out = Path(f"/artifacts/runs/{name}")
                metrics = out / "metrics.json"
                if not metrics.exists():
                    run([
                        sys.executable, "train_concept_models.py", "train", "--data", str(data),
                        "--arch", architecture, "--out", str(out), "--epochs", str(epochs),
                        "--patience", str(epochs), "--lr", str(learning_rate), "--batch-size", "256",
                        "--seed", str(seed), "--backend", "torch-concepts", "--device", "cuda",
                        "--tensorboard-dir", f"/artifacts/tensorboard/{name}",
                    ])
                    results.commit()
                completed.append({"run": name, "metrics": str(metrics)})
    return completed


@app.local_entrypoint()
def sweep(epochs: int = 50, learning_rate: float = 1e-3) -> None:
    """Prepare each dataset once, then run CBM/CEM across seeds sequentially on one T4."""
    for dataset in DATASETS:
        print(prepare_dataset.remote(dataset))
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                print(train_one.remote(dataset, architecture, seed, epochs, learning_rate))


@app.local_entrypoint()
def sweep_torch_concepts(epochs: int = 50, learning_rate: float = 1e-3) -> None:
    """Compare vanilla and residual CBMs through Torch Concepts' low-level layers."""
    for dataset in DATASETS:
        print(prepare_dataset.remote(dataset))
        for architecture in ("cbm", "residual-cbm"):
            for seed in SEEDS:
                print(train_one.remote(dataset, architecture, seed, epochs, learning_rate, "torch-concepts"))


@app.local_entrypoint()
def cebab_cbm_mse(epochs: int = 50, learning_rate: float = 1e-3) -> None:
    """Controlled 5-class CEBaB CBM comparison with a categorical MSE task loss."""
    print(prepare_dataset.remote("cebab"))
    print(cebab_cbm_mse_all.remote(epochs, learning_rate))


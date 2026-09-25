"""
Launch the QA Modal jobs for MAQA and AmbigQA.

Usage:
    export HF_TOKEN=...
    python scripts/run_qa_sweep.py

This is intentionally minimal: it just triggers the two Modal workloads
back-to-back and streams their output through the Modal CLI.
"""
from __future__ import annotations

import os
import subprocess


def main() -> int:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("HF_TOKEN or HUGGINGFACE_TOKEN is not set.")
        print("Set one of them before running this script.")
        return 2

    for which in ("maqa", "ambigqa"):
        print(f"\n=== Running Modal QA sweep: {which} ===")
        result = subprocess.run(
            ["modal", "run", "modal_credal_runs.py", "--which", which],
            check=False,
        )
        if result.returncode != 0:
            print(f"\n{which} failed with exit code {result.returncode}")
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

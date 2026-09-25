#!/usr/bin/env python3
"""Run a command while preserving enough context to reproduce it later."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot() -> dict:
    if subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        check=False,
    ).returncode:
        return {"available": False}

    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    ).stdout
    return {
        "available": True,
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "status": run_text(["git", "status", "--short"]),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_stat": run_text(["git", "diff", "--stat", "HEAD"]),
    }


def tracked_files(paths: list[str], command: list[str]) -> dict:
    candidates = list(paths)
    candidates.extend(token for token in command if token.endswith(".py"))
    records = {}
    for raw_path in dict.fromkeys(candidates):
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            records[str(path)] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute an experiment and save a reproducibility manifest."
    )
    parser.add_argument("--name", required=True, help="Stable experiment name")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--track",
        action="append",
        default=[],
        help="Extra code/config/data file to checksum; may be repeated",
    )
    parser.add_argument(
        "--root",
        default="repro_runs",
        help="Directory for run manifests (default: repro_runs)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")

    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_seed{args.seed}"
    run_dir = Path(args.root) / args.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
        "name": args.name,
        "run_id": run_id,
        "seed": args.seed,
        "started_at_utc": started.isoformat(),
        "working_directory": str(Path.cwd()),
        "command": command,
        "command_shell": shlex.join(command),
        "git": git_snapshot(),
        "tracked_files": tracked_files(args.track, command),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    freeze = run_text([sys.executable, "-m", "pip", "freeze"])
    (run_dir / "requirements.freeze.txt").write_text(freeze + "\n")

    print(f"[repro] manifest: {run_dir / 'manifest.json'}", flush=True)
    completed = subprocess.run(command, check=False)

    ended = dt.datetime.now(dt.timezone.utc)
    manifest.update({
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "exit_code": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
    })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

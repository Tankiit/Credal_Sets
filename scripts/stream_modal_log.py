"""Stream `modal app logs <app>` into a file, dropping intermediate progress-bar redraws.

    python scripts/stream_modal_log.py <app_id> <dest>

Exits when the app stops. Modal keeps only the tail of a finished app's logs,
so this must run while the app is alive to capture its full output.
"""
import re
import subprocess
import sys

app, dest = sys.argv[1:]
MODAL = "/Users/tanmoy/anaconda3/envs/torch-multimodal/bin/modal"
bar = re.compile(r"\d+%\|.*\| *\d+/\d+")
with open(dest, "a", buffering=1) as f:
    proc = subprocess.Popen([MODAL, "app", "logs", app], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    for raw in proc.stdout:
        for line in raw.rstrip("\n").split("\r"):
            line = line.replace("\x1b[1A", "").rstrip()
            if not line or (bar.search(line) and "100%|" not in line):
                continue
            f.write(line + "\n")
    proc.wait()
    f.write(f"[stream ended, exit {proc.returncode}]\n")

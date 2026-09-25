#!/usr/bin/env bash
# After collect_campaign.sh exits, gather every log of the retraining campaign
# into outputs/icml_2026_reeval/logs/ and push it:
#   modal/<group>.log         stdout/stderr of each Modal training app
#   eval/<run>.log            local re-evaluation output
#   training_history/<run>.json  per-epoch train/val metrics from Modal
#   campaign_collect.log      the collector's own log
# Progress-bar redraws are stripped. Safe to rerun.
#   nohup caffeinate -i bash scripts/collect_campaign_logs.sh > /dev/null 2>&1 &
set -uo pipefail

REPO=/Users/tanmoy/research/Credal_Sets/neurips-credal
PY=/Users/tanmoy/anaconda3/envs/torch-multimodal/bin/python
MODAL=/Users/tanmoy/anaconda3/envs/torch-multimodal/bin/modal
OUT=$REPO/outputs/icml_2026_reeval
LOGS=$OUT/logs
cd "$REPO" || exit 1

# Modal apps of the campaign launch (2026-09-25 16:41-16:42 CEST), in launch order.
APPS=(
  "A_cebab_fixed_distilbert:ap-amsmm7mUSL63mQDZeiO3rX"
  "A_cebab_fixed_roberta:ap-HjADcnORUh0Qw3Borlmsu5"
  "B_cebab_fixed_noale:ap-UNV3aM1sqdUbcnY3eC2Gv5"
  "C_cebab_fixed_decorr5:ap-QMqbvTm37s6ygho0c4PTv7"
  "D_seed42_hatexplain_goemotions_maqa:ap-W01lFAAoUmoeonw00Kezbq"
  "E_ambigqa:ap-aO0NU81eQX3Yu6nG5GZE4t"
)

# 1. Wait for the results collector to finish.
while pgrep -f "bash scripts/collect_campaign.sh" > /dev/null; do sleep 120; done

mkdir -p "$LOGS"/{modal,eval,training_history}

# 2. Modal app logs are captured live by scripts/stream_modal_log.py (one per
#    app, started while the apps ran, because Modal keeps only the tail of a
#    finished app's logs). Wait for those streams to end; if one is missing,
#    fall back to whatever tail Modal still returns.
for i in $(seq 1 60); do pgrep -f stream_modal_log.py > /dev/null || break; sleep 60; done
for entry in "${APPS[@]}"; do
  name=${entry%%:*}; app=${entry#*:}
  [ -s "$LOGS/modal/$name.log" ] && continue
  "$PY" scripts/stream_modal_log.py "$app" "$LOGS/modal/$name.log" &
  pid=$!; sleep 120; kill "$pid" 2>/dev/null
done

# 3. Local evaluation logs, per-epoch histories, and the collector log.
for f in "$OUT"/.eval_*.log; do
  [ -e "$f" ] || continue
  r=$(basename "$f" .log); r=${r#.eval_}
  "$PY" -c "import sys; t=open(sys.argv[1], errors='replace').read(); open(sys.argv[2], 'w').write('\n'.join(l.split('\r')[-1] for l in t.splitlines()) + '\n')" "$f" "$LOGS/eval/$r.log"
done
for h in "$REPO"/checkpoints_from_modal/icml_2026/*/training_history.json; do
  r=$(basename "$(dirname "$h")")
  case "$r" in *_fixed*|*seed42_100ep|ambigqa_*) cp "$h" "$LOGS/training_history/$r.json" ;; esac
done
for r in "$REPO"/checkpoints_from_modal/icml_2026/{maqa_distilbert_seed42_100ep,ambigqa_distilbert_seed*_100ep}; do
  [ -s "$r/final_results.json" ] && cp "$r/final_results.json" "$LOGS/training_history/$(basename "$r")_final_results.json"
done
cp "$OUT/campaign_collect.log" "$LOGS/campaign_collect.log" 2>/dev/null || true
du -sh "$LOGS" > "$LOGS/SIZE.txt"

# 4. Commit and push the logs.
git add "$LOGS" "$OUT/campaign_collect.log" scripts/collect_campaign_logs.sh scripts/stream_modal_log.py
if ! git diff --cached --quiet; then
  git commit -q -m "Add retraining campaign logs (Modal training, evaluation, histories)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  git push -q origin neurips-credal || echo "push failed; run: git push origin neurips-credal" >> "$LOGS/campaign_collect.log"
fi
osascript -e 'display notification "Campaign logs pushed to GitHub" with title "Credal CBM campaign"' 2>/dev/null || true

#!/usr/bin/env bash
# Collect the retraining campaign unattended: wait for the Modal runs, download
# them, re-evaluate with the 532fd05 pipeline, compute every metric and
# baseline, write CAMPAIGN_RESULTS.md, and push the results to GitHub.
#
# Safe to rerun: finished downloads and evaluations are skipped.
#   nohup caffeinate -i bash scripts/collect_campaign.sh > outputs/icml_2026_reeval/campaign_collect.log 2>&1 &
set -uo pipefail

REPO=/Users/tanmoy/research/Credal_Sets/neurips-credal
VCBM=/Users/tanmoy/research/Credal_Sets/Variational_CBM
WT=/Users/tanmoy/research/Credal_Sets/.wt532_eval
PY=/Users/tanmoy/anaconda3/envs/torch-multimodal/bin/python
MODAL=/Users/tanmoy/anaconda3/envs/torch-multimodal/bin/modal
VOL=icml-2026-credal-results
CKPT=$REPO/checkpoints_from_modal/icml_2026
OUT=$REPO/outputs/icml_2026_reeval
MAX_WAIT_H=${MAX_WAIT_H:-10}

RUNS=(
  cebab_3class_distilbert_seed{42,123,2024}_100ep_fixed
  cebab_3class_roberta_base_seed{42,123,2024}_100ep_fixed
  cebab_3class_distilbert_seed{42,123,2024}_100ep_fixed_noale
  cebab_3class_distilbert_seed{42,123,2024}_100ep_fixed_decorr5
  hatexplain_distilbert_seed42_100ep goemotions_distilbert_seed42_100ep maqa_distilbert_seed42_100ep
  ambigqa_distilbert_seed{42,123,2024}_100ep
)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
cd "$REPO" || exit 1

# 1. Wait until every run has written test_metrics.json (or give up after MAX_WAIT_H).
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while :; do
  pending=()
  for r in "${RUNS[@]}"; do
    [ -f "$CKPT/$r/test_metrics.json" ] && continue
    "$MODAL" volume ls "$VOL" "/$r" 2>/dev/null | grep -q test_metrics.json || pending+=("$r")
  done
  [ ${#pending[@]} -eq 0 ] && { log "all ${#RUNS[@]} runs finished on Modal"; break; }
  if [ "$(date +%s)" -ge "$deadline" ]; then log "deadline reached; still missing: ${pending[*]}"; break; fi
  log "waiting on ${#pending[@]} runs: ${pending[*]}"; sleep 300
done

# 2. Download finished runs. best_model.pt is rewritten during training, so a
#    run is fetched only once its final test_metrics.json exists on the volume.
ready=()
for r in "${RUNS[@]}"; do
  if [ ! -s "$CKPT/$r/test_metrics.json" ]; then
    rm -rf "$CKPT/$r" "$OUT/$r"; mkdir -p "$CKPT/$r"
    "$MODAL" volume get "$VOL" "/$r/test_metrics.json" "$CKPT/$r/test_metrics.json" >/dev/null 2>&1 || true
    if [ ! -s "$CKPT/$r/test_metrics.json" ]; then rm -rf "$CKPT/$r"; log "not finished: $r"; continue; fi
    for f in best_model.pt run_metadata.json final_results.json training_history.json; do
      "$MODAL" volume get "$VOL" "/$r/$f" "$CKPT/$r/$f" >/dev/null 2>&1 || true
    done
  fi
  if [ -s "$CKPT/$r/best_model.pt" ]; then ready+=("$r"); else log "no checkpoint: $r"; fi
done
log "downloaded ${#ready[@]} finished runs"

# 3. Evaluation worktree at the training commit (+ MAQA support files).
if [ ! -d "$WT" ]; then
  git -C "$VCBM" worktree add --detach "$WT" 532fd05 || exit 1
fi
cp "$VCBM"/{v7b_complete_integration.py,load_maqa_real.py,maqa_credal_loss_v7b.py,maqa_credal_loss_v7b_fixed.py,maqa_fixed_config.py} "$WT"/
[ -e "$WT/data" ] || ln -s "$VCBM/data" "$WT/data"

# 4. Re-evaluate each run once. The corrected-label default runs also get 10
#    MC-dropout passes for the baseline table.
for r in "${ready[@]}"; do
  [ -f "$OUT/$r/test_arrays.npz" ] && continue
  mc=0; [[ "$r" == *_100ep_fixed ]] && mc=10
  log "evaluating $r (MC passes: $mc)"
  REEVAL_MC=$mc "$PY" scripts/reeval_icml_modal.py --code-dir "$WT" --ckpt-root "$CKPT" --out "$OUT" --runs "$r" \
    > "$OUT/.eval_$r.log" 2>&1 || log "EVAL FAILED: $r (see $OUT/.eval_$r.log)"
done

# 5. Metrics, masked rescore, diagnostics, sanity sweep, baselines, report.
"$PY" scripts/summarize_icml_reeval.py --root "$OUT" > /dev/null 2>&1 || log "summarize failed"
"$PY" scripts/rescore_cebab_masked.py --root "$OUT" > /dev/null 2>&1 || log "masked rescore failed"
"$PY" scripts/extra_diagnostics.py --root "$OUT" > /dev/null 2>&1 || log "diagnostics failed"
"$PY" scripts/sanity_sweep_reeval.py --root "$OUT" > /dev/null 2>&1 || log "sanity sweep failed"
for g in "cebab_3class_distilbert_seed{}_100ep_fixed" "cebab_3class_roberta_base_seed{}_100ep_fixed" \
         "cebab_3class_distilbert_seed{}_100ep_fixed_noale" "cebab_3class_distilbert_seed{}_100ep_fixed_decorr5" \
         "ambigqa_distilbert_seed{}_100ep"; do
  "$PY" scripts/ensemble_baselines.py --root "$OUT" --group "$g" > /dev/null 2>&1 || log "baselines skipped: $g"
done
"$PY" scripts/campaign_report.py --root "$OUT" --out CAMPAIGN_RESULTS.md > /dev/null || log "report failed"
log "report written: $REPO/CAMPAIGN_RESULTS.md"

# 6. Commit and push results only (the paper draft is git-ignored).
[ -n "${NO_PUSH:-}" ] && { log "NO_PUSH set; skipping commit"; exit 0; }
git add CAMPAIGN_RESULTS.md scripts outputs/icml_2026_reeval modal_icml_2026_multiseed.py
if ! git diff --cached --quiet; then
  git commit -q -m "Collect retraining campaign: corrected CEBaB labels, seed 42, ablations, AmbigQA*, baselines

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && log "committed $(git rev-parse --short HEAD)"
  git push -q origin neurips-credal && log "pushed to origin/neurips-credal" || log "PUSH FAILED; run: git push origin neurips-credal"
fi
osascript -e 'display notification "CAMPAIGN_RESULTS.md is ready" with title "Credal CBM campaign"' 2>/dev/null || true
log "done"

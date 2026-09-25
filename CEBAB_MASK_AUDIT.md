# CEBaB label audit, masked rescore, and invalid-value sweep

Date: 2026-09-25. Applies to the ICML_2026 checkpoints (commit `532fd05`, seeds 123/2024)
behind `tab:main-results` in `paper/iclr2027_revised.tex`.

Reproduce:

```bash
python scripts/audit_cebab_majority.py --code-dir <532fd05 worktree> \
    --saved-test outputs/icml_2026_reeval/cebab_3class_seed123_100ep/test_arrays.npz
python scripts/rescore_cebab_masked.py --root outputs/icml_2026_reeval   # -> cebab_masked_rescore.json
python scripts/sanity_sweep_reeval.py  --root outputs/icml_2026_reeval   # -> sanity_sweep.json
```

## 1. What was wrong

CEBaB aspect majorities take five values, not three: `Negative`, `Positive`, `unknown`,
`no majority` (train only) and `''` (aspect not annotated, 0 annotators). The 532fd05
loader (`process_cebab_raw`) only maps the first three:

| Raw aspect value | Loader concept label | Loader entropy target H | Correct treatment |
|---|---|---|---|
| `''` (0 annotators) | **0 = Negative** | **0.0** (fallback distribution `[0,1,0]`) | no concept target, H undefined |
| `no majority` | **0 = Negative** | real entropy | no concept target, H valid |

Test split (1,689 reviews, 6,756 aspect slots):

| Aspect | Annotated | Unannotated (`''`) | Binary target (Neg/Pos) | `unknown` majority |
|---|---|---|---|---|
| food | 1,589 | 100 | 1,275 | 314 |
| service | 1,297 | 392 | 875 | 422 |
| ambiance | 1,101 | 588 | 546 | 555 |
| noise | 947 | 742 | 362 | 585 |
| **total** | 4,934 | **1,822** | **3,058** | 1,876 |

- 1,822 test aspect slots were scored as Negative with no annotation. The models predict Negative on 90–93% of them, which inflates concept accuracy.
- Those same slots carry H = 0 ("perfect agreement") in the ρ(U_ale, H) target. 38% of all per-aspect H = 0 entries are unannotated.
- Every review has at least one annotated aspect. Unannotated slots occur in original reviews too (227 of 272 test originals), not only in counterfactual edits.
- Training was affected as well. The concept loss uses `concept_labels != 1`, so phantom Negatives were trained as Negative. `HybridCredalCBM` (`VCBM.py`) fits the ambiguity head to every entry, including the H = 0 fallbacks.
- Where annotations exist, H recomputed from the raw labels matches the loader and the saved arrays exactly (max |diff| = 0). The bug is confined to the unannotated and no-majority slots.

## 2. Fix

**Evaluation, applied now without retraining.** A per-aspect validity mask is rebuilt from the raw annotations. Before rescoring, the script checks that saved `concept_labels` equal the loader's mapping and that every unannotated slot has H = 0.
- H is averaged over annotated aspects only.
- U_ale is averaged over the same aspects for the matched ρ.
- Concept accuracy uses Negative/Positive targets only.

**Code, for future runs** (`load_cebab_direct.py`, `VCBM.py`, `models/slvm_base.py`, `training/metrics.py`):
- **Concept labels:** unannotated and `no majority` aspects get label 1, so every `concept_labels != 1` loss and metric skips them. New `concept_valid` and `entropy_valid` fields are added.
- **Entropy target:** H is NaN for unannotated aspects (it was 0.0). `entropy_weights` equals `entropy_valid`, and both aleatoric losses mask NaN targets.
- **Reviews without a majority rating** are dropped. Previously they got a silent 3-star label in the five-class path; the three-class path already dropped them.
- **Undefined Spearman ρ:** `compute_uncertainty_correlations` now returns NaN for a constant or too-short series (it used to return 0.0), and averages AU and H over the same valid entries.
- **Side effect:** the U-supervision ablation (`concept_labels == 1`) will now also count unannotated aspects as "unknown".

**Not fixed by this:** the current checkpoints were trained on the contaminated labels. The rescored numbers below correct the evaluation only. Removing the contamination from training requires retraining the four CEBaB runs.

## 3. Sanity sweep over all 10 Table 1 runs

| Run | Finding | Status in Table 1 |
|---|---|---|
| CEBaB ×4 | no NaN/inf; probabilities sum to 1; all 3 classes predicted; EU and AU have ≥1,675 distinct values | valid after masking |
| CEBaB ×4 | per-aspect H takes 4 values only {0, .455, .613, .865} (5 annotators, 3 labels); 71% of entries are 0 | coarse but genuine; heavy ties |
| HateXplain s123 | AU: 2/2 columns constant (sd < 1e-4), 202 distinct scores out of 1,924 | ρ(EU,AU) and ρ(AU,H) are **not valid** |
| HateXplain s2024 | AU: 1/2 columns constant, 763 distinct scores | ρ(EU,AU) and ρ(AU,H) are **not valid** |
| HateXplain | H is binary: 0 (974) or 0.579 (950); no three-way splits in test | ρ with a binary target; ties |
| GoEmotions ×2 | AU: 28/28 columns constant, only 2–4 distinct scores out of 5,427 | ρ(EU,AU) is **not valid** |
| MAQA* ×2 | EU spread very narrow (sd 0.0045) but 463 distinct values; AU without H input is fine | valid |
| MAQA* ×2 | ambiguity head with H input saturates at its 1.5 cap on 4 and 1 examples | not reported (leaky variant) |
| code | `training/metrics.py` returned ρ = 0.0 for a constant series | fixed: now NaN |

## 4. Corrected numbers, by paper location

"Matched" means U_ale and H are both averaged over each review's annotated aspects. "Pooled" means per-aspect pairs over all annotated slots. All values are mean ± sample SD over seeds 123 and 2024.

### Table 1 (`tab:main-results`, l. 208–212)

| Row / column | Old | New |
|---|---|---|
| CEBaB DistilBERT ρ(U_ale,H) | 0.13 ± 0.01 | **0.23 ± 0.04** (pooled 0.22 ± 0.02) |
| CEBaB RoBERTa ρ(U_ale,H) | 0.18 ± 0.03 | **0.30 ± 0.03** (pooled 0.25 ± 0.02) |
| HateXplain ρ(U_epi,U_ale) | 0.07 ± 0.13 | **—** (AU head constant) |
| HateXplain ρ(U_ale,H) | −0.01 ± 0.03† | **—** (AU head constant) |
| GoEmotions ρ(U_epi,U_ale) | 0.01 ± 0.01 | **—** (AU head constant) |
| All other cells | | unchanged (task accuracy, ρ(EU,err), both AUROCs, MAQA*) |

### Text, l. 255

The sentence "ρ(U_ale,H) is 0.13 with DistilBERT and 0.18 with RoBERTa; the 95% intervals for every seed exclude zero but also exclude 0.25" **no longer holds**. The new per-seed bootstrap 95% CIs (matched):
- DistilBERT: [.211, .303] and [.158, .249]
- RoBERTa: [.233, .325] and [.279, .365]

Three of the four intervals contain or exceed 0.25.

"Most of this association comes from the food concept" also **no longer holds**. Masked per-aspect ρ ranges over the four runs:

| Aspect | Old ρ | Masked ρ |
|---|---|---|
| food | .23–.31 | .28–.36 |
| service | .05–.08 | **.20–.22** |
| ambiance | .04–.09 | .04–.13 |
| noise | .03–.05 | .06–.13 |

### Stratification (`tab:ambiguity-strat`, l. 266–285; appendix l. 729)

Same rule as the paper: low = H = 0, and the rest are split at the 2/3 quantile of H.
- **Cut:** moves from 0.228 to 0.288.
- **Low stratum:** keeps the same 418 reviews. They are unanimous on their annotated aspects, but only 42 of them have all four aspects annotated.

| Encoder | Score | Low (n=418) | Med (695 → **686**) | High (576 → **585**, H ≥ 0.29) | Δ(H−L) old → new |
|---|---|---|---|---|---|
| DistilBERT | EU | 0.68 | 0.64 | **0.60** | −0.08 → −0.08 |
| DistilBERT | MaxProb | 0.79 | **0.78** | **0.72** | −0.05 → **−0.07** |
| RoBERTa | EU | 0.69 | **0.68** | **0.60** | −0.06 → **−0.10** |
| RoBERTa | MaxProb | 0.83 | **0.81** | 0.77 | −0.06 → −0.06 |

- DistilBERT errors per seed: 88/85, **163/163**, **158/149**.
- The qualitative claim, that both scores lose discrimination as ambiguity rises, still holds and is stronger for RoBERTa EU.

### Quadrants (`tab:quadrant`, l. 866)

Only the mean-H row changes; accuracies and AURCs don't depend on H.

| | Trust | Data | Review | Abstain |
|---|---|---|---|---|
| Mean H, DistilBERT (old) | 0.13 | 0.15 | 0.15 | 0.18 |
| Mean H, DistilBERT (new) | **0.18** | **0.19** | **0.22** | **0.25** |
| Mean H, RoBERTa (new) | 0.16 | 0.18 | 0.24 | 0.26 |

### Per-concept table (`tab:cebab-per-concept`, l. 651–661), DistilBERT, seed 123 / 2024

| Concept | Known labels old → new | Accuracy % old → new | ρ(σ_ale,j, H_j) old → new |
|---|---|---|---|
| Food | 1,375 → **1,275** | 76.5/77.5 → **77.0/78.1** | .261/.227 → **.320/.279** |
| Service | 1,267 → **875** | 80.9/79.0 → **76.1/73.6** | .061/.049 → **.213/.202** |
| Ambiance | 1,134 → **546** | 79.3/79.1 → **70.3/71.6** | .055/.044 → **.120/.105** |
| Noise | 1,104 → **362** | 87.3/87.9 → **71.8/70.2** | .045/.025 → **.072/.062** |
| Mean | 4,880 → **3,058** | 80.7/80.6 → **73.8/73.4** | — |

### Encoder ablation (`tab:encoder-ablation`, l. 750–752)

| Encoder | Concept acc old → new | ρ(U_ale,H) old → new |
|---|---|---|
| DistilBERT | 80.7 ± 0.1 → **73.6 ± 0.3** | 0.13 ± 0.01 → **0.23 ± 0.04** |
| RoBERTa | 82.1 ± 0.4 → **74.5 ± 1.1** | 0.18 ± 0.03 → **0.30 ± 0.03** |
| DeBERTa-v3 | 79.4 ± 0.1 → **cannot rescore** | 0.05 ± 0.06 → **cannot rescore** |

DeBERTa has no saved test arrays; it must be re-evaluated from its checkpoint.

### Per-seed table (`tab:per-seed`, l. 707–712)

| Row | ρ(AU,H) old | ρ(AU,H) new, matched, with bootstrap 95% CI |
|---|---|---|
| CEBaB DistilBERT s123 | .140 | .258 [.211, .303] |
| CEBaB DistilBERT s2024 | .125 | .205 [.158, .249] |
| CEBaB RoBERTa s123 | .165 | .278 [.233, .325] |
| CEBaB RoBERTa s2024 | .204 | .323 [.279, .365] |
| HateXplain rows | ρ(EU,AU), ρ(AU,H) | **—** |

### Method text (appendix l. 645) that disagrees with the code

- **H normalization.** The paper says H is divided by log(number of distinct labels observed), so an even two-way split gives H = 1. The code divides by log 3 in every case, which puts the maximum at 0.865. Fix the text, or change the code and rescore.
- **Averaging.** "U_ale and H are both averaged over the four concepts" becomes "over the annotated concepts".
- **Label counts.** "4,880 known and 1,876 unknown concept labels" becomes 3,058 binary targets, 1,876 unknown-majority labels, and 1,822 unannotated slots (excluded).
- **Missing annotations.** Add a sentence saying that aspects without annotations or without a majority are excluded from concept targets and from H.

### Values that cannot be rescored from saved outputs

These come from runs without saved per-example arrays that used the same loader:
- **Earlier five-class campaign:** `tab:matrix-runs` and the sweep tables at l. 765 onwards. Concept accuracy 84.9–95.0 and ρ(AU,H) 0.34–0.37 were computed with phantom Negatives. The five-class path also gave reviews without a majority rating a 3-star label.
- **Other encoders:** DeBERTa and ModernBERT in `tab:encoder-ablation`, `RESULTS.md` and `tmp_modal_pull/concept_metrics/`. Their concept accuracies and ρ(AU,H) include unannotated slots.

## 5. What does not change

- Task accuracy, ρ(EU,AU) for CEBaB and MAQA*, ρ(EU,err), AUROC(EU) and AUROC(MaxProb) on every dataset.
- The reduction check in `reduction_check.txt`.
- All MAQA* values.
- The HateXplain and GoEmotions EU columns.

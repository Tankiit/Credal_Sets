"""Pre-results registry for the twin/intervention study (created 2026-10-09).

Thresholds are deliberately stated before the expanded 12-run evaluation is
aggregated.  B2 remains exploratory because it has only twelve trained runs.
"""

EXPS = {
    "A1": dict(
        claim=("For residual-logit CBMs at mix=1, mean intervened label agreement "
               "is below 0.95 in every trained-seed cell on CEBaB and GoEmotions."),
        falsified_if=("Any CEBaB or GoEmotions trained-seed mean is >= 0.95, or its "
                       "20-twin 95% bootstrap confidence interval includes 0.99."),
        table="tab:main",
    ),
    "A2": dict(
        claim=("The measured intervened twin-minus-original logit gap equals the "
               "Proposition 2 closed form to float32 numerical precision."),
        falsified_if="Maximum absolute discrepancy exceeds 1e-4 on any evaluated cell.",
        table="tab:prop2",
    ),
    "A3": dict(
        claim=("At mix=1, twin full-representation CKA and kNN are no higher than "
               "the 75th percentile of the corresponding cross-seed similarities."),
        falsified_if=("Both twin CKA and twin kNN exceed the cross-seed 75th percentile "
                       "for every trained seed of a configuration."),
        fig="fig:realism",
    ),
    "A4": dict(
        claim=("At least five nontrivial fixed CEBaB or GoEmotions examples have a "
               "different intervened predicted label across the 20 twins at mix=1."),
        falsified_if="Fewer than five such examples are found after inspecting all 20 twins.",
        table="tab:examples",
    ),
    "B1": dict(
        claim=("On validated CEBaB original-to-aspect-edit pairs, function-preserving "
               "twins can produce different targeted concept-effect estimates and label changes."),
        falsified_if=("Across the 1,002 Positive/Negative test edits, every twin has mean "
                       "absolute expected-rating effect gap below 0.01 and zero label-change disagreements."),
        table="tab:cebab-effects",
        status="primary NLP analysis; loader validated 2026-10-09",
    ),
    "B2": dict(
        claim=("Across the 12 residual-logit runs, agreement drop is monotonically "
               "associated with residual-head reliance ||H_r||/||H||."),
        falsified_if="Spearman rho <= 0; report regardless as exploratory.",
        status="exploratory; n=12",
    ),
    "B3": dict(
        claim="Intervened agreement decreases as the number of edited concepts increases.",
        falsified_if="The pooled monotone trend is non-negative.",
        fig="fig:budget",
    ),
}

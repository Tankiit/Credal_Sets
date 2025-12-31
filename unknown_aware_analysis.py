"""
Unknown-Aware Intervention Analysis

====================================

The key insight: CEBaB has 52% average unknown rate!

- food: 25% unknown
- service: 45% unknown
- ambiance: 63% unknown
- noise: 76% unknown

Hypothesis: GT interventions hurt because we're "fixing" unknown
concepts to arbitrary values, destroying information the model learned.

This script separates analysis by known vs unknown status.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy import stats

def analyze_interventions_by_known_status(
    concept_probs: np.ndarray,      # [N, K] predicted concept probabilities
    gt_concepts: np.ndarray,        # [N, K] ground truth (0, 1, 2 for ternary)
    is_unknown: np.ndarray,         # [N, K] boolean mask for unknown concepts
    labels: np.ndarray,             # [N] task labels
    classifier_weights: np.ndarray, # [C, K] classifier weights
    concept_names: Optional[list] = None
) -> Dict:
    """
    Run intervention analysis separately for:
    1. ALL samples (baseline)
    2. Samples where concept is KNOWN
    3. Samples where concept is UNKNOWN
    """
    
    print("\n" + "="*70)
    print("UNKNOWN-AWARE INTERVENTION ANALYSIS")
    print("="*70)
    
    n_samples, n_concepts = concept_probs.shape
    n_classes = classifier_weights.shape[0]
    
    if concept_names is None:
        concept_names = [f"Concept_{i}" for i in range(n_concepts)]
    
    # Convert GT to probability scale (0, 0.5, 1 for ternary)
    gt_probs = gt_concepts / 2.0 if gt_concepts.max() > 1 else gt_concepts
    
    results = {}
    
    # Baseline accuracy
    baseline_logits = concept_probs @ classifier_weights.T
    baseline_preds = baseline_logits.argmax(axis=1)
    baseline_acc = (baseline_preds == labels).mean()
    
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print(f"\nUnknown rates per concept:")
    for c in range(n_concepts):
        unknown_rate = is_unknown[:, c].mean()
        print(f"  {concept_names[c]}: {unknown_rate:.1%} unknown")
    
    avg_unknown = is_unknown.mean()
    print(f"\nAverage unknown rate: {avg_unknown:.1%}")
    
    results['baseline_acc'] = float(baseline_acc)
    results['avg_unknown_rate'] = float(avg_unknown)
    
    # === ANALYSIS 1: Oracle intervention (ALL concepts to GT) ===
    print("\n" + "-"*50)
    print("ORACLE INTERVENTION (Fix ALL concepts)")
    print("-"*50)
    
    oracle_logits = gt_probs @ classifier_weights.T
    oracle_preds = oracle_logits.argmax(axis=1)
    oracle_acc = (oracle_preds == labels).mean()
    oracle_delta = oracle_acc - baseline_acc
    
    print(f"  All samples:  {oracle_acc:.4f} ({oracle_delta:+.4f})")
    results['oracle_all'] = {'acc': float(oracle_acc), 'delta': float(oracle_delta)}
    
    # Separate by samples with ALL concepts known vs ANY unknown
    all_known_mask = ~is_unknown.any(axis=1)
    any_unknown_mask = is_unknown.any(axis=1)
    
    print(f"\n  Samples with ALL concepts known: {all_known_mask.sum()} ({all_known_mask.mean():.1%})")
    print(f"  Samples with ANY concept unknown: {any_unknown_mask.sum()} ({any_unknown_mask.mean():.1%})")
    
    if all_known_mask.sum() > 10:
        # Baseline for known-only samples
        known_baseline_acc = (baseline_preds[all_known_mask] == labels[all_known_mask]).mean()
        known_oracle_acc = (oracle_preds[all_known_mask] == labels[all_known_mask]).mean()
        known_delta = known_oracle_acc - known_baseline_acc
        
        print(f"\n  Known-only samples:")
        print(f"    Baseline: {known_baseline_acc:.4f}")
        print(f"    Oracle:   {known_oracle_acc:.4f} ({known_delta:+.4f})")
        
        if known_delta > 0:
            print(f"    GT HELPS on known samples!")
        else:
            print(f"    GT still hurts even on known samples")
        
        results['oracle_known_only'] = {
            'n_samples': int(all_known_mask.sum()),
            'baseline_acc': float(known_baseline_acc),
            'oracle_acc': float(known_oracle_acc),
            'delta': float(known_delta)
        }
    
    if any_unknown_mask.sum() > 10:
        # Baseline for unknown samples
        unknown_baseline_acc = (baseline_preds[any_unknown_mask] == labels[any_unknown_mask]).mean()
        unknown_oracle_acc = (oracle_preds[any_unknown_mask] == labels[any_unknown_mask]).mean()
        unknown_delta = unknown_oracle_acc - unknown_baseline_acc
        
        print(f"\n  Unknown-containing samples:")
        print(f"    Baseline: {unknown_baseline_acc:.4f}")
        print(f"    Oracle:   {unknown_oracle_acc:.4f} ({unknown_delta:+.4f})")
        
        if unknown_delta < 0:
            print(f"    GT hurts more on unknown samples (as expected)")
        
        results['oracle_unknown'] = {
            'n_samples': int(any_unknown_mask.sum()),
            'baseline_acc': float(unknown_baseline_acc),
            'oracle_acc': float(unknown_oracle_acc),
            'delta': float(unknown_delta)
        }
    
    # === ANALYSIS 2: Per-concept intervention ===
    print("\n" + "-"*50)
    print("PER-CONCEPT INTERVENTION")
    print("-"*50)
    
    print(f"\n{'Concept':<12} {'Unknown%':<10} {'All Δ':<10} {'Known Δ':<10} {'Unknown Δ':<10}")
    print("-"*52)
    
    results['per_concept'] = {}
    
    for c in range(n_concepts):
        # Intervene on this concept only
        intervened = concept_probs.copy()
        intervened[:, c] = gt_probs[:, c]
        
        int_logits = intervened @ classifier_weights.T
        int_preds = int_logits.argmax(axis=1)
        
        # All samples
        all_acc = (int_preds == labels).mean()
        all_delta = all_acc - baseline_acc
        
        # Known samples for this concept
        known_c = ~is_unknown[:, c]
        unknown_c = is_unknown[:, c]
        
        known_delta_str = "N/A"
        unknown_delta_str = "N/A"
        
        if known_c.sum() > 10:
            known_base = (baseline_preds[known_c] == labels[known_c]).mean()
            known_int = (int_preds[known_c] == labels[known_c]).mean()
            known_delta = known_int - known_base
            known_delta_str = f"{known_delta:+.4f}"
        
        if unknown_c.sum() > 10:
            unknown_base = (baseline_preds[unknown_c] == labels[unknown_c]).mean()
            unknown_int = (int_preds[unknown_c] == labels[unknown_c]).mean()
            unknown_delta = unknown_int - unknown_base
            unknown_delta_str = f"{unknown_delta:+.4f}"
        
        unknown_rate = is_unknown[:, c].mean()
        print(f"{concept_names[c]:<12} {unknown_rate:>8.1%} {all_delta:>+9.4f} "
              f"{known_delta_str:>9} {unknown_delta_str:>9}")
        
        results['per_concept'][concept_names[c]] = {
            'unknown_rate': float(unknown_rate),
            'all_delta': float(all_delta),
            'known_delta': float(known_delta) if known_c.sum() > 10 else None,
            'unknown_delta': float(unknown_delta) if unknown_c.sum() > 10 else None
        }
    
    # === ANALYSIS 3: What does model predict for unknowns? ===
    print("\n" + "-"*50)
    print("MODEL PREDICTIONS FOR UNKNOWN CONCEPTS")
    print("-"*50)
    
    print(f"\n{'Concept':<12} {'Known μ':<10} {'Unknown μ':<12} {'Difference':<12}")
    print("-"*46)
    
    for c in range(n_concepts):
        known_c = ~is_unknown[:, c]
        unknown_c = is_unknown[:, c]
        
        if known_c.sum() > 10 and unknown_c.sum() > 10:
            known_mean = concept_probs[known_c, c].mean()
            unknown_mean = concept_probs[unknown_c, c].mean()
            diff = unknown_mean - known_mean
            
            print(f"{concept_names[c]:<12} {known_mean:>8.4f} {unknown_mean:>10.4f} {diff:>+10.4f}")
            
            results['per_concept'][concept_names[c]]['known_pred_mean'] = float(known_mean)
            results['per_concept'][concept_names[c]]['unknown_pred_mean'] = float(unknown_mean)
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    oracle_all_delta = results['oracle_all']['delta']
    
    if 'oracle_known_only' in results:
        oracle_known_delta = results['oracle_known_only']['delta']
        
        if oracle_known_delta > 0 and oracle_all_delta < 0:
            print("""
HYPOTHESIS CONFIRMED: Unknown labels cause intervention paradox!
  - Oracle on ALL samples: {:.1%} (hurts)
  - Oracle on KNOWN-ONLY:  {:.1%} (helps!)
  
The model has learned to handle unknowns differently than GT.
When we "fix" unknown concepts to GT values, we destroy this
learned representation, causing accuracy to drop.

PAPER FRAMING:
"We find that concept interventions decrease accuracy when
unknown concept labels are common ({}% unknown in CEBaB).
When restricting to samples with known concepts, oracle
intervention improves accuracy by {:.1%}, validating our
uncertainty decomposition on labeled data."
""".format(oracle_all_delta, oracle_known_delta, 
           avg_unknown*100, oracle_known_delta*100))
            
        elif oracle_known_delta < 0:
            print("""
Unknown labels do NOT explain the paradox.
Even on samples with ALL concepts known, oracle intervention hurts.
This suggests deeper concept-label misalignment issues.
""")
    
    return results

def stratified_uncertainty_analysis(
    disagreement: np.ndarray,   # [N, K] epistemic
    ambiguity: np.ndarray,      # [N, K] aleatoric  
    is_unknown: np.ndarray,     # [N, K] unknown mask
    labels: np.ndarray,         # [N]
    preds: np.ndarray,          # [N]
    concept_names: Optional[list] = None
) -> Dict:
    """
    Analyze uncertainty patterns separately for known vs unknown concepts.
    """
    
    print("\n" + "="*70)
    print("STRATIFIED UNCERTAINTY ANALYSIS")
    print("="*70)
    
    n_samples, n_concepts = disagreement.shape
    errors = (preds != labels).astype(float)
    
    if concept_names is None:
        concept_names = [f"Concept_{i}" for i in range(n_concepts)]
    
    results = {}
    
    print(f"\n{'Concept':<12} {'Known ρ(epi,err)':<18} {'Unknown ρ(epi,err)':<18}")
    print("-"*48)
    
    for c in range(n_concepts):
        known_c = ~is_unknown[:, c]
        unknown_c = is_unknown[:, c]
        
        known_corr_str = "N/A"
        unknown_corr_str = "N/A"
        
        if known_c.sum() > 30:
            corr, p = stats.spearmanr(disagreement[known_c, c], errors[known_c])
            known_corr_str = f"{corr:+.3f} (p={p:.2e})"
        
        if unknown_c.sum() > 30:
            corr, p = stats.spearmanr(disagreement[unknown_c, c], errors[unknown_c])
            unknown_corr_str = f"{corr:+.3f} (p={p:.2e})"
        
        print(f"{concept_names[c]:<12} {known_corr_str:<18} {unknown_corr_str:<18}")
    
    # Overall analysis
    print("\n--- Overall (mean across concepts) ---")
    
    mean_epi = disagreement.mean(axis=1)
    mean_ale = ambiguity.mean(axis=1)
    
    # Samples with mostly known vs mostly unknown
    unknown_count = is_unknown.sum(axis=1)
    mostly_known = unknown_count <= 1
    mostly_unknown = unknown_count >= 3
    
    print(f"\nMostly known (≤1 unknown): {mostly_known.sum()} samples")
    if mostly_known.sum() > 30:
        corr_epi, p_epi = stats.spearmanr(mean_epi[mostly_known], errors[mostly_known])
        corr_ale, p_ale = stats.spearmanr(mean_ale[mostly_known], errors[mostly_known])
        print(f"  ρ(epistemic, error): {corr_epi:.3f} (p={p_epi:.2e})")
        print(f"  ρ(aleatoric, error): {corr_ale:.3f} (p={p_ale:.2e})")
    
    print(f"\nMostly unknown (≥3 unknown): {mostly_unknown.sum()} samples")
    if mostly_unknown.sum() > 30:
        corr_epi, p_epi = stats.spearmanr(mean_epi[mostly_unknown], errors[mostly_unknown])
        corr_ale, p_ale = stats.spearmanr(mean_ale[mostly_unknown], errors[mostly_unknown])
        print(f"  ρ(epistemic, error): {corr_epi:.3f} (p={p_epi:.2e})")
        print(f"  ρ(aleatoric, error): {corr_ale:.3f} (p={p_ale:.2e})")
    
    return results

# =============================================================================
# INTEGRATION WITH YOUR ABLATION CODE
# =============================================================================

def add_unknown_analysis_to_ablation(data: Dict, classifier_weights: np.ndarray) -> Dict:
    """
    Add unknown-aware analysis to existing ablation results.
    
    Call this with the 'data' dict from collect_predictions() and
    the classifier weights from model.classifier.W
    """
    
    concept_probs = data['concept_probs']
    gt_concepts = data['concepts']
    is_unknown = data['is_unknown'].astype(bool)
    labels = data['labels']
    preds = data['preds']
    disagreement = data['disagreement']
    ambiguity = data['ambiguity']
    
    # Get concept names if available (CEBaB has 4: food, service, ambiance, noise)
    concept_names = ['food', 'service', 'ambiance', 'noise']  # CEBaB default
    
    results = {}
    
    # Unknown-aware intervention analysis
    results['intervention'] = analyze_interventions_by_known_status(
        concept_probs, gt_concepts, is_unknown, labels,
        classifier_weights, concept_names
    )
    
    # Stratified uncertainty analysis
    results['uncertainty'] = stratified_uncertainty_analysis(
        disagreement, ambiguity, is_unknown, labels, preds, concept_names
    )
    
    return results

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("To use this analysis, call add_unknown_analysis_to_ablation()")
    print("with your model outputs.")
    print("="*70)
    
    print("""
Example usage:
    from unknown_aware_analysis import add_unknown_analysis_to_ablation
    
    # After running collect_predictions()
    data = collect_predictions(model, encoder, test_loader, device, model_type)
    
    # Get classifier weights
    classifier_weights = model.classifier.W.detach().cpu().numpy()
    
    # Run unknown-aware analysis
    unknown_results = add_unknown_analysis_to_ablation(data, classifier_weights)
""")


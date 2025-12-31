#!/usr/bin/env python3
"""
Combine all LLM ablation analysis outputs into a single comprehensive file.

Collects all ablation_results.json files from LLM models and combines them
into a structured JSON file with organized metrics.
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict

def load_ablation_results(ablation_dir):
    """Load ablation results from a directory."""
    results_file = os.path.join(ablation_dir, "ablation_results.json")
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {results_file}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None

def find_all_llm_ablations(ablation_base_dir):
    """Find all LLM model ablation directories."""
    llm_models = []
    
    for model_dir in Path(ablation_base_dir).iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Check if it's an LLM model (contains LLM model names)
        is_llm = any(keyword in model_name.lower() for keyword in 
                    ['mistral', 'llama', 'phi-3', 'phi3', 'gpt', 'llm'])
        
        if not is_llm:
            continue
        
        # Find dataset subdirectories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            results = load_ablation_results(str(dataset_dir))
            
            if results:
                llm_models.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'path': str(dataset_dir),
                    'results': results
                })
            else:
                print(f"Warning: No results found for {model_name}/{dataset_name}")
    
    return llm_models

def extract_summary_metrics(results):
    """Extract key summary metrics from full results."""
    summary = {}
    
    # Calibration
    if 'calibration' in results:
        cal = results['calibration']
        summary['calibration'] = {
            'original_ece': cal.get('original_ece'),
            'original_mce': cal.get('original_mce'),
            'calibrated_ece': cal.get('calibrated_ece'),
            'optimal_temperature': cal.get('optimal_temperature'),
            'improvement_percent': cal.get('improvement_percent'),
            'mean_credal_width': cal.get('mean_credal_width')
        }
    
    # Ensemble
    if 'weighted_ensemble' in results:
        ens = results['weighted_ensemble']
        summary['ensemble'] = {
            'uniform_ensemble_acc': ens.get('uniform_ensemble_acc'),
            'weighted_ensemble_acc': ens.get('weighted_ensemble_acc'),
            'best_single_head_acc': ens.get('best_single_head_acc'),
            'head_accuracies': ens.get('head_accuracies'),
            'optimal_weights': ens.get('optimal_weights'),
            'optimal_temperature': ens.get('optimal_temperature')
        }
    
    # Interventions
    if 'interventions' in results:
        interv = results['interventions']
        summary['interventions'] = {
            'baseline_acc': interv.get('baseline_acc'),
            'fix_top1_epistemic_acc': interv.get('fix_top1_epistemic_acc'),
            'fix_top3_epistemic_acc': interv.get('fix_top3_epistemic_acc'),
            'fix_top5_epistemic_acc': interv.get('fix_top5_epistemic_acc'),
            'fix_top1_aleatoric_acc': interv.get('fix_top1_aleatoric_acc'),
            'fix_top3_aleatoric_acc': interv.get('fix_top3_aleatoric_acc'),
            'fix_top5_aleatoric_acc': interv.get('fix_top5_aleatoric_acc'),
            'fix_all_acc': interv.get('fix_all_acc')
        }
        
        # Calculate gains
        if summary['interventions']['baseline_acc']:
            baseline = summary['interventions']['baseline_acc']
            summary['interventions']['epistemic_top1_gain'] = (
                summary['interventions']['fix_top1_epistemic_acc'] - baseline
                if summary['interventions']['fix_top1_epistemic_acc'] else None
            )
            summary['interventions']['aleatoric_top1_gain'] = (
                summary['interventions']['fix_top1_aleatoric_acc'] - baseline
                if summary['interventions']['fix_top1_aleatoric_acc'] else None
            )
    
    # Error distance (if available)
    if 'error_distance' in results:
        ed = results['error_distance']
        summary['error_distance'] = {
            'by_distance': ed.get('by_distance', {}),
            'validation': ed.get('validation', {}),
            'correlations': ed.get('correlations', {})
        }
    
    # Unknown-aware (if available)
    if 'unknown_aware' in results:
        ua = results['unknown_aware']
        summary['unknown_aware'] = {}
        
        if 'intervention' in ua:
            summary['unknown_aware']['intervention'] = ua['intervention']
        if 'uncertainty' in ua:
            summary['unknown_aware']['uncertainty'] = ua['uncertainty']
    
    # Head contributions
    if 'head_contributions' in results:
        hc = results['head_contributions']
        summary['head_contributions'] = {
            'head_accuracies': {k: v for k, v in hc.items() if k.startswith('head_') and k.endswith('_acc')},
            'concept_head_agreement': {k: v for k, v in hc.items() if k.startswith('concept_') and k.endswith('_head_agreement')},
            'quartile_error_rates': {k: v for k, v in hc.items() if k.startswith('Q') and k.endswith('_error_rate')}
        }
    
    # Concept importance
    if 'concept_importance' in results:
        ci = results['concept_importance']
        summary['concept_importance'] = {
            'corr_importance_epistemic': ci.get('corr_importance_epistemic'),
            'corr_importance_aleatoric': ci.get('corr_importance_aleatoric'),
            'importance_ranks': {k: v for k, v in ci.items() if k.startswith('importance_rank_')}
        }
    
    # Selective prediction
    if 'selective_prediction' in results:
        sp = results['selective_prediction']
        summary['selective_prediction'] = {
            'aurc_epistemic': sp.get('aurc_epistemic'),
            'aurc_aleatoric': sp.get('aurc_aleatoric'),
            'aurc_total': sp.get('aurc_total'),
            'aurc_max_prob_baseline': sp.get('aurc_max_prob_(baseline)')
        }
    
    # Head ablation
    if 'head_ablation' in results:
        ha = results['head_ablation']
        summary['head_ablation'] = {
            k: v for k, v in ha.items() if k.endswith('_acc') or k.endswith('_corr')
        }
    
    return summary

def combine_all_results(llm_models):
    """Combine all results into a structured format."""
    combined = {
        'metadata': {
            'total_models': len(set(m['model'] for m in llm_models)),
            'total_datasets': len(set(m['dataset'] for m in llm_models)),
            'total_combinations': len(llm_models)
        },
        'by_model': defaultdict(dict),
        'by_dataset': defaultdict(list),
        'all_results': []
    }
    
    for item in llm_models:
        model = item['model']
        dataset = item['dataset']
        
        # Extract summary metrics
        summary = extract_summary_metrics(item['results'])
        
        # Store in by_model structure
        combined['by_model'][model][dataset] = {
            'path': item['path'],
            'summary': summary,
            'full_results': item['results']  # Include full results
        }
        
        # Store in by_dataset structure
        combined['by_dataset'][dataset].append({
            'model': model,
            'path': item['path'],
            'summary': summary
        })
        
        # Store in all_results
        combined['all_results'].append({
            'model': model,
            'dataset': dataset,
            'path': item['path'],
            'summary': summary
        })
    
    # Convert defaultdicts to regular dicts for JSON serialization
    combined['by_model'] = dict(combined['by_model'])
    combined['by_dataset'] = dict(combined['by_dataset'])
    
    return combined

def create_human_readable_summary(combined, output_file):
    """Create a human-readable text summary."""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LLM MODEL ABLATION ANALYSIS - COMBINED RESULTS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total Models: {combined['metadata']['total_models']}\n")
        f.write(f"Total Datasets: {combined['metadata']['total_datasets']}\n")
        f.write(f"Total Model-Dataset Combinations: {combined['metadata']['total_combinations']}\n\n")
        
        # By dataset
        f.write("="*100 + "\n")
        f.write("RESULTS BY DATASET\n")
        f.write("="*100 + "\n\n")
        
        for dataset in sorted(combined['by_dataset'].keys()):
            f.write(f"\n{'='*100}\n")
            f.write(f"DATASET: {dataset.upper()}\n")
            f.write(f"{'='*100}\n\n")
            
            for item in combined['by_dataset'][dataset]:
                model = item['model']
                summary = item['summary']
                
                f.write(f"\nModel: {model}\n")
                f.write("-"*100 + "\n")
                
                # Calibration
                if 'calibration' in summary:
                    cal = summary['calibration']
                    f.write("Calibration:\n")
                    f.write(f"  Original ECE: {cal.get('original_ece', 'N/A')}\n")
                    f.write(f"  Calibrated ECE: {cal.get('calibrated_ece', 'N/A')}\n")
                    f.write(f"  Optimal Temperature: {cal.get('optimal_temperature', 'N/A')}\n")
                    f.write(f"  Improvement: {cal.get('improvement_percent', 'N/A')}%\n")
                
                # Ensemble
                if 'ensemble' in summary:
                    ens = summary['ensemble']
                    f.write("\nEnsemble:\n")
                    f.write(f"  Uniform: {ens.get('uniform_ensemble_acc', 0)*100:.2f}%\n" if ens.get('uniform_ensemble_acc') else "  Uniform: N/A\n")
                    f.write(f"  Weighted: {ens.get('weighted_ensemble_acc', 0)*100:.2f}%\n" if ens.get('weighted_ensemble_acc') else "  Weighted: N/A\n")
                    f.write(f"  Best Single: {ens.get('best_single_head_acc', 0)*100:.2f}%\n" if ens.get('best_single_head_acc') else "  Best Single: N/A\n")
                
                # Interventions
                if 'interventions' in summary:
                    interv = summary['interventions']
                    f.write("\nInterventions:\n")
                    f.write(f"  Baseline: {interv.get('baseline_acc', 0)*100:.2f}%\n" if interv.get('baseline_acc') else "  Baseline: N/A\n")
                    f.write(f"  Epistemic Top-1 Gain: {interv.get('epistemic_top1_gain', 0)*100:+.2f}%\n" if interv.get('epistemic_top1_gain') is not None else "  Epistemic Top-1 Gain: N/A\n")
                    f.write(f"  Aleatoric Top-1 Gain: {interv.get('aleatoric_top1_gain', 0)*100:+.2f}%\n" if interv.get('aleatoric_top1_gain') is not None else "  Aleatoric Top-1 Gain: N/A\n")
                    f.write(f"  Oracle (All): {interv.get('fix_all_acc', 0)*100:.2f}%\n" if interv.get('fix_all_acc') else "  Oracle (All): N/A\n")
                
                # Unknown-aware
                if 'unknown_aware' in summary and 'intervention' in summary['unknown_aware']:
                    ua = summary['unknown_aware']['intervention']
                    f.write("\nUnknown-Aware Analysis:\n")
                    if 'oracle_all' in ua:
                        f.write(f"  Oracle All Delta: {ua['oracle_all'].get('delta', 0)*100:+.2f}%\n" if ua['oracle_all'].get('delta') is not None else "  Oracle All Delta: N/A\n")
                    if 'oracle_known_only' in ua:
                        f.write(f"  Oracle Known-Only Delta: {ua['oracle_known_only'].get('delta', 0)*100:+.2f}%\n" if ua['oracle_known_only'].get('delta') is not None else "  Oracle Known-Only Delta: N/A\n")
                
                f.write("\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combine LLM ablation outputs")
    parser.add_argument("--ablation_dir", type=str, default="./ablations",
                       help="Base directory containing ablation results")
    parser.add_argument("--output_file", type=str, default="./ablations/combined_llm_results.json",
                       help="Output JSON file for combined results")
    parser.add_argument("--summary_file", type=str, default="./ablations/combined_llm_summary.txt",
                       help="Output text file for human-readable summary")
    
    args = parser.parse_args()
    
    print("Finding LLM model ablation results...")
    llm_models = find_all_llm_ablations(args.ablation_dir)
    
    if not llm_models:
        print(f"No LLM ablation results found in {args.ablation_dir}")
        return
    
    print(f"Found {len(llm_models)} LLM model-dataset combinations")
    
    # Combine all results
    print("Combining results...")
    combined = combine_all_results(llm_models)
    
    # Save combined JSON
    print(f"Saving combined results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(combined, f, indent=2)
    
    # Create human-readable summary
    print(f"Creating human-readable summary at {args.summary_file}...")
    create_human_readable_summary(combined, args.summary_file)
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total Models: {combined['metadata']['total_models']}")
    print(f"Total Datasets: {combined['metadata']['total_datasets']}")
    print(f"Total Combinations: {combined['metadata']['total_combinations']}")
    print(f"\nCombined JSON: {args.output_file}")
    print(f"Human-readable summary: {args.summary_file}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()


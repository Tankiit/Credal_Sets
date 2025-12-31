#!/usr/bin/env python3
"""
Summarize statistics from LLM model ablation analyses.

Collects key metrics from all LLM model ablation results and creates
a comprehensive summary table.
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
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None

def extract_key_metrics(results):
    """Extract key metrics from ablation results."""
    metrics = {}
    
    # Calibration
    if 'calibration' in results:
        cal = results['calibration']
        metrics['original_ece'] = cal.get('original_ece', None)
        metrics['calibrated_ece'] = cal.get('calibrated_ece', None)
        metrics['optimal_temperature'] = cal.get('optimal_temperature', None)
        metrics['improvement_percent'] = cal.get('improvement_percent', None)
    
    # Ensemble
    if 'weighted_ensemble' in results:
        ens = results['weighted_ensemble']
        metrics['uniform_ensemble_acc'] = ens.get('uniform_ensemble_acc', None)
        metrics['weighted_ensemble_acc'] = ens.get('weighted_ensemble_acc', None)
        metrics['best_single_head_acc'] = ens.get('best_single_head_acc', None)
    
    # Interventions
    if 'interventions' in results:
        interv = results['interventions']
        metrics['baseline_acc'] = interv.get('baseline_acc', None)
        metrics['fix_top1_epistemic_acc'] = interv.get('fix_top1_epistemic_acc', None)
        metrics['fix_top1_aleatoric_acc'] = interv.get('fix_top1_aleatoric_acc', None)
        metrics['fix_all_acc'] = interv.get('fix_all_acc', None)
        
        if metrics['baseline_acc'] and metrics['fix_top1_epistemic_acc']:
            metrics['epistemic_gain'] = metrics['fix_top1_epistemic_acc'] - metrics['baseline_acc']
        if metrics['baseline_acc'] and metrics['fix_top1_aleatoric_acc']:
            metrics['aleatoric_gain'] = metrics['fix_top1_aleatoric_acc'] - metrics['baseline_acc']
    
    # Error distance (if available)
    if 'error_distance' in results and 'validation' in results['error_distance']:
        ed = results['error_distance']['validation']
        metrics['epistemic_validates'] = ed.get('epistemic_validates', None)
        metrics['aleatoric_validates'] = ed.get('aleatoric_validates', None)
    
    # Unknown-aware (if available)
    if 'unknown_aware' in results and 'intervention' in results['unknown_aware']:
        ua = results['unknown_aware']['intervention']
        if 'oracle_all' in ua:
            metrics['oracle_all_delta'] = ua['oracle_all'].get('delta', None)
        if 'oracle_known_only' in ua:
            metrics['oracle_known_delta'] = ua['oracle_known_only'].get('delta', None)
    
    return metrics

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
    
    return llm_models

def print_summary_table(llm_models):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("LLM MODEL ABLATION SUMMARY")
    print("="*100)
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for item in llm_models:
        by_dataset[item['dataset']].append(item)
    
    for dataset in sorted(by_dataset.keys()):
        print(f"\n{'='*100}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*100}")
        
        items = by_dataset[dataset]
        
        # Print header
        print(f"\n{'Model':<30} {'Baseline':<10} {'Uniform':<10} {'Weighted':<10} {'Best Single':<12} {'ECE Orig':<10} {'ECE Cal':<10} {'Temp':<8} {'Epist Gain':<12} {'Aleat Gain':<12}")
        print("-"*100)
        
        for item in items:
            metrics = extract_key_metrics(item['results'])
            model_short = item['model'].split('_')[0] if '_' in item['model'] else item['model'][:30]
            
            baseline = f"{metrics.get('baseline_acc', 0)*100:.1f}%" if metrics.get('baseline_acc') else "N/A"
            uniform = f"{metrics.get('uniform_ensemble_acc', 0)*100:.1f}%" if metrics.get('uniform_ensemble_acc') else "N/A"
            weighted = f"{metrics.get('weighted_ensemble_acc', 0)*100:.1f}%" if metrics.get('weighted_ensemble_acc') else "N/A"
            best_single = f"{metrics.get('best_single_head_acc', 0)*100:.1f}%" if metrics.get('best_single_head_acc') else "N/A"
            ece_orig = f"{metrics.get('original_ece', 0):.4f}" if metrics.get('original_ece') is not None else "N/A"
            ece_cal = f"{metrics.get('calibrated_ece', 0):.4f}" if metrics.get('calibrated_ece') is not None else "N/A"
            temp = f"{metrics.get('optimal_temperature', 0):.2f}" if metrics.get('optimal_temperature') is not None else "N/A"
            epi_gain = f"{metrics.get('epistemic_gain', 0)*100:+.1f}%" if metrics.get('epistemic_gain') is not None else "N/A"
            ale_gain = f"{metrics.get('aleatoric_gain', 0)*100:+.1f}%" if metrics.get('aleatoric_gain') is not None else "N/A"
            
            print(f"{model_short:<30} {baseline:<10} {uniform:<10} {weighted:<10} {best_single:<12} {ece_orig:<10} {ece_cal:<10} {temp:<8} {epi_gain:<12} {ale_gain:<12}")
        
        # Print detailed metrics for each model
        print(f"\n{'='*100}")
        print("DETAILED METRICS")
        print(f"{'='*100}")
        
        for item in items:
            metrics = extract_key_metrics(item['results'])
            model_short = item['model'].split('_')[0] if '_' in item['model'] else item['model'][:30]
            
            print(f"\n{model_short} ({item['dataset']}):")
            print(f"  Baseline Accuracy: {metrics.get('baseline_acc', 0)*100:.2f}%" if metrics.get('baseline_acc') else "  Baseline Accuracy: N/A")
            print(f"  Uniform Ensemble: {metrics.get('uniform_ensemble_acc', 0)*100:.2f}%" if metrics.get('uniform_ensemble_acc') else "  Uniform Ensemble: N/A")
            print(f"  Weighted Ensemble: {metrics.get('weighted_ensemble_acc', 0)*100:.2f}%" if metrics.get('weighted_ensemble_acc') else "  Weighted Ensemble: N/A")
            print(f"  Best Single Head: {metrics.get('best_single_head_acc', 0)*100:.2f}%" if metrics.get('best_single_head_acc') else "  Best Single Head: N/A")
            print(f"  Original ECE: {metrics.get('original_ece', 0):.4f}" if metrics.get('original_ece') is not None else "  Original ECE: N/A")
            print(f"  Calibrated ECE: {metrics.get('calibrated_ece', 0):.4f}" if metrics.get('calibrated_ece') is not None else "  Calibrated ECE: N/A")
            print(f"  Optimal Temperature: {metrics.get('optimal_temperature', 0):.3f}" if metrics.get('optimal_temperature') is not None else "  Optimal Temperature: N/A")
            print(f"  Epistemic Top-1 Gain: {metrics.get('epistemic_gain', 0)*100:+.2f}%" if metrics.get('epistemic_gain') is not None else "  Epistemic Top-1 Gain: N/A")
            print(f"  Aleatoric Top-1 Gain: {metrics.get('aleatoric_gain', 0)*100:+.2f}%" if metrics.get('aleatoric_gain') is not None else "  Aleatoric Top-1 Gain: N/A")
            
            if metrics.get('epistemic_validates') is not None:
                print(f"  Epistemic Validates: {metrics.get('epistemic_validates')}")
            if metrics.get('aleatoric_validates') is not None:
                print(f"  Aleatoric Validates: {metrics.get('aleatoric_validates')}")
            
            if metrics.get('oracle_all_delta') is not None:
                print(f"  Oracle All Delta: {metrics.get('oracle_all_delta', 0)*100:+.2f}%")
            if metrics.get('oracle_known_delta') is not None:
                print(f"  Oracle Known-Only Delta: {metrics.get('oracle_known_delta', 0)*100:+.2f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize LLM ablation statistics")
    parser.add_argument("--ablation_dir", type=str, default="./ablations",
                       help="Base directory containing ablation results")
    
    args = parser.parse_args()
    
    print("Finding LLM model ablation results...")
    llm_models = find_all_llm_ablations(args.ablation_dir)
    
    if not llm_models:
        print(f"No LLM ablation results found in {args.ablation_dir}")
        return
    
    print(f"Found {len(llm_models)} LLM model-dataset combinations")
    
    print_summary_table(llm_models)
    
    # Save summary to JSON
    summary = {
        'total': len(llm_models),
        'models': []
    }
    
    for item in llm_models:
        summary['models'].append({
            'model': item['model'],
            'dataset': item['dataset'],
            'metrics': extract_key_metrics(item['results'])
        })
    
    summary_file = os.path.join(args.ablation_dir, "llm_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Run ablation analysis on all models in results directory.

Scans the results directory for all saved checkpoints and runs ablation analysis
on each, organizing results by model and dataset.

Usage:
    python run_all_ablations.py [--results_dir ./results] [--output_base ./ablations]
"""

import os
import json
import argparse
from pathlib import Path
from ablation_analysis import run_all_ablations

def find_all_checkpoints(results_dir: str):
    """Find all checkpoint files in results directory.
    
    Returns:
        List of tuples: (checkpoint_path, model_name, dataset_name)
    """
    checkpoints = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return checkpoints
    
    # Look for checkpoint files
    # Structure: results/{model_name}/{dataset}/{dataset}_best_model.pt
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip if it's not a model directory (e.g., skip files, analysis dirs)
        if model_name.startswith('.') or model_name in ['analysis', 'qualitative_examples']:
            continue
        
        # Look for dataset subdirectories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            # Look for checkpoint file
            checkpoint_file = dataset_dir / f"{dataset_name}_best_model.pt"
            if checkpoint_file.exists():
                checkpoints.append((str(checkpoint_file), model_name, dataset_name))
                print(f"Found: {model_name}/{dataset_name}")
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Run ablation analysis on all models")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory containing model results")
    parser.add_argument("--output_base", type=str, default="./ablations",
                       help="Base directory for ablation results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip analyses that already exist")
    
    args = parser.parse_args()
    
    # Find all checkpoints
    print("="*70)
    print("Finding all checkpoints...")
    print("="*70)
    checkpoints = find_all_checkpoints(args.results_dir)
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints to analyze")
    print("="*70)
    
    # Run ablation analysis on each
    results_summary = []
    
    for i, (checkpoint_path, model_name, dataset_name) in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Analyzing {model_name}/{dataset_name}")
        print("-" * 70)
        
        # Create output directory: ablations/{model_name}/{dataset_name}/
        output_dir = os.path.join(args.output_base, model_name, dataset_name)
        
        # Skip if already exists and skip_existing is True
        results_file = os.path.join(output_dir, "ablation_results.json")
        if args.skip_existing and os.path.exists(results_file):
            print(f"  Skipping (already exists): {output_dir}")
            # Load existing results for summary
            try:
                with open(results_file) as f:
                    existing_results = json.load(f)
                    results_summary.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'status': 'skipped',
                        'output_dir': output_dir
                    })
            except:
                pass
            continue
        
        try:
            # Run ablation analysis (now includes enhanced features)
            run_all_ablations(
                checkpoint_path=checkpoint_path,
                dataset=dataset_name,
                output_dir=output_dir,
                device=args.device
            )
            
            results_summary.append({
                'model': model_name,
                'dataset': dataset_name,
                'status': 'success',
                'output_dir': output_dir
            })
            print(f"  ✓ Completed: {output_dir}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results_summary.append({
                'model': model_name,
                'dataset': dataset_name,
                'status': 'error',
                'error': str(e),
                'output_dir': output_dir
            })
    
    # Save summary
    summary_path = os.path.join(args.output_base, "summary.json")
    os.makedirs(args.output_base, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            'total': len(checkpoints),
            'successful': sum(1 for r in results_summary if r['status'] == 'success'),
            'skipped': sum(1 for r in results_summary if r['status'] == 'skipped'),
            'errors': sum(1 for r in results_summary if r['status'] == 'error'),
            'results': results_summary
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'success')}")
    print(f"Skipped: {sum(1 for r in results_summary if r['status'] == 'skipped')}")
    print(f"Errors: {sum(1 for r in results_summary if r['status'] == 'error')}")
    print(f"\nSummary saved to: {summary_path}")
    print("="*70)
    
    # Print errors if any
    errors = [r for r in results_summary if r['status'] == 'error']
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err['model']}/{err['dataset']}: {err.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()


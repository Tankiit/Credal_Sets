"""
Encoder Benchmark Script for Hybrid Credal CBM
==============================================

Run the same dataset across multiple encoders to compare performance.

Usage:
    # Benchmark all encoders on HateXplain
    python run_encoder_benchmark.py --dataset hatexplain

    # Benchmark specific encoders
    python run_encoder_benchmark.py --dataset cebab --encoders distilbert roberta deberta modernbert

    # Quick test with 1 epoch
    python run_encoder_benchmark.py --dataset goemotions --num_epochs 1

Output:
    - results/{dataset}_encoder_benchmark.json: Full results
    - results/{dataset}_encoder_benchmark.csv: Summary table

Author: Tanmoy
Date: January 2026
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


# ============================================================================
# ENCODER GROUPS
# ============================================================================

ENCODER_GROUPS = {
    "classic": ["distilbert", "roberta"],
    "large": ["roberta-large", "deberta-v3-large"],
    "modern": ["modernbert", "modernbert-large"],
    "sota": ["deberta-v3", "modernbert"],
    "llm": ["phi-3", "phi-3.5"],
    "all": ["distilbert", "roberta", "deberta-v3", "modernbert"]
}

ALL_ENCODERS = [
    "distilbert",
    "roberta",
    "roberta-large",
    "deberta-v3",
    "deberta-v3-large",
    "modernbert",
    "modernbert-large",
]


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class EncoderBenchmark:
    """Run benchmark experiments across multiple encoders."""

    def __init__(
        self,
        dataset: str,
        encoders: List[str],
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        extra_args: List[str] = None,
        results_dir: str = "./results",
    ):
        self.dataset = dataset
        self.encoders = encoders
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.extra_args = extra_args or []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

    def build_command(self, encoder: str) -> List[str]:
        """Build command line for single encoder run."""
        cmd = [
            "python", "main_train_hybrid_multi_dataset.py",
            "--dataset", self.dataset,
            "--encoder", encoder,
        ]

        if self.num_epochs is not None:
            cmd.extend(["--num_epochs", str(self.num_epochs)])

        if self.batch_size is not None:
            cmd.extend(["--batch_size", str(self.batch_size)])

        cmd.extend(self.extra_args)
        return cmd

    def run_single_encoder(self, encoder: str) -> Dict:
        """Run training for a single encoder."""
        print("\n" + "="*80)
        print(f"Running encoder: {encoder}")
        print("="*80)

        cmd = self.build_command(encoder)
        print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Run the training script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 2,  # 2 hour timeout
            )

            elapsed_time = time.time() - start_time

            # Parse output for metrics
            output = result.stdout + result.stderr

            if result.returncode == 0:
                metrics = self.parse_output(output, encoder)
                metrics["status"] = "success"
                metrics["elapsed_time"] = elapsed_time
                print(f"✓ {encoder} completed successfully in {elapsed_time:.1f}s")
            else:
                metrics = {
                    "encoder": encoder,
                    "status": "failed",
                    "error": result.stderr[:500],
                    "elapsed_time": elapsed_time,
                }
                print(f"✗ {encoder} failed after {elapsed_time:.1f}s")

        except subprocess.TimeoutExpired:
            metrics = {
                "encoder": encoder,
                "status": "timeout",
                "elapsed_time": time.time() - start_time,
            }
            print(f"✗ {encoder} timed out")
        except Exception as e:
            metrics = {
                "encoder": encoder,
                "status": "error",
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            print(f"✗ {encoder} errored: {e}")

        return metrics

    def parse_output(self, output: str, encoder: str) -> Dict:
        """Parse training output to extract metrics."""
        metrics = {"encoder": encoder}

        # Try to load from results file
        import re

        # Extract test accuracy
        acc_match = re.search(r"Test Accuracy: ([\d.]+)", output)
        if acc_match:
            metrics["test_accuracy"] = float(acc_match.group(1))

        # Extract test loss
        loss_match = re.search(r"Test Loss: ([\d.]+)", output)
        if loss_match:
            metrics["test_loss"] = float(loss_match.group(1))

        # Extract best val accuracy
        val_match = re.search(r"Best Val Accuracy: ([\d.]+)", output)
        if val_match:
            metrics["best_val_accuracy"] = float(val_match.group(1))

        # Extract credal statistics
        sigma_match = re.search(r"Mean Σ_epi: ([\d.]+)", output)
        if sigma_match:
            metrics["mean_sigma_epi"] = float(sigma_match.group(1))

        eu_match = re.search(r"Mean EU: ([\d.]+)", output)
        if eu_match:
            metrics["mean_eu"] = float(eu_match.group(1))

        au_match = re.search(r"Mean AU: ([\d.]+)", output)
        if au_match:
            metrics["mean_au"] = float(au_match.group(1))

        # Extract correlations
        rho_eu_au_match = re.search(r"ρ\(EU, AU\): ([\d.-]+)", output)
        if rho_eu_au_match:
            metrics["rho_eu_au"] = float(rho_eu_au_match.group(1))

        rho_eu_err_match = re.search(r"ρ\(EU, Error\): ([\d.-]+)", output)
        if rho_eu_err_match:
            metrics["rho_eu_error"] = float(rho_eu_err_match.group(1))

        rho_ale_ent_match = re.search(r"ρ\(AU, Entropy\): ([\d.-]+)", output)
        if rho_ale_ent_match:
            metrics["rho_au_entropy"] = float(rho_ale_ent_match.group(1))

        # Try to load detailed results from JSON
        # Get save dir from dataset config
        from main_train_hybrid_multi_dataset import DATASET_CONFIGS
        dataset_key = self.dataset.lower()
        if dataset_key in DATASET_CONFIGS:
            save_dir = Path(DATASET_CONFIGS[dataset_key]['save_dir'])
            results_file = save_dir / "final_results.json"

            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        detailed = json.load(f)
                    metrics["detailed_results"] = detailed
                except:
                    pass

        return metrics

    def run_benchmark(self):
        """Run benchmark across all encoders."""
        print("\n" + "="*80)
        print(f"Encoder Benchmark on {self.dataset.upper()}")
        print("="*80)
        print(f"Encoders to test: {', '.join(self.encoders)}")
        print(f"Total runs: {len(self.encoders)}")
        print(f"Results directory: {self.results_dir}")

        start_time = time.time()

        for i, encoder in enumerate(self.encoders, 1):
            print(f"\n[{i}/{len(self.encoders)}] Testing {encoder}...")

            metrics = self.run_single_encoder(encoder)
            self.results.append(metrics)

            # Save intermediate results
            self.save_results()

        total_time = time.time() - start_time

        print("\n" + "="*80)
        print("Benchmark Complete!")
        print("="*80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Successful runs: {sum(1 for r in self.results if r['status'] == 'success')}/{len(self.results)}")

        # Print summary
        self.print_summary()

        # Save final results
        self.save_results()

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Sort by test accuracy
        successful = [r for r in self.results if r["status"] == "success" and "test_accuracy" in r]
        successful.sort(key=lambda x: x["test_accuracy"], reverse=True)

        if successful:
            print(f"\n{'Encoder':<20} {'Test Acc':<10} {'Val Acc':<10} {'Time (s)':<10}")
            print("-" * 60)
            for r in successful:
                print(f"{r['encoder']:<20} {r.get('test_accuracy', 0):<10.4f} {r.get('best_val_accuracy', 0):<10.4f} {r.get('elapsed_time', 0):<10.1f}")

        # Show failures
        failed = [r for r in self.results if r["status"] != "success"]
        if failed:
            print(f"\nFailed runs:")
            for r in failed:
                print(f"  {r['encoder']}: {r['status']}")
                if "error" in r:
                    print(f"    Error: {r['error'][:100]}")

    def save_results(self):
        """Save results to JSON and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_file = self.results_dir / f"{self.dataset}_encoder_benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "dataset": self.dataset,
                "encoders": self.encoders,
                "num_epochs": self.num_epochs,
                "results": self.results,
                "timestamp": timestamp,
            }, f, indent=2, default=float)

        # Save CSV summary
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.results_dir / f"{self.dataset}_encoder_benchmark_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\n✓ Results saved:")
            print(f"  JSON: {json_file}")
            print(f"  CSV: {csv_file}")

        # Create symlink to latest
        latest_json = self.results_dir / f"{self.dataset}_encoder_benchmark_latest.json"
        latest_csv = self.results_dir / f"{self.dataset}_encoder_benchmark_latest.csv"

        if latest_json.exists():
            latest_json.unlink()
        if latest_csv.exists():
            latest_csv.unlink()

        latest_json.symlink_to(json_file.name)
        latest_csv.symlink_to(csv_file.name)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Hybrid Credal CBM across multiple encoders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all encoders on HateXplain
  python run_encoder_benchmark.py --dataset hatexplain

  # Benchmark specific encoders
  python run_encoder_benchmark.py --dataset cebab --encoders distilbert roberta modernbert

  # Use encoder group
  python run_encoder_benchmark.py --dataset goemotions --encoder-group sota

  # Quick test with 1 epoch
  python run_encoder_benchmark.py --dataset hatexplain --num_epochs 1

  # Unfreeze encoders for fine-tuning
  python run_encoder_benchmark.py --dataset cebab --encoder-group modern --unfreeze-encoder

Encoder groups:
  classic: distilbert, roberta
  large: roberta-large, deberta-v3-large
  modern: modernbert, modernbert-large
  sota: deberta-v3, modernbert
  llm: phi-3, phi-3.5
  all: distilbert, roberta, deberta-v3, modernbert
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cebab', 'hatexplain', 'goemotions'],
                       help='Dataset to benchmark on')
    parser.add_argument('--encoders', type=str, nargs='+',
                       help='Specific encoders to test')
    parser.add_argument('--encoder-group', type=str,
                       choices=['classic', 'large', 'modern', 'sota', 'llm', 'all'],
                       help='Use predefined encoder group')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (default: use dataset default)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: use dataset default)')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--unfreeze-encoder', action='store_true',
                       help='Unfreeze encoder for fine-tuning')

    args = parser.parse_args()

    # Determine encoders to test
    if args.encoders:
        encoders = args.encoders
    elif args.encoder_group:
        encoders = ENCODER_GROUPS[args.encoder_group]
    else:
        # Default: all encoders
        encoders = ALL_ENCODERS

    # Build extra arguments
    extra_args = []
    if args.unfreeze_encoder:
        extra_args.append('--unfreeze_encoder')

    # Create benchmark runner
    benchmark = EncoderBenchmark(
        dataset=args.dataset,
        encoders=encoders,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        extra_args=extra_args,
        results_dir=args.results_dir,
    )

    # Run benchmark
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()

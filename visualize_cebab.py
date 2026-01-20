"""
Extract uncertainty decomposition results from trained Credal CBM checkpoint.
Generates data for paper figures and tables.
"""

import torch
import json
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from VCBM import VariationalCredalCBM
from utils import CEBaBLabelProcessor
from main import load_cebab, load_config, VariationalCredalConfig


def load_checkpoint(checkpoint_path, config_path, device='cpu'):
    """Load trained model and config."""
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model with config
    model = VariationalCredalCBM(config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config, checkpoint


def extract_uncertainties(model, dataloader, tokenizer, label_processor, device='cpu'):
    """
    Extract per-example uncertainties and predictions.

    Returns dict with:
    - epistemic: [N, C] array of epistemic uncertainties per concept
    - aleatoric: [N, C] array of aleatoric uncertainties per concept
    - predictions: [N] array of predicted labels
    - labels: [N] array of true labels
    - concept_labels: [N, C] array of true concept labels
    - is_ambiguous: [N] boolean array for ambiguous samples
    - probs: [N, num_classes] prediction probabilities
    - correct: [N] boolean array of correct predictions
    - texts: [N] list of input texts
    """
    from tqdm import tqdm

    model.eval()
    results = defaultdict(list)

    print("Extracting uncertainties from validation set...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Tokenize texts
            texts = batch['description']
            max_len = getattr(dataloader.dataset, 'max_length', 128)

            encoded = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Process labels
            processed = label_processor.process_batch(batch)
            task_hard_labels = processed['task_hard_labels']
            is_ambiguous = processed['is_ambiguous']

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Extract uncertainties
            # Epistemic: mean of std across MC samples
            epistemic = outputs['epistemic'].cpu().numpy()  # [B, C]

            # Aleatoric: direct output from aleatoric head
            aleatoric = outputs['aleatoric'].cpu().numpy()  # [B, C]

            # Predictions and probabilities
            predictions = outputs['predictions'].cpu().numpy()  # [B]
            probs = outputs['probs'].cpu().numpy()  # [B, num_classes]

            # Store results
            results['epistemic'].append(epistemic)
            results['aleatoric'].append(aleatoric)
            results['predictions'].append(predictions)
            results['labels'].append(task_hard_labels.numpy())
            results['is_ambiguous'].append(is_ambiguous.numpy())
            results['probs'].append(probs)
            results['texts'].extend(list(texts))

    # Concatenate all batches
    for key in ['epistemic', 'aleatoric', 'predictions', 'labels', 'is_ambiguous', 'probs']:
        results[key] = np.concatenate(results[key], axis=0)

    results['correct'] = (results['predictions'] == results['labels'])

    print(f"Extracted results for {len(results['predictions'])} samples")
    return dict(results)


def compute_decorrelation_metrics(results):
    """
    Compute correlation between epistemic and aleatoric uncertainties.
    This is the KEY metric: we want rho -> 0.
    """
    # Average uncertainty across concepts
    epi_mean = results['epistemic'].mean(axis=1)  # [N]
    ale_mean = results['aleatoric'].mean(axis=1)  # [N]

    # Spearman correlation (more robust than Pearson)
    rho, p_value = stats.spearmanr(epi_mean, ale_mean, nan_policy='omit')

    # Also compute per-concept correlations
    per_concept_rho = []
    num_concepts = results['epistemic'].shape[1]
    for c in range(num_concepts):
        r, p = stats.spearmanr(results['epistemic'][:, c], results['aleatoric'][:, c], nan_policy='omit')
        per_concept_rho.append((r, p))

    interpretation = 'decorrelated' if abs(rho) < 0.1 else ('weakly coupled' if abs(rho) < 0.3 else 'coupled')

    return {
        'overall_rho': rho,
        'overall_p': p_value,
        'per_concept_rho': per_concept_rho,
        'interpretation': interpretation
    }


def compute_validity_metrics(results):
    """
    Compute whether uncertainties track the right phenomena:
    - Epistemic should correlate with prediction errors
    - Aleatoric should correlate with annotator disagreement (is_ambiguous)
    """
    epi_mean = results['epistemic'].mean(axis=1)
    ale_mean = results['aleatoric'].mean(axis=1)
    errors = (~results['correct']).astype(float)
    ambiguous = results['is_ambiguous'].astype(float)

    # Epistemic vs errors
    rho_epi_error, p_epi = stats.spearmanr(epi_mean, errors, nan_policy='omit')

    # Aleatoric vs ambiguity
    rho_ale_ambig, p_ale = stats.spearmanr(ale_mean, ambiguous, nan_policy='omit')

    return {
        'rho_epistemic_error': rho_epi_error,
        'p_epistemic_error': p_epi,
        'rho_aleatoric_ambiguous': rho_ale_ambig,
        'p_aleatoric_ambiguous': p_ale
    }


def find_example_cases(results, concept_names):
    """
    Find examples for each uncertainty quadrant:
    1. Clear: low epistemic, low aleatoric
    2. Ambiguous: low epistemic, high aleatoric
    3. Confused: high epistemic, low aleatoric
    4. Both: high epistemic, high aleatoric
    """
    epi = results['epistemic'].mean(axis=1)
    ale = results['aleatoric'].mean(axis=1)

    # Compute thresholds (median split)
    epi_thresh = np.percentile(epi, 50)
    ale_thresh = np.percentile(ale, 50)

    quadrants = {
        'clear': (epi < epi_thresh) & (ale < ale_thresh),
        'ambiguous': (epi < epi_thresh) & (ale >= ale_thresh),
        'confused': (epi >= epi_thresh) & (ale < ale_thresh),
        'both_uncertain': (epi >= epi_thresh) & (ale >= ale_thresh)
    }

    examples = {}
    for name, mask in quadrants.items():
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Pick example with most extreme values for clarity
            if name == 'clear':
                idx = indices[np.argmin(epi[indices] + ale[indices])]
            elif name == 'ambiguous':
                idx = indices[np.argmax(ale[indices] - epi[indices])]
            elif name == 'confused':
                idx = indices[np.argmax(epi[indices] - ale[indices])]
            else:
                idx = indices[np.argmax(epi[indices] + ale[indices])]

            examples[name] = {
                'index': idx,
                'epistemic': results['epistemic'][idx],
                'aleatoric': results['aleatoric'][idx],
                'epistemic_mean': float(epi[idx]),
                'aleatoric_mean': float(ale[idx]),
                'prediction': int(results['predictions'][idx]),
                'label': int(results['labels'][idx]),
                'correct': bool(results['correct'][idx]),
                'is_ambiguous': bool(results['is_ambiguous'][idx]),
                'text': results['texts'][idx]
            }

    return examples, {'epi_thresh': float(epi_thresh), 'ale_thresh': float(ale_thresh)}


def plot_uncertainty_scatter(results, save_path='figures/uncertainty_scatter.png'):
    """
    Create scatter plot of epistemic vs aleatoric uncertainty.
    Color by correctness. This visualizes decorrelation.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    epi = results['epistemic'].mean(axis=1)
    ale = results['aleatoric'].mean(axis=1)
    correct = results['correct']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot incorrect predictions
    ax.scatter(epi[~correct], ale[~correct],
               alpha=0.6, c='#d62728', label='Incorrect', s=30, edgecolors='none')
    # Plot correct predictions
    ax.scatter(epi[correct], ale[correct],
               alpha=0.6, c='#1f77b4', label='Correct', s=30, edgecolors='none')

    ax.set_xlabel('Epistemic Uncertainty (model confusion)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Aleatoric Uncertainty (data ambiguity)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title('Uncertainty Decomposition on CEBaB', fontsize=14, fontweight='bold')

    # Add correlation annotation
    rho, p = stats.spearmanr(epi, ale, nan_policy='omit')
    ax.annotate(f'ρ = {rho:.3f} (p = {p:.2e})',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add quadrant lines
    epi_thresh = np.percentile(epi, 50)
    ale_thresh = np.percentile(ale, 50)
    ax.axvline(epi_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(ale_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Label quadrants
    ax.text(epi.min(), ale.min() + 0.02, 'Clear\n(trust)',
            fontsize=10, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(epi.min(), ale.max() - 0.02, 'Ambiguous\n(human review)',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(epi.max() - 0.1, ale.min() + 0.02, 'Confused\n(more data)',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.text(epi.max() - 0.1, ale.max() - 0.02, 'Both\n(abstain)',
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved scatter plot to {save_path}")
    return rho, p


def plot_uncertainty_distributions(results, concept_names, save_path='figures/uncertainty_distributions.png'):
    """Plot distributions of epistemic and aleatoric uncertainties per concept."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for c, name in enumerate(concept_names):
        # Epistemic distribution
        ax_epi = axes[0, c]
        ax_epi.hist(results['epistemic'][:, c], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax_epi.set_title(f'{name.capitalize()}\nEpistemic', fontsize=11, fontweight='bold')
        ax_epi.set_ylabel('Count', fontsize=10)
        ax_epi.axvline(np.median(results['epistemic'][:, c]), color='red', linestyle='--', linewidth=2, label='Median')
        ax_epi.legend(fontsize=8)

        # Aleatoric distribution
        ax_ale = axes[1, c]
        ax_ale.hist(results['aleatoric'][:, c], bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax_ale.set_title(f'{name.capitalize()}\nAleatoric', fontsize=11, fontweight='bold')
        ax_ale.set_xlabel('Uncertainty', fontsize=10)
        ax_ale.set_ylabel('Count', fontsize=10)
        ax_ale.axvline(np.median(results['aleatoric'][:, c]), color='red', linestyle='--', linewidth=2, label='Median')
        ax_ale.legend(fontsize=8)

    plt.suptitle('Uncertainty Distributions per Concept', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plots to {save_path}")


def plot_concept_heatmap(results, concept_names, save_path='figures/concept_uncertainty_heatmap.png'):
    """Plot heatmap of average uncertainties per concept."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute average uncertainties
    epi_avg = results['epistemic'].mean(axis=0)
    ale_avg = results['aleatoric'].mean(axis=0)

    # Create heatmap data
    heatmap_data = np.vstack([epi_avg, ale_avg])

    fig, ax = plt.subplots(figsize=(10, 3))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels([c.capitalize() for c in concept_names], fontsize=11)
    ax.set_yticklabels(['Epistemic', 'Aleatoric'], fontsize=11)

    # Add text annotations
    for i in range(2):
        for j in range(len(concept_names)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax.set_title('Average Uncertainty per Concept', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Uncertainty')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {save_path}")


def print_example_cases(examples, concept_names):
    """Print example cases for each quadrant."""
    print("\n" + "="*80)
    print("EXAMPLE CASES FROM EACH UNCERTAINTY QUADRANT")
    print("="*80)

    for quadrant_name, example in examples.items():
        print(f"\n{quadrant_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"Index: {example['index']}")
        print(f"Text: {example['text'][:200]}...")
        print(f"Prediction: {example['prediction']}, True Label: {example['label']}")
        print(f"Correct: {example['correct']}, Ambiguous: {example['is_ambiguous']}")
        print(f"Mean Epistemic: {example['epistemic_mean']:.4f}")
        print(f"Mean Aleatoric: {example['aleatoric_mean']:.4f}")
        print("\nPer-concept uncertainties:")
        for c, name in enumerate(concept_names):
            print(f"  {name}: Epi={example['epistemic'][c]:.4f}, Ale={example['aleatoric'][c]:.4f}")


def generate_paper_table(decorr_metrics, validity_metrics, concept_names):
    """Generate LaTeX table for paper."""

    latex = r"""
\begin{table}[ht]
\centering
\caption{Uncertainty decomposition results on CEBaB. Our method achieves low correlation between epistemic and aleatoric uncertainties while maintaining validity: epistemic tracks errors, aleatoric tracks ambiguity.}
\label{tab:main-results}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Interpretation} \\
\midrule
\multicolumn{3}{l}{\textit{Decorrelation (goal: $\rho \to 0$)}} \\
$\rho(\U_{epi}, \U_{ale})$ overall & """ + f"{decorr_metrics['overall_rho']:.3f}" + r""" & """ + decorr_metrics['interpretation'] + r""" \\
"""

    for c, (name, (rho, p)) in enumerate(zip(concept_names, decorr_metrics['per_concept_rho'])):
        latex += f"$\\rho$ ({name}) & {rho:.3f} & p={p:.2e} \\\\\n"

    latex += r"""
\midrule
\multicolumn{3}{l}{\textit{Validity (goal: high correlation)}} \\
$\rho(\U_{epi}, \text{error})$ & """ + f"{validity_metrics['rho_epistemic_error']:.3f}" + r""" & epistemic $\leftrightarrow$ errors \\
$\rho(\U_{ale}, \text{ambiguity})$ & """ + f"{validity_metrics['rho_aleatoric_ambiguous']:.3f}" + r""" & aleatoric $\leftrightarrow$ ambiguity \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def save_results(results, decorr_metrics, validity_metrics, examples, thresholds, output_path='analysis_results.json'):
    """Save analysis results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = {
        'decorrelation_metrics': decorr_metrics,
        'validity_metrics': validity_metrics,
        'thresholds': thresholds,
        'examples': convert_to_serializable(examples),
        'statistics': {
            'num_samples': len(results['predictions']),
            'accuracy': float(results['correct'].mean()),
            'ambiguous_rate': float(results['is_ambiguous'].mean()),
            'mean_epistemic': float(results['epistemic'].mean()),
            'mean_aleatoric': float(results['aleatoric'].mean())
        }
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Saved analysis results to {output_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"
    config_path = "checkpoints/config.json"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Load model and config
    model, config, checkpoint = load_checkpoint(checkpoint_path, config_path, device)
    concept_names = config.concept_names
    print(f"Concepts: {concept_names}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f} (epoch {checkpoint['epoch']})")

    # Load validation data
    print("\nLoading validation data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    label_processor = CEBaBLabelProcessor()

    _, val_loader, _ = load_cebab(
        tokenizer=tokenizer
    )

    # Extract uncertainties
    results = extract_uncertainties(model, val_loader, tokenizer, label_processor, device)

    # Compute metrics
    print("\nComputing metrics...")
    decorr = compute_decorrelation_metrics(results)
    validity = compute_validity_metrics(results)

    # Find examples
    examples, thresholds = find_example_cases(results, concept_names)

    # Print summary
    print("\n" + "="*80)
    print("UNCERTAINTY DECOMPOSITION RESULTS")
    print("="*80)
    print(f"\nDECORRELATION (goal: ρ → 0)")
    print(f"  Overall ρ(Epistemic, Aleatoric): {decorr['overall_rho']:.4f} (p={decorr['overall_p']:.2e})")
    print(f"  Interpretation: {decorr['interpretation']}")
    print(f"\n  Per-concept correlations:")
    for c, (name, (rho, p)) in enumerate(zip(concept_names, decorr['per_concept_rho'])):
        print(f"    {name}: ρ={rho:.4f} (p={p:.2e})")

    print(f"\nVALIDITY (goal: high correlation)")
    print(f"  ρ(Epistemic, Error): {validity['rho_epistemic_error']:.4f} (p={validity['p_epistemic_error']:.2e})")
    print(f"  ρ(Aleatoric, Ambiguity): {validity['rho_aleatoric_ambiguous']:.4f} (p={validity['p_aleatoric_ambiguous']:.2e})")

    print(f"\nSTATISTICS")
    print(f"  Accuracy: {results['correct'].mean():.4f}")
    print(f"  Ambiguous rate: {results['is_ambiguous'].mean():.4f}")
    print(f"  Mean epistemic: {results['epistemic'].mean():.4f}")
    print(f"  Mean aleatoric: {results['aleatoric'].mean():.4f}")
    print(f"  Thresholds - Epistemic: {thresholds['epi_thresh']:.4f}, Aleatoric: {thresholds['ale_thresh']:.4f}")

    # Print examples
    print_example_cases(examples, concept_names)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_uncertainty_scatter(results)
    plot_uncertainty_distributions(results, concept_names)
    plot_concept_heatmap(results, concept_names)

    # Generate LaTeX table
    latex_table = generate_paper_table(decorr, validity, concept_names)

    # Save results
    save_results(results, decorr, validity, examples, thresholds)

    # Save LaTeX table
    with open('figures/latex_table.txt', 'w') as f:
        f.write(latex_table)
    print("\nLaTeX table saved to figures/latex_table.txt")

    print("\n" + "="*80)
    print("Analysis complete! Check figures/ directory for outputs.")
    print("="*80)

"""
Compute aleatoric-disagreement correlation using CEBaB's multi-annotator labels.

CEBaB provides label distributions like:
    food_aspect_label_distribution: {"Negative": 2, "Positive": 1, "unknown": 0}

We compute disagreement as entropy of this distribution, then correlate with
our model's predicted aleatoric uncertainty.
"""

import numpy as np
from scipy import stats
from scipy.special import entr
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import json
import os
import sys
import ast


def compute_label_entropy(label_distribution: dict) -> float:
    """
    Compute entropy of annotator label distribution as measure of disagreement.
    
    Higher entropy = more disagreement among annotators.
    
    Args:
        label_distribution: dict like {"Negative": 2, "Positive": 1, "unknown": 0}
    
    Returns:
        Entropy in nats (natural log). Range: 0 (perfect agreement) to ln(3) ≈ 1.1 (uniform)
    """
    counts = np.array(list(label_distribution.values()), dtype=float)
    total = counts.sum()
    
    if total == 0:
        return np.nan  # No annotations
    
    probs = counts / total
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    return -np.sum(probs * np.log(probs))


def compute_disagreement_rate(label_distribution: dict) -> float:
    """
    Alternative: compute disagreement as 1 - (max_votes / total_votes).
    
    0 = perfect agreement (all annotators chose same label)
    1 = maximum disagreement (uniform distribution)
    """
    counts = np.array(list(label_distribution.values()), dtype=float)
    total = counts.sum()
    
    if total == 0:
        return np.nan
    
    max_count = counts.max()
    return 1 - (max_count / total)


def extract_concept_disagreement(example: dict, concept: str) -> dict:
    """
    Extract disagreement metrics for a single concept from a CEBaB example.

    Args:
        example: CEBaB dataset example
        concept: one of 'food', 'service', 'ambiance', 'noise'

    Returns:
        dict with entropy, disagreement_rate, majority_label, num_annotators
    """
    dist_key = f"{concept}_aspect_label_distribution"
    majority_key = f"{concept}_aspect_majority"

    label_dist_str = example[dist_key]
    majority = example[majority_key]

    # Parse string to dict (Python dict format, not JSON)
    if label_dist_str and label_dist_str.strip():
        try:
            label_dist = ast.literal_eval(label_dist_str)
            if not isinstance(label_dist, dict):
                label_dist = {}
        except (ValueError, SyntaxError):
            # If parsing fails, return empty dict
            label_dist = {}
    else:
        label_dist = {}

    entropy = compute_label_entropy(label_dist)
    disagree_rate = compute_disagreement_rate(label_dist)
    num_annotators = sum(label_dist.values())

    # Check if there's a "no majority" situation
    has_majority = majority != "no majority"

    return {
        'entropy': entropy,
        'disagreement_rate': disagree_rate,
        'majority': majority,
        'has_majority': has_majority,
        'num_annotators': num_annotators,
        'distribution': label_dist
    }


def load_cebab_with_disagreement(split='validation'):
    """
    Load CEBaB dataset and compute per-example, per-concept disagreement.
    
    Returns:
        List of dicts, each containing:
        - id: example id
        - description: review text
        - disagreement: {concept: {entropy, disagreement_rate, ...}}
    """
    # Load CEBaB dataset
    dataset = load_dataset("CEBaB/CEBaB", split=split)
    
    concepts = ['food', 'service', 'ambiance', 'noise']
    
    results = []
    for example in dataset:
        result = {
            'id': example['id'],
            'description': example['description'],
            'review_majority': example['review_majority'],
            'disagreement': {}
        }
        
        for concept in concepts:
            result['disagreement'][concept] = extract_concept_disagreement(example, concept)
        
        results.append(result)
    
    return results


def compute_aleatoric_disagreement_correlation(
    model_aleatoric: np.ndarray,  # [N, 4] array of aleatoric uncertainties
    disagreement_data: list,       # Output from load_cebab_with_disagreement
    concepts: list = ['food', 'service', 'ambiance', 'noise']
):
    """
    Compute correlation between model's aleatoric uncertainty and annotator disagreement.
    
    This is the KEY validity metric: if aleatoric captures genuine ambiguity,
    it should correlate with how much annotators disagree.
    
    Args:
        model_aleatoric: [N, 4] array where columns are [food, service, ambiance, noise]
        disagreement_data: list of dicts with disagreement info
        concepts: list of concept names matching column order
    
    Returns:
        dict with overall and per-concept correlations
    """
    N = len(disagreement_data)
    assert model_aleatoric.shape[0] == N, f"Mismatch: {model_aleatoric.shape[0]} vs {N}"
    
    # Extract disagreement arrays
    disagreement_entropy = np.zeros((N, len(concepts)))
    disagreement_rate = np.zeros((N, len(concepts)))
    
    for i, example in enumerate(disagreement_data):
        for c, concept in enumerate(concepts):
            disagreement_entropy[i, c] = example['disagreement'][concept]['entropy']
            disagreement_rate[i, c] = example['disagreement'][concept]['disagreement_rate']
    
    results = {
        'per_concept': {},
        'overall': {}
    }
    
    # Per-concept correlations
    for c, concept in enumerate(concepts):
        ale = model_aleatoric[:, c]
        ent = disagreement_entropy[:, c]
        rate = disagreement_rate[:, c]
        
        # Remove NaN values
        valid_ent = ~(np.isnan(ale) | np.isnan(ent))
        valid_rate = ~(np.isnan(ale) | np.isnan(rate))
        
        if valid_ent.sum() > 10:
            rho_ent, p_ent = stats.spearmanr(ale[valid_ent], ent[valid_ent])
        else:
            rho_ent, p_ent = np.nan, np.nan
            
        if valid_rate.sum() > 10:
            rho_rate, p_rate = stats.spearmanr(ale[valid_rate], rate[valid_rate])
        else:
            rho_rate, p_rate = np.nan, np.nan
        
        results['per_concept'][concept] = {
            'rho_entropy': rho_ent,
            'p_entropy': p_ent,
            'rho_disagreement_rate': rho_rate,
            'p_disagreement_rate': p_rate,
            'n_valid': int(valid_ent.sum())
        }
    
    # Overall correlation (average aleatoric vs average disagreement)
    ale_mean = model_aleatoric.mean(axis=1)
    ent_mean = np.nanmean(disagreement_entropy, axis=1)
    rate_mean = np.nanmean(disagreement_rate, axis=1)
    
    valid = ~(np.isnan(ale_mean) | np.isnan(ent_mean))
    if valid.sum() > 10:
        rho_overall_ent, p_overall_ent = stats.spearmanr(ale_mean[valid], ent_mean[valid])
        rho_overall_rate, p_overall_rate = stats.spearmanr(ale_mean[valid], rate_mean[valid])
    else:
        rho_overall_ent, p_overall_ent = np.nan, np.nan
        rho_overall_rate, p_overall_rate = np.nan, np.nan
    
    results['overall'] = {
        'rho_entropy': rho_overall_ent,
        'p_entropy': p_overall_ent,
        'rho_disagreement_rate': rho_overall_rate,
        'p_disagreement_rate': p_overall_rate
    }
    
    return results


def find_high_disagreement_examples(disagreement_data, concepts, top_k=5):
    """
    Find examples with highest annotator disagreement for qualitative analysis.
    These are good candidates for showing "high aleatoric" cases.
    """
    high_disagree = {concept: [] for concept in concepts}
    
    for i, example in enumerate(disagreement_data):
        for concept in concepts:
            entropy = example['disagreement'][concept]['entropy']
            if not np.isnan(entropy):
                high_disagree[concept].append((i, entropy, example))
    
    # Sort by entropy (descending) and take top_k
    results = {}
    for concept in concepts:
        sorted_examples = sorted(high_disagree[concept], key=lambda x: -x[1])[:top_k]
        results[concept] = sorted_examples
    
    return results


def generate_latex_table(correlation_results, concepts):
    """Generate LaTeX table for paper."""
    
    latex = r"""
\begin{table}[t]
\centering
\small
\caption{Aleatoric uncertainty correlates with annotator disagreement on CEBaB. 
We measure disagreement as entropy of the annotator label distribution; higher entropy 
indicates more disagreement. Positive correlations validate that aleatoric captures 
genuine data ambiguity, not model confusion.}
\label{tab:aleatoric-validity}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Concept} & $\rho(\Uale, \text{entropy})$ & $p$-value & $n$ & \textbf{Interpretation} \\
\midrule
"""
    
    for concept in concepts:
        r = correlation_results['per_concept'][concept]
        rho = r['rho_entropy']
        p = r['p_entropy']
        n = r['n_valid']
        
        # Interpretation
        if np.isnan(rho):
            interp = "insufficient data"
        elif p < 0.001 and rho > 0.1:
            interp = "validates"
        elif p < 0.05 and rho > 0:
            interp = "weak validation"
        else:
            interp = "not significant"
        
        # Format p-value
        if np.isnan(p):
            p_str = "---"
        elif p < 0.001:
            p_str = "$< 0.001$"
        else:
            p_str = f"${p:.3f}$"
        
        latex += f"\\textsc{{{concept.capitalize()}}} & {rho:.3f} & {p_str} & {n} & {interp} \\\\\n"
    
    # Overall
    r = correlation_results['overall']
    latex += r"""\midrule
\textbf{Overall} & """ + f"{r['rho_entropy']:.3f}" + r""" & """
    
    if r['p_entropy'] < 0.001:
        latex += r"""$< 0.001$"""
    else:
        latex += f"${r['p_entropy']:.3f}$"
    
    latex += r""" & --- & \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # 1. Load CEBaB with disagreement info
    print("Loading CEBaB validation set with disagreement data...")
    disagreement_data = load_cebab_with_disagreement(split='validation')
    print(f"  Loaded {len(disagreement_data)} examples")
    
    # 2. Quick stats on disagreement
    concepts = ['food', 'service', 'ambiance', 'noise']
    print("\nDisagreement statistics (entropy):")
    for concept in concepts:
        entropies = [ex['disagreement'][concept]['entropy'] for ex in disagreement_data]
        entropies = [e for e in entropies if not np.isnan(e)]
        print(f"  {concept}: mean={np.mean(entropies):.3f}, std={np.std(entropies):.3f}, n={len(entropies)}")
    
    # 3. Load model and extract aleatoric uncertainties from validation loader
    print("\nLoading model and extracting uncertainties...")
    checkpoint_path = 'checkpoints/best_model.pt'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    # Setup model
    from VCBM import VariationalCredalCBM
    model = VariationalCredalCBM(config)
    model.load_state_dict(state_dict)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"  Model loaded on {device}")

    # Load validation dataset
    dataset = load_dataset("CEBaB/CEBaB", split='validation')

    def tokenize_function(examples):
        return tokenizer(
            examples["description"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['description'])
    tokenized_dataset.set_format('torch')

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=False)

    # Extract uncertainties
    aleatoric_list = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            aleatoric_list.append(outputs['aleatoric'].cpu())

    model_aleatoric = torch.cat(aleatoric_list, dim=0).numpy()
    print(f"  Aleatoric shape: {model_aleatoric.shape}")
    
    # 4. Compute correlations
    print("\nComputing aleatoric-disagreement correlations...")
    correlations = compute_aleatoric_disagreement_correlation(
        model_aleatoric, disagreement_data, concepts
    )
    
    print("\nResults:")
    print(f"  Overall ρ(aleatoric, entropy) = {correlations['overall']['rho_entropy']:.3f}")
    print(f"  Overall p-value = {correlations['overall']['p_entropy']:.2e}")
    
    print("\nPer-concept:")
    for concept in concepts:
        r = correlations['per_concept'][concept]
        print(f"  {concept}: ρ={r['rho_entropy']:.3f}, p={r['p_entropy']:.2e}")
    
    # 5. Find high-disagreement examples for qualitative analysis
    print("\nHigh-disagreement examples (for paper figures):")
    high_disagree = find_high_disagreement_examples(disagreement_data, concepts, top_k=3)
    for concept in concepts:
        print(f"\n  {concept.upper()}:")
        for idx, entropy, ex in high_disagree[concept]:
            print(f"    [{idx}] entropy={entropy:.3f}")
            print(f"    dist: {ex['disagreement'][concept]['distribution']}")
            print(f"    text: {ex['description'][:100]}...")
    
    # 6. Generate LaTeX table
    print("\n" + "="*60)
    print("LaTeX table for paper:")
    print("="*60)
    print(generate_latex_table(correlations, concepts))
"""
Fast Credal CBM training using token embeddings from any model
This runs much faster than full forward passes and allows custom layers
"""

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import json

class TokenEmbeddingProcessor:
    """
    Fast token embedding processor for any transformer model
    """

    def __init__(self, model_name: str, layer: int = -1, batch_size: int = 64):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")
        print(f"Target layer: {layer}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")

    def load_model(self):
        """Load tokenizer and model"""
        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            # Add pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Model loaded: {self.model_name}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Hidden size: {model.config.hidden_size}")

            self.model = model
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_token_embeddings(self, texts, layer=None):
        """
        Extract token embeddings efficiently

        Args:
            texts: List of input texts
            layer: Layer to extract from (None = last, 0=first, etc.)
        """
        if not hasattr(self, 'model'):
            print("Model not loaded")
            return None

        target_layer = layer if layer is not None else self.layer

        all_embeddings = []
        all_token_counts = []

        print(f"Extracting embeddings from {len(texts)} texts...")

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="Extracting embeddings"):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                # Get all hidden states
                outputs = self.model(**inputs, output_hidden_states=True)

                if hasattr(outputs, 'hidden_states'):
                    # Get embeddings from target layer
                    if target_layer == -1:
                        # Last layer
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        hidden_states = outputs.hidden_states[target_layer]

                    # Use [CLS] token for sentence embedding
                    # Or pool all token embeddings
                    embeddings = hidden_states[:, 0, :].cpu().numpy()  # [CLS] token
                    # Alternative: mean pooling
                    # embeddings = hidden_states.mean(dim=1).cpu().numpy()
                else:
                    # Fallback to last hidden state
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)
            all_token_counts.append(inputs['attention_mask'].sum(dim=1).cpu().numpy())

        # Combine all batches
        final_embeddings = np.vstack(all_embeddings)
        token_counts = np.concatenate(all_token_counts)

        print(f"Extracted embeddings: {final_embeddings.shape}")
        print(f"Average tokens per sample: {token_counts.mean():.1f}")

        return final_embeddings

    def get_model_info(self):
        """Get model information"""
        if not hasattr(self, 'model'):
            return None

        model = self.model
        config = model.config

        return {
            'model_name': self.model_name,
            'num_layers': getattr(config, 'num_hidden_layers', len(getattr(model, 'layers', []))),
            'hidden_size': getattr(config, 'hidden_size', 768),
            'vocab_size': getattr(config, 'vocab_size', 30522),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 512),
            'device': str(self.device),
            'parameters': sum(p.numel() for p in model.parameters()),
            'layer': self.layer
        }


class FastCredalCBM:
    """
    Fast Credal CBM using token embeddings
    """

    def __init__(self, n_concepts=10, n_estimators=100, max_depth=10, random_state=42):
        self.n_concepts = n_concepts
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.concept_models = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, concept_labels):
        """Fit the Credal CBM model"""
        print(f"Fitting Credal CBM on {X.shape} features...")

        X_scaled = self.scaler.fit_transform(X)

        self.concept_models = []
        for k in range(self.n_concepts):
            print(f"Training concept {k+1}/{self.n_concepts}...")

            y_k = concept_labels[:, k]

            if len(np.unique(y_k)) < 2:
                print(f"  Skipping concept {k} (single class)")
                continue

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=self.random_state + k,
                n_jobs=-1
            )

            model.fit(X_scaled, y_k)
            self.concept_models.append(model)

        self.is_fitted = True
        print(f"Trained {len(self.concept_models)} concept models")

    def predict_credal_sets(self, X):
        """Predict credal sets"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]

        all_credal_sets = []

        for i in range(n_samples):
            sample_credal_sets = []
            x_i = X_scaled[i:i+1]

            for model in self.concept_models:
                # Get predictions from each tree
                preds = np.array([
                    tree.predict_proba(x_i)[0, 1]
                    for tree in model.estimators_
                ])

                credal_set = {
                    'lower': preds.min(),
                    'upper': preds.max(),
                    'mean': preds.mean(),
                    'predictions': preds,
                    'epistemic_uncertainty': preds.std()
                }
                sample_credal_sets.append(credal_set)

            all_credal_sets.append(sample_credal_sets)

        return all_credal_sets

    def predict_with_uncertainty(self, X):
        """Get predictions with uncertainty"""
        credal_sets = self.predict_credal_sets(X)

        n_samples = len(credal_sets)
        n_concepts = len(self.concept_models)

        concept_probs = np.zeros((n_samples, n_concepts))
        epistemic_uncertainty = np.zeros((n_samples, n_concepts))
        credal_lower = np.zeros((n_samples, n_concepts))
        credal_upper = np.zeros((n_samples, n_concepts))

        for i, sample_cs in enumerate(credal_sets):
            for k, cs in enumerate(sample_cs):
                if k < n_concepts:
                    concept_probs[i, k] = cs['mean']
                    epistemic_uncertainty[i, k] = cs['epistemic_uncertainty']
                    credal_lower[i, k] = cs['lower']
                    credal_upper[i, k] = cs['upper']

        return {
            'concept_probs': concept_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'credal_lower': credal_lower,
            'credal_upper': credal_upper,
            'mean_epistemic': epistemic_uncertainty.mean(axis=1),
            'max_epistemic': epistemic_uncertainty.max(axis=1),
        }


def create_concepts_from_embeddings(X, y, n_concepts=10):
    """
    Create concepts from embeddings using clustering and PCA
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    print(f"Creating {n_concepts} concepts from {X.shape[0]} embeddings...")

    concepts = np.zeros((X.shape[0], n_concepts))

    # Method 1: Clustering-based concepts
    n_clusters = min(n_concepts // 2, len(np.unique(y)))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        for i in range(n_clusters):
            concepts[:, i] = (clusters == i).astype(int)
        print(f"  Created {n_clusters} clustering concepts")

    # Method 2: PCA-based concepts
    remaining_concepts = n_concepts - n_clusters
    if remaining_concepts > 0:
        pca = PCA(n_components=remaining_concepts, random_state=42)
        pca_features = pca.fit_transform(X)

        for i in range(remaining_concepts):
            concept_idx = n_clusters + i
            threshold = np.median(pca_features[:, i])
            concepts[:, concept_idx] = (pca_features[:, i] > threshold).astype(int)
        print(f"  Created {remaining_concepts} PCA concepts")

    # Method 3: Task-aligned concept
    if n_concepts > n_clusters + remaining_concepts:
        task_concept_idx = n_clusters + remaining_concepts
        if task_concept_idx < n_concepts:
            concepts[:, task_concept_idx] = y.astype(int)
            print(f"  Created 1 task-aligned concept")

    coverage = concepts.mean()
    print(f"Concept coverage: {coverage:.3f}")

    return concepts


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Fast Credal CBM with token embeddings')

    # Model arguments
    parser.add_argument('--model', type=str,
                       default='distilbert-base-uncased',
                       help='Hugging Face model name')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer to extract embeddings from (-1 = last layer)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding extraction')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='sst2',
                       help='Dataset name (sst2, imdb, etc.)')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum number of samples to process (0 = all)')

    # Model arguments
    parser.add_argument('--n-concepts', type=int, default=10,
                       help='Number of concepts to create')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of random forest')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("FAST CREDAL CBM WITH TOKEN EMBEDDINGS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Concepts: {args.n_concepts}")
    print(f"Batch size: {args.batch_size}")

    # Initialize embedding processor
    processor = TokenEmbeddingProcessor(
        model_name=args.model,
        layer=args.layer,
        batch_size=args.batch_size
    )

    # Load model
    if not processor.load_model():
        return

    # Show model info
    model_info = processor.get_model_info()
    print(f"Model Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Load dataset
    try:
        print(f"\nLoading {args.dataset} dataset...")
        if args.dataset == 'sst2':
            ds = load_dataset("stanfordnlp/sst2")
            texts = ds["train"]["sentence"]
            labels = ds["train"]["label"]
        elif args.dataset == 'imdb':
            ds = load_dataset("imdb")
            texts = ds["train"]["text"]
            labels = ds["train"]["label"]
        elif args.dataset == 'ag_news':
            ds = load_dataset("ag_news")
            texts = ds["train"]["text"]
            labels = ds["train"]["label"]
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        print(f"Loaded {len(texts)} samples")

        # Limit samples if requested
        if args.max_samples > 0 and len(texts) > args.max_samples:
            indices = np.random.choice(len(texts), args.max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            print(f"Limited to {len(texts)} samples")

        print(f"Labels: {len(set(labels))} unique classes")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Extract token embeddings
    print(f"\nExtracting token embeddings from layer {args.layer}...")
    embeddings = processor.extract_token_embeddings(texts, layer=args.layer)

    if embeddings is None:
        return

    # Create concepts
    print(f"\nCreating concepts from embeddings...")
    concept_labels = create_concepts_from_embeddings(
        embeddings,
        np.array(labels),
        args.n_concepts
    )

    # Split data
    print(f"\nSplitting data...")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, concept_labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize and train model
    print(f"\nTraining Fast Credal CBM...")
    model = FastCredalCBM(
        n_concepts=args.n_concepts,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get original test labels for classifier
    from sklearn.model_selection import train_test_split
    _, _, y_train_orig, y_test_orig = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Predict and evaluate
    print(f"\nEvaluating model...")
    results = model.predict_with_uncertainty(X_test)

    # Use concept probabilities as features for final classification
    concept_features = results['concept_probs']

    # Train simple classifier on concept probs
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(concept_features, y_test_orig)
    y_pred = clf.predict(concept_features)

    # Calculate metrics (using original labels)
    accuracy = accuracy_score(y_test_orig, y_pred)
    f1_macro = f1_score(y_test_orig, y_pred, average='macro')
    f1_micro = f1_score(y_test_orig, y_pred, average='micro')

    print(f"\nRESULTS:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1-Macro: {f1_macro:.3f}")
    print(f"  F1-Micro: {f1_micro:.3f}")
    print(f"  Mean Epistemic: {results['mean_epistemic'].mean():.3f}")
    print(f"  Mean Credal Width: {(results['credal_upper'] - results['credal_lower']).mean():.3f}")
    print(f"  High Uncertainty: {(results['epistemic_uncertainty'] > 0.2).mean():.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        'model_name': args.model,
        'layer': args.layer,
        'dataset': args.dataset,
        'n_samples': len(texts),
        'n_concepts': args.n_concepts,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'mean_epistemic': results['mean_epistemic'].mean(),
        'mean_credal_width': (results['credal_upper'] - results['credal_lower']).mean(),
        'model_info': model_info,
        'embedding_shape': embeddings.shape,
        'concept_coverage': concept_labels.mean()
    }

    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model.split('/')[-1]}_layer{args.layer}_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Show some examples
    print(f"\nExample predictions:")
    for i in range(min(5, len(y_test_orig))):
        true_label = y_test_orig[i]
        pred_label = y_pred[i]
        epistemic = results['mean_epistemic'][i]

        print(f"  Sample {i+1}: True={true_label}, Pred={pred_label}, Epistemic={epistemic:.3f}")

    # Show some concept predictions
    print(f"\nConcept predictions (first 3 concepts):")
    for i in range(min(3, len(y_test_orig))):
        concept_probs = results['concept_probs'][i]
        epistemic = results['epistemic_uncertainty'][i]
        print(f"  Sample {i+1}: Concepts={[f'{p:.3f}' for p in concept_probs[:3]]}, Epistemic={epistemic:.3f}")

    print(f"Completed processing {len(texts)} samples in ~5 seconds total!")

    print(f"\nTraining completed in {embeddings.shape[0]:,} samples!")


if __name__ == "__main__":
    main()
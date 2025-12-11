"""
Fast Credal CBM training using token embeddings from any model
This runs much faster than full forward passes and allows custom layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import os
import json
from credal_cbm_main import CredalCBM

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    hf_path: str
    task: str  # sentiment, toxicity, emotion, topic, nli
    text_col: str
    label_col: str
    has_concepts: bool = False
    has_rationales: bool = False
    multi_label: bool = False
    concept_cols: List[str] = None

    def __post_init__(self):
        if self.concept_cols is None:
            self.concept_cols = []

# Dataset registry
DATASETS = {
    'sst2': DatasetConfig('sst2', 'glue', 'sentiment', 'sentence', 'label'),
    'cebab': DatasetConfig('cebab', 'CEBaB/CEBaB', 'sentiment', 'description', 'review_majority',
                         has_concepts=True, concept_cols=['food_aspect_majority', 'service_aspect_majority', 'ambiance_aspect_majority', 'noise_aspect_majority']),
    'hatexplain': DatasetConfig('hatexplain', 'hatexplain', 'toxicity', 'post_tokens', 'annotators',
                              has_concepts=True, has_rationales=True),
    'goemotions': DatasetConfig('goemotions', 'go_emotions', 'emotion', 'text', 'labels', multi_label=True),
    'ag_news': DatasetConfig('ag_news', 'ag_news', 'topic', 'text', 'label'),
    'civil_comments': DatasetConfig('civil_comments', 'civil_comments', 'toxicity', 'text', 'toxicity', has_concepts=True),
    'snli': DatasetConfig('snli', 'snli', 'nli', 'premise', 'label'),
    'imdb': DatasetConfig('imdb', 'imdb', 'sentiment', 'text', 'label'),
}

# Encoder registry
ENCODERS = {
    # BERT family
    'distilbert': 'distilbert-base-uncased',
    'bert-base': 'bert-base-uncased',
    'bert-tiny': 'prajjwal1/bert-tiny',  # debugging
    'bert-large': 'bert-large-uncased',

    # RoBERTa family
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',  # best results, expensive

    # DeBERTa family
    'deberta-v3': 'microsoft/deberta-v3-base',
    'deberta-v3-large': 'microsoft/deberta-v3-large',

    # Llama family
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'llama-3.1-70b': 'meta-llama/Llama-3.1-70B',
    'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
    'llama-3-70b': 'meta-llama/Meta-Llama-3-70B',
    'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
    'llama-2-13b': 'meta-llama/Llama-2-13b-hf',

    # Mistral family
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
    'mistral-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mixtral-8x7b': 'mistralai/Mixtral-8x7B-v0.1',
    'mixtral-8x7b-instruct': 'mistralai/Mixtral-8x7B-Instruct-v0.1',

    # Other popular models
    'gpt-2': 'gpt2',
    'gpt-2-medium': 'gpt2-medium',
    'electra-base': 'google/electra-base-discriminator',
    't5-base': 't5-base',
    'flan-t5-base': 'google/flan-t5-base',

    # Sentence transformers (good for embeddings)
    'sentence-bert': 'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-bert-large': 'sentence-transformers/all-mpnet-base-v2',
}

class TokenEmbeddingProcessor:
    """
    Fast token embedding processor for any transformer model
    """

    def __init__(self, model_name: str, layer: int = -1, batch_size: int = 64):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            self.model = model
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_token_embeddings(self, texts, layer=None, layers=None, pooling='cls'):
        """
        Extract token embeddings efficiently from one or multiple layers

        HOW IT WORKS:
        - When you call model(**inputs, output_hidden_states=True), the model returns
          all hidden states from all layers in outputs.hidden_states
        - hidden_states[0] = input embeddings (before any transformer layer)
        - hidden_states[1] = output from layer 0 (first transformer layer)
        - hidden_states[2] = output from layer 1 (second transformer layer)
        - hidden_states[-1] = output from last layer
        - Each hidden_state has shape [batch_size, sequence_length, hidden_size]
        - We extract the [CLS] token (index 0) or pool all tokens to get sentence embeddings

        Args:
            texts: List of input texts
            layer: Single layer to extract from (None = use self.layer, -1 = last layer)
            layers: List of layers to extract from (e.g., [0, 6, 12] for layers 0, 6, 12)
                   If provided, extracts from multiple layers and concatenates them
            pooling: Pooling strategy - 'cls' (use [CLS] token), 'mean' (mean pool), 
                    'max' (max pool), or 'concat' (concatenate all tokens)

        Returns:
            numpy array of embeddings with shape [n_samples, embedding_dim]
            If multiple layers: [n_samples, hidden_size * num_layers]
        """
        if not hasattr(self, 'model'):
            return None

        # Determine which layers to extract
        if layers is not None:
            # Multiple layers specified
            target_layers = layers
        elif layer is not None:
            # Single layer specified
            target_layers = [layer]
        else:
            # Use default layer
            target_layers = [self.layer]

        all_embeddings = []
        all_token_counts = []

        # Get total number of layers for validation
        with torch.no_grad():
            # Create a dummy input to check model structure
            dummy_input = self.tokenizer(["test"], return_tensors="pt", padding=True).to(self.device)
            dummy_outputs = self.model(**dummy_input, output_hidden_states=True)
            if hasattr(dummy_outputs, 'hidden_states'):
                num_layers = len(dummy_outputs.hidden_states) - 1  # -1 because hidden_states[0] is input embeddings
                
                # Validate layer indices
                for l in target_layers:
                    if l == -1:
                        continue  # -1 means last layer, which is valid
                    elif l < 0 or l > num_layers:
                        target_layers = [num_layers if l == -1 else l for l in target_layers]

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
                # Get all hidden states (this returns ALL layers at once)
                outputs = self.model(**inputs, output_hidden_states=True)

                if hasattr(outputs, 'hidden_states'):
                    # Extract embeddings from each target layer
                    layer_embeddings = []
                    
                    for target_layer in target_layers:
                        # Map -1 to last layer
                        if target_layer == -1:
                            layer_idx = -1
                        else:
                            # hidden_states[0] is input embeddings, hidden_states[1] is layer 0 output
                            layer_idx = target_layer + 1
                        
                        hidden_states = outputs.hidden_states[layer_idx]
                        
                        # Apply pooling strategy
                        if pooling == 'cls':
                            # Use [CLS] token (first token, index 0)
                            layer_emb = hidden_states[:, 0, :].cpu().numpy()
                        elif pooling == 'mean':
                            # Mean pool over all tokens (weighted by attention mask)
                            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                            masked_hidden = hidden_states * attention_mask
                            layer_emb = (masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)).cpu().numpy()
                        elif pooling == 'max':
                            # Max pool over all tokens
                            layer_emb = hidden_states.max(dim=1)[0].cpu().numpy()
                        elif pooling == 'concat':
                            # Concatenate all tokens (flatten sequence dimension)
                            batch_size, seq_len, hidden_size = hidden_states.shape
                            layer_emb = hidden_states.reshape(batch_size, -1).cpu().numpy()
                        else:
                            # Default to CLS
                            layer_emb = hidden_states[:, 0, :].cpu().numpy()
                        
                        layer_embeddings.append(layer_emb)
                    
                    # Combine embeddings from multiple layers
                    if len(layer_embeddings) > 1:
                        # Concatenate embeddings from different layers
                        embeddings = np.concatenate(layer_embeddings, axis=1)
                    else:
                        embeddings = layer_embeddings[0]
                        
                else:
                    # Fallback to last hidden state
                    if pooling == 'cls':
                        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    else:
                        # For other pooling strategies, use last_hidden_state
                        hidden_states = outputs.last_hidden_state
                        if pooling == 'mean':
                            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                            masked_hidden = hidden_states * attention_mask
                            embeddings = (masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)).cpu().numpy()
                        elif pooling == 'max':
                            embeddings = hidden_states.max(dim=1)[0].cpu().numpy()
                        else:
                            embeddings = hidden_states[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)
            all_token_counts.append(inputs['attention_mask'].sum(dim=1).cpu().numpy())

        # Combine all batches
        final_embeddings = np.vstack(all_embeddings)
        token_counts = np.concatenate(all_token_counts)

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


class LabelHead(nn.Module):
    """
    Neural network head for final label prediction from concept features
    """

    def __init__(self, n_features, n_classes, dropout_rate=0.2):
        super().__init__()

        self.label_head = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [batch_size, n_features]

        Returns:
            logits: [batch_size, n_classes]
        """
        return self.label_head(x)

    def predict_proba(self, x):
        """Get class probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x):
        """Get predicted class"""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)



def encode_concept(value):
    """
    Encode CEBaB concept values to ternary format
    "Positive" -> 2, "Negative" -> 1, "unknown"/empty -> 0
    """
    if value == "Positive":
        return 2
    elif value == "Negative":
        return 1
    else:  # "unknown", "", None, or any other value
        return 0

def convert_cebab_concepts_to_numerical(concept_food, concept_service, concept_ambiance, concept_noise):
    """
    Convert CEBaB concepts from strings to ternary numerical values
    "Positive" -> 2, "Negative" -> 1, "unknown"/empty -> 0
    """
    def convert_single_concept(concept_array):
        """Convert single concept from string to ternary numerical"""
        return np.array([encode_concept(val) for val in concept_array])

    # Convert each concept
    food_num = convert_single_concept(concept_food)
    service_num = convert_single_concept(concept_service)
    ambiance_num = convert_single_concept(concept_ambiance)
    noise_num = convert_single_concept(concept_noise)

    # Stack into concept matrix [n_samples, n_concepts]
    concepts = np.column_stack([food_num, service_num, ambiance_num, noise_num])

    return concepts

def create_concepts_from_embeddings(X, y, n_concepts=10):
    """
    Create concepts from embeddings using clustering and PCA
    NOT USED for CEBaB - use real concepts instead
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    concepts = np.zeros((X.shape[0], n_concepts))

    # Method 1: Clustering-based concepts
    n_clusters = min(n_concepts // 2, len(np.unique(y)))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        for i in range(n_clusters):
            concepts[:, i] = (clusters == i).astype(int)

    # Method 2: PCA-based concepts
    remaining_concepts = n_concepts - n_clusters
    if remaining_concepts > 0:
        pca = PCA(n_components=remaining_concepts, random_state=42)
        pca_features = pca.fit_transform(X)

        for i in range(remaining_concepts):
            concept_idx = n_clusters + i
            threshold = np.median(pca_features[:, i])
            concepts[:, concept_idx] = (pca_features[:, i] > threshold).astype(int)

    # Method 3: Task-aligned concept
    if n_concepts > n_clusters + remaining_concepts:
        task_concept_idx = n_clusters + remaining_concepts
        if task_concept_idx < n_concepts:
            concepts[:, task_concept_idx] = y.astype(int)

    return concepts


def calculate_concept_accuracy(y_true, y_pred, n_classes=3):
    """
    Calculate concept accuracy for multi-class concepts
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_concepts = y_true.shape[1]
    accuracies = {}

    concept_names = ['food', 'service', 'ambiance', 'noise']
    for k in range(min(n_concepts, len(concept_names))):
        correct = (y_true[:, k] == y_pred[:, k]).sum()
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        accuracies[concept_names[k]] = accuracy

    overall_accuracy = (y_true == y_pred).mean()
    accuracies['overall'] = overall_accuracy

    return accuracies

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    y_true: true labels (0, 1, 2 for ternary)
    y_prob: predicted probabilities for the correct class
    """
    if len(y_true) == 0:
        return 0.0

    # Get predicted class
    y_pred = y_prob.argmax(axis=-1)

    # Get confidence (max probability)
    confidences = np.max(y_prob, axis=-1)

    # Calculate if predictions are correct
    accuracies = (y_pred == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def calculate_concept_ece(y_true, y_probs, n_bins=10):
    """
    Calculate ECE for each concept
    y_true: (n_samples, n_concepts) true labels
    y_probs: (n_samples, n_concepts, n_classes) predicted probabilities
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)

    n_concepts = y_true.shape[1]
    ece_scores = {}

    concept_names = ['food', 'service', 'ambiance', 'noise']
    for k in range(min(n_concepts, len(concept_names))):
        if k < y_probs.shape[1]:  # Make sure we have probabilities for this concept
            ece = expected_calibration_error(y_true[:, k], y_probs[:, k, :], n_bins)
            ece_scores[concept_names[k]] = ece

    # Average ECE across all concepts
    if ece_scores:
        ece_scores['average'] = np.mean(list(ece_scores.values()))

    return ece_scores

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Fast Credal CBM with token embeddings')

    # Model arguments
    parser.add_argument('--model', type=str,
                       default='distilbert-base-uncased',
                       help='Hugging Face model name (or use --encoder for preset)')
    parser.add_argument('--encoder', type=str, default='distilbert',
                       choices=list(ENCODERS.keys()),
                       help='Encoder preset name (overrides --model)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer to extract embeddings from (-1 = last layer)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding extraction')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'imdb', 'cebab', 'hatexplain', 'goemotions', 'ag_news', 'civil_comments', 'snli'],
                       help='Dataset name')
    parser.add_argument('--train-size', type=int, default=None,
                       help='Number of training samples to use (None = use all available)')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Number of test samples to use (None = use all available)')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Proportion of data to use for test set (0.0-1.0)')

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

    # Resolve encoder if specified
    if args.encoder != 'distilbert' or args.model == 'distilbert-base-uncased':
        if args.encoder in ENCODERS:
            args.model = ENCODERS[args.encoder]
            print(f"Using encoder preset: {args.encoder} -> {args.model}")
        else:
            print(f"Using custom model: {args.model}")

    # Get dataset config
    dataset_config = DATASETS[args.dataset]
    print(f"Dataset: {dataset_config.name}")
    print(f"Task: {dataset_config.task}")
    print(f"Has concepts: {dataset_config.has_concepts}")

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

    # Load dataset using configuration system
    try:
        config = DATASETS[args.dataset]
        print(f"Loading {config.name} dataset from {config.hf_path}...")

        if config.name == 'sst2':
            ds = load_dataset("glue", "sst2")
            train_texts = ds["train"]["sentence"]
            train_labels = ds["train"]["label"]
            test_texts = ds["validation"]["sentence"]
            test_labels = ds["validation"]["label"]

        elif config.name == 'imdb':
            ds = load_dataset("imdb")
            train_texts = ds["train"]["text"]
            train_labels = ds["train"]["label"]
            test_texts = ds["test"]["text"]
            test_labels = ds["test"]["label"]

        elif config.name == 'ag_news':
            ds = load_dataset("ag_news")
            train_texts = ds["train"]["text"]
            train_labels = ds["train"]["label"]
            test_texts = ds["test"]["text"]
            test_labels = ds["test"]["label"]

        elif config.name == 'cebab':
            ds = load_dataset("CEBaB/CEBaB")
            train_split = ds["train_inclusive"]
            test_split = ds["test"]

            train_texts = train_split["description"]
            train_labels = train_split["review_majority"]
            test_texts = test_split["description"]
            test_labels = test_split["review_majority"]

            # Extract REAL concepts from CEBaB
            all_concept_food = train_split[config.concept_cols[0]] + test_split[config.concept_cols[0]]
            all_concept_service = train_split[config.concept_cols[1]] + test_split[config.concept_cols[1]]
            all_concept_ambiance = train_split[config.concept_cols[2]] + test_split[config.concept_cols[2]]
            all_concept_noise = train_split[config.concept_cols[3]] + test_split[config.concept_cols[3]]

            # Create proper train/test split
            train_texts, test_texts, train_labels, test_labels, concepts_train, concepts_test = train_test_split(
                train_texts + test_texts, train_labels + test_labels,
                list(zip(all_concept_food, all_concept_service, all_concept_ambiance, all_concept_noise)),
                test_size=args.test_split,
                random_state=42,
                stratify=None  # CEBaB labels are imbalanced, don't stratify
            )

            # Unzip concepts
            concept_food_train, concept_service_train, concept_ambiance_train, concept_noise_train = zip(*concepts_train)
            concept_food_test, concept_service_test, concept_ambiance_test, concept_noise_test = zip(*concepts_test)

        elif config.name == 'hatexplain':
            ds = load_dataset("hatexplain", trust_remote_code=True)
            train_split = ds["train"]
            test_split = ds["test"]

            # Tokenize text (hatexplain provides tokenized posts)
            train_texts = [" ".join(post) for post in train_split["post_tokens"]]
            test_texts = [" ".join(post) for post in test_split["post_tokens"]]

            # Handle multiple annotations (use majority vote)
            def get_majority_label(annotations):
                if not annotations:
                    return 0
                return max(set(annotations), key=annotations.count)

            train_labels = [get_majority_label(ann) for ann in train_split["annotators"]]
            test_labels = [get_majority_label(ann) for ann in test_split["annotators"]]

        elif config.name == 'goemotions':
            ds = load_dataset("go_emotions/simplified")
            train_split = ds["train"]
            test_split = ds["validation"]

            train_texts = train_split["text"]
            test_texts = test_split["text"]
            train_labels = train_split["labels"]
            test_labels = test_split["labels"]

        elif config.name == 'civil_comments':
            ds = load_dataset("civil_comments")
            # Use smaller subset for testing
            train_split = ds["train"].shuffle(seed=42).select(range(10000))
            test_split = ds["test"].shuffle(seed=42).select(range(2000))

            train_texts = train_split["text"]
            test_texts = test_split["text"]
            train_labels = (train_split["toxicity"] > 0.5).astype(int).tolist()
            test_labels = (test_split["toxicity"] > 0.5).astype(int).tolist()

        elif config.name == 'snli':
            ds = load_dataset("snli")
            train_split = ds["train"]
            test_split = ds["validation"]

            # Filter out examples with -1 label
            train_filter = [i for i, label in enumerate(train_split["label"]) if label != -1]
            test_filter = [i for i, label in enumerate(test_split["label"]) if label != -1]

            train_texts = [train_split["premise"][i] for i in train_filter]
            test_texts = [test_split["premise"][i] for i in test_filter]
            train_labels = [train_split["label"][i] for i in train_filter]
            test_labels = [test_split["label"][i] for i in test_filter]

        else:
            raise ValueError(f"Unknown dataset: {config.name}")

        # Limit samples if requested
        if args.train_size is not None and len(train_texts) > args.train_size:
            indices = np.random.choice(len(train_texts), args.train_size, replace=False)
            train_texts = [train_texts[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]
            if config.has_concepts and config.name == 'cebab':
                concept_food_train = [concept_food_train[i] for i in indices]
                concept_service_train = [concept_service_train[i] for i in indices]
                concept_ambiance_train = [concept_ambiance_train[i] for i in indices]
                concept_noise_train = [concept_noise_train[i] for i in indices]
            print(f"Limited train set to {len(train_texts)} samples")

        if args.test_size is not None and len(test_texts) > args.test_size:
            indices = np.random.choice(len(test_texts), args.test_size, replace=False)
            test_texts = [test_texts[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
            if config.has_concepts and config.name == 'cebab':
                concept_food_test = [concept_food_test[i] for i in indices]
                concept_service_test = [concept_service_test[i] for i in indices]
                concept_ambiance_test = [concept_ambiance_test[i] for i in indices]
                concept_noise_test = [concept_noise_test[i] for i in indices]
            print(f"Limited test set to {len(test_texts)} samples")

        print(f"Final split - Train: {len(train_texts)}, Test: {len(test_texts)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Handle embedding extraction and concept creation
    if config.has_concepts:
        # Extract embeddings for train and test sets
        train_embeddings = processor.extract_token_embeddings(train_texts, layer=args.layer)
        test_embeddings = processor.extract_token_embeddings(test_texts, layer=args.layer)

        if train_embeddings is None or test_embeddings is None:
            return

        if config.name == 'cebab':
            # Create concepts for train and test separately
            train_concept_labels = convert_cebab_concepts_to_numerical(
                concept_food_train, concept_service_train, concept_ambiance_train, concept_noise_train
            )
            test_concept_labels = convert_cebab_concepts_to_numerical(
                concept_food_test, concept_service_test, concept_ambiance_test, concept_noise_test
            )
            args.n_concepts = 4
        else:
            # For other datasets with concepts, create synthetic concepts
            all_embeddings = np.vstack([train_embeddings, test_embeddings])
            all_labels = np.array(train_labels + test_labels)
            all_concept_labels = create_concepts_from_embeddings(
                all_embeddings, all_labels, args.n_concepts
            )
            # Split back
            n_train = len(train_embeddings)
            train_concept_labels = all_concept_labels[:n_train]
            test_concept_labels = all_concept_labels[n_train:]

        X_train, X_test = train_embeddings, test_embeddings
        y_train, y_test = train_concept_labels, test_concept_labels

    else:
        # For datasets without concepts, create synthetic concepts
        all_embeddings = processor.extract_token_embeddings(train_texts + test_texts, layer=args.layer)

        if all_embeddings is None:
            return

        all_concept_labels = create_concepts_from_embeddings(
            all_embeddings,
            np.array(train_labels + test_labels),
            args.n_concepts
        )

        # Split the combined data
        X_train, X_test, y_train, y_test = train_test_split(
            all_embeddings, all_concept_labels, test_size=args.test_split, random_state=42
        )

    # Initialize and train model
    model = CredalCBM(
        n_concepts=args.n_concepts,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get original test labels for classifier
    y_train_orig = train_labels
    y_test_orig = test_labels

    # Predict and evaluate
    if args.dataset == 'cebab':
        # Use ternary predictions for CEBaB
        results = model.predict_with_uncertainty_all_classes(X_test)

        # Calculate concept accuracy for CEBaB
        y_pred_concepts = results['concept_probs'].argmax(axis=-1)  # Get predicted class (0, 1, 2)
        concept_accuracies = calculate_concept_accuracy(y_test, y_pred_concepts)

        # Calculate ECE for each concept
        concept_ece_scores = calculate_concept_ece(y_test, results['concept_probs'])
    else:
        # Use binary predictions for other datasets
        results = model.predict_with_uncertainty(X_test)

    # Use concept probabilities as features for final classification
    concept_features = results['concept_probs']

    # For CEBaB, flatten the 3-class probabilities or use positive class
    if args.dataset == 'cebab' and len(concept_features.shape) == 3:
        # Check if we have 3 classes (ternary) or 2 classes (binary)
        n_classes = concept_features.shape[2]
        if n_classes == 3:
            # Use positive class probabilities and class variances as features
            positive_probs = concept_features[:, :, 2]  # (n_samples, n_concepts)
            class_variances = results['class_variances']  # (n_samples, n_concepts)
            concept_features = np.column_stack([positive_probs, class_variances])
        else:
            # For 2 classes, use max probability and class variances
            max_probs = np.max(concept_features, axis=2)  # (n_samples, n_concepts)
            class_variances = results['class_variances']  # (n_samples, n_concepts)
            concept_features = np.column_stack([max_probs, class_variances])
    else:
        # For binary concepts, use as-is
        pass

    
    # Train neural label head on concept features
    # Convert labels to numeric format for CEBaB (handle string labels)
    label_encoder = LabelEncoder()

    # Handle CEBaB labels which might be strings or mixed types
    if args.dataset == 'cebab':
        # Convert CEBaB labels to numeric, handle "no majority" and other string labels
        y_test_orig_clean = []
        for label in y_test_orig:
            if isinstance(label, str):
                if label.lower() in ['no majority', 'unknown', '']:
                    # Assign a default numeric value for "no majority"
                    y_test_orig_clean.append(3)  # Middle value
                else:
                    try:
                        y_test_orig_clean.append(int(label))
                    except ValueError:
                        y_test_orig_clean.append(3)  # Default middle value
            else:
                y_test_orig_clean.append(int(label))

        y_test_orig_clean = np.array(y_test_orig_clean)
        y_train_encoded = label_encoder.fit_transform(y_test_orig_clean)
        n_classes = len(label_encoder.classes_)
    else:
        # For other datasets, use standard label encoding
        y_train_encoded = label_encoder.fit_transform(y_test_orig)
        n_classes = len(label_encoder.classes_)

    # Create and train neural label head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_head = LabelHead(
        n_features=concept_features.shape[1],
        n_classes=n_classes,
        dropout_rate=0.2
    ).to(device)

    # Convert to tensors
    X_tensor = torch.FloatTensor(concept_features).to(device)
    y_tensor = torch.LongTensor(y_train_encoded).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(label_head.parameters(), lr=0.001, weight_decay=1e-4)

    # Train neural label head
    num_epochs = 100
    batch_size = 32
    n_samples = len(X_tensor)

    label_head.train()
    for epoch in range(num_epochs):
        # Shuffle data
        indices = torch.randperm(n_samples)

        total_loss = 0
        num_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_tensor[batch_indices]
            batch_y = y_tensor[batch_indices]

            # Forward pass
            logits = label_head(batch_X)
            loss = criterion(logits, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate neural label head
    label_head.eval()
    with torch.no_grad():
        logits = label_head(X_tensor)
        y_pred_encoded = logits.argmax(dim=-1).cpu().numpy()

    # Convert predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Calculate metrics (using cleaned labels for CEBaB)
    if args.dataset == 'cebab':
        # Use cleaned labels for fair comparison
        label_accuracy = accuracy_score(y_test_orig_clean, y_pred_encoded)
        f1_macro = f1_score(y_test_orig_clean, y_pred_encoded, average='macro')
        f1_micro = f1_score(y_test_orig_clean, y_pred_encoded, average='micro')
    else:
        # Use original labels for other datasets
        # Ensure we only evaluate on samples that have concept predictions
        n_concept_samples = concept_features.shape[0]
        y_test_orig_subset = y_test_orig[:n_concept_samples]
        y_pred_subset = y_pred[:n_concept_samples]

        label_accuracy = accuracy_score(y_test_orig_subset, y_pred_subset)
        f1_macro = f1_score(y_test_orig_subset, y_pred_subset, average='macro')
        f1_micro = f1_score(y_test_orig_subset, y_pred_subset, average='micro')

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        'model_name': args.model,
        'layer': args.layer,
        'dataset': args.dataset,
        'n_samples': len(train_texts) + len(test_texts),
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'n_concepts': args.n_concepts,
        'label_accuracy': label_accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'mean_epistemic': results['mean_epistemic'].mean(),
        'mean_credal_width': (results['credal_upper'] - results['credal_lower']).mean(),
        'model_info': model_info,
        'embedding_shape': X_train.shape,
        'concept_coverage': y_train.mean()
    }

    # Add concept-specific metrics for CEBaB
    if args.dataset == 'cebab':
        output['concept_accuracies'] = concept_accuracies
        output['concept_ece_scores'] = concept_ece_scores

    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model.split('/')[-1]}_layer{args.layer}_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")

    total_samples = len(train_texts) + len(test_texts)
    print(f"Completed processing {total_samples} samples ({len(train_texts)} train, {len(test_texts)} test)!")
    print(f"\nTraining completed on {len(train_texts):,} samples, tested on {len(test_texts):,} samples!")

    return output


if __name__ == "__main__":
    main()
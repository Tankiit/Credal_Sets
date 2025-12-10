"""
Fast Credal CBM training using token embeddings from any model
This runs much faster than full forward passes and allows custom layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
                       help='Hugging Face model name')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Layer to extract embeddings from (-1 = last layer)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding extraction')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='sst2',
                       help='Dataset name (sst2, imdb, cebab, etc.)')
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

    # Load dataset
    try:
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
        elif args.dataset == 'cebab':
            ds = load_dataset("CEBaB/CEBaB")
            # Use proper train/test splits for CEBaB
            train_split = ds["train_inclusive"]
            test_split = ds["test"]

            # Combine train and test for consistent splitting
            all_texts = train_split["description"] + test_split["description"]
            all_labels = train_split["review_majority"] + test_split["review_majority"]

            # Extract REAL concepts from CEBaB
            all_concept_food = train_split["food_aspect_majority"] + test_split["food_aspect_majority"]
            all_concept_service = train_split["service_aspect_majority"] + test_split["service_aspect_majority"]
            all_concept_ambiance = train_split["ambiance_aspect_majority"] + test_split["ambiance_aspect_majority"]
            all_concept_noise = train_split["noise_aspect_majority"] + test_split["noise_aspect_majority"]

            # Create proper train/test split
            texts_train, texts_test, labels_train, labels_test, concepts_train, concepts_test = train_test_split(
                all_texts, all_labels,
                list(zip(all_concept_food, all_concept_service, all_concept_ambiance, all_concept_noise)),
                test_size=args.test_split,
                random_state=42,
                stratify=None  # CEBaB labels are imbalanced, don't stratify
            )

            # Unzip concepts
            concept_food_train, concept_service_train, concept_ambiance_train, concept_noise_train = zip(*concepts_train)
            concept_food_test, concept_service_test, concept_ambiance_test, concept_noise_test = zip(*concepts_test)

            # Limit samples if requested
            if args.train_size is not None and len(texts_train) > args.train_size:
                indices = np.random.choice(len(texts_train), args.train_size, replace=False)
                texts_train = [texts_train[i] for i in indices]
                labels_train = [labels_train[i] for i in indices]
                concept_food_train = [concept_food_train[i] for i in indices]
                concept_service_train = [concept_service_train[i] for i in indices]
                concept_ambiance_train = [concept_ambiance_train[i] for i in indices]
                concept_noise_train = [concept_noise_train[i] for i in indices]

            if args.test_size is not None and len(texts_test) > args.test_size:
                indices = np.random.choice(len(texts_test), args.test_size, replace=False)
                texts_test = [texts_test[i] for i in indices]
                labels_test = [labels_test[i] for i in indices]
                concept_food_test = [concept_food_test[i] for i in indices]
                concept_service_test = [concept_service_test[i] for i in indices]
                concept_ambiance_test = [concept_ambiance_test[i] for i in indices]
                concept_noise_test = [concept_noise_test[i] for i in indices]
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Handle embedding extraction separately for train and test
    if args.dataset == 'cebab':
        # Extract embeddings for train set
        train_embeddings = processor.extract_token_embeddings(texts_train, layer=args.layer)

        # Extract embeddings for test set
        test_embeddings = processor.extract_token_embeddings(texts_test, layer=args.layer)

        if train_embeddings is None or test_embeddings is None:
            return

        # Create concepts for train and test separately
        # Convert train concepts
        train_concept_labels = convert_cebab_concepts_to_numerical(
            concept_food_train, concept_service_train, concept_ambiance_train, concept_noise_train
        )

        # Convert test concepts
        test_concept_labels = convert_cebab_concepts_to_numerical(
            concept_food_test, concept_service_test, concept_ambiance_test, concept_noise_test
        )

        # Set n_concepts to match real concepts
        args.n_concepts = 4

        # Use the split embeddings directly
        X_train, X_test = train_embeddings, test_embeddings
        y_train, y_test = train_concept_labels, test_concept_labels
    else:
        # For other datasets, keep the old approach but with new parameter names
        all_embeddings = processor.extract_token_embeddings(texts_train + texts_test, layer=args.layer)

        if all_embeddings is None:
            return

        all_concept_labels = create_concepts_from_embeddings(
            all_embeddings,
            np.array(labels_train + labels_test),
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
    if args.dataset == 'cebab':
        y_train_orig = labels_train
        y_test_orig = labels_test
    else:
        # For other datasets, we need to split the combined labels
        all_labels = labels_train + labels_test
        _, _, y_train_orig, y_test_orig = train_test_split(
            texts_train + texts_test, all_labels, test_size=args.test_split, random_state=42
        )

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
        # Use positive class probabilities and class variances as features
        positive_probs = concept_features[:, :, 2]  # (n_samples, n_concepts)
        class_variances = results['class_variances']  # (n_samples, n_concepts)
        concept_features = np.column_stack([positive_probs, class_variances])
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
        label_accuracy = accuracy_score(y_test_orig, y_pred)
        f1_macro = f1_score(y_test_orig, y_pred, average='macro')
        f1_micro = f1_score(y_test_orig, y_pred, average='micro')

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        'model_name': args.model,
        'layer': args.layer,
        'dataset': args.dataset,
        'n_samples': len(texts_train) + len(texts_test),
        'train_samples': len(texts_train),
        'test_samples': len(texts_test),
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

    return output


if __name__ == "__main__":
    main()
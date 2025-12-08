"""
Training script for Credal CBM experiments
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from config import (
    ExperimentConfig, ConfigManager,
    get_dataset_info, get_encoder_info, get_credal_info
)
from credal_cbm_main import CredalCBM
from dataclasses import asdict


class DataProcessor:
    """Handle data loading and preprocessing for different datasets"""

    def __init__(self, dataset_config, encoder_config):
        self.dataset_config = dataset_config
        self.encoder_config = encoder_config
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def load_huggingface_data(self):
        """Load data from Hugging Face datasets"""
        import datasets
        print(f"Loading {self.dataset_config.name} from Hugging Face...")

        dataset = datasets.load_dataset(
            self.dataset_config.hf_path,
            split=self.dataset_config.train_split
        )

        # Extract texts and labels
        texts = [item[self.dataset_config.text_column] for item in dataset]

        if self.dataset_config.task_type.value == "nli":
            # Handle NLI datasets with premise/hypothesis
            premises = [item['premise'] for item in dataset]
            hypotheses = [item['hypothesis'] for item in dataset]
            texts = [f"{prem} [SEP] {hyp}" for prem, hyp in zip(premises, hypotheses)]

        labels = [item[self.dataset_config.label_column] for item in dataset]

        # For multi-label tasks (like GoEmotions)
        if self.dataset_config.name == "GoEmotions":
            # Convert multi-label to single-label for simplicity
            labels = [lbl[0] if lbl else 0 for lbl in labels]

        return texts, labels

  
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the specified encoder"""
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f"Encoding texts with {self.encoder_config.name}...")

        tokenizer = AutoTokenizer.from_pretrained(self.encoder_config.hf_path)
        model = AutoModel.from_pretrained(self.encoder_config.hf_path)
        model.eval()

        features = []
        batch_size = self.encoder_config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.encoder_config.max_length,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token representation
                batch_features = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(batch_features)

        return np.vstack(features)

    def process_labels(self, labels: List[Any]) -> np.ndarray:
        """Process labels into encoded format"""
        if not self.is_fitted:
            self.label_encoder.fit(labels)
            self.is_fitted = True

        return self.label_encoder.transform(labels)

    def split_data(self, texts: List[str], labels: List[Any],
                   train_size: float, val_size: float, test_size: float) -> Tuple:
        """Split data into train/val/test sets"""
        # Ensure proportions sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6

        # First split: train + val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            texts, labels, test_size=test_size,
            stratify=labels, random_state=42
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            stratify=y_train_val, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_task_concepts(self, X_features: np.ndarray, y_labels: np.ndarray, n_concepts: int) -> np.ndarray:
        """Create task-based synthetic concepts that are more meaningful than random"""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        print("Creating task-based concepts using feature clustering...")

        concept_labels = np.zeros((len(X_features), n_concepts))

        # Method 1: Clustering-based concepts
        n_clusters = min(n_concepts // 2, len(np.unique(y_labels)))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_features)

            # Create binary concepts from clusters
            for i in range(n_clusters):
                concept_labels[:, i] = (clusters == i).astype(int)

        # Method 2: PCA-based concepts
        remaining_concepts = n_concepts - n_clusters
        if remaining_concepts > 0:
            pca = PCA(n_components=remaining_concepts, random_state=42)
            pca_features = pca.fit_transform(X_features)

            # Convert PCA features to binary concepts using median threshold
            for i in range(remaining_concepts):
                concept_idx = n_clusters + i
                threshold = np.median(pca_features[:, i])
                concept_labels[:, concept_idx] = (pca_features[:, i] > threshold).astype(int)

        return concept_labels


class EnhancedCredalCBMTrainer:
    """
    Enhanced Credal CBM Trainer with epoch-based training and progress tracking
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.models = {}

    def train_with_epochs(
        self,
        model: CredalCBM,
        X_train: np.ndarray,
        concept_labels_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        concept_labels_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 10,
        patience: int = 3,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train Credal CBM with epoch-based training loop

        Args:
            model: CredalCBM model instance
            X_train: Training features
            concept_labels_train: Training concept labels
            y_train: Training task labels
            X_val: Validation features (optional)
            concept_labels_val: Validation concept labels (optional)
            y_val: Validation task labels (optional)
            epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        print(f"Starting Enhanced Credal CBM training for {epochs} epochs...")

        # Initialize training history
        training_history = {
            'concept_losses': [],
            'overall_losses': [],
            'concept_accuracies': [],
            'val_concept_losses': [],
            'val_overall_losses': [],
            'val_concept_accuracies': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        # Progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training Epochs", disable=not verbose)

        for epoch in epoch_pbar:
            # Train concept predictors
            epoch_metrics = self._train_epoch(
                model, X_train, concept_labels_train, y_train, verbose
            )

            # Store training metrics
            training_history['concept_losses'].append(epoch_metrics['concept_loss'])
            training_history['overall_losses'].append(epoch_metrics['overall_loss'])
            training_history['concept_accuracies'].append(epoch_metrics['concept_accuracy'])

            # Validation if available
            if X_val is not None and concept_labels_val is not None:
                val_metrics = self._validate_epoch(
                    model, X_val, concept_labels_val, y_val, verbose
                )

                training_history['val_concept_losses'].append(val_metrics['concept_loss'])
                training_history['val_overall_losses'].append(val_metrics['overall_loss'])
                training_history['val_concept_accuracies'].append(val_metrics['concept_accuracy'])

                # Early stopping
                current_val_loss = val_metrics['overall_loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    break

                # Update progress bar description
                epoch_pbar.set_postfix({
                    'Train Loss': f"{epoch_metrics['concept_loss']:.4f}",
                    'Val Loss': f"{val_metrics['concept_loss']:.4f}",
                    'Patience': f"{patience_counter}/{patience}"
                })
            else:
                # Update progress bar description for train only
                epoch_pbar.set_postfix({
                    'Train Loss': f"{epoch_metrics['concept_loss']:.4f}",
                    'Train Acc': f"{epoch_metrics['concept_accuracy']:.3f}"
                })

        epoch_pbar.close()
        print(f"\n✅ Training completed!")

        return training_history

    def _train_epoch(
        self,
        model: CredalCBM,
        X: np.ndarray,
        concept_labels: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Train for one epoch with progress tracking"""

        # Scale features
        if not model.is_fitted:
            X_scaled = model.scaler.fit_transform(X)
            model.is_fitted = True
        else:
            X_scaled = model.scaler.transform(X)

        # Train concept models
        if not model.concept_models:
            model.concept_models = []

            # Progress bar for initial training
            concept_pbar = tqdm(
                range(model.n_concepts),
                desc="Training Concepts",
                disable=not verbose,
                leave=False
            )

            for k in concept_pbar:
                y_k = concept_labels[:, k]

                if len(np.unique(y_k)) < 2:
                    continue

                # Create model with warm_start for incremental learning
                rf_model = RandomForestClassifier(
                    n_estimators=model.n_estimators,
                    max_depth=model.max_depth,
                    min_samples_leaf=5,
                    random_state=model.random_state + k,
                    n_jobs=-1,
                    warm_start=True
                )

                rf_model.fit(X_scaled, y_k)
                model.concept_models.append(rf_model)

                concept_pbar.set_postfix({'Concept': f"{k+1}/{model.n_concepts}"})

            concept_pbar.close()
        else:
            # Incremental learning: add more trees to existing models
            for k, rf_model in enumerate(model.concept_models):
                if k < concept_labels.shape[1]:
                    y_k = concept_labels[:, k]

                    # Add more trees to existing model
                    current_trees = len(rf_model.estimators_)
                    target_trees = min(current_trees + 10, model.n_estimators * 2)

                    if current_trees < target_trees:
                        rf_model.n_estimators = target_trees
                        rf_model.fit(X_scaled, y_k)

        # Calculate training metrics
        return self._calculate_epoch_metrics(model, X, concept_labels, y, "train")

    def _validate_epoch(
        self,
        model: CredalCBM,
        X: np.ndarray,
        concept_labels: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Validate for one epoch"""
        return self._calculate_epoch_metrics(model, X, concept_labels, y, "val")

    def _calculate_epoch_metrics(
        self,
        model: CredalCBM,
        X: np.ndarray,
        concept_labels: np.ndarray,
        y: np.ndarray,
        mode: str = "train"
    ) -> Dict[str, float]:
        """Calculate metrics for an epoch"""

        X_scaled = model.scaler.transform(X) if model.is_fitted else X

        # Get concept predictions
        if not model.concept_models:
            return {'concept_loss': 1.0, 'overall_loss': 1.0, 'concept_accuracy': 0.0}

        concept_predictions = []
        concept_losses = []

        # Progress bar for prediction
        pred_pbar = tqdm(
            model.concept_models,
            desc=f"Calculating {mode} metrics",
            disable=True,  # Disable to avoid too much output
            leave=False
        )

        for k, rf_model in enumerate(pred_pbar):
            if k < concept_labels.shape[1]:
                y_k = concept_labels[:, k]

                # Predict probabilities
                try:
                    probs = rf_model.predict_proba(X_scaled)[:, 1]
                    pred_labels = (probs > 0.5).astype(int)

                    # Calculate binary cross-entropy loss
                    eps = 1e-15  # Prevent log(0)
                    loss = -np.mean(y_k * np.log(probs + eps) + (1 - y_k) * np.log(1 - probs + eps))
                    concept_losses.append(loss)

                    concept_predictions.append(probs)
                except:
                    concept_losses.append(1.0)
                    concept_predictions.append(np.zeros(len(X)))

        pred_pbar.close()

        # Overall metrics
        avg_concept_loss = np.mean(concept_losses) if concept_losses else 1.0

        # Train simple classifier on concept predictions for task performance
        if len(concept_predictions) > 0:
            concept_probs = np.column_stack(concept_predictions)
            try:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(random_state=model.random_state, max_iter=1000)
                clf.fit(concept_probs, y)
                task_accuracy = clf.score(concept_probs, y)
            except:
                task_accuracy = 0.0
        else:
            task_accuracy = 0.0

        # Overall loss combines concept loss and task performance
        overall_loss = avg_concept_loss + (1.0 - task_accuracy)

        # Calculate concept accuracy
        concept_acc = 0.0
        if model.concept_models:
            accs = []
            for k, rf_model in enumerate(model.concept_models):
                if k < concept_labels.shape[1]:
                    y_k = concept_labels[:, k]
                    pred_labels = (rf_model.predict(X_scaled) > 0.5).astype(int)
                    acc = np.mean(pred_labels == y_k)
                    accs.append(acc)
            concept_acc = np.mean(accs) if accs else 0.0

        return {
            'concept_loss': avg_concept_loss,
            'overall_loss': overall_loss,
            'concept_accuracy': concept_acc
        }

    def end_to_end_train(
        self,
        model: CredalCBM,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 10,
        auto_concepts: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        End-to-end training with automatic concept discovery

        Args:
            model: CredalCBM model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            auto_concepts: Whether to automatically generate concepts
            verbose: Whether to print progress
        """

        if auto_concepts:
            print("🔍 Discovering concepts from data...")

            # Progress bar for concept discovery
            concept_pbar = tqdm(
                desc="Discovering Concepts",
                total=3,  # Three methods of concept discovery
                disable=not verbose
            )

            concept_labels_train = self._discover_concepts(model, X_train, y_train)
            concept_pbar.update(1)

            if X_val is not None:
                concept_labels_val = self._discover_concepts(model, X_val, y_val)
                concept_pbar.update(1)
            else:
                concept_labels_val = None
                concept_pbar.update(1)

            concept_pbar.update(1)  # Final step
            concept_pbar.close()
        else:
            # Assume concept labels are provided externally
            print("📋 Using provided concept labels...")
            raise NotImplementedError("External concept labels not implemented yet")

        # Train with discovered concepts
        print("🚀 Starting training with discovered concepts...")
        history = self.train_with_epochs(
            model=model,
            X_train=X_train,
            concept_labels_train=concept_labels_train,
            y_train=y_train,
            X_val=X_val,
            concept_labels_val=concept_labels_val,
            y_val=y_val,
            epochs=epochs,
            patience=3,
            verbose=verbose
        )

        return history

    def _discover_concepts(
        self,
        model: CredalCBM,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Automatically discover concepts from data using clustering and PCA
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        print(f"Discovering {model.n_concepts} concepts from {X.shape[0]} samples...")

        concept_labels = np.zeros((X.shape[0], model.n_concepts))

        # Method 1: Clustering-based concepts
        n_clusters = min(model.n_concepts // 2, len(np.unique(y)))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=model.random_state, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Create binary concepts from clusters
            for i in range(n_clusters):
                concept_labels[:, i] = (clusters == i).astype(int)

        # Method 2: PCA-based concepts
        remaining_concepts = model.n_concepts - n_clusters
        if remaining_concepts > 0:
            pca = PCA(n_components=remaining_concepts, random_state=model.random_state)
            pca_features = pca.fit_transform(X)

            # Convert PCA features to binary concepts
            for i in range(remaining_concepts):
                concept_idx = n_clusters + i
                threshold = np.median(pca_features[:, i])
                concept_labels[:, concept_idx] = (pca_features[:, i] > threshold).astype(int)

        # Method 3: Task-aligned concepts (based on labels)
        if model.n_concepts > n_clusters + remaining_concepts:
            task_concept_idx = n_clusters + remaining_concepts
            if task_concept_idx < model.n_concepts:
                concept_labels[:, task_concept_idx] = y.astype(int)

        print(f"✅ Concept coverage: {concept_labels.mean():.3f}")
        return concept_labels

    def plot_training_history(
        self,
        training_history: Dict[str, List[float]],
        save_path: str = None
    ):
        """Plot training history with enhanced visualization"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credal CBM Training History', fontsize=16, fontweight='bold')

        # Concept loss
        axes[0, 0].plot(training_history['concept_losses'], label='Train', color='blue', linewidth=2, marker='o', markersize=3)
        if training_history['val_concept_losses']:
            axes[0, 0].plot(training_history['val_concept_losses'], label='Validation', color='red', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_title('Concept Loss Over Epochs', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Overall loss
        axes[0, 1].plot(training_history['overall_losses'], label='Train', color='blue', linewidth=2, marker='o', markersize=3)
        if training_history['val_overall_losses']:
            axes[0, 1].plot(training_history['val_overall_losses'], label='Validation', color='red', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title('Overall Loss Over Epochs', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Concept accuracy
        axes[1, 0].plot(training_history['concept_accuracies'], label='Train', color='blue', linewidth=2, marker='o', markersize=3)
        if training_history['val_concept_accuracies']:
            axes[1, 0].plot(training_history['val_concept_accuracies'], label='Validation', color='red', linewidth=2, marker='s', markersize=3)
        axes[1, 0].set_title('Concept Accuracy Over Epochs', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Combined view
        axes[1, 1].plot(training_history['concept_losses'], label='Concept Loss', alpha=0.7, linewidth=2)
        axes[1, 1].plot(training_history['overall_losses'], label='Overall Loss', alpha=0.7, linewidth=2)
        if training_history['val_concept_losses']:
            axes[1, 1].plot(training_history['val_concept_losses'], label='Val Concept Loss', alpha=0.7, linewidth=2, linestyle='--')
            axes[1, 1].plot(training_history['val_overall_losses'], label='Val Overall Loss', alpha=0.7, linewidth=2, linestyle='--')
        axes[1, 1].set_title('All Losses Combined', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plot saved to {save_path}")

        plt.show()

    def run_single_experiment(self, dataset_name: str, encoder_name: str,
                            credal_method: str) -> Dict[str, Any]:
        """Run a single experiment configuration (compatible with original trainer)"""
        # Get configurations
        dataset_config = get_dataset_info(dataset_name)
        encoder_config = get_encoder_info(encoder_name)
        credal_config = get_credal_info(credal_method)

        # Initialize data processor
        processor = DataProcessor(dataset_config, encoder_config)

        # Load and process data
        try:
            print(f"\n[1] Loading and processing {dataset_name}...")
            texts, labels = processor.load_huggingface_data()

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                texts, labels,
                self.config.train_size,
                self.config.val_size,
                self.config.test_size
            )

            # Encode labels
            y_train_enc = processor.process_labels(y_train)
            y_val_enc = processor.process_labels(y_val)
            y_test_enc = processor.process_labels(y_test)

            print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        except Exception as e:
            print(f"Error processing data: {e}")
            return {"error": str(e)}

        # Encode texts
        try:
            print(f"[2] Encoding texts with {encoder_name}...")
            X_train_enc = processor.encode_texts(X_train)
            X_val_enc = processor.encode_texts(X_val)
            X_test_enc = processor.encode_texts(X_test)

        except Exception as e:
            print(f"Error encoding texts: {e}")
            return {"error": str(e)}

        # Create concept labels
        try:
            print(f"[3] Creating concept labels...")
            n_concepts = 10
            if dataset_config.has_concepts and dataset_config.concept_columns:
                concept_labels_train = processor.create_task_concepts(X_train_enc, y_train_enc, n_concepts)
                concept_labels_val = processor.create_task_concepts(X_val_enc, y_val_enc, n_concepts)
            else:
                concept_labels_train = self._discover_concepts(None, X_train_enc, y_train_enc)
                concept_labels_val = self._discover_concepts(None, X_val_enc, y_val_enc)

        except Exception as e:
            print(f"Error creating concepts: {e}")
            return {"error": str(e)}

        # Initialize and train model
        try:
            print(f"[4] Training Credal CBM...")
            model = CredalCBM(
                n_concepts=n_concepts,
                n_estimators=credal_config.n_estimators,
                max_depth=credal_config.max_depth,
                random_state=self.config.seed
            )

            # Train the model
            model.fit(X_train_enc, concept_labels_train)

            # For enhanced training, also run epoch-based training
            if hasattr(self, 'epochs'):
                print(f"[5] Running enhanced epoch-based training...")
                training_history = self.train_with_epochs(
                    model=model,
                    X_train=X_train_enc,
                    concept_labels_train=concept_labels_train,
                    y_train=y_train_enc,
                    X_val=X_val_enc,
                    concept_labels_val=concept_labels_val,
                    y_val=y_val_enc,
                    epochs=getattr(self, 'epochs', 3),
                    patience=getattr(self, 'patience', 3)
                )

        except Exception as e:
            print(f"Error training model: {e}")
            return {"error": str(e)}

        # Evaluate model
        try:
            print(f"[6] Evaluating model...")
            results = self.evaluate_credal_model(
                model, X_test_enc, y_test_enc, y_test,
                processor.label_encoder, {}
            )

            # Add metadata to results
            results.update({
                "dataset": dataset_name,
                "encoder": encoder_name,
                "credal_method": credal_method,
                "dataset_config": asdict(dataset_config),
                "encoder_config": asdict(encoder_config),
                "credal_config": asdict(credal_config),
                "n_samples": len(X_test),
                "n_features": X_test_enc.shape[1]
            })

            if 'training_history' in locals():
                results['training_history'] = training_history

        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {"error": str(e)}

        return results

    def evaluate_credal_model(self, model, X_test, y_test_enc, y_test_orig,
                             label_encoder, gpu_info) -> Dict[str, Any]:
        """Evaluate the Credal CBM model"""

        # Get predictions with uncertainty
        pred_results = model.predict_with_uncertainty(X_test)

        # Use concept predictions for final classification
        concept_probs = pred_results['concept_probs']

        # Train classifier on concept probabilities
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(concept_probs, y_test_enc)
        y_pred = classifier.predict(concept_probs)

        # Calculate metrics
        accuracy = accuracy_score(y_test_enc, y_pred)
        f1_macro = f1_score(y_test_enc, y_pred, average='macro')
        f1_micro = f1_score(y_test_enc, y_pred, average='micro')

        # Uncertainty metrics
        epistemic_uncertainty = pred_results['mean_epistemic']
        credal_width = (pred_results['credal_upper'] - pred_results['credal_lower']).mean(axis=1)
        concept_disagreement = pred_results['epistemic_uncertainty'].mean(axis=1)

        # Correlation metrics
        from scipy.stats import spearmanr
        uncertainty_corr, p_value = spearmanr(epistemic_uncertainty, concept_disagreement)

        results = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "mean_epistemic": epistemic_uncertainty.mean(),
            "std_epistemic": epistemic_uncertainty.std(),
            "mean_credal_width": credal_width.mean(),
            "uncertainty_correlation": uncertainty_corr,
            "uncertainty_pvalue": p_value,
            "precise_predictions": (credal_width < 0.1).mean(),
            "high_uncertainty": (epistemic_uncertainty > 0.2).mean(),
        }

        return results

    def save_results(self, results: Dict, output_dir: str):
        """Save experiment results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"enhanced_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        def convert_recursive(obj):
            if isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(item) for item in obj]
            else:
                return convert_numpy(obj)

        json_results = convert_recursive(results)

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"✅ Enhanced results saved to {results_file}")

        # Save summary
        summary_file = os.path.join(output_dir, f"enhanced_summary_{timestamp}.txt")
        self.save_summary(results, summary_file)

    def save_summary(self, results: Dict, filepath: str):
        """Save a human-readable summary"""
        with open(filepath, 'w') as f:
            f.write("ENHANCED CREDAL CBM EXPERIMENT RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for exp_key, exp_result in results.items():
                if isinstance(exp_result, dict) and 'error' not in exp_result:
                    f.write(f"Experiment: {exp_key}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Dataset: {exp_result.get('dataset', 'N/A')}\n")
                    f.write(f"Encoder: {exp_result.get('encoder', 'N/A')}\n")
                    f.write(f"Credal Method: {exp_result.get('credal_method', 'N/A')}\n\n")

                    # Performance metrics
                    f.write("Performance Metrics:\n")
                    f.write(f"  Accuracy: {exp_result.get('accuracy', 0):.3f}\n")
                    f.write(f"  F1-Macro: {exp_result.get('f1_macro', 0):.3f}\n")
                    f.write(f"  F1-Micro: {exp_result.get('f1_micro', 0):.3f}\n\n")

                    # Uncertainty metrics
                    f.write("Uncertainty Metrics:\n")
                    f.write(f"  Mean Epistemic: {exp_result.get('mean_epistemic', 0):.3f}\n")
                    f.write(f"  Mean Credal Width: {exp_result.get('mean_credal_width', 0):.3f}\n")
                    f.write(f"  Uncertainty Correlation: {exp_result.get('uncertainty_correlation', 0):.3f}\n\n")

                    # Enhanced training metrics
                    if 'final_train_loss' in exp_result:
                        f.write("Enhanced Training Metrics:\n")
                        f.write(f"  Final Train Loss: {exp_result.get('final_train_loss', 0):.4f}\n")
                        f.write(f"  Epochs Completed: {exp_result.get('epochs', 0)}\n\n")

                    f.write("\n" + "=" * 50 + "\n\n")


class CredalCBMTrainer:
    """Main training class for Credal CBM experiments"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.models = {}

    def run_single_experiment(self, dataset_name: str, encoder_name: str,
                            credal_method: str) -> Dict[str, Any]:
        """Run a single experiment configuration"""
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {dataset_name} | {encoder_name} | {credal_method}")
        print(f"{'='*80}")

        # Get configurations
        dataset_config = get_dataset_info(dataset_name)
        encoder_config = get_encoder_info(encoder_name)
        credal_config = get_credal_info(credal_method)

        # Initialize data processor
        processor = DataProcessor(dataset_config, encoder_config)

        # Load and process data
        try:
            texts, labels = processor.load_huggingface_data()

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                texts, labels,
                self.config.train_size,
                self.config.val_size,
                self.config.test_size
            )

            # Encode labels
            y_train_enc = processor.process_labels(y_train)
            y_val_enc = processor.process_labels(y_val)
            y_test_enc = processor.process_labels(y_test)

            print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        except Exception as e:
            print(f"Error processing data: {e}")
            return {"error": str(e)}

        # Encode texts first (needed for concept creation)
        print("Encoding texts...")
        X_train_enc = processor.encode_texts(X_train)
        X_val_enc = processor.encode_texts(X_val)
        X_test_enc = processor.encode_texts(X_test)

        # Load concept labels if available, otherwise use task-based concepts
        n_concepts = 10
        n_train = len(X_train)

        if dataset_config.has_concepts and dataset_config.concept_columns:
            # Load real concept annotations from dataset
            print(f"Loading concept annotations for {dataset_config.name}...")
            try:
                import datasets
                full_dataset = datasets.load_dataset(
                    dataset_config.hf_path,
                    split=dataset_config.train_split
                )

                # Extract concept columns
                concept_labels_train = []
                for item in full_dataset[:n_train]:
                    concept_row = []
                    for concept_col in dataset_config.concept_columns:
                        # Convert to binary if needed
                        concept_val = item.get(concept_col, 0)
                        if isinstance(concept_val, str):
                            # Handle categorical concepts
                            concept_val = 1 if concept_val.lower() in ['yes', 'true', '1', 'positive'] else 0
                        elif isinstance(concept_val, (int, float)):
                            concept_val = 1 if concept_val > 0.5 else 0
                        concept_row.append(concept_val)
                    concept_labels_train.append(concept_row)

                concept_labels_train = np.array(concept_labels_train)
                print(f"Loaded {len(dataset_config.concept_columns)} real concepts")

            except Exception as e:
                print(f"Could not load concept annotations: {e}")
                print("Using task-based synthetic concepts...")
                concept_labels_train = processor.create_task_concepts(X_train_enc, y_train_enc, n_concepts)
        else:
            # Create task-based synthetic concepts (more meaningful than random)
            print(f"Creating {n_concepts} task-based synthetic concepts...")
            concept_labels_train = processor.create_task_concepts(X_train_enc, y_train_enc, n_concepts)

        # Initialize and train Credal CBM
        print("Training Credal CBM...")
        credal_model = CredalCBM(
            n_concepts=n_concepts,
            n_estimators=credal_config.n_estimators,
            max_depth=credal_config.max_depth,
            random_state=self.config.seed
        )

        credal_model.fit(X_train_enc, concept_labels_train)

        # Evaluate on test set
        print("Evaluating model...")
        results = self.evaluate_credal_model(
            credal_model, X_test_enc, y_test_enc, y_test, processor.label_encoder
        )

        # Add metadata to results
        results.update({
            "dataset": dataset_name,
            "encoder": encoder_name,
            "credal_method": credal_method,
            "dataset_config": asdict(dataset_config),
            "encoder_config": asdict(encoder_config),
            "credal_config": asdict(credal_config),
            "n_samples": len(X_test),
            "n_features": X_test_enc.shape[1]
        })

        return results

    def evaluate_credal_model(self, model, X_test, y_test_enc, y_test_orig, label_encoder):
        """Evaluate the Credal CBM model"""

        # Get predictions with uncertainty
        pred_results = model.predict_with_uncertainty(X_test)

        # Use concept predictions for final classification
        concept_probs = pred_results['concept_probs']

        # Train a simple classifier on concept probabilities to make predictions
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(concept_probs, y_test_enc)
        y_pred = classifier.predict(concept_probs)

        # Calculate metrics
        accuracy = accuracy_score(y_test_enc, y_pred)
        f1_macro = f1_score(y_test_enc, y_pred, average='macro')
        f1_micro = f1_score(y_test_enc, y_pred, average='micro')

        # Uncertainty metrics
        epistemic_uncertainty = pred_results['mean_epistemic']
        credal_width = (pred_results['credal_upper'] - pred_results['credal_lower']).mean(axis=1)

        # For uncertainty correlation, use disagreement between concept predictions
        # This is a proxy for true aleatoric uncertainty when multi-annotator data is unavailable
        concept_disagreement = pred_results['epistemic_uncertainty'].mean(axis=1)

        from scipy.stats import spearmanr
        uncertainty_corr, p_value = spearmanr(epistemic_uncertainty, concept_disagreement)

        results = {
            # Classification metrics
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,

            # Uncertainty metrics
            "mean_epistemic": epistemic_uncertainty.mean(),
            "std_epistemic": epistemic_uncertainty.std(),
            "mean_credal_width": credal_width.mean(),
            "uncertainty_correlation": uncertainty_corr,
            "uncertainty_pvalue": p_value,

            # Coverage metrics
            "precise_predictions": (credal_width < 0.1).mean(),
            "high_uncertainty": (epistemic_uncertainty > 0.2).mean(),

            # Detailed predictions (for analysis)
            "predictions": {
                "concept_probs": concept_probs.tolist(),
                "epistemic_uncertainty": epistemic_uncertainty.tolist(),
                "credal_width": credal_width.tolist(),
                "y_true": y_test_enc.tolist(),
                "y_pred": y_pred.tolist()
            }
        }

        return results

    def run_cross_validation(self, dataset_name: str, encoder_name: str,
                           credal_method: str, n_folds: int = 5) -> Dict[str, Any]:
        """Run cross-validation experiment"""
        print(f"\nRunning {n_folds}-fold cross validation...")

        # Load and process data
        dataset_config = get_dataset_info(dataset_name)
        encoder_config = get_encoder_info(encoder_name)
        processor = DataProcessor(dataset_config, encoder_config)

        texts, labels = processor.load_huggingface_data()
        X_enc = processor.encode_texts(texts)
        y_enc = processor.process_labels(labels)

        # Create concept labels from encoded features and labels
        n_concepts = 10
        concept_labels = processor.create_task_concepts(X_enc, y_enc, n_concepts)

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_enc, y_enc)):
            print(f"Fold {fold + 1}/{n_folds}")

            X_train_fold, X_val_fold = X_enc[train_idx], X_enc[val_idx]
            y_train_fold, y_val_fold = y_enc[train_idx], y_enc[val_idx]
            concept_train_fold = concept_labels[train_idx]

            # Train model
            credal_config = get_credal_info(credal_method)
            model = CredalCBM(
                n_concepts=n_concepts,
                n_estimators=credal_config.n_estimators,
                max_depth=credal_config.max_depth,
                random_state=self.config.seed
            )
            model.fit(X_train_fold, concept_train_fold)

            # Evaluate
            fold_result = self.evaluate_credal_model(
                model, X_val_fold, y_val_fold,
                processor.label_encoder.inverse_transform(y_val_fold),
                processor.label_encoder
            )
            fold_result["fold"] = fold + 1
            fold_results.append(fold_result)

        # Aggregate results
        aggregated = self.aggregate_cv_results(fold_results)
        return aggregated

    def aggregate_cv_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics = ["accuracy", "f1_macro", "f1_micro", "mean_epistemic",
                  "uncertainty_correlation", "mean_credal_width"]

        aggregated = {
            "fold_results": fold_results,
            "mean_metrics": {},
            "std_metrics": {}
        }

        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated["mean_metrics"][metric] = np.mean(values)
                aggregated["std_metrics"][metric] = np.std(values)

        return aggregated

    def save_results(self, results: Dict, output_dir: str):
        """Save experiment results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        # Recursively convert all numpy objects
        def convert_recursive(obj):
            if isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(item) for item in obj]
            else:
                return convert_numpy(obj)

        json_results = convert_recursive(results)

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {results_file}")

        # Save summary
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        self.save_summary(results, summary_file)

    def save_summary(self, results: Dict, filepath: str):
        """Save a human-readable summary"""
        with open(filepath, 'w') as f:
            f.write("CREDAL CBM EXPERIMENT RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for exp_key, exp_result in results.items():
                if isinstance(exp_result, dict) and 'error' not in exp_result:
                    f.write(f"Experiment: {exp_key}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Dataset: {exp_result.get('dataset', 'N/A')}\n")
                    f.write(f"Encoder: {exp_result.get('encoder', 'N/A')}\n")
                    f.write(f"Credal Method: {exp_result.get('credal_method', 'N/A')}\n\n")

                    # Key metrics
                    f.write("Performance Metrics:\n")
                    f.write(f"  Accuracy: {exp_result.get('accuracy', 0):.3f}\n")
                    f.write(f"  F1-Macro: {exp_result.get('f1_macro', 0):.3f}\n")
                    f.write(f"  F1-Micro: {exp_result.get('f1_micro', 0):.3f}\n\n")

                    # Uncertainty metrics
                    f.write("Uncertainty Metrics:\n")
                    f.write(f"  Mean Epistemic: {exp_result.get('mean_epistemic', 0):.3f}\n")
                    f.write(f"  Mean Credal Width: {exp_result.get('mean_credal_width', 0):.3f}\n")
                    f.write(f"  Uncertainty Correlation: {exp_result.get('uncertainty_correlation', 0):.3f}\n\n")

                    f.write("\n" + "=" * 50 + "\n\n")


def main():
    """Main training function with enhanced epoch-based training"""
    parser = argparse.ArgumentParser(description='Train Credal CBM experiments with Enhanced Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--dataset', type=str, nargs='+',
                       default=['sst2'], help='Datasets to run')
    parser.add_argument('--encoder', type=str, nargs='+',
                       default=['distilbert'], help='Encoders to use')
    parser.add_argument('--credal-method', type=str, nargs='+',
                       default=['random_forest'], help='Credal methods to test')
    parser.add_argument('--cross-validation', action='store_true',
                       help='Run cross-validation instead of single split')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')

    # Enhanced training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs for enhanced trainer')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--enhanced-training', action='store_true',
                       help='Use enhanced epoch-based training with progress bars')
    parser.add_argument('--end-to-end', action='store_true',
                       help='Use end-to-end training with automatic concept discovery')
    parser.add_argument('--plot-training', action='store_true',
                       help='Plot training curves and save to file')
    parser.add_argument('--auto-concepts', action='store_true',
                       help='Use automatic concept discovery')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        config = ConfigManager.get_full_config()

    # Override with command line arguments
    if args.dataset != ['sst2']:  # Default not changed
        config.default_datasets = args.dataset
    if args.encoder != ['distilbert']:
        config.default_encoder = args.encoder[0]

    # Enhanced training mode
    if args.enhanced_training:
        print("🚀 ENHANCED CREDAL CBM TRAINING PIPELINE")
        print("=" * 60)
        print(f"Datasets: {config.default_datasets}")
        print(f"Encoders: {args.encoder}")
        print(f"Credal Methods: {args.credal_method}")
        print(f"Epochs: {args.epochs}")
        print(f"Patience: {args.patience}")
        print(f"End-to-End: {args.end_to_end}")
        print(f"Auto Concepts: {args.auto_concepts}")
        print(f"Plot Training: {args.plot_training}")
        print(f"Output Dir: {args.output_dir}")

        # Initialize enhanced trainer
        enhanced_trainer = EnhancedCredalCBMTrainer(config)

        # Run enhanced experiments
        all_results = {}
        all_histories = {}

        for dataset in tqdm(config.default_datasets, desc="Datasets"):
            for encoder in tqdm(args.encoder, desc=f"Encoders for {dataset}", leave=False):
                for credal_method in tqdm(args.credal_method, desc=f"Methods for {encoder}", leave=False):

                    exp_key = f"{dataset}_{encoder}_{credal_method}"

                    try:
                        print(f"\n{'='*80}")
                        print(f"ENHANCED TRAINING: {exp_key}")
                        print(f"{'='*80}")

                        # Run single experiment with enhanced training
                        results = enhanced_trainer.run_single_experiment(
                            dataset, encoder, credal_method
                        )

                        if 'error' not in results and args.end_to_end:
                            # Add enhanced training with epochs and concept discovery
                            print(f"🔄 Running enhanced epoch-based training...")

                            # Get data for enhanced training
                            dataset_config = get_dataset_info(dataset)
                            encoder_config = get_encoder_info(encoder)
                            credal_config = get_credal_info(credal_method)

                            processor = DataProcessor(dataset_config, encoder_config)
                            texts, labels = processor.load_huggingface_data()

                            # Split data
                            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                                texts, labels,
                                0.7, 0.15, 0.15  # 70% train, 15% val, 15% test
                            )

                            # Encode texts
                            print("📝 Encoding texts...")
                            X_train_enc = processor.encode_texts(tqdm(X_train, "Encoding train"))
                            X_val_enc = processor.encode_texts(tqdm(X_val, "Encoding val"))
                            X_test_enc = processor.encode_texts(tqdm(X_test, "Encoding test"))

                            # Encode labels
                            y_train_enc = processor.process_labels(y_train)
                            y_val_enc = processor.process_labels(y_val)
                            y_test_enc = processor.process_labels(y_test)

                            # Initialize model
                            model = CredalCBM(
                                n_concepts=10,
                                n_estimators=credal_config.n_estimators,
                                max_depth=credal_config.max_depth,
                                random_state=config.seed
                            )

                            # Run enhanced training
                            if args.end_to_end or args.auto_concepts:
                                training_history = enhanced_trainer.end_to_end_train(
                                    model=model,
                                    X_train=X_train_enc,
                                    y_train=y_train_enc,
                                    X_val=X_val_enc,
                                    y_val=y_val_enc,
                                    epochs=args.epochs,
                                    auto_concepts=True,
                                    verbose=True
                                )
                            else:
                                # Use provided concept labels
                                concept_labels_train = processor.create_task_concepts(X_train_enc, y_train_enc, 10)
                                concept_labels_val = processor.create_task_concepts(X_val_enc, y_val_enc, 10)

                                training_history = enhanced_trainer.train_with_epochs(
                                    model=model,
                                    X_train=X_train_enc,
                                    concept_labels_train=concept_labels_train,
                                    y_train=y_train_enc,
                                    X_val=X_val_enc,
                                    concept_labels_val=concept_labels_val,
                                    y_val=y_val_enc,
                                    epochs=args.epochs,
                                    patience=args.patience,
                                    verbose=True
                                )

                            # Evaluate trained model
                            final_results = enhanced_trainer.evaluate_credal_model(
                                model, X_test_enc, y_test_enc, y_test, processor.label_encoder
                            )

                            # Combine results
                            enhanced_results = {
                                **results,
                                **final_results,
                                'enhanced_training': True,
                                'epochs': len(training_history.get('concept_losses', [])),
                                'final_train_loss': training_history['concept_losses'][-1] if training_history['concept_losses'] else None,
                                'final_val_loss': training_history['val_concept_losses'][-1] if training_history.get('val_concept_losses') else None,
                            }

                            all_results[exp_key] = enhanced_results
                            all_histories[exp_key] = training_history

                            # Plot training history
                            if args.plot_training:
                                plot_path = f"{args.output_dir}/training_curves_{exp_key}.png"
                                enhanced_trainer.plot_training_history(training_history, plot_path)

                            print(f"✅ Enhanced training completed for {exp_key}")
                            print(f"  Final Train Loss: {enhanced_results['final_train_loss']:.4f}")
                            print(f"  Epochs Completed: {enhanced_results['epochs']}")

                        else:
                            all_results[exp_key] = results

                        # Print quick summary
                        if 'error' not in results:
                            print(f"\n📊 {exp_key} Results:")
                            print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
                            print(f"  Uncertainty Correlation: {results.get('uncertainty_correlation', 0):.3f}")
                            print(f"  Mean Credal Width: {results.get('mean_credal_width', 0):.3f}")
                        else:
                            print(f"\n❌ {exp_key} FAILED: {results['error']}")

                    except Exception as e:
                        print(f"\n❌ {exp_key} FAILED: {str(e)}")
                        all_results[exp_key] = {'error': str(e)}

        # Save enhanced results
        enhanced_trainer.save_results(all_results, args.output_dir)

        # Save training histories
        if all_histories:
            import json
            history_path = os.path.join(args.output_dir, "training_histories.json")
            with open(history_path, 'w') as f:
                json.dump(all_histories, f, indent=2)
            print(f"📊 Training histories saved to {history_path}")

    else:
        # Original training mode
        print("CREDAL CBM EXPERIMENT PIPELINE")
        print("=" * 60)
        print(f"Datasets: {config.default_datasets}")
        print(f"Encoders: {args.encoder}")
        print(f"Credal Methods: {args.credal_method}")
        print(f"Cross Validation: {args.cross_validation}")
        print(f"Output Dir: {args.output_dir}")

        # Initialize trainer
        trainer = CredalCBMTrainer(config)

        # Run experiments
        all_results = {}

        for dataset in config.default_datasets:
            for encoder in args.encoder:
                for credal_method in args.credal_method:

                    exp_key = f"{dataset}_{encoder}_{credal_method}"

                    if args.cross_validation:
                        results = trainer.run_cross_validation(
                            dataset, encoder, credal_method, args.cv_folds
                        )
                    else:
                        results = trainer.run_single_experiment(
                            dataset, encoder, credal_method
                        )

                    all_results[exp_key] = results

                    # Print quick summary
                    if 'error' not in results:
                        print(f"\n{exp_key} Results:")
                        print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
                        print(f"  Uncertainty Correlation: {results.get('uncertainty_correlation', 0):.3f}")
                        print(f"  Mean Credal Width: {results.get('mean_credal_width', 0):.3f}")
                    else:
                        print(f"\n{exp_key} FAILED: {results['error']}")

        # Save results
        trainer.save_results(all_results, args.output_dir)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
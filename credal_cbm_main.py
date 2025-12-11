"""
Credal Concept Bottleneck Model (Credal CBM) - Fixed Implementation

A corrected implementation of Credal CBM for ambiguous tasks using sklearn ensembles.
Supports both binary and multi-class (ternary) concept predictions.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse


@dataclass
class CredalSet:
    """
    Credal set representation for concept predictions
    """
    lower: float
    upper: float
    mean: float
    predictions: np.ndarray
    class_probs: Optional[np.ndarray] = None  # For multi-class: [n_trees, n_classes]

    @property
    def epistemic_uncertainty(self) -> float:
        """Ensemble disagreement as epistemic uncertainty"""
        return self.predictions.std()

    @property
    def interval_width(self) -> float:
        """Width of credal interval"""
        return self.upper - self.lower


class CredalCBM:
    """
    Main Credal CBM implementation using Random Forest ensembles
    Supports both binary and multi-class concept prediction
    """

    def __init__(
        self,
        n_concepts: int = 10,
        n_classes: int = 2,  # Number of classes per concept (2 for binary, 3 for ternary)
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.concept_models = []
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Track which concepts were actually trained
        self.valid_concept_indices = []

    def fit(self, X: np.ndarray, concept_labels: np.ndarray):
        """
        Fit the Credal CBM model

        Args:
            X: Feature matrix [n_samples, n_features]
            concept_labels: Concept labels [n_samples, n_concepts]
                           For binary: {0, 1}
                           For ternary: {0=Unknown, 1=Negative, 2=Positive}
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train separate ensemble for each concept
        self.concept_models = []
        self.valid_concept_indices = []

        for k in range(self.n_concepts):
            y_k = concept_labels[:, k]

            # Check number of unique classes
            unique_classes = np.unique(y_k)
            n_unique = len(unique_classes)

            # Skip if concept has only one class
            if n_unique < 2:
                print(f"Warning: Concept {k} has only {n_unique} class(es), skipping")
                self.concept_models.append(None)
                continue

            # Create and fit Random Forest
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=self.random_state + k,  # Different seed per concept
                n_jobs=-1
            )

            model.fit(X_scaled, y_k)
            self.concept_models.append(model)
            self.valid_concept_indices.append(k)

        self.is_fitted = True
        print(f"Trained {len(self.valid_concept_indices)}/{self.n_concepts} concepts successfully")

    def predict_credal_sets_single_sample(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None
    ) -> List[CredalSet]:
        """
        Predict credal sets for a single sample

        Args:
            x: Single sample [1, n_features] (already scaled)
            target_class: If specified, get credal set for this class only
                         If None, return full probability distribution

        Returns:
            List of credal sets, one per concept
        """
        credal_sets = []

        for model in self.concept_models:
            if model is None:
                # Concept was skipped during training
                credal_sets.append(CredalSet(
                    lower=0.0,
                    upper=0.0,
                    mean=0.0,
                    predictions=np.array([0.0])
                ))
                continue

            # Get predictions from each tree
            tree_probs = np.array([
                tree.predict_proba(x)[0] for tree in model.estimators_
            ])  # Shape: [n_trees, n_classes]

            if target_class is not None:
                # Extract probabilities for target class only
                if target_class < tree_probs.shape[1]:
                    tree_preds = tree_probs[:, target_class]
                else:
                    # Class doesn't exist in this model
                    tree_preds = np.zeros(len(tree_probs))

                credal_set = CredalSet(
                    lower=tree_preds.min(),
                    upper=tree_preds.max(),
                    mean=tree_preds.mean(),
                    predictions=tree_preds,
                    class_probs=None
                )
            else:
                # Return full distribution
                # Use mean probability across trees for each class
                mean_probs = tree_probs.mean(axis=0)

                # Epistemic uncertainty = variance in probability predictions
                # Use entropy of mean distribution as the main prediction
                epistemic = tree_probs.std(axis=0).mean()

                credal_set = CredalSet(
                    lower=mean_probs.min(),
                    upper=mean_probs.max(),
                    mean=mean_probs[mean_probs.argmax()],  # Max class probability
                    predictions=mean_probs,
                    class_probs=tree_probs  # Store all tree predictions
                )

            credal_sets.append(credal_set)

        return credal_sets

    def predict_with_uncertainty_all_classes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions with uncertainty for all classes (multi-class version)

        Returns:
            Dict with:
                - concept_probs: [n_samples, n_concepts, n_classes] probability distributions
                - epistemic_uncertainty: [n_samples, n_concepts] per-concept epistemic uncertainty
                - class_variances: [n_samples, n_concepts] variance across class predictions
                - credal_lower: [n_samples, n_concepts] lower bound of credal set
                - credal_upper: [n_samples, n_concepts] upper bound of credal set
                - mean_epistemic: [n_samples] average epistemic uncertainty per sample
                - max_epistemic: [n_samples] maximum epistemic uncertainty per sample
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        n_concepts = len(self.concept_models)

        # Initialize arrays for all classes
        concept_probs = np.zeros((n_samples, n_concepts, self.n_classes))
        epistemic_uncertainty = np.zeros((n_samples, n_concepts))
        credal_lower = np.zeros((n_samples, n_concepts))
        credal_upper = np.zeros((n_samples, n_concepts))

        # Process each sample
        for i in range(n_samples):
            x_i = X_scaled[i:i+1]

            # Get credal sets for all classes
            for k, model in enumerate(self.concept_models):
                if model is None:
                    continue

                # Get predictions from all trees
                tree_probs = np.array([
                    tree.predict_proba(x_i)[0] for tree in model.estimators_
                ])  # Shape: [n_trees, n_classes]

                # Ensure we have the right number of classes
                n_model_classes = tree_probs.shape[1]

                # Mean probabilities across trees
                mean_probs = tree_probs.mean(axis=0)
                # Handle case where model has different number of classes than expected
                actual_classes = min(n_model_classes, min(mean_probs.shape[0], self.n_classes))
                concept_probs[i, k, :actual_classes] = mean_probs[:actual_classes]

                # Epistemic uncertainty: standard deviation across trees
                # Average std across all classes
                epistemic_uncertainty[i, k] = tree_probs.std(axis=0).mean()

                # Credal bounds: min/max of predicted class
                predicted_class = mean_probs.argmax()
                class_preds = tree_probs[:, predicted_class]
                credal_lower[i, k] = class_preds.min()
                credal_upper[i, k] = class_preds.max()

        # Compute sample-level uncertainty metrics
        # Class variance: variance across class probabilities for each concept
        class_variances = np.var(concept_probs, axis=2)

        # Mean epistemic: average across concepts
        mean_epistemic = epistemic_uncertainty.mean(axis=1)

        # Max epistemic: maximum across concepts
        max_epistemic = epistemic_uncertainty.max(axis=1)

        return {
            'concept_probs': concept_probs,  # (n_samples, n_concepts, n_classes)
            'epistemic_uncertainty': epistemic_uncertainty,  # (n_samples, n_concepts)
            'credal_lower': credal_lower,
            'credal_upper': credal_upper,
            'mean_epistemic': mean_epistemic,  # (n_samples,)
            'max_epistemic': max_epistemic,  # (n_samples,)
            'class_variances': class_variances,  # (n_samples, n_concepts)
        }

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions with uncertainty decomposition (binary version for compatibility)
        Uses the highest probability class for each concept

        Returns:
            Dict with concept probabilities, epistemic uncertainty, and credal bounds
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get full multi-class predictions
        full_results = self.predict_with_uncertainty_all_classes(X)

        n_samples = X.shape[0]
        n_concepts = len(self.concept_models)

        # Extract the predicted class probability (max across classes)
        concept_probs = np.zeros((n_samples, n_concepts))
        for i in range(n_samples):
            for k in range(n_concepts):
                concept_probs[i, k] = full_results['concept_probs'][i, k].max()

        return {
            'concept_probs': concept_probs,  # (n_samples, n_concepts)
            'epistemic_uncertainty': full_results['epistemic_uncertainty'],
            'credal_lower': full_results['credal_lower'],
            'credal_upper': full_results['credal_upper'],
            'mean_epistemic': full_results['mean_epistemic'],
            'max_epistemic': full_results['max_epistemic'],
        }


def prepare_label_features(
    concept_results: Dict[str, np.ndarray],
    strategy: str = 'all_probs'
) -> np.ndarray:
    """
    Prepare features for label prediction from concept predictions

    Args:
        concept_results: Output from predict_with_uncertainty_all_classes
        strategy: Feature construction strategy
            - 'all_probs': Flatten all class probabilities [n_concepts * n_classes features]
            - 'summary': Use summary statistics [n_concepts * 4 features]
            - 'positive_only': Use only positive class + uncertainty [n_concepts * 2 features]

    Returns:
        Feature matrix [n_samples, n_features]
    """
    concept_probs = concept_results['concept_probs']  # (n_samples, n_concepts, n_classes)
    epistemic = concept_results['epistemic_uncertainty']  # (n_samples, n_concepts)

    n_samples, n_concepts, n_classes = concept_probs.shape

    if strategy == 'all_probs':
        # Flatten all probabilities
        features = concept_probs.reshape(n_samples, -1)  # (n_samples, n_concepts * n_classes)
        # Add epistemic uncertainty
        features = np.column_stack([features, epistemic])

    elif strategy == 'summary':
        # For each concept: [max_prob, argmax, entropy, epistemic]
        max_probs = concept_probs.max(axis=2)  # (n_samples, n_concepts)
        argmax = concept_probs.argmax(axis=2)  # (n_samples, n_concepts)

        # Entropy: -sum(p * log(p))
        entropy = -np.sum(
            concept_probs * np.log(concept_probs + 1e-10),
            axis=2
        )  # (n_samples, n_concepts)

        features = np.column_stack([
            max_probs,
            argmax,
            entropy,
            epistemic
        ])

    elif strategy == 'positive_only':
        # For ternary: use positive class probability + epistemic
        if n_classes == 3:
            positive_probs = concept_probs[:, :, 2]  # (n_samples, n_concepts)
        else:
            # For binary, use class 1
            positive_probs = concept_probs[:, :, -1]

        features = np.column_stack([positive_probs, epistemic])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return features


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 100,
    n_concepts: int = 10,
    n_classes: int = 2,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing

    Returns:
        X: Features
        concept_labels: Concept labels (multi-class)
        true_uncertainty: Ground truth uncertainty per sample
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Create concept labels with varying difficulty
    concept_labels = np.zeros((n_samples, n_concepts), dtype=int)
    true_uncertainty = np.zeros(n_samples)

    for i in range(n_samples):
        # Sample difficulty level (affects all concepts for this sample)
        difficulty = np.random.beta(2, 5)  # Most samples are easy
        true_uncertainty[i] = difficulty

        for k in range(n_concepts):
            # Base probability depends on difficulty
            if difficulty < 0.3:  # Easy sample
                # Strongly favor one class
                class_probs = np.random.dirichlet([0.1] * n_classes)
                dominant_class = np.random.randint(n_classes)
                class_probs[dominant_class] = 0.8
                class_probs = class_probs / class_probs.sum()
            elif difficulty < 0.7:  # Medium sample
                class_probs = np.random.dirichlet([1.0] * n_classes)
            else:  # Hard sample - uniform distribution
                class_probs = np.ones(n_classes) / n_classes

            # Sample class
            concept_labels[i, k] = np.random.choice(n_classes, p=class_probs)

    return X, concept_labels, true_uncertainty


def evaluate_model(
    model: CredalCBM,
    X_test: np.ndarray,
    true_uncertainty: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the Credal CBM model
    """
    from scipy.stats import spearmanr

    # Get predictions
    results = model.predict_with_uncertainty(X_test)

    predicted_epistemic = results['mean_epistemic']

    # Calculate correlation with true uncertainty
    correlation, p_value = spearmanr(predicted_epistemic, true_uncertainty)

    metrics = {
        'uncertainty_correlation': correlation,
        'uncertainty_pvalue': p_value,
        'mean_epistemic': predicted_epistemic.mean(),
        'std_epistemic': predicted_epistemic.std(),
        'mean_true_uncertainty': true_uncertainty.mean(),
        'mean_credal_width': (results['credal_upper'] - results['credal_lower']).mean()
    }

    return metrics


if __name__ == "__main__":
    main()
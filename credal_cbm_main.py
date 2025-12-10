"""
Credal Concept Bottleneck Model (Credal CBM) - Main Implementation

A simplified implementation of Credal CBM for ambiguous tasks using sklearn ensembles.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple
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
    Includes training loop for end-to-end learning
    """

    def __init__(
        self,
        n_concepts: int = 10,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.n_concepts = n_concepts
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.concept_models = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, concept_labels: np.ndarray):
        """
        Fit the Credal CBM model

        Args:
            X: Feature matrix [n_samples, n_features]
            concept_labels: Binary concept labels [n_samples, n_concepts]
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train separate ensemble for each concept
        self.concept_models = []
        for k in range(self.n_concepts):
            y_k = concept_labels[:, k]

            # Skip if concept has only one class
            if len(np.unique(y_k)) < 2:
                continue

            # Create and fit Random Forest
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            )

            model.fit(X_scaled, y_k)
            self.concept_models.append(model)

        self.is_fitted = True

    def predict_credal_sets(self, X: np.ndarray, target_class: int = 2) -> List[List[CredalSet]]:
        """
        Predict credal sets for each sample and concept

        Args:
            X: Feature matrix [n_samples, n_features]
            target_class: Which class to get probability for (0=unknown, 1=negative, 2=positive)

        Returns:
            List of credal sets [n_samples][n_concepts]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]

        all_credal_sets = []

        for i in range(n_samples):
            sample_credal_sets = []
            x_i = X_scaled[i:i+1]

            for model in self.concept_models:
                # Get predictions from each tree for the target class
                tree_preds = np.array([
                    tree.predict_proba(x_i)[0, target_class] if target_class < len(tree.classes_) else 0.0
                    for tree in model.estimators_
                ])

                # Create credal set
                credal_set = CredalSet(
                    lower=tree_preds.min(),
                    upper=tree_preds.max(),
                    mean=tree_preds.mean(),
                    predictions=tree_preds
                )

                sample_credal_sets.append(credal_set)

            all_credal_sets.append(sample_credal_sets)

        return all_credal_sets

    def predict_with_uncertainty_all_classes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions with uncertainty for all 3 classes (ternary)

        Returns:
            Dict with concept probabilities for all classes, epistemic uncertainty, and credal bounds
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        n_concepts = len(self.concept_models)
        n_classes = 3  # unknown=0, negative=1, positive=2

        # Initialize arrays for all classes
        concept_probs = np.zeros((n_samples, n_concepts, n_classes))
        epistemic_uncertainty = np.zeros((n_samples, n_concepts))
        credal_lower = np.zeros((n_samples, n_concepts))
        credal_upper = np.zeros((n_samples, n_concepts))

        # Get predictions for each class
        for target_class in range(n_classes):
            credal_sets = self.predict_credal_sets(X, target_class=target_class)

            for i, sample_cs in enumerate(credal_sets):
                for k, cs in enumerate(sample_cs):
                    if k < n_concepts:
                        concept_probs[i, k, target_class] = cs.mean
                        if target_class == 2:  # For positive class, store uncertainty metrics
                            epistemic_uncertainty[i, k] = cs.epistemic_uncertainty
                            credal_lower[i, k] = cs.lower
                            credal_upper[i, k] = cs.upper

        # Compute overall epistemic uncertainty (variance across classes)
        class_variances = np.var(concept_probs, axis=2)
        overall_epistemic = np.mean(class_variances, axis=1)
        max_epistemic = np.max(class_variances, axis=1)

        return {
            'concept_probs': concept_probs,  # (n_samples, n_concepts, 3)
            'epistemic_uncertainty': epistemic_uncertainty,  # (n_samples, n_concepts)
            'credal_lower': credal_lower,
            'credal_upper': credal_upper,
            'mean_epistemic': overall_epistemic,
            'max_epistemic': max_epistemic,
            'class_variances': class_variances,
        }

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions with uncertainty decomposition

        Returns:
            Dict with concept probabilities, epistemic uncertainty, and credal bounds
        """
        credal_sets = self.predict_credal_sets(X)

        n_samples = len(credal_sets)
        n_concepts = len(self.concept_models)

        # Initialize arrays
        concept_probs = np.zeros((n_samples, n_concepts))
        epistemic_uncertainty = np.zeros((n_samples, n_concepts))
        credal_lower = np.zeros((n_samples, n_concepts))
        credal_upper = np.zeros((n_samples, n_concepts))

        # Fill arrays from credal sets
        for i, sample_cs in enumerate(credal_sets):
            for k, cs in enumerate(sample_cs):
                concept_probs[i, k] = cs.mean
                epistemic_uncertainty[i, k] = cs.epistemic_uncertainty
                credal_lower[i, k] = cs.lower
                credal_upper[i, k] = cs.upper

        return {
            'concept_probs': concept_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'credal_lower': credal_lower,
            'credal_upper': credal_upper,
            'mean_epistemic': epistemic_uncertainty.mean(axis=1),
            'max_epistemic': epistemic_uncertainty.max(axis=1),
        }

    

def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 100,
    n_concepts: int = 10,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing

    Returns:
        X: Features
        concept_labels: Binary concept labels
        true_uncertainty: Ground truth uncertainty per sample
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Create concept labels with varying difficulty
    concept_labels = np.zeros((n_samples, n_concepts))
    true_uncertainty = np.zeros(n_samples)

    for i in range(n_samples):
        # Sample difficulty level (affects all concepts for this sample)
        difficulty = np.random.beta(2, 5)  # Most samples are easy
        true_uncertainty[i] = difficulty

        for k in range(n_concepts):
            # Base probability depends on difficulty
            if difficulty < 0.3:  # Easy sample
                base_prob = np.random.choice([0.1, 0.9])
            elif difficulty < 0.7:  # Medium sample
                base_prob = np.random.choice([0.3, 0.7])
            else:  # Hard sample
                base_prob = 0.5

            # Add noise
            prob = base_prob + np.random.normal(0, noise_level)
            prob = np.clip(prob, 0.1, 0.9)

            # Generate binary label
            concept_labels[i, k] = np.random.binomial(1, prob)

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


def main():
    """
    Main function to demonstrate Credal CBM
    """
    parser = argparse.ArgumentParser(description='Credal CBM Demo')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n-features', type=int, default=100, help='Number of features')
    parser.add_argument('--n-concepts', type=int, default=10, help='Number of concepts')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees in ensemble')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')

    args = parser.parse_args()

    # Create synthetic data
    X, concept_labels, true_uncertainty = create_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_concepts=args.n_concepts
    )

    # Split data
    n_test = int(args.n_samples * args.test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train = concept_labels[:-n_test]
    uncertainty_test = true_uncertainty[-n_test:]

    # Initialize and fit model
    model = CredalCBM(
        n_concepts=args.n_concepts,
        n_estimators=args.n_estimators,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, uncertainty_test)

    # Show example predictions
    credal_sets = model.predict_credal_sets(X_test[:3])

    return results


if __name__ == "__main__":
    main()
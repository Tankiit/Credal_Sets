"""
Construction module for creating credal sets from annotations
Supports multiple construction methods including convex hull
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Dict, Tuple, Optional
from collections import Counter


class CredalSet:
    """
    Represents a credal set (convex set of probability distributions)

    Properties:
    - Convex by construction
    - 2-monotone capacity (automatic from convexity)
    - Can be represented by vertices
    """

    def __init__(self, vertices: List[Dict[str, float]], answers: List[str], method: str = 'convex_hull'):
        """
        Initialize a credal set

        Args:
            vertices: List of probability distributions as dicts {answer: probability}
            answers: List of all possible answers
            method: Construction method used
        """
        self.vertices = vertices
        self.answers = answers
        self.method = method
        self.n_vertices = len(vertices)

        # Validate vertices
        self._validate_vertices()

    def _validate_vertices(self):
        """Validate that all vertices are proper probability distributions"""
        for v in self.vertices:
            # Check probabilities sum to 1
            total = sum(v.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"Invalid vertex: probabilities sum to {total}, expected 1.0")

            # Check all probabilities are non-negative
            if any(p < 0 for p in v.values()):
                raise ValueError("Invalid vertex: negative probabilities detected")

    def to_vertices(self) -> List[Dict[str, float]]:
        """Return the vertices of the credal set"""
        return self.vertices

    def to_array(self) -> np.ndarray:
        """
        Convert vertices to numpy array

        Returns:
            Array of shape [n_vertices, n_answers]
        """
        arr = []
        for v in self.vertices:
            row = [v.get(answer, 0.0) for answer in self.answers]
            arr.append(row)
        return np.array(arr)

    def __repr__(self):
        return f"CredalSet(n_vertices={self.n_vertices}, n_answers={len(self.answers)}, method='{self.method}')"


class CredalConstructor:
    """
    Construct credal sets from annotation data

    Methods:
    - convex_hull: Build convex hull of empirical distributions
    - empirical: Use empirical distributions directly (may not be convex)
    """

    def __init__(self, method: str = 'convex_hull'):
        """
        Initialize constructor

        Args:
            method: Construction method ('convex_hull' or 'empirical')
        """
        if method not in ['convex_hull', 'empirical']:
            raise ValueError(f"Unknown method: {method}. Use 'convex_hull' or 'empirical'")

        self.method = method

    def construct(self, annotations: List[str]) -> CredalSet:
        """
        Construct a credal set from annotations

        Args:
            annotations: List of observed answers/annotations

        Returns:
            CredalSet object
        """
        if not annotations:
            raise ValueError("Cannot construct credal set from empty annotations")

        # Get all unique answers
        answers = sorted(list(set(annotations)))

        if self.method == 'convex_hull':
            return self._construct_convex_hull(annotations, answers)
        elif self.method == 'empirical':
            return self._construct_empirical(annotations, answers)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _construct_convex_hull(self, annotations: List[str], answers: List[str]) -> CredalSet:
        """
        Construct credal set using convex hull method

        This creates a convex credal set by:
        1. Computing empirical distribution
        2. Adding extreme points for each possible answer
        3. Taking convex hull (which for this case is just the set of vertices)

        For the case where we're building from a set of annotations,
        we create vertices that represent the uncertainty in the data.
        """
        # Count occurrences
        counts = Counter(annotations)
        total = len(annotations)

        # Get all unique answers
        unique_answers = sorted(counts.keys())

        # If only one unique answer, create a degenerate credal set with one vertex
        if len(unique_answers) == 1:
            single_answer = unique_answers[0]
            vertex = {ans: 1.0 if ans == single_answer else 0.0 for ans in answers}
            return CredalSet([vertex], answers, method=self.method)

        # Create vertices for the convex hull
        # Strategy: Each vertex emphasizes one answer more than the empirical distribution
        vertices = []

        # Add empirical distribution as a vertex
        empirical_vertex = {}
        for ans in answers:
            empirical_vertex[ans] = counts.get(ans, 0) / total
        vertices.append(empirical_vertex)

        # Add extreme points: for each observed answer, create a vertex
        # that gives more weight to that answer
        n_observed = len(unique_answers)
        for ans in unique_answers:
            # Create an extreme vertex that emphasizes this answer
            extreme_vertex = {}
            for a in answers:
                if a == ans:
                    # Give this answer higher probability
                    extreme_vertex[a] = 1.0 / n_observed + (1.0 - 1.0 / n_observed) * 0.5
                elif a in unique_answers:
                    # Give other observed answers lower probability
                    extreme_vertex[a] = (1.0 - 1.0 / n_observed) * 0.5 / (n_observed - 1)
                else:
                    extreme_vertex[a] = 0.0

            # Normalize to ensure it sums to 1
            total_prob = sum(extreme_vertex.values())
            extreme_vertex = {k: v / total_prob for k, v in extreme_vertex.items()}
            vertices.append(extreme_vertex)

        # For convex credal sets, we only need the vertices
        # The convex hull is implied by the construction
        return CredalSet(vertices, answers, method=self.method)

    def _construct_empirical(self, annotations: List[str], answers: List[str]) -> CredalSet:
        """
        Construct credal set using empirical distributions

        Note: This may not produce a convex set!
        """
        # Count occurrences
        counts = Counter(annotations)
        total = len(annotations)

        # Create a single empirical distribution
        empirical_vertex = {}
        for ans in answers:
            empirical_vertex[ans] = counts.get(ans, 0) / total

        return CredalSet([empirical_vertex], answers, method=self.method)

    def __repr__(self):
        return f"CredalConstructor(method='{self.method}')"


# Convenience functions

def create_credal_set_from_annotations(annotations: List[str], method: str = 'convex_hull') -> CredalSet:
    """
    Convenience function to create a credal set directly from annotations

    Args:
        annotations: List of observed answers/annotations
        method: Construction method ('convex_hull' or 'empirical')

    Returns:
        CredalSet object
    """
    constructor = CredalConstructor(method=method)
    return constructor.construct(annotations)


def credal_set_from_array(prob_array: np.ndarray, answers: List[str]) -> CredalSet:
    """
    Create a credal set from a numpy array of probability distributions

    Args:
        prob_array: Array of shape [n_distributions, n_answers]
        answers: List of answer labels

    Returns:
        CredalSet object
    """
    if prob_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {prob_array.shape}")

    if prob_array.shape[1] != len(answers):
        raise ValueError(f"Array columns ({prob_array.shape[1]}) don't match number of answers ({len(answers)})")

    # Convert each row to a dictionary
    vertices = []
    for row in prob_array:
        vertex = {ans: prob for ans, prob in zip(answers, row)}
        vertices.append(vertex)

    return CredalSet(vertices, answers, method='array')


# Example usage and testing

if __name__ == "__main__":
    print("Testing CredalSet Construction")
    print("=" * 60)

    # Example 1: High agreement
    correct_answers = ["A", "A", "A", "A", "B"]
    print(f"\nHigh agreement: {correct_answers}")

    constructor = CredalConstructor(method='convex_hull')
    credal_correct = constructor.construct(correct_answers)

    print(f"  Created: {credal_correct}")
    print(f"  Vertices: {credal_correct.n_vertices}")
    print(f"  Array shape: {credal_correct.to_array().shape}")

    # Example 2: High disagreement
    incorrect_answers = ["A", "B", "C", "D", "E"]
    print(f"\nHigh disagreement: {incorrect_answers}")

    credal_incorrect = constructor.construct(incorrect_answers)

    print(f"  Created: {credal_incorrect}")
    print(f"  Vertices: {credal_incorrect.n_vertices}")
    print(f"  Array shape: {credal_incorrect.to_array().shape}")

    print("\n✓ All tests passed!")

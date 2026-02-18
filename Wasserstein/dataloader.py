"""
CREDENCE Multi-Dataset Dataloader v2
=====================================

Verified against actual HuggingFace dataset structures.

Dataset Field Reference (from HF documentation):
================================================

CEBaB:
  - description: str (review text)
  - review_majority: str ("1", "2", "3", "4", "5", "no majority")
  - food_aspect_majority: str ("Positive", "Negative", "unknown", "")
  - service_aspect_majority: str
  - ambiance_aspect_majority: str
  - noise_aspect_majority: str
  - Splits: train_inclusive, train_exclusive, validation, test

HateXplain:
  - post_tokens: list[str] (tokenized text)
  - annotators: dict with keys:
      - label: list[int] (0=hatespeech, 1=normal, 2=offensive)
      - target: list[list[str]] (target communities per annotator)
  - rationales: list[list[int]] (token-level rationales)
  - Splits: train, validation, test

GoEmotions (simplified):
  - text: str
  - labels: list[int] (emotion indices 0-27, multi-label)
  - id: str
  - Splits: train, validation, test

Civil Comments:
  - text: str
  - toxicity: float (0.0-1.0)
  - severe_toxicity: float
  - obscene: float
  - threat: float
  - insult: float
  - identity_attack: float
  - sexual_explicit: float
  - Splits: train, validation, test
  - Note: identity columns NOT in default config

SST-2 (glue):
  - sentence: str
  - label: int (0=negative, 1=positive)
  - idx: int
  - Splits: train, validation, test

SST-5 (SetFit/sst5):
  - text: str
  - label: int (0-4)
  - label_text: str
  - Splits: train, validation, test

ChaosNLI:
  - premise: str
  - hypothesis: str
  - label: int (0=entailment, 1=neutral, 2=contradiction, -1=unlabeled)
  - Splits: train, validation, test

TID-8:
  - premise: str
  - hypothesis: str
  - label: int (0=entailment, 1=neutral, 2=contradiction)
  - Splits: train, validation, test

IMDB:
  - text: str
  - label: int (0=negative, 1=positive)
  - Splits: train, test (no validation - need to create)

Yelp Review Full:
  - text: str
  - label: int (0-4 for 1-5 stars)
  - Splits: train, test
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import warnings
import math

# =============================================================================
# CEBaB ALEATORIC UNCERTAINTY UTILITIES
# =============================================================================

# CEBaB-specific constants
CEBAB_LABELS_3 = ["Negative", "unknown", "Positive"]
CEBAB_LABEL_TO_IDX = {l: i for i, l in enumerate(CEBAB_LABELS_3)}
CEBAB_ASPECT_KEYS = ["food", "ambiance", "service", "noise"]  # K=4 concepts


def _dist_from_label_counts(label_dist) -> np.ndarray:
    """
    Convert CEBaB label distribution to probability array.

    Args:
        label_dist: dict(str->int) or str, e.g. {"Negative": 2, "Positive": 1, "unknown": 1}
                   If string, attempts to parse it as JSON

    Returns:
        [3] array with probabilities over (Negative, unknown, Positive)
    """
    import json

    counts = np.zeros(3, dtype=np.float32)
    total = 0.0

    # Handle if label_dist is a string (JSON)
    if isinstance(label_dist, str):
        try:
            label_dist = json.loads(label_dist)
        except:
            # If parsing fails, return unknown fallback
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Handle if label_dist is None or not a dict
    if not isinstance(label_dist, dict):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Process the distribution
    for k, v in label_dist.items():
        if k in CEBAB_LABEL_TO_IDX:
            counts[CEBAB_LABEL_TO_IDX[k]] += float(v)
            total += float(v)

    if total <= 0:
        # Fallback: uniform (max entropy), or put mass on "unknown"
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return counts / total


def _entropy(p: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Shannon entropy of probability distribution.

    Args:
        p: probability array
        eps: small constant for numerical stability

    Returns:
        entropy in nats
    """
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def build_cebab_aleatoric_targets(example: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build aleatoric uncertainty targets for CEBaB example.

    Uses the annotator label distributions to compute:
      1. Probability distribution over (Neg, Unk, Pos) for each concept
      2. Normalized entropy (uncertainty) for each concept

    Args:
        example: CEBaB dataset example with *_aspect_label_distribution fields

    Returns:
        concept_dist: [K, 3] probabilities over (Negative, unknown, Positive)
        concept_entropy: [K] normalized entropy in [0, 1]
    """
    dists = []
    ents = []

    for asp in CEBAB_ASPECT_KEYS:
        # Get label distribution for this aspect
        ld = example.get(f"{asp}_aspect_label_distribution", None)

        # Convert to probability distribution
        p = _dist_from_label_counts(ld)  # [3]

        # Compute normalized entropy
        H = _entropy(p)
        H_norm = H / math.log(3.0)  # Normalize to [0, 1]

        dists.append(p)
        ents.append(H_norm)

    return np.stack(dists, axis=0), np.array(ents, dtype=np.float32)


# =============================================================================
# DATASET REGISTRY - Verified against HuggingFace
# =============================================================================

DATASET_INFO = {
    # =========================================================================
    # SENTIMENT ANALYSIS
    # =========================================================================
    "cebab": {
        "hf_path": "CEBaB/CEBaB",
        "hf_name": None,
        "train_split": "train_inclusive",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "description",
        "label_field": "review_majority",
        "concept_fields": [
            "food_aspect_majority",
            "service_aspect_majority",
            "ambiance_aspect_majority",
            "noise_aspect_majority"
        ],
        "concept_names": ["food", "service", "ambiance", "noise"],
        "num_classes": 3,  # Using ternary: negative/neutral/positive
        "has_concepts": True,
        "has_multi_annotator": True,
        "task": "sentiment",
        "class_names": ["negative", "neutral", "positive"],
    },

    "sst2": {
        "hf_path": "glue",
        "hf_name": "sst2",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",  # SST-2 test has no labels
        "text_field": "sentence",
        "label_field": "label",
        "num_classes": 2,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "sentiment",
        "class_names": ["negative", "positive"],
    },

    "sst5": {
        "hf_path": "SetFit/sst5",
        "hf_name": None,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "num_classes": 5,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "sentiment",
        "class_names": ["very_negative", "negative", "neutral", "positive", "very_positive"],
        "is_ordinal": True,
    },

    "imdb": {
        "hf_path": "imdb",
        "hf_name": None,
        "train_split": "train",
        "val_split": "test[:5000]",  # Create val from test
        "test_split": "test[5000:]",
        "text_field": "text",
        "label_field": "label",
        "num_classes": 2,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "sentiment",
        "class_names": ["negative", "positive"],
    },

    "yelp": {
        "hf_path": "yelp_review_full",
        "hf_name": None,
        "train_split": "train[:50000]",  # Subsample for speed
        "val_split": "test[:5000]",
        "test_split": "test[5000:15000]",
        "text_field": "text",
        "label_field": "label",
        "num_classes": 5,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "sentiment",
        "class_names": ["1_star", "2_star", "3_star", "4_star", "5_star"],
        "is_ordinal": True,
    },

    # =========================================================================
    # TOXICITY DETECTION
    # =========================================================================
    "hatexplain": {
        "hf_path": "hatexplain",
        "hf_name": None,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "post_tokens",  # List of tokens, needs joining
        "label_field": "annotators",  # Dict with 'label' key
        "num_classes": 3,
        "has_concepts": True,
        "concept_names": ["has_target", "is_offensive"],
        "has_multi_annotator": True,
        "num_annotators": 3,
        "task": "toxicity",
        "class_names": ["hatespeech", "normal", "offensive"],
        # Note: HF uses 0=hatespeech, 1=normal, 2=offensive
    },

    "civil_comments": {
        "hf_path": "google/civil_comments",
        "hf_name": None,
        "train_split": "train[:50000]",
        "val_split": "validation[:5000]",
        "test_split": "test[:10000]",
        "text_field": "text",
        "label_field": "toxicity",  # Float 0-1
        "num_classes": 2,
        "has_concepts": True,
        "concept_names": ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"],
        "concept_fields": ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"],
        "has_multi_annotator": True,
        "task": "toxicity",
        "class_names": ["non_toxic", "toxic"],
    },

    # =========================================================================
    # EMOTION DETECTION
    # =========================================================================
    "goemotions": {
        "hf_path": "google-research-datasets/go_emotions",
        "hf_name": "simplified",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "text",
        "label_field": "labels",  # List of ints (multi-label)
        "num_classes": 28,
        "has_concepts": True,
        "concept_names": [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ],
        "has_multi_annotator": True,
        "task": "emotion",
    },

    # =========================================================================
    # NATURAL LANGUAGE INFERENCE
    # =========================================================================
    "chaosnli": {
        "hf_path": "chaosnli",
        "hf_name": None,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "num_classes": 3,
        "has_concepts": False,
        "has_multi_annotator": True,
        "task": "nli",
        "class_names": ["entailment", "neutral", "contradiction"],
    },

    "tid8": {
        "hf_path": "MichiganNLP/TID-8",
        "hf_name": "commitmentbank-ann",  # TID-8 requires a config name
        "train_split": "train",
        "val_split": "test[:50%]",  # Split test: first 50% for validation
        "test_split": "test[50%:]",  # Second 50% for test
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "num_classes": 3,
        "has_concepts": False,
        "has_multi_annotator": True,
        "task": "nli",
        "class_names": ["entailment", "neutral", "contradiction"],
    },

    # =========================================================================
    # SNLI with Annotator Disagreement Concepts
    # =========================================================================
    "snli": {
        "hf_path": "stanfordnlp/snli",
        "hf_name": None,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "num_classes": 3,
        "has_concepts": True,
        "concept_names": ["entailment", "neutral", "contradiction"],
        "has_multi_annotator": True,
        "task": "nli",
        "class_names": ["entailment", "neutral", "contradiction"],
    },
}

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    label_type: str = "ternary"  # For CEBaB: "binary" | "ternary" | "5way"
    max_length: int = 128
    tokenizer_name: str = "distilbert-base-uncased"
    batch_size: int = 16
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    # If True, Dataset.__getitem__ returns raw text fields; tokenization occurs in collate_fn (batched, dynamic padding)
    defer_tokenization: bool = False

# =============================================================================
# DATASET CLASS
# =============================================================================

class CredenceDataset(Dataset):
    """Universal dataset class for CREDENCE."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        config: DatasetConfig,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.config = config
        self.info = DATASET_INFO[dataset_name]

        # Will be populated by loaders
        self.examples = []
        self._load_data(split)

        # Get metadata after loading
        self.num_classes = self._get_num_classes()
        self.num_concepts = self._get_num_concepts()
        self.concept_names = self.info.get("concept_names", [])

        print(f"  Loaded {len(self.examples)} examples from {dataset_name}/{split}")

    def _get_num_concepts(self) -> int:
        """Get number of concepts."""
        if self.examples and len(self.examples[0]["concepts"]) > 0:
            return len(self.examples[0]["concepts"])
        return len(self.info.get("concept_names", []))

    def _load_data(self, split: str):
        """Load data from HuggingFace."""
        info = self.info

        # Get split name
        split_key = f"{split}_split"
        split_name = info.get(split_key, split)

        # Load from HuggingFace
        hf_name = info.get("hf_name")
        try:
            if hf_name:
                # When hf_name is provided, pass it as the 'name' parameter
                ds = hf_load_dataset(info["hf_path"], name=hf_name, split=split_name)
            else:
                ds = hf_load_dataset(info["hf_path"], split=split_name)
        except Exception as e:
            print(f"  Error loading {self.dataset_name}/{split}: {e}")
            return

        # Route to appropriate loader
        loader_map = {
            "cebab": self._load_cebab,
            "hatexplain": self._load_hatexplain,
            "civil_comments": self._load_civil_comments,
            "goemotions": self._load_goemotions,
            "chaosnli": self._load_nli,
            "tid8": self._load_tid8,
            "snli": self._load_snli,
        }

        loader = loader_map.get(self.dataset_name, self._load_generic)
        loader(ds)

    def _load_cebab(self, ds):
        """
        Load CEBaB dataset with aleatoric uncertainty information.

        Fields:
          - description: str
          - review_majority: str ("1"-"5", "no majority")
          - *_aspect_majority: str ("Positive", "Negative", "unknown", "")
          - *_aspect_label_distribution: dict with annotator counts

        Concept encoding (ternary):
          - 0 = Negative
          - 1 = Unknown (can't tell / empty)
          - 2 = Positive

        Aleatoric uncertainty:
          - concept_distributions: [K, 3] probability distributions from annotators
          - concept_entropy: [K] normalized entropy (0=certain, 1=uncertain)
        """
        concept_fields = self.info["concept_fields"]

        for sample in ds:
            # Get label
            label = self._encode_cebab_label(sample)
            if label == -1:
                continue

            text = sample["description"]
            if not text or len(text.strip()) == 0:
                continue

            # Encode concepts (ternary majority vote)
            concepts = []
            for field in concept_fields:
                value = sample.get(field, "")
                if value == "Positive":
                    concepts.append(2)
                elif value == "Negative":
                    concepts.append(0)
                else:  # "unknown" or ""
                    concepts.append(1)

            concepts = np.array(concepts, dtype=np.int64)
            is_unknown = (concepts == 1).astype(np.float32)

            # Build aleatoric uncertainty targets from annotator distributions
            concept_distributions, concept_entropy = build_cebab_aleatoric_targets(sample)

            self.examples.append({
                "text": text,
                "label": label,
                "concepts": concepts,
                "is_unknown": is_unknown,
                "concept_distributions": concept_distributions,  # [K, 3] probs
                "concept_entropy": concept_entropy,              # [K] normalized entropy
            })

    def _encode_cebab_label(self, sample) -> int:
        """Encode CEBaB review_majority to label."""
        value = sample.get("review_majority", "")

        if not isinstance(value, str) or value == "no majority":
            return -1

        # Extract star rating
        try:
            star = int(value.strip())
        except ValueError:
            return -1

        if star < 1 or star > 5:
            return -1

        if self.config.label_type == "binary":
            return 0 if star <= 2 else 1
        elif self.config.label_type == "ternary":
            if star <= 2:
                return 0  # Negative
            elif star == 3:
                return 1  # Neutral
            else:
                return 2  # Positive
        else:  # 5way
            return star - 1

    def _load_hatexplain(self, ds):
        """
        Load HateXplain dataset.

        Fields:
          - post_tokens: list[str]
          - annotators: dict
              - label: list[int] (0=hatespeech, 1=normal, 2=offensive)
              - target: list[list[str]]

        Concept encoding (ternary):
          - Concept 1 (has_target): 0=No, 1=Disagree, 2=Yes
          - Concept 2 (is_offensive): 0=Normal, 1=Disagree, 2=Offensive/Hate
        """
        for sample in ds:
            # Get tokens and join to text
            tokens = sample.get("post_tokens", [])
            if not tokens:
                continue
            text = " ".join(tokens)
            if len(text.strip()) == 0:
                continue

            # Get labels from annotators
            annotators = sample.get("annotators", {})
            labels = annotators.get("label", [])
            targets = annotators.get("target", [])

            if not labels:
                continue

            # Majority vote for label
            label_counts = Counter(labels)
            majority_label, majority_count = label_counts.most_common(1)[0]

            # Check agreement
            total = len(labels)
            has_disagreement = (majority_count / total) < 1.0

            # Check if any annotator found a target
            has_any_target = any(
                len(t) > 0 for t in targets if isinstance(t, list)
            )

            # Concept 1: Has target (ternary)
            if has_disagreement:
                target_concept = 1  # Disagreement
            elif has_any_target:
                target_concept = 2  # Yes
            else:
                target_concept = 0  # No

            # Concept 2: Is offensive (ternary)
            # 0=hatespeech, 1=normal, 2=offensive in HF
            if has_disagreement:
                offensive_concept = 1  # Disagreement
            elif majority_label in [0, 2]:  # hatespeech or offensive
                offensive_concept = 2  # Offensive
            else:
                offensive_concept = 0  # Normal

            concepts = np.array([target_concept, offensive_concept], dtype=np.int64)
            is_unknown = (concepts == 1).astype(np.float32)

            self.examples.append({
                "text": text,
                "label": majority_label,
                "concepts": concepts,
                "is_unknown": is_unknown,
            })

    def _load_civil_comments(self, ds):
        """
        Load Civil Comments dataset.

        Fields:
          - text: str
          - toxicity: float (0-1)
          - severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit: float

        Concept encoding (ternary):
          - 0 = Low (< 0.1)
          - 1 = Medium (0.1 - 0.5) - borderline/unclear
          - 2 = High (>= 0.5)
        """
        concept_fields = self.info.get("concept_fields", [])

        for sample in ds:
            text = sample.get("text", "")
            if not text or len(text.strip()) == 0:
                continue

            toxicity = sample.get("toxicity", 0.0)
            if toxicity is None:
                toxicity = 0.0

            # Binary label
            label = 1 if toxicity >= 0.5 else 0

            # Concepts from toxicity subtypes
            concepts = []
            for field in concept_fields:
                val = sample.get(field, 0.0)
                if val is None:
                    val = 0.0

                if val >= 0.5:
                    concepts.append(2)  # High
                elif val >= 0.1:
                    concepts.append(1)  # Medium/borderline
                else:
                    concepts.append(0)  # Low

            concepts = np.array(concepts, dtype=np.int64)
            is_unknown = (concepts == 1).astype(np.float32)

            self.examples.append({
                "text": text,
                "label": label,
                "concepts": concepts,
                "is_unknown": is_unknown,
            })

    def _load_goemotions(self, ds):
        """
        Load GoEmotions (simplified) dataset.

        Fields:
          - text: str
          - labels: list[int] (emotion indices 0-27)

        Multi-label to single-label: take first label.
        Concepts: emotion presence (ternary per emotion).
        """
        num_emotions = 28

        for sample in ds:
            text = sample.get("text", "")
            labels = sample.get("labels", [])

            if not text or len(text.strip()) == 0:
                continue
            if not labels:
                continue

            # Primary label is first
            label = labels[0]
            if label >= num_emotions:
                continue

            # Multi-label indicates ambiguity
            is_ambiguous = len(labels) > 1

            # Concepts: emotion presence (ternary)
            concepts = np.zeros(num_emotions, dtype=np.int64)
            for l in labels:
                if l < num_emotions:
                    if is_ambiguous:
                        concepts[l] = 1  # Present but ambiguous
                    else:
                        concepts[l] = 2  # Clearly present

            is_unknown = (concepts == 1).astype(np.float32)

            self.examples.append({
                "text": text,
                "label": label,
                "concepts": concepts,
                "is_unknown": is_unknown,
            })

    def _load_nli(self, ds):
        """
        Load NLI dataset (ChaosNLI).

        Fields:
          - premise: str
          - hypothesis: str
          - label: int (0=entailment, 1=neutral, 2=contradiction, -1=skip)
        """
        for sample in ds:
            label = sample.get("label", -1)
            if label == -1:  # Skip unlabeled
                continue

            premise = sample.get("premise", "")
            hypothesis = sample.get("hypothesis", "")

            if not premise or not hypothesis:
                continue

            text = f"{premise} [SEP] {hypothesis}"

            self.examples.append({
                "text": text,
                "label": label,
                "concepts": np.array([], dtype=np.int64),
                "is_unknown": np.array([], dtype=np.float32),
            })

    def _load_tid8(self, ds):
        """
        Load TID-8 dataset (MichiganNLP/TID-8).

        Fields:
          - Context: str (premise)
          - Target: str (hypothesis)
          - answer_label: int (0, 1, 2, 3, -3, -1, -2)
          - question: str (Context</s>Target</s>Prompt format)
        """
        for sample in ds:
            # TID-8 uses Context and Target instead of premise/hypothesis
            premise = sample.get("Context", "")
            hypothesis = sample.get("Target", "")

            # If Context/Target not available, try parsing from question field
            if not premise or not hypothesis:
                question = sample.get("question", "")
                if question and "</s>" in question:
                    parts = question.split("</s>")
                    if len(parts) >= 2:
                        premise = parts[0].strip()
                        hypothesis = parts[1].strip()

            if not premise or not hypothesis:
                continue

            # Get label - TID-8 uses answer_label
            label = sample.get("answer_label", -1)

            # TID-8 labels: 0, 1, 2, 3, -3, -1, -2
            # Map to standard NLI labels: 0=entailment, 1=neutral, 2=contradiction
            # Based on TID-8 documentation: 0=entailment, 1=neutral, 2=contradiction, 3=unknown
            # Negative values might be invalid/unknown, skip them
            if label < 0 or label > 2:
                continue  # Skip invalid labels (3, -3, -1, -2)

            text = f"{premise} [SEP] {hypothesis}"

            self.examples.append({
                "text": text,
                "label": int(label),  # Already in correct format (0, 1, 2)
                "concepts": np.array([], dtype=np.int64),
                "is_unknown": np.array([], dtype=np.float32),
            })

    def _load_snli(self, ds):
        """
        Load SNLI dataset with annotator disagreement concepts.

        Fields (from HuggingFace stanfordnlp/snli):
          - premise: str
          - hypothesis: str
          - label: int (0=entailment, 1=neutral, 2=contradiction, -1=unlabeled)
          - annotator_labels: list of 5 strings (when available)

        Concept encoding (ternary):
          Concepts = annotator proportions per class, discretized:
            - 0 (low):  proportion <= 0.2
            - 1 (unknown): 0.2 < proportion < 0.6  (disagreement!)
            - 2 (high): proportion >= 0.6

          The "unknown" concepts capture annotator disagreement (epistemic signal).
        """
        LABEL_MAP_LOCAL = {"entailment": 0, "neutral": 1, "contradiction": 2}
        low_thresh = 0.2
        high_thresh = 0.6
        fallback_count = 0

        for sample in ds:
            label = sample.get("label", -1)
            if label == -1:
                continue

            premise = sample.get("premise", "")
            hypothesis = sample.get("hypothesis", "")
            if not premise or not hypothesis:
                continue

            text = f"{premise} [SEP] {hypothesis}"

            # Get annotator labels if available
            ann_labels = sample.get("annotator_labels", None)
            if ann_labels:
                # Filter valid entries
                ann_labels = [a for a in ann_labels
                              if a and a.strip() and a.strip().lower() in LABEL_MAP_LOCAL]

            if ann_labels and len(ann_labels) >= 3:
                # Compute proportions from annotator votes
                n = len(ann_labels)
                counts = np.zeros(3, dtype=np.float32)
                for lbl in ann_labels:
                    counts[LABEL_MAP_LOCAL[lbl.strip().lower()]] += 1
                proportions = counts / n

                # Convert to ternary concepts
                concepts = np.zeros(3, dtype=np.int64)
                for k in range(3):
                    if proportions[k] >= high_thresh:
                        concepts[k] = 2  # high agreement
                    elif proportions[k] > low_thresh:
                        concepts[k] = 1  # disagreement/unknown
                    else:
                        concepts[k] = 0  # low

                is_unknown = (concepts == 1).astype(np.float32)
            else:
                # Fallback: use gold label as deterministic concept
                concepts = np.array([0, 0, 0], dtype=np.int64)
                concepts[label] = 2
                is_unknown = np.zeros(3, dtype=np.float32)
                proportions = np.zeros(3, dtype=np.float32)
                proportions[label] = 1.0
                fallback_count += 1

            self.examples.append({
                "text": text,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "concepts": concepts,
                "is_unknown": is_unknown,
            })

        if fallback_count > 0:
            print(f"    SNLI: {fallback_count} examples used fallback (gold label only)")

    def _load_generic(self, ds):
        """
        Generic loader for simple classification datasets.

        Handles:
          - SST-2: sentence, label
          - SST-5: text, label
          - IMDB: text, label
          - Yelp: text, label
        """
        text_field = self.info["text_field"]
        label_field = self.info["label_field"]

        for sample in ds:
            # Handle text field (could be list for NLI)
            if isinstance(text_field, list):
                parts = [sample.get(f, "") for f in text_field]
                text = " [SEP] ".join(parts)
            else:
                text = sample.get(text_field, "")

            if not text or len(text.strip()) == 0:
                continue

            label = sample.get(label_field)
            if label is None or label == -1:
                continue

            # Ensure int
            if isinstance(label, float):
                label = int(label)

            self.examples.append({
                "text": text,
                "label": label,
                "concepts": np.array([], dtype=np.int64),
                "is_unknown": np.array([], dtype=np.float32),
            })

    def _get_num_classes(self) -> int:
        """Get number of classes."""
        if self.dataset_name == "cebab":
            if self.config.label_type == "binary":
                return 2
            elif self.config.label_type == "ternary":
                return 3
            else:
                return 5
        return self.info.get("num_classes", 2)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # If deferring tokenization, just return raw fields (handled by collate_fn)
        if self.config.defer_tokenization:
            item = {
                'text': ex.get('text', None),
                'premise': ex.get('premise', None),
                'hypothesis': ex.get('hypothesis', None),
                'labels': torch.tensor(ex["label"], dtype=torch.long),
            }
        else:
            # Tokenize per-item (legacy path)
            encoding = self.tokenizer(
                ex["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(ex["label"], dtype=torch.long),
            }

        # Add concepts if available
        if len(ex["concepts"]) > 0:
            item['concept_labels'] = torch.tensor(ex["concepts"], dtype=torch.long)
            item['is_unknown'] = torch.tensor(ex["is_unknown"], dtype=torch.float)

            # Add aleatoric uncertainty targets if available (CEBaB only)
            if "concept_distributions" in ex:
                item['concept_distributions'] = torch.tensor(
                    ex["concept_distributions"], dtype=torch.float
                )  # [K, 3]
                item['concept_entropy'] = torch.tensor(
                    ex["concept_entropy"], dtype=torch.float
                )  # [K]
        else:
            # Placeholder for datasets without concepts
            item['concept_labels'] = torch.zeros(1, dtype=torch.long)
            item['is_unknown'] = torch.zeros(1, dtype=torch.float)

        return item

# =============================================================================
# MAIN LOADING FUNCTION
# =============================================================================

def load_dataset_splits(
    dataset_name: str,
    config: Optional[DatasetConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Dict]:
    """Load train/val/test splits for a dataset."""

    if config is None:
        config = DatasetConfig()

    if dataset_name not in DATASET_INFO:
        available = list(DATASET_INFO.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    info = DATASET_INFO[dataset_name]

    print(f"\nLoading dataset: {dataset_name}")
    print(f"  HF path: {info['hf_path']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load splits
    # Enable batched, dynamic padding tokenization for SNLI to speed up training significantly.
    use_batched_collate = dataset_name in ("snli", "chaosnli")
    if use_batched_collate:
        # Make a shallow copy-like config enabling deferred tokenization
        cfg = DatasetConfig(
            label_type=config.label_type,
            max_length=config.max_length,
            tokenizer_name=config.tokenizer_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers if config.num_workers > 0 else 2,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples,
            max_test_samples=config.max_test_samples,
            defer_tokenization=True,
        )
    else:
        cfg = config

    train_ds = CredenceDataset(dataset_name, "train", tokenizer, cfg)
    val_ds = CredenceDataset(dataset_name, "val", tokenizer, cfg)
    test_ds = CredenceDataset(dataset_name, "test", tokenizer, cfg)

    # Apply sample limits
    if config.max_train_samples and len(train_ds.examples) > config.max_train_samples:
        train_ds.examples = train_ds.examples[:config.max_train_samples]
    if config.max_val_samples and len(val_ds.examples) > config.max_val_samples:
        val_ds.examples = val_ds.examples[:config.max_val_samples]
    if config.max_test_samples and len(test_ds.examples) > config.max_test_samples:
        test_ds.examples = test_ds.examples[:config.max_test_samples]

    # Metadata
    metadata = {
        "dataset_name": dataset_name,
        "task": info.get("task", "classification"),
        "num_classes": train_ds.num_classes,
        "num_concepts": train_ds.num_concepts,
        "concept_names": train_ds.concept_names,
        "class_names": info.get("class_names", []),
        "has_concepts": info.get("has_concepts", False),
        "has_multi_annotator": info.get("has_multi_annotator", False),
        "is_ordinal": info.get("is_ordinal", False),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }

    # Label distribution
    if train_ds.examples:
        label_counts = Counter([ex["label"] for ex in train_ds.examples])
        metadata["label_distribution"] = dict(label_counts)

    # Unknown rates
    if train_ds.num_concepts > 0 and train_ds.examples:
        unknown_arrs = [ex["is_unknown"] for ex in train_ds.examples if len(ex["is_unknown"]) > 0]
        if unknown_arrs:
            unknown_rates = np.array(unknown_arrs).mean(axis=0)
            metadata["unknown_rates"] = {
                name: float(r) for name, r in zip(train_ds.concept_names[:len(unknown_rates)], unknown_rates)
            }

    # Print summary
    print(f"\n  Summary:")
    print(f"    Task: {metadata['task']}")
    print(f"    Classes: {metadata['num_classes']} {metadata['class_names']}")
    print(f"    Concepts: {metadata['num_concepts']} {metadata['concept_names'][:5]}{'...' if len(metadata['concept_names']) > 5 else ''}")
    print(f"    Sizes: train={metadata['train_size']}, val={metadata['val_size']}, test={metadata['test_size']}")
    if "label_distribution" in metadata:
        print(f"    Labels: {metadata['label_distribution']}")
    if "unknown_rates" in metadata:
        rates_str = ", ".join(f"{k}:{v:.2f}" for k, v in list(metadata["unknown_rates"].items())[:4])
        print(f"    Unknown rates: {rates_str}")

    # Check for empty datasets
    if len(train_ds) == 0 and len(val_ds) == 0 and len(test_ds) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' failed to load: all splits are empty. "
            f"This usually means the dataset doesn't exist on HuggingFace Hub or cannot be accessed. "
            f"Please check if the dataset name '{info.get('hf_path', dataset_name)}' is correct."
        )
    if len(train_ds) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' has no training examples. Cannot proceed with training."
        )

    # Collate function for dynamic, batched tokenization (speeds up SNLI)
    def build_collate(tokenizer, max_length):
        def collate_fn(batch):
            if batch and (batch[0].get('premise') is not None and batch[0].get('hypothesis') is not None):
                premises = [b['premise'] for b in batch]
                hyps = [b['hypothesis'] for b in batch]
                enc = tokenizer(
                    premises,
                    hyps,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors='pt',
                )
            else:
                texts = [b['text'] for b in batch]
                enc = tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors='pt',
                )

            out = {
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'labels': torch.stack([b['labels'] for b in batch], dim=0),
            }

            # Optional concept fields
            if 'concept_labels' in batch[0]:
                out['concept_labels'] = torch.stack([b['concept_labels'] for b in batch], dim=0)
            if 'is_unknown' in batch[0]:
                out['is_unknown'] = torch.stack([b['is_unknown'] for b in batch], dim=0)

            return out
        return collate_fn

    # Create loaders
    if use_batched_collate:
        collate = build_collate(tokenizer, cfg.max_length)
        nw = cfg.num_workers
        bs = cfg.batch_size
    else:
        collate = None
        nw = config.num_workers
        bs = config.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate
    )

    return train_loader, val_loader, test_loader, tokenizer, metadata

# =============================================================================
# RECOMMENDED CONFIGS
# =============================================================================

def get_recommended_config(dataset_name: str) -> Dict[str, Any]:
    """Get recommended training config for each dataset."""

    configs = {
        "cebab": {
            "epochs": 40,
            "lr": 1e-4,
            "batch_size": 16,
            "label_type": "ternary",
            "concept_weight": 1.0,
            "aleatoric_weight": 0.5,
        },
        "hatexplain": {
            "epochs": 30,
            "lr": 1e-4,
            "batch_size": 16,
            "label_type": "default",
            "concept_weight": 1.0,
            "aleatoric_weight": 0.5,
        },
        "civil_comments": {
            "epochs": 15,
            "lr": 1e-4,
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.5,
            "aleatoric_weight": 0.5,
        },
        "goemotions": {
            "epochs": 20,
            "lr": 5e-5,
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.5,
            "aleatoric_weight": 0.5,
        },
        "sst2": {
            "epochs": 20,
            "lr": 1e-4,
            "batch_size": 32,
            "label_type": "binary",
            "concept_weight": 0.0,
            "aleatoric_weight": 0.0,
        },
        "sst5": {
            "epochs": 30,
            "lr": 1e-4,
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.0,
            "aleatoric_weight": 0.0,
        },
        "chaosnli": {
            "epochs": 10,
            "lr": 2e-5,
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.0,
            "aleatoric_weight": 0.0,
        },
        "tid8": {
            "epochs": 10,
            "lr": 2e-5,
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.0,
            "aleatoric_weight": 0.0,
        },
    }

    return configs.get(dataset_name, {
        "epochs": 20,
        "lr": 1e-4,
        "batch_size": 16,
        "label_type": "default",
        "concept_weight": 1.0,
        "aleatoric_weight": 0.5,
    })

def list_datasets():
    """List all available datasets with their properties."""

    print("\n" + "="*80)
    print("AVAILABLE DATASETS")
    print("="*80)

    # Group by task
    by_task = {}
    for name, info in DATASET_INFO.items():
        task = info.get("task", "other")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(name)

    for task in ["sentiment", "toxicity", "emotion", "nli", "topic"]:
        if task not in by_task:
            continue

        print(f"\n{task.upper()}:")
        print("-" * 60)

        for ds in by_task[task]:
            info = DATASET_INFO[ds]
            concepts = "✓" if info.get("has_concepts") else "✗"
            multi_ann = "✓" if info.get("has_multi_annotator") else "✗"
            n_cls = info.get("num_classes", "?")
            n_con = len(info.get("concept_names", []))

            print(f"  {ds:20} classes={n_cls:2}  concepts={n_con:2} ({concepts})  multi_ann={multi_ann}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    list_datasets()

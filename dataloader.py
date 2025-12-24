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
                ds = hf_load_dataset(info["hf_path"], name=hf_name, split=split_name, trust_remote_code=True)
            else:
                ds = hf_load_dataset(info["hf_path"], split=split_name, trust_remote_code=True)
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
        }
        
        loader = loader_map.get(self.dataset_name, self._load_generic)
        loader(ds)
    
    def _load_cebab(self, ds):
        """
        Load CEBaB dataset.
        
        Fields:
          - description: str
          - review_majority: str ("1"-"5", "no majority")
          - *_aspect_majority: str ("Positive", "Negative", "unknown", "")
        
        Concept encoding (ternary):
          - 0 = Negative
          - 1 = Unknown (can't tell / empty)
          - 2 = Positive
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
            
            # Encode concepts (ternary)
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
            
            self.examples.append({
                "text": text,
                "label": label,
                "concepts": concepts,
                "is_unknown": is_unknown,
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
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load splits
    train_ds = CredenceDataset(dataset_name, "train", tokenizer, config)
    val_ds = CredenceDataset(dataset_name, "val", tokenizer, config)
    test_ds = CredenceDataset(dataset_name, "test", tokenizer, config)
    
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
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
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

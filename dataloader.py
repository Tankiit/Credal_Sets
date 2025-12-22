"""
CREDENCE Multi-Dataset Dataloader (Fixed)

==========================================

Key fixes for HateXplain:

1. Use ternary concepts (0=Neg, 1=Unknown, 2=Pos) like CEBaB

2. is_unknown is binary (0 or 1), not continuous

3. Multiple concepts instead of single binary

Supports:

- Sentiment: CEBaB, SST-2, SST-5, IMDB, Yelp, Amazon

- Toxicity: HateXplain, Civil Comments

- Emotion: GoEmotions

- NLI: MNLI, SNLI

- Topic: AG News

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
# DATASET REGISTRY
# =============================================================================

DATASET_INFO = {
    # =========================================================================
    # SENTIMENT ANALYSIS
    # =========================================================================
    "cebab": {
        "hf_path": "CEBaB/CEBaB",
        "train_split": "train_inclusive",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "description",
        "label_field": "review_majority",
        "concept_names": ["food", "service", "ambiance", "noise"],
        "concept_fields": ["food_aspect_majority", "service_aspect_majority",
                          "ambiance_aspect_majority", "noise_aspect_majority"],
        "num_classes_binary": 2,
        "num_classes_ternary": 3,
        "num_classes_5way": 5,
        "has_concepts": True,
        "has_multi_annotator": True,
        "task": "sentiment",
    },
    "sst2": {
        "hf_path": "glue",
        "hf_name": "sst2",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "validation",
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
        "train_split": "train",
        "val_split": "test[:5000]",
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
        "train_split": "train[:50000]",
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
    "amazon": {
        "hf_path": "amazon_polarity",
        "train_split": "train[:50000]",
        "val_split": "test[:5000]",
        "test_split": "test[5000:15000]",
        "text_field": "content",
        "label_field": "label",
        "num_classes": 2,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "sentiment",
        "class_names": ["negative", "positive"],
    },
    
    # =========================================================================
    # TOXICITY DETECTION
    # =========================================================================
    "hatexplain": {
        "hf_path": "hatexplain",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "post_tokens",
        "label_field": "annotators",
        "num_classes": 3,
        "has_concepts": True,
        "concept_names": ["has_target", "is_offensive"],  # Two ternary concepts
        "has_multi_annotator": True,
        "num_annotators": 3,
        "task": "toxicity",
        "class_names": ["normal", "offensive", "hatespeech"],
    },
    "civil_comments": {
        "hf_path": "civil_comments",
        "train_split": "train[:50000]",
        "val_split": "validation[:5000]",
        "test_split": "test[:10000]",
        "text_field": "text",
        "label_field": "toxicity",
        "num_classes": 2,
        "has_concepts": True,
        "concept_names": ["male", "female", "christian", "muslim", "jewish",
                         "black", "white", "psychiatric_or_mental_illness"],
        "concept_fields": ["male", "female", "christian", "muslim", "jewish",
                          "black", "white", "psychiatric_or_mental_illness"],
        "has_multi_annotator": True,
        "task": "toxicity",
        "class_names": ["non_toxic", "toxic"],
    },
    
    # =========================================================================
    # EMOTION DETECTION
    # =========================================================================
    "goemotions": {
        "hf_path": "go_emotions",
        "hf_name": "simplified",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": "text",
        "label_field": "labels",
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
        "num_annotators": 3,
        "task": "emotion",
    },
    
    # =========================================================================
    # NATURAL LANGUAGE INFERENCE
    # =========================================================================
    "mnli": {
        "hf_path": "glue",
        "hf_name": "mnli",
        "train_split": "train",
        "val_split": "validation_matched",
        "test_split": "validation_mismatched",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "num_classes": 3,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "nli",
        "class_names": ["entailment", "neutral", "contradiction"],
    },
    "snli": {
        "hf_path": "snli",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "text_field": ["premise", "hypothesis"],
        "label_field": "label",
        "num_classes": 3,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "nli",
        "class_names": ["entailment", "neutral", "contradiction"],
    },
    
    # =========================================================================
    # TOPIC CLASSIFICATION
    # =========================================================================
    "ag_news": {
        "hf_path": "ag_news",
        "train_split": "train",
        "val_split": "test[:5000]",
        "test_split": "test[5000:]",
        "text_field": "text",
        "label_field": "label",
        "num_classes": 4,
        "has_concepts": False,
        "has_multi_annotator": False,
        "task": "topic",
        "class_names": ["world", "sports", "business", "sci_tech"],
    },
}

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    label_type: str = "ternary"  # "binary" | "ternary" | "5way" | "default"
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
        
        # Load data
        self.examples = []
        self._load_data(split)
        
        # Get metadata
        self.num_classes = self._get_num_classes()
        self.num_concepts = len(self.info.get("concept_names", []))
        self.concept_names = self.info.get("concept_names", [])
        
        print(f"Loaded {len(self.examples)} from {dataset_name}/{split}")
    
    def _load_data(self, split: str):
        """Load data from HuggingFace."""
        info = self.info
        
        # Handle split name
        split_name = info.get(f"{split}_split", split)
        
        # Load from HuggingFace
        hf_name = info.get("hf_name", None)
        if hf_name:
            ds = hf_load_dataset(info["hf_path"], hf_name, split=split_name)
        else:
            ds = hf_load_dataset(info["hf_path"], split=split_name)
        
        # Process based on dataset
        if self.dataset_name == "cebab":
            self._load_cebab(ds)
        elif self.dataset_name == "hatexplain":
            self._load_hatexplain(ds)
        elif self.dataset_name == "civil_comments":
            self._load_civil_comments(ds)
        elif self.dataset_name == "goemotions":
            self._load_goemotions(ds)
        elif self.dataset_name in ["mnli", "snli"]:
            self._load_nli(ds)
        else:
            self._load_generic(ds)
    
    def _load_cebab(self, ds):
        """Load CEBaB dataset with concepts."""
        for sample in ds:
            label = self._encode_cebab_label(sample)
            if label == -1:
                continue
            
            text = sample["description"]
            concepts = []
            is_unknown = []
            
            for col in self.info["concept_fields"]:
                value = sample[col]
                if value == "Positive":
                    concepts.append(2)
                    is_unknown.append(0.0)
                elif value == "Negative":
                    concepts.append(0)
                    is_unknown.append(0.0)
                else:  # Unknown
                    concepts.append(1)
                    is_unknown.append(1.0)
            
            self.examples.append({
                "text": text,
                "label": label,
                "concepts": np.array(concepts, dtype=np.int64),
                "is_unknown": np.array(is_unknown, dtype=np.float32),
            })
    
    def _encode_cebab_label(self, sample) -> int:
        """Encode CEBaB label based on config."""
        value = sample.get('review_majority', '')
        if not isinstance(value, str):
            return -1
        value = value.lower().strip()
        
        # Extract star rating
        star = None
        for s in ["1", "2", "3", "4", "5"]:
            if s in value:
                star = int(s)
                break
        
        if star is None:
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
        Load HateXplain with multi-annotator labels.
        
        FIXED: Uses ternary concept encoding (like CEBaB):
        - 0 = Negative (no target / normal)
        - 1 = Unknown (annotators disagree)  
        - 2 = Positive (has target / offensive)
        
        Concepts: [has_target, is_offensive]
        """
        for sample in ds:
            # Get majority label from annotators
            labels = sample["annotators"]["label"]
            if len(labels) == 0:
                continue
            
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]
            majority_count = label_counts.most_common(1)[0][1]
            
            # Tokens to text
            text = " ".join(sample["post_tokens"])
            if len(text.strip()) == 0:
                continue
            
            # Compute annotator agreement
            total_annotations = len(labels)
            agreement_ratio = majority_count / total_annotations
            
            # Determine if there's disagreement
            has_disagreement = agreement_ratio < 1.0  # Not unanimous
            
            # Get target communities
            targets = sample.get("annotators", {}).get("target", [])
            all_targets = []
            for t_list in targets:
                if isinstance(t_list, list):
                    all_targets.extend(t_list)
            has_any_target = len(all_targets) > 0
            
            # Concept 1: Has target community (ternary)
            if has_disagreement:
                target_concept = 1  # Unknown - annotators disagree
            elif has_any_target:
                target_concept = 2  # Positive - clear target
            else:
                target_concept = 0  # Negative - no target
            
            # Concept 2: Is offensive (based on label, ternary)
            if has_disagreement:
                offensive_concept = 1  # Unknown
            elif majority_label in [1, 2]:  # offensive or hatespeech
                offensive_concept = 2  # Positive
            else:
                offensive_concept = 0  # Negative (normal)
            
            # Build concepts array (ternary like CEBaB)
            concepts = np.array([target_concept, offensive_concept], dtype=np.int64)
            
            # is_unknown: 1.0 if concept==1 (unknown), 0.0 otherwise
            is_unknown = (concepts == 1).astype(np.float32)
            
            self.examples.append({
                "text": text,
                "label": majority_label,
                "concepts": concepts,
                "is_unknown": is_unknown,
            })
    
    def _load_civil_comments(self, ds):
        """
        Load Civil Comments with identity attributes.
        
        FIXED: Uses ternary encoding for identity mentions:
        - 0 = Not mentioned
        - 1 = Unclear/borderline  
        - 2 = Clearly mentioned
        """
        concept_fields = self.info["concept_fields"]
        
        for sample in ds:
            text = sample["text"]
            if len(text.strip()) == 0:
                continue
                
            toxicity = sample["toxicity"]
            
            # Binary label
            label = 1 if toxicity >= 0.5 else 0
            
            # Identity attributes as ternary concepts
            concepts = []
            for field in concept_fields:
                val = sample.get(field, 0) or 0
                if val >= 0.5:
                    concepts.append(2)  # Clearly mentioned
                elif val >= 0.1:
                    concepts.append(1)  # Borderline/unclear
                else:
                    concepts.append(0)  # Not mentioned
            
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
        Load GoEmotions (multi-label to single-label).
        
        FIXED: Uses ternary encoding:
        - 0 = Emotion not present
        - 1 = Ambiguous (multiple emotions)
        - 2 = Emotion present
        """
        for sample in ds:
            text = sample["text"]
            labels = sample["labels"]
            
            if len(labels) == 0:
                continue
            
            # Take first label as primary
            label = labels[0]
            
            # Multi-label indicates ambiguity
            is_ambiguous = len(labels) > 1
            
            # Emotions as ternary concepts
            concepts = np.zeros(28, dtype=np.int64)
            for l in labels:
                if l < 28:
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
        """Load NLI dataset (premise + hypothesis)."""
        for sample in ds:
            label = sample["label"]
            if label == -1:
                continue
            
            premise = sample["premise"]
            hypothesis = sample["hypothesis"]
            text = f"{premise} [SEP] {hypothesis}"
            
            self.examples.append({
                "text": text,
                "label": label,
                "concepts": np.array([], dtype=np.int64),
                "is_unknown": np.array([], dtype=np.float32),
            })
    
    def _load_generic(self, ds):
        """Generic loader for simple text classification."""
        text_field = self.info["text_field"]
        label_field = self.info["label_field"]
        
        for sample in ds:
            text = sample[text_field]
            label = sample[label_field]
            
            if label is None or label == -1:
                continue
            
            self.examples.append({
                "text": text,
                "label": label,
                "concepts": np.array([], dtype=np.int64),
                "is_unknown": np.array([], dtype=np.float32),
            })
    
    def _get_num_classes(self) -> int:
        """Get number of classes based on config."""
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
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")
    
    info = DATASET_INFO[dataset_name]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load splits
    train_ds = CredenceDataset(dataset_name, "train", tokenizer, config)
    val_ds = CredenceDataset(dataset_name, "val", tokenizer, config)
    test_ds = CredenceDataset(dataset_name, "test", tokenizer, config)
    
    # Apply sample limits
    if config.max_train_samples:
        train_ds.examples = train_ds.examples[:config.max_train_samples]
    if config.max_val_samples:
        val_ds.examples = val_ds.examples[:config.max_val_samples]
    if config.max_test_samples:
        test_ds.examples = test_ds.examples[:config.max_test_samples]
    
    # Metadata
    metadata = {
        "dataset_name": dataset_name,
        "task": info.get("task", "classification"),
        "num_classes": train_ds.num_classes,
        "num_concepts": train_ds.num_concepts,
        "concept_names": train_ds.concept_names,
        "has_concepts": info.get("has_concepts", False),
        "has_multi_annotator": info.get("has_multi_annotator", False),
        "is_ordinal": info.get("is_ordinal", False),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    
    # Compute stats
    label_counts = Counter([ex["label"] for ex in train_ds.examples])
    metadata["label_distribution"] = dict(label_counts)
    
    if train_ds.num_concepts > 0:
        unknown_rates = np.array([ex["is_unknown"] for ex in train_ds.examples]).mean(axis=0)
        metadata["unknown_rates"] = {name: float(r) for name, r in zip(train_ds.concept_names, unknown_rates)}
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Task: {metadata['task']}")
    print(f"  Classes: {metadata['num_classes']} - {info.get('class_names', [])}")
    print(f"  Concepts: {metadata['num_concepts']} ({metadata['concept_names']})")
    print(f"  Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}")
    print(f"  Label distribution: {metadata['label_distribution']}")
    if 'unknown_rates' in metadata:
        print(f"  Unknown rates: {metadata['unknown_rates']}")
    
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
# DATASET-SPECIFIC CONFIGS
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
            "epochs": 30,  # Smaller dataset
            "lr": 1e-4,
            "batch_size": 16,
            "label_type": "default",
            "concept_weight": 1.0,
            "aleatoric_weight": 0.5,
        },
        "sst2": {
            "epochs": 20,
            "lr": 1e-4,
            "batch_size": 32,
            "label_type": "binary",
            "concept_weight": 0.0,  # No concepts
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
        "goemotions": {
            "epochs": 20,
            "lr": 5e-5,  # Lower LR for many classes
            "batch_size": 32,
            "label_type": "default",
            "concept_weight": 0.5,
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
    """List all available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 80)
    
    by_task = {}
    for name, info in DATASET_INFO.items():
        task = info.get("task", "other")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(name)
    
    for task, datasets in by_task.items():
        print(f"\n{task.upper()}:")
        for ds in datasets:
            info = DATASET_INFO[ds]
            concepts = "✓" if info.get("has_concepts") else "✗"
            multi_ann = "✓" if info.get("has_multi_annotator") else "✗"
            n_cls = info.get("num_classes", info.get("num_classes_ternary", "?"))
            print(f"  {ds:20} classes={n_cls:2}  concepts={concepts}  multi_ann={multi_ann}")

if __name__ == "__main__":
    list_datasets()


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict, List, Any
from abc import ABC, abstractmethod
import warnings


# ============================================================================
# BASE CONFIGURATION
# ============================================================================

@dataclass
class DatasetConfig:
    
    # Label configuration
    label_type: str = "default"  # Dataset-specific default
    
    # Concept configuration  
    concept_encoding: Literal["ternary", "binary"] = "ternary"
    
    # Tokenization
    max_length: int = 128
    tokenizer_name: str = "distilbert-base-uncased"
    
    # Data loading
    batch_size: int = 16
    max_samples: Optional[int] = None  # Limit samples (for debugging)
    
    # Splits
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


# ============================================================================
# DATASET REGISTRY
# ============================================================================

DATASET_INFO = {
    "cebab": {
        "hf_path": "CEBaB/CEBaB",
        "task": "sentiment",
        "has_concepts": True,
        "has_multi_annotator": True,
        "concept_names": ["food", "service", "ambiance", "noise"],
        "label_types": ["binary", "binary_with_neutral", "ternary", "multiclass"],
        "default_label_type": "multiclass",
        "num_classes": {"binary": 2, "binary_with_neutral": 2, "ternary": 3, "multiclass": 5},
        "train_split": "train_inclusive",
    },
    "hatexplain": {
        "hf_path": "Hate-speech-CNERG/hatexplain",
        "task": "hate_speech",
        "has_concepts": True,
        "has_multi_annotator": True,
        "concept_names": ["target_race", "target_religion", "target_gender", 
                         "target_sexuality", "target_origin", "target_disability"],
        "label_types": ["binary", "multiclass"],
        "default_label_type": "multiclass",
        "num_classes": {"binary": 2, "multiclass": 3},
    },
    "goemotions": {
        "hf_path": "go_emotions",
        "task": "emotion",
        "has_concepts": True,  # Emotions ARE the concepts
        "has_multi_annotator": True,
        "concept_names": [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ],
        "label_types": ["multilabel", "multiclass_dominant"],
        "default_label_type": "multilabel",
        "num_classes": {"multilabel": 28, "multiclass_dominant": 28},
        "hf_subset": "simplified",
    },
    "civil_comments": {
        "hf_path": "pietrolesci/civilcomments-wilds",
        "task": "toxicity",
        "has_concepts": True,  # Identity attributes
        "has_multi_annotator": True,
        "concept_names": ["male", "female", "LGBTQ", "christian", "muslim",
                         "jewish", "black", "white", "psychiatric_or_mental_illness"],
        "label_types": ["binary", "regression"],
        "default_label_type": "binary",
        "num_classes": {"binary": 2, "regression": 1},
    },
    "sst2": {
        "hf_path": "glue",
        "hf_subset": "sst2",
        "task": "sentiment",
        "has_concepts": False,
        "has_multi_annotator": False,
        "concept_names": [],  # Will use synthetic concepts
        "label_types": ["binary"],
        "default_label_type": "binary",
        "num_classes": {"binary": 2},
        "val_split": "validation",
        "test_split": "validation",  # SST-2 has no public test set
    },
    "ag_news": {
        "hf_path": "ag_news",
        "task": "topic",
        "has_concepts": False,
        "has_multi_annotator": False,
        "concept_names": [],  # Will use synthetic concepts
        "label_types": ["multiclass"],
        "default_label_type": "multiclass",
        "num_classes": {"multiclass": 4},
    },
}


# ============================================================================
# BASE DATASET CLASS
# ============================================================================

class BaseNLPDataset(Dataset, ABC):
    
    def __init__(self, split_data, tokenizer, config: DatasetConfig):
        self.tokenizer = tokenizer
        self.config = config
        
        self.texts = []
        self.labels = []
        self.concepts = []
        self.annotator_labels = []  # For multi-annotator datasets
        
        self._load_data(split_data)
    
    @abstractmethod
    def _load_data(self, split_data):
        pass
    
    @abstractmethod
    def _encode_label(self, sample) -> int:
        pass
    
    @abstractmethod
    def _encode_concepts(self, sample) -> np.ndarray:
        pass
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'concept_labels': torch.tensor(self.concepts[idx], dtype=torch.long)
        }
    
        # Add annotator labels if available
        if self.annotator_labels:
            item['annotator_labels'] = torch.tensor(
                self.annotator_labels[idx], dtype=torch.float
            )
        
        return item
    
    def get_stats(self) -> Dict:
        stats = {
            'num_samples': len(self.texts),
            'num_classes': len(set(self.labels)),
            'num_concepts': self.concepts.shape[1] if len(self.concepts) > 0 else 0,
            'label_distribution': dict(Counter(self.labels)),
        }
        
        if len(self.concepts) > 0:
            # Unknown rate per concept (assuming 1 = unknown)
            unknown_rates = {}
            for i in range(self.concepts.shape[1]):
                unknown_rates[f'concept_{i}'] = (self.concepts[:, i] == 1).mean()
            stats['unknown_rates'] = unknown_rates
            stats['overall_unknown_rate'] = (self.concepts == 1).mean()
        
        return stats


# ============================================================================
# CEBAB DATASET
# ============================================================================

class CEBaBDataset(BaseNLPDataset):
    
    CONCEPT_NAMES = ['food', 'service', 'ambiance', 'noise']
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            label = self._encode_label(split_data[i])
            if label == -1:
                continue
            
            self.texts.append(split_data['description'][i])
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(split_data[i]))
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        value = sample['review_majority']
        label_type = self.config.label_type
        
        if not isinstance(value, str):
            return -1
        
        value = value.lower().strip()
        
        if label_type == "binary":
            if "5" in value or "4" in value:
                return 1
            elif "1" in value or "2" in value:
                return 0
            return -1  # Skip neutral
        
        elif label_type == "binary_with_neutral":
            if "5" in value or "4" in value:
                return 1
            elif "1" in value or "2" in value or "3" in value:
                return 0
            return -1
        
        elif label_type == "ternary":
            if "5" in value or "4" in value:
                return 2
            elif "3" in value:
                return 1
            elif "1" in value or "2" in value:
                return 0
            return -1
        
        elif label_type == "multiclass":
            for i, star in enumerate(["1", "2", "3", "4", "5"]):
                if star in value:
                    return i
            return -1
        
        return -1
    
    def _encode_concepts(self, sample) -> np.ndarray:
        concepts = []
        for col in ['food_aspect_majority', 'service_aspect_majority',
                    'ambiance_aspect_majority', 'noise_aspect_majority']:
            value = sample[col]
            if value == "Positive":
                concepts.append(2)
            elif value == "Negative":
                concepts.append(0)
            else:  # unknown
                concepts.append(1)
        return np.array(concepts)


# ============================================================================
# HATEXPLAIN DATASET
# ============================================================================

class HateXplainDataset(BaseNLPDataset):
    
    CONCEPT_NAMES = ['target_race', 'target_religion', 'target_gender',
                     'target_sexuality', 'target_origin', 'target_disability',
                     'target_age', 'target_politics']
    
    TARGET_GROUPS = [
        'African', 'Arab', 'Asian', 'Caucasian', 'Hispanic', 'Indian',
        'Christian', 'Hindu', 'Jewish', 'Muslim', 'Buddhist',
        'Men', 'Women', 'Homosexual', 'Bisexual', 'Heterosexual',
        'Immigrant', 'Refugee', 'Indigenous', 'Disabled', 'Economic'
    ]
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            sample = split_data[i]
            label = self._encode_label(sample)
            
            if label == -1:
                continue
            
            # Join tokens into text
            text = " ".join(sample['post_tokens'])
            self.texts.append(text)
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(sample))
            
            # Store annotator labels for disagreement analysis
            if 'annotators' in sample:
                ann_labels = [ann['label'] for ann in sample['annotators']]
                self.annotator_labels.append(ann_labels)
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        label_type = self.config.label_type
        
        # Get annotator labels
        if 'annotators' in sample:
            ann_labels = [ann['label'] for ann in sample['annotators']]
            # Majority vote
            label_counts = Counter(ann_labels)
            majority_label = label_counts.most_common(1)[0][0]
        else:
            majority_label = sample.get('label', -1)
        
        if label_type == "binary":
            # 0 = normal, 1 = hate/offensive
            if majority_label == 0:  # normal
                return 0
            elif majority_label in [1, 2]:  # offensive or hate
                return 1
            return -1
        
        elif label_type == "multiclass":
            # 0 = normal, 1 = offensive, 2 = hate speech
            if majority_label in [0, 1, 2]:
                return majority_label
            return -1
        
        return -1
    
    def _encode_concepts(self, sample) -> np.ndarray:
        """
        Encode target group concepts
        0 = not targeted, 1 = unknown/ambiguous, 2 = targeted
        """
        concepts = np.ones(len(self.CONCEPT_NAMES), dtype=np.int64)  # Default: unknown
        
        if 'target' in sample:
            targets = sample['target']
            
            # Check each concept category
            for i, concept in enumerate(self.CONCEPT_NAMES):
                category = concept.replace('target_', '')
                
                # Check if any target in this category is mentioned
                is_targeted = False
                for target in targets:
                    if self._target_matches_category(target, category):
                        is_targeted = True
                        break
                
                if is_targeted:
                    concepts[i] = 2  # Targeted
                elif len(targets) > 0:
                    concepts[i] = 0  # Other target, not this category
                # else: keep as 1 (unknown)
        
        return concepts
    
    def _target_matches_category(self, target: str, category: str) -> bool:
        target = target.lower()
        category = category.lower()
        
        mappings = {
            'race': ['african', 'arab', 'asian', 'caucasian', 'hispanic', 'indian', 'black', 'white'],
            'religion': ['christian', 'hindu', 'jewish', 'muslim', 'buddhist', 'islam', 'jew'],
            'gender': ['men', 'women', 'male', 'female', 'woman', 'man'],
            'sexuality': ['homosexual', 'gay', 'lesbian', 'bisexual', 'lgbtq', 'queer'],
            'origin': ['immigrant', 'refugee', 'indigenous', 'native', 'foreigner'],
            'disability': ['disabled', 'disability', 'handicap', 'retard', 'mental'],
            'age': ['old', 'young', 'boomer', 'millennial', 'elderly'],
            'politics': ['liberal', 'conservative', 'democrat', 'republican', 'political']
        }
        
        if category in mappings:
            return any(keyword in target for keyword in mappings[category])
        return False


# ============================================================================
# GOEMOTIONS DATASET
# ============================================================================

class GoEmotionsDataset(BaseNLPDataset):
    
    EMOTION_NAMES = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            sample = split_data[i]
            label = self._encode_label(sample)
            
            if label == -1:
                continue
            
            self.texts.append(sample['text'])
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(sample))
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        label_type = self.config.label_type
        labels = sample['labels']  # List of emotion indices
        
        if len(labels) == 0:
            return -1
        
        if label_type == "multilabel":
            # For multilabel, return the first label (will handle separately)
            return labels[0]
        
        elif label_type == "multiclass_dominant":
            # Return the first (dominant) emotion
            return labels[0]
        
        return -1
    
    def _encode_concepts(self, sample) -> np.ndarray:
        """
        Encode emotions as concepts
        0 = not present, 1 = uncertain (multi-label), 2 = present
        """
        labels = sample['labels']
        concepts = np.zeros(len(self.EMOTION_NAMES), dtype=np.int64)
        
        for label_idx in labels:
            if label_idx < len(concepts):
                concepts[label_idx] = 2  # Present
        
        # If multiple emotions, mark ambiguity
        if len(labels) > 1:
            # Mark non-present emotions as uncertain (could be present)
            for i in range(len(concepts)):
                if concepts[i] == 0:
                    concepts[i] = 1  # Uncertain
        
        return concepts
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # For multilabel, also return full label vector
        if self.config.label_type == "multilabel":
            # concepts already encode presence/absence
            pass
        
        return item


# ============================================================================
# CIVIL COMMENTS DATASET
# ============================================================================

class CivilCommentsDataset(BaseNLPDataset):
    
    IDENTITY_COLS = [
        'male', 'female', 'transgender', 'other_gender',
        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
        'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
        'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
        'physical_disability', 'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness', 'other_disability'
    ]
    
    # Simplified concept names (grouped)
    CONCEPT_NAMES = ['male', 'female', 'LGBTQ', 'christian', 'muslim',
                     'jewish', 'black', 'white', 'disability']
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            sample = split_data[i]
            label = self._encode_label(sample)
            
            if label == -1:
                continue
            
            self.texts.append(sample['text'])
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(sample))
            
            # Store toxicity score for uncertainty analysis
            self.annotator_labels.append([sample['toxicity']])
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        toxicity = sample['toxicity']
        
        if self.config.label_type == "binary":
            # Threshold at 0.5
            return 1 if toxicity >= 0.5 else 0
        
        elif self.config.label_type == "regression":
            # Return binned toxicity (0-4)
            return min(int(toxicity * 5), 4)
        
        return -1
    
    def _encode_concepts(self, sample) -> np.ndarray:
        """
        Encode identity concepts
        0 = not mentioned, 1 = uncertain, 2 = mentioned
        """
        concepts = []
        
        # Grouped concept mappings
        mappings = {
            'male': ['male'],
            'female': ['female'],
            'LGBTQ': ['homosexual_gay_or_lesbian', 'bisexual', 'transgender'],
            'christian': ['christian'],
            'muslim': ['muslim'],
            'jewish': ['jewish'],
            'black': ['black'],
            'white': ['white'],
            'disability': ['physical_disability', 'intellectual_or_learning_disability',
                          'psychiatric_or_mental_illness']
        }
        
        for concept_name in self.CONCEPT_NAMES:
            cols = mappings.get(concept_name, [concept_name])
            
            # Check if any identity in this group is mentioned
            max_score = 0
            for col in cols:
                if col in sample and sample[col] is not None:
                    max_score = max(max_score, sample[col])
            
            if max_score >= 0.5:
                concepts.append(2)  # Mentioned
            elif max_score > 0:
                concepts.append(1)  # Uncertain
            else:
                concepts.append(0)  # Not mentioned
        
        return np.array(concepts)


# ============================================================================
# SST-2 DATASET (No ground-truth concepts)
# ============================================================================

class SST2Dataset(BaseNLPDataset):
    
    # Synthetic concept definitions (keyword-based)
    CONCEPT_NAMES = ['positive_words', 'negative_words', 'intensity', 
                     'negation', 'subjective']
    
    POSITIVE_KEYWORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'love', 'best', 'fantastic', 'beautiful', 'perfect'}
    NEGATIVE_KEYWORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst',
                         'hate', 'boring', 'disappointing', 'poor', 'waste'}
    INTENSITY_KEYWORDS = {'very', 'extremely', 'incredibly', 'absolutely', 
                          'totally', 'really', 'so', 'quite'}
    NEGATION_KEYWORDS = {'not', "n't", 'never', 'no', 'nothing', 'neither', 'none'}
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            sample = split_data[i]
            label = self._encode_label(sample)
            
            if label == -1:
                continue
            
            self.texts.append(sample['sentence'])
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(sample))
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        return sample['label']  # Already 0 or 1
    
    def _encode_concepts(self, sample) -> np.ndarray:
        text = sample['sentence'].lower()
        words = set(text.split())
        
        concepts = []
        
        # Positive words
        pos_count = len(words & self.POSITIVE_KEYWORDS)
        concepts.append(2 if pos_count >= 2 else (1 if pos_count == 1 else 0))
        
        # Negative words
        neg_count = len(words & self.NEGATIVE_KEYWORDS)
        concepts.append(2 if neg_count >= 2 else (1 if neg_count == 1 else 0))
        
        # Intensity
        int_count = len(words & self.INTENSITY_KEYWORDS)
        concepts.append(2 if int_count >= 1 else 0)
        
        # Negation
        neg_present = len(words & self.NEGATION_KEYWORDS) > 0
        concepts.append(2 if neg_present else 0)
        
        # Subjective (based on pronouns and opinion words)
        subjective_words = {'i', 'my', 'me', 'think', 'feel', 'believe', 'opinion'}
        subj_count = len(words & subjective_words)
        concepts.append(2 if subj_count >= 1 else 1)  # Most reviews are subjective
        
        return np.array(concepts)


# ============================================================================
# AG NEWS DATASET (No ground-truth concepts)
# ============================================================================

class AGNewsDataset(BaseNLPDataset):
    
    CONCEPT_NAMES = ['business_terms', 'tech_terms', 'sports_terms', 
                     'world_terms', 'named_entities']
    
    BUSINESS_KEYWORDS = {'market', 'stock', 'company', 'price', 'profit', 
                         'sales', 'ceo', 'investor', 'trade', 'economy'}
    TECH_KEYWORDS = {'software', 'computer', 'internet', 'microsoft', 'google',
                     'technology', 'digital', 'online', 'web', 'app'}
    SPORTS_KEYWORDS = {'game', 'team', 'win', 'player', 'season', 
                       'coach', 'championship', 'score', 'league', 'match'}
    WORLD_KEYWORDS = {'country', 'government', 'president', 'minister', 'war',
                      'peace', 'united', 'nations', 'international', 'military'}
    
    LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    def _load_data(self, split_data):
        for i in range(len(split_data)):
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                break
            
            sample = split_data[i]
            label = self._encode_label(sample)
            
            if label == -1:
                continue
            
            self.texts.append(sample['text'])
            self.labels.append(label)
            self.concepts.append(self._encode_concepts(sample))
        
        self.concepts = np.array(self.concepts)
    
    def _encode_label(self, sample) -> int:
        return sample['label']  # 0-3
    
    def _encode_concepts(self, sample) -> np.ndarray:
        text = sample['text'].lower()
        words = set(text.split())
        
        concepts = []
        
        # Business terms
        biz_count = len(words & self.BUSINESS_KEYWORDS)
        concepts.append(2 if biz_count >= 2 else (1 if biz_count == 1 else 0))
        
        # Tech terms
        tech_count = len(words & self.TECH_KEYWORDS)
        concepts.append(2 if tech_count >= 2 else (1 if tech_count == 1 else 0))
        
        # Sports terms
        sports_count = len(words & self.SPORTS_KEYWORDS)
        concepts.append(2 if sports_count >= 2 else (1 if sports_count == 1 else 0))
        
        # World terms
        world_count = len(words & self.WORLD_KEYWORDS)
        concepts.append(2 if world_count >= 2 else (1 if world_count == 1 else 0))
        
        # Named entities (rough heuristic: capitalized words)
        original_words = sample['text'].split()
        caps_count = sum(1 for w in original_words if w[0].isupper() and len(w) > 1)
        concepts.append(2 if caps_count >= 3 else (1 if caps_count >= 1 else 0))
        
        return np.array(concepts)


# ============================================================================
# DATASET FACTORY
# ============================================================================

DATASET_CLASSES = {
    'cebab': CEBaBDataset,
    'hatexplain': HateXplainDataset,
    'goemotions': GoEmotionsDataset,
    'civil_comments': CivilCommentsDataset,
    'sst2': SST2Dataset,
    'ag_news': AGNewsDataset,
}


def load_dataset_splits(
    dataset_name: str,
    config: Optional[DatasetConfig] = None,
    tokenizer_name: str = "distilbert-base-uncased",
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer, Dict]:
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_INFO.keys())}")
    
    info = DATASET_INFO[dataset_name]
    
    # Create config with dataset-specific defaults
    if config is None:
        config = DatasetConfig()
    
    if config.label_type == "default":
        config.label_type = info['default_label_type']
    
    # Override splits if specified in dataset info
    if 'train_split' in info:
        config.train_split = info['train_split']
    if 'val_split' in info:
        config.val_split = info['val_split']
    if 'test_split' in info:
        config.test_split = info['test_split']
    
    # Load from HuggingFace
    if 'hf_subset' in info:
        ds = load_dataset(info['hf_path'], info['hf_subset'])
    else:
        ds = load_dataset(info['hf_path'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get dataset class
    DatasetClass = DATASET_CLASSES[dataset_name]
    
    # Create datasets
    train_dataset = DatasetClass(ds[config.train_split], tokenizer, config)
    val_dataset = DatasetClass(ds[config.val_split], tokenizer, config)
    test_dataset = DatasetClass(ds[config.test_split], tokenizer, config)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Collect metadata
    metadata = {
        'dataset_name': dataset_name,
        'task': info['task'],
        'num_classes': info['num_classes'][config.label_type],
        'num_concepts': len(train_dataset.concepts[0]) if len(train_dataset.concepts) > 0 else 0,
        'concept_names': info['concept_names'] or DatasetClass.CONCEPT_NAMES,
        'has_ground_truth_concepts': info['has_concepts'],
        'has_multi_annotator': info['has_multi_annotator'],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_stats': train_dataset.get_stats(),
    }
    
    return train_loader, val_loader, test_loader, tokenizer, metadata


def print_dataset_summary(metadata: Dict):
    pass



# ============================================================
# Unambiguous loaders with 3 upgrades:
#  A) Negation-aware lexical filter
#  B) Teacher confidence gate
#  C) Multi-criteria scoring + threshold + class balancing
# ============================================================

import re
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ----------------------------
# 0) Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


# ----------------------------
# 1) Upgrade A: Negation-aware lexical scoring
# ----------------------------

class NegationAwareLexicon:
    """
    Counts strong pos/neg hits, but discounts words in a negation window.
    Simple & robust enough for fast filtering.
    """

    def __init__(
        self,
        strong_positive_words=None,
        strong_negative_words=None,
        negation_words=None,
        window: int = 3,
    ):
        self.strong_positive_words = strong_positive_words or {
            "amazing", "excellent", "fantastic", "perfect", "incredible",
            "wonderful", "outstanding", "delicious", "superb", "best"
        }
        self.strong_negative_words = strong_negative_words or {
            "terrible", "awful", "horrible", "disgusting", "worst",
            "dreadful", "inedible", "pathetic", "abysmal", "revolting"
        }
        self.negation_words = negation_words or {
            "not", "n't", "never", "no", "hardly", "scarcely", "barely"
        }
        self.window = window

        self._token_re = re.compile(r"[A-Za-z']+")

    def tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_re.findall(text)]

    def _is_negated(self, toks: List[str], idx: int) -> bool:
        lo = max(0, idx - self.window)
        for j in range(lo, idx):
            if toks[j] in self.negation_words:
                return True
        return False

    def score(self, text: str) -> Dict[str, Any]:
        """
        Returns:
          pos_hits, neg_hits: negation-aware counts
          lex_margin: pos_hits - neg_hits
          lex_strength: pos_hits + neg_hits
        """
        toks = self.tokenize(text)
        pos_hits, neg_hits = 0, 0

        for i, tok in enumerate(toks):
            if tok in self.strong_positive_words:
                if not self._is_negated(toks, i):
                    pos_hits += 1
            elif tok in self.strong_negative_words:
                if not self._is_negated(toks, i):
                    neg_hits += 1

        return {
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "lex_margin": pos_hits - neg_hits,
            "lex_strength": pos_hits + neg_hits,
            "n_tokens": len(toks),
        }


# ----------------------------
# 2) Upgrade B: Teacher confidence gate
# ----------------------------

@dataclass
class TeacherConfig:
    # Strong open sentiment teacher (works for SST/Yelp reasonably well)
    # You can swap to something else if you prefer.
    teacher_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: int = 64
    max_length: int = 256
    device: Optional[str] = None  # "cuda" / "cpu" / None(auto)


class SentimentTeacher:
    """
    A thin wrapper around a sequence classification model that returns:
      pred_label in {0=neg,1=pos} and confidence in [0,1]
    Supports teachers with 2 or 3 labels (NEG/NEU/POS).
    """

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.teacher_name)
        self.model.eval()

        if cfg.device is None:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        # Try to map teacher labels to NEG/NEU/POS
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.num_labels = self.model.config.num_labels

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Returns probabilities [B, num_labels] as numpy.
        """
        all_probs = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits  # [B, L]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def _to_binary(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts multi-class probs to binary {neg,pos}:
          - if 2 labels: direct
          - if 3 labels: drop/ignore neutral by comparing neg vs pos
        Returns:
          pred_bin: [B] in {0,1}
          conf_bin: [B] confidence for predicted bin
        """
        if probs.shape[1] == 2:
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            return pred.astype(int), conf

        # 3 labels case: assume labels contain NEG/NEU/POS somewhere
        # We'll find indices by string matching; fallback to [0,1,2] = NEG,NEU,POS
        labels = [self.id2label.get(i, str(i)).lower() for i in range(probs.shape[1])]
        neg_idx = next((i for i, s in enumerate(labels) if "neg" in s), 0)
        pos_idx = next((i for i, s in enumerate(labels) if "pos" in s), probs.shape[1]-1)

        neg_p = probs[:, neg_idx]
        pos_p = probs[:, pos_idx]

        pred = (pos_p >= neg_p).astype(int)
        conf = np.maximum(pos_p, neg_p)
        return pred, conf

    @torch.no_grad()
    def predict_binary(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(texts)
        return self._to_binary(probs)


# ----------------------------
# 3) Upgrade C: Multi-criteria scoring
# ----------------------------

@dataclass
class UnambiguousScoringConfig:
    # gates
    min_tokens: int = 6
    min_lex_strength: int = 1          # at least one strong word hit (after negation handling)
    teacher_conf_threshold: float = 0.95

    # scoring weights
    w_extreme: float = 1.0
    w_lex: float = 0.7
    w_teacher: float = 1.2
    w_agreement: float = 0.0           # set >0 if you have agreement signals

    # final threshold on total score
    score_threshold: float = 2.2


def normalize01(x: float, cap: float = 5.0) -> float:
    # keep in [0,1] with a soft cap
    return float(min(x, cap) / cap)


# ----------------------------
# 4) Dataset-specific label extraction (binary extremes)
# ----------------------------

def extract_binary_extreme_label(dataset_name: str, ex: Dict[str, Any]) -> Optional[int]:
    """
    Returns 0/1 for negative/positive if example is an extreme, else None.
    """
    if dataset_name == "sst5":
        # SetFit/sst5 labels: 0..4; keep 0 and 4
        if ex["label"] == 0:
            return 0
        if ex["label"] == 4:
            return 1
        return None

    if dataset_name == "yelp5":
        # yelp_review_full label: 0..4; keep 0 and 4
        if ex["label"] == 0:
            return 0
        if ex["label"] == 4:
            return 1
        return None

    if dataset_name == "cebab":
        # CEBaB: use review_majority + aspect consistency
        rating = str(ex.get("review_majority", "")).lower()
        if ("1" in rating) or ("2" in rating):
            return 0
        if ("4" in rating) or ("5" in rating):
            return 1
        return None

    raise ValueError(f"Unknown dataset_name={dataset_name}")


def get_text_field(dataset_name: str, ex: Dict[str, Any]) -> str:
    if dataset_name == "sst5":
        return ex["text"]
    if dataset_name == "yelp5":
        return ex["text"]
    if dataset_name == "cebab":
        return ex["description"]
    raise ValueError(dataset_name)


def cebab_aspect_agreement_score(ex: Dict[str, Any]) -> float:
    """
    Optional: returns 0..1 agreement/consistency proxy for CEBaB.
    Here: if >=3 known aspects and all same polarity -> 1.0 else 0.0
    """
    aspects = [
        ex.get("food_aspect_majority"),
        ex.get("service_aspect_majority"),
        ex.get("ambiance_aspect_majority"),
        ex.get("noise_aspect_majority"),
    ]
    known = [a for a in aspects if a in ["Positive", "Negative"]]
    if len(known) < 3:
        return 0.0
    if all(a == "Positive" for a in known) or all(a == "Negative" for a in known):
        return 1.0
    return 0.0


# ----------------------------
# 5) Build filtered examples list (with the 3 upgrades)
# ----------------------------

def build_unambiguous_examples(
    *,
    dataset_name: str,
    split: str,
    n_per_class: int,
    lex: NegationAwareLexicon,
    teacher: SentimentTeacher,
    cfg: UnambiguousScoringConfig,
    max_text_len_chars: int = 512,
) -> List[Dict[str, Any]]:

    # Load HF dataset
    if dataset_name == "sst5":
        ds = load_dataset("SetFit/sst5", split=split)
    elif dataset_name == "yelp5":
        ds = load_dataset("yelp_review_full", split=split)
    elif dataset_name == "cebab":
        ds = load_dataset("CEBaB/CEBaB", split=split)
    else:
        raise ValueError(dataset_name)

    # First pass: keep only extreme-label candidates + lexical gates
    candidates: List[Dict[str, Any]] = []
    for ex in ds:
        y = extract_binary_extreme_label(dataset_name, ex)
        if y is None:
            continue

        text = get_text_field(dataset_name, ex)
        text = text.strip()
        if len(text) == 0:
            continue
        if len(text) > max_text_len_chars:
            text = text[:max_text_len_chars]

        lex_stats = lex.score(text)

        # Gate 1: minimum tokens
        if lex_stats["n_tokens"] < cfg.min_tokens:
            continue

        # Gate 2: minimum lexical strength (after negation handling)
        if lex_stats["lex_strength"] < cfg.min_lex_strength:
            continue

        # Optional: for CEBaB enforce aspect consistency early
        agree = 0.0
        if dataset_name == "cebab":
            agree = cebab_aspect_agreement_score(ex)
            if agree < 1.0:
                continue

        candidates.append({
            "text": text,
            "label": y,
            "source": dataset_name,
            "lex_stats": lex_stats,
            "agreement": agree,
        })

    if len(candidates) == 0:
        return []

    # Teacher pass (batched)
    texts = [c["text"] for c in candidates]
    t_pred, t_conf = teacher.predict_binary(texts)  # arrays

    # Second pass: teacher gate + scoring
    scored: List[Dict[str, Any]] = []
    for c, tp, tc in zip(candidates, t_pred.tolist(), t_conf.tolist()):
        y = c["label"]

        # Gate 3: teacher must agree with label
        if int(tp) != int(y):
            continue

        # Gate 4: teacher confidence threshold
        if float(tc) < cfg.teacher_conf_threshold:
            continue

        # Multi-criteria score
        # extreme = 1 always here (since we filtered to extremes), but keep structure
        extreme_score = 1.0

        # lexical: reward margin in correct direction + strength
        lex_margin = c["lex_stats"]["lex_margin"]
        lex_strength = c["lex_stats"]["lex_strength"]

        # direction correctness: for positive, margin should be positive; for negative, margin negative
        dir_ok = (lex_margin > 0) if y == 1 else (lex_margin < 0)
        lex_dir = 1.0 if dir_ok else 0.0
        lex_str = normalize01(lex_strength, cap=4.0)   # 0..1
        lex_score = 0.6 * lex_dir + 0.4 * lex_str      # 0..1-ish

        teacher_score = float(tc)                       # already 0..1

        agreement_score = float(c["agreement"])         # 0..1 (only used if cfg.w_agreement > 0)

        total = (
            cfg.w_extreme * extreme_score +
            cfg.w_lex * lex_score +
            cfg.w_teacher * teacher_score +
            cfg.w_agreement * agreement_score
        )

        if total < cfg.score_threshold:
            continue

        scored.append({
            "text": c["text"],
            "label": y,
            "source": dataset_name,
            "scenario": "unambiguous",
            "teacher_conf": float(tc),
            "lex_strength": int(lex_strength),
            "lex_margin": int(lex_margin),
            "score": float(total),
        })

    # Balance classes & take top by score
    pos = [x for x in scored if x["label"] == 1]
    neg = [x for x in scored if x["label"] == 0]

    pos.sort(key=lambda z: z["score"], reverse=True)
    neg.sort(key=lambda z: z["score"], reverse=True)

    pos = pos[:n_per_class]
    neg = neg[:n_per_class]

    final = pos + neg
    random.shuffle(final)
    return final


# ----------------------------
# 6) Torch Dataset that tokenizes with your target tokenizer
# ----------------------------

class TokenizedTextDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
            # optional metadata (can remove if you want pure tensors)
            "source": ex.get("source", "unknown"),
            "score": ex.get("score", 0.0),
            "teacher_conf": ex.get("teacher_conf", 0.0),
        }
        return item


# ----------------------------
# 7) Public API: build train + OOD loaders
# ----------------------------

@dataclass
class LoaderBuildConfig:
    # Target tokenizer used for embeddings/LLM
    target_tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 128
    batch_size: int = 16
    seed: int = 0

    # Which datasets
    train_source: str = "sst5"   # "sst5" | "yelp5" | "cebab"
    train_split: str = "train"   # sst5: train, yelp: train, cebab: train_inclusive
    ood_source: str = "yelp5"
    ood_split: str = "test"

    n_per_class_train: int = 500
    n_per_class_ood: int = 500

    # Teacher
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    # Unambiguous scoring settings
    scoring: UnambiguousScoringConfig = field(default_factory=UnambiguousScoringConfig)


def build_unambiguous_train_and_ood_loaders(cfg: LoaderBuildConfig):
    set_seed(cfg.seed)

    # Lexicon + teacher
    lex = NegationAwareLexicon()
    teacher = SentimentTeacher(cfg.teacher)

    # Build example sets (filtered + scored)
    train_examples = build_unambiguous_examples(
        dataset_name=cfg.train_source,
        split=cfg.train_split,
        n_per_class=cfg.n_per_class_train,
        lex=lex,
        teacher=teacher,
        cfg=cfg.scoring,
    )
    ood_examples = build_unambiguous_examples(
        dataset_name=cfg.ood_source,
        split=cfg.ood_split,
        n_per_class=cfg.n_per_class_ood,
        lex=lex,
        teacher=teacher,
        cfg=cfg.scoring,
    )
    for ex in ood_examples:
        ex["scenario"] = "ood_unambiguous"

    # Target tokenizer (must match your embedding model vocab!)
    tok = AutoTokenizer.from_pretrained(cfg.target_tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = TokenizedTextDataset(train_examples, tok, max_length=cfg.max_length)
    ood_ds = TokenizedTextDataset(ood_examples, tok, max_length=cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    ood_loader = DataLoader(ood_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    meta = {
        "train_source": cfg.train_source,
        "ood_source": cfg.ood_source,
        "train_size": len(train_ds),
        "ood_size": len(ood_ds),
        "teacher_name": cfg.teacher.teacher_name,
        "teacher_conf_threshold": cfg.scoring.teacher_conf_threshold,
        "score_threshold": cfg.scoring.score_threshold,
        "target_tokenizer_name": cfg.target_tokenizer_name,
    }

    return train_loader, ood_loader, tok, meta


# # ----------------------------
# # 8) Example usage
# # ----------------------------

# if __name__ == "__main__":
#     cfg = LoaderBuildConfig(
#         target_tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         max_length=128,
#         batch_size=16,
#         seed=0,

#         train_source="sst5",
#         train_split="train",
#         ood_source="yelp5",
#         ood_split="test",

#         n_per_class_train=200,
#         n_per_class_ood=200,

#         # tighten if you want “very clean” sets
#         scoring=UnambiguousScoringConfig(
#             min_tokens=6,
#             min_lex_strength=1,
#             teacher_conf_threshold=0.97,
#             score_threshold=2.35,
#             w_extreme=1.0,
#             w_lex=0.7,
#             w_teacher=1.2,
#         ),
#     )

    # train_loader, ood_loader, tok, meta = build_unambiguous_train_and_ood_loaders(cfg)
    # print(meta)

    # b = next(iter(train_loader))
    # print("Train batch:", b["input_ids"].shape, b["labels"].shape, b["source"][:3], b["teacher_conf"][:3])

    # b2 = next(iter(ood_loader))
    # print("OOD batch:", b2["input_ids"].shape, b2["labels"].shape, b2["source"][:3], b2["teacher_conf"][:3])

# ============================================================================
# MAIN: DEMO ALL DATASETS
# ============================================================================

def main(
    dataset_name: str = "cebab",
    config: Optional[DatasetConfig] = None,
    tokenizer_name: str = "distilbert-base-uncased",
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer, Dict]:
    train_loader, val_loader, test_loader, tokenizer, metadata = \
        load_dataset_splits(dataset_name, config, tokenizer_name, batch_size)
    
    return train_loader, val_loader, test_loader, tokenizer, metadata


# ============================================================================
# EXAMPLE USAGE PATTERNS
# ============================================================================

# Example 1: Load CEBaB with default settings
# train_loader, val_loader, test_loader, tokenizer, metadata = main("cebab")

# Example 2: Load SST-2 with custom batch size
# train_loader, val_loader, test_loader, tokenizer, metadata = main("sst2", batch_size=32)

# Example 3: Load AG News with custom tokenizer
# train_loader, val_loader, test_loader, tokenizer, metadata = main(
#     "ag_news", 
#     tokenizer_name="bert-base-uncased",
#     batch_size=64
# )

# Example 4: Load CEBaB with custom config (binary labels)
# config = DatasetConfig(label_type="binary", max_length=256, batch_size=32)
# train_loader, val_loader, test_loader, tokenizer, metadata = main("cebab", config=config)

# Example 5: Load GoEmotions with multilabel
# config = DatasetConfig(label_type="multilabel", batch_size=16)
# train_loader, val_loader, test_loader, tokenizer, metadata = main("goemotions", config=config)

# Example 6: Load HateXplain with multiclass
# train_loader, val_loader, test_loader, tokenizer, metadata = main(
#     "hatexplain",
#     config=DatasetConfig(label_type="multiclass", batch_size=16)
# )

# Example 7: Load Civil Comments
# train_loader, val_loader, test_loader, tokenizer, metadata = main("civil_comments")

# Example 8: Direct usage of load_dataset_splits (returns all three loaders)
# train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
#     "cebab",
#     config=DatasetConfig(label_type="ternary", batch_size=32),
#     tokenizer_name="roberta-base",
#     batch_size=32
# )


if __name__ == "__main__":
    # Example: Load CEBaB dataset
    train_loader, val_loader, test_loader, tokenizer, metadata = main("cebab")
    
    # Access the loaders
    # batch = next(iter(train_loader))
    # print(f"Train batch shape: {batch['input_ids'].shape}")
    # print(f"Number of classes: {metadata['num_classes']}")
    # print(f"Number of concepts: {metadata['num_concepts']}")

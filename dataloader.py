
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

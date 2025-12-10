"""
Configuration file for Credal CBM experiments
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import os


class TaskType(Enum):
    """Task types for different datasets"""
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    EMOTION = "emotion"
    TOXICITY = "toxicity"
    NLI = "nli"
    QA = "qa"
    CONCEPT_CLASSIFICATION = "concept_classification"


class EnsembleType(Enum):
    """Types of sklearn ensembles for credal sets"""
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    BAGGING = "bagging"
    GRADIENT_BOOSTING = "gradient_boosting"
    ADABOOST = "adaboost"


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    hf_path: str
    task_type: TaskType
    num_classes: int
    text_column: str
    label_column: str
    has_concepts: bool = False
    has_multi_annotator: bool = False  # Ground-truth aleatoric!
    expected_aleatoric: str = "low"  # low, medium, high
    concept_columns: Optional[List[str]] = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    notes: str = ""


@dataclass
class ModelConfig:
    """Configuration for encoder model"""
    name: str
    hf_path: str
    hidden_size: int
    max_length: int = 512
    is_fast: bool = True
    batch_size: int = 32
    unsloth: bool = False  # Whether to use Unsloth optimization


@dataclass
class CredalConfig:
    """Configuration specific to Credal CBM"""
    ensemble_type: EnsembleType = EnsembleType.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 5
    random_state: int = 42

    # Credal set parameters
    epistemic_threshold: float = 0.1  # Below this = precise prediction
    aleatoric_estimation: str = "disagreement"  # disagreement, entropy, variance

    # Feature scaling
    use_standard_scaling: bool = True

    # Concept-specific settings
    n_concepts: int = 10
    concept_threshold: float = 0.5  # For binary concept predictions


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""

    # Experiment metadata
    experiment_name: str = "credal_cbm_experiment"
    output_dir: str = "./results"
    seed: int = 42

    # Training parameters
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    cross_validation_folds: int = 5

    # Datasets - Priority-based selection
    datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        # === TIER 1: MUST HAVE ===
        "sst2": DatasetConfig(
            name="SST-2",
            hf_path="stanfordnlp/sst2",
            task_type=TaskType.SENTIMENT,
            num_classes=2,
            text_column="sentence",
            label_column="label",
            expected_aleatoric="low",
            train_split="train",
            validation_split="validation",
            test_split="validation",
            notes="Binary sentiment, movie reviews, standard benchmark"
        ),
        "imdb": DatasetConfig(
            name="IMDB",
            hf_path="imdb",
            task_type=TaskType.SENTIMENT,
            num_classes=2,
            text_column="text",
            label_column="label",
            expected_aleatoric="low",
            notes="Longer movie reviews, good for cross-domain from SST-2"
        ),
        "ag_news": DatasetConfig(
            name="AG News",
            hf_path="ag_news",
            task_type=TaskType.TOPIC,
            num_classes=4,
            text_column="text",
            label_column="label",
            expected_aleatoric="low",
            notes="Topic classification: World, Sports, Business, Sci/Tech"
        ),

        # === TIER 2: SHOULD HAVE ===
        "goemotions": DatasetConfig(
            name="GoEmotions",
            hf_path="go_emotions",
            task_type=TaskType.EMOTION,
            num_classes=28,
            text_column="text",
            label_column="labels",
            expected_aleatoric="high",
            notes="Reddit comments, 28 emotions, HIGH aleatoric expected"
        ),
        "civil_comments": DatasetConfig(
            name="Civil Comments",
            hf_path="civil_comments",
            task_type=TaskType.TOXICITY,
            num_classes=2,
            text_column="text",
            label_column="toxicity",
            has_multi_annotator=True,
            expected_aleatoric="medium",
            notes="Multi-annotator labels = ground-truth aleatoric uncertainty"
        ),
        "cebab": DatasetConfig(
            name="CEBaB",
            hf_path="CEBaB/CEBaB",
            task_type=TaskType.SENTIMENT,
            num_classes=2,
            text_column="review",
            label_column="label",
            has_concepts=True,
            concept_columns=["food", "service", "ambiance", "noise"],
            expected_aleatoric="medium",
            notes="Has causal concepts - good for concept supervision"
        ),
    })

    # Encoder models
    encoders: Dict[str, ModelConfig] = field(default_factory=lambda: {
        # Standard Transformer Models
        "distilbert": ModelConfig(
            name="DistilBERT",
            hf_path="distilbert-base-uncased",
            hidden_size=768,
            max_length=512,
            is_fast=True,
            batch_size=32
        ),
        "bert": ModelConfig(
            name="BERT",
            hf_path="bert-base-uncased",
            hidden_size=768,
            max_length=512,
            is_fast=True,
            batch_size=16
        ),
        "roberta": ModelConfig(
            name="RoBERTa",
            hf_path="roberta-base",
            hidden_size=768,
            max_length=512,
            is_fast=True,
            batch_size=16
        ),

        # Unsloth GPU-Optimized Models
        "unsloth_llama_3b": ModelConfig(
            name="Unsloth Llama-3-8B",
            hf_path="unsloth/llama-3-8b-bnb-4bit",
            hidden_size=4096,  # Llama-3 has larger hidden size
            max_length=2048,
            is_fast=True,
            batch_size=4,  # Smaller batch for larger model
            unsloth=True
        ),
        "unsloth_mistral_7b": ModelConfig(
            name="Unsloth Mistral-7B",
            hf_path="unsloth/mistral-7b-bnb-4bit",
            hidden_size=4096,
            max_length=2048,
            is_fast=True,
            batch_size=4,
            unsloth=True
        ),
        "unsloth_phi_3b": ModelConfig(
            name="Unsloth Phi-3-mini-4K",
            hf_path="unsloth/Phi-3-mini-4k-instruct",
            hidden_size=3072,
            max_length=4096,
            is_fast=True,
            batch_size=8,
            unsloth=True
        ),
        "unsloth_qwen_2b": ModelConfig(
            name="Unsloth Qwen2-1.5B",
            hf_path="unsloth/Qwen2-1.5B-bnb-4bit",
            hidden_size=2048,
            max_length=2048,
            is_fast=True,
            batch_size=8,
            unsloth=True
        ),
        "unsloth_gemma_2b": ModelConfig(
            name="Unsloth Gemma-2B",
            hf_path="unsloth/gemma-2b-bnb-4bit",
            hidden_size=2048,
            max_length=2048,
            is_fast=True,
            batch_size=8,
            unsloth=True
        ),
    })

    # Credal CBM configurations
    credal_configs: Dict[str, CredalConfig] = field(default_factory=lambda: {
        "random_forest": CredalConfig(
            ensemble_type=EnsembleType.RANDOM_FOREST,
            n_estimators=100,
            max_depth=10
        ),
        "extra_trees": CredalConfig(
            ensemble_type=EnsembleType.EXTRA_TREES,
            n_estimators=100,
            max_depth=10
        ),
        "bagging": CredalConfig(
            ensemble_type=EnsembleType.BAGGING,
            n_estimators=50,
            max_depth=None
        ),
        "gradient_boosting": CredalConfig(
            ensemble_type=EnsembleType.GRADIENT_BOOSTING,
            n_estimators=100,
            max_depth=5
        ),
    })

    # Default selections for experiments
    default_datasets: List[str] = field(default_factory=lambda: [
        "sst2", "imdb", "ag_news", "goemotions", "civil_comments"
    ])

    default_encoder: str = "distilbert"
    default_credal_config: str = "random_forest"

    # Uncertainty methods to compare
    uncertainty_methods: List[str] = field(default_factory=lambda: [
        "credal_sklearn",       # Sklearn ensemble credal sets
        "mc_dropout",           # Monte Carlo dropout
        "deep_ensemble",        # Deep ensembles
        "temperature_scaling",  # Calibration
        "deterministic",        # Baseline
    ])

    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_macro", "f1_micro", "precision", "recall",
        "uncertainty_correlation", "calibration_error", "sharpness",
        "epistemic_coverage", "credal_set_size", "decomposition_quality"
    ])

    # Experiment settings
    run_cross_validation: bool = True
    save_predictions: bool = True
    save_credal_sets: bool = True
    generate_plots: bool = True
    verbose: bool = True


class ConfigManager:
    """Utility class for managing configurations"""

    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str):
        """Save configuration to JSON file"""
        def convert_to_dict(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj

        config_dict = convert_to_dict(config)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def load_config(filepath: str) -> ExperimentConfig:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct enums and nested objects
        # This is a simplified version - you might need to enhance this
        # based on your specific use case

        datasets = {}
        for name, ds_dict in config_dict.get('datasets', {}).items():
            ds_dict['task_type'] = TaskType(ds_dict['task_type'])
            datasets[name] = DatasetConfig(**ds_dict)

        encoders = {}
        for name, enc_dict in config_dict.get('encoders', {}).items():
            encoders[name] = ModelConfig(**enc_dict)

        credal_configs = {}
        for name, cred_dict in config_dict.get('credal_configs', {}).items():
            cred_dict['ensemble_type'] = EnsembleType(cred_dict['ensemble_type'])
            credal_configs[name] = CredalConfig(**cred_dict)

        config_dict['datasets'] = datasets
        config_dict['encoders'] = encoders
        config_dict['credal_configs'] = credal_configs

        return ExperimentConfig(**config_dict)

    @staticmethod
    def get_quick_start_config() -> ExperimentConfig:
        """Get a quick start configuration for testing"""
        config = ExperimentConfig()
        config.experiment_name = "credal_cbm_quick_start"
        config.default_datasets = ["sst2"]  # Start with just one dataset
        config.train_size = 0.7  # Smaller for faster testing
        config.val_size = 0.15
        config.test_size = 0.15
        config.cross_validation_folds = 3
        return config

    @staticmethod
    def get_full_config() -> ExperimentConfig:
        """Get full configuration with all datasets"""
        return ExperimentConfig()


# Global default configuration
DEFAULT_CONFIG = ExperimentConfig()


# Convenience functions
def get_dataset_info(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name"""
    return DEFAULT_CONFIG.datasets.get(dataset_name)


def get_encoder_info(encoder_name: str) -> ModelConfig:
    """Get encoder configuration by name"""
    return DEFAULT_CONFIG.encoders.get(encoder_name)


def get_credal_info(config_name: str) -> CredalConfig:
    """Get credal configuration by name"""
    return DEFAULT_CONFIG.credal_configs.get(config_name)


def list_available_datasets() -> List[str]:
    """List all available datasets"""
    return list(DEFAULT_CONFIG.datasets.keys())


def list_available_encoders() -> List[str]:
    """List all available encoders"""
    return list(DEFAULT_CONFIG.encoders.keys())


def list_available_credal_methods() -> List[str]:
    """List all available credal methods"""
    return list(DEFAULT_CONFIG.credal_configs.keys())
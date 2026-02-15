"""
Load HuggingFace Datasets and Create DataLoaders

This script:
1. Handles Hugging Face authentication
2. Loads datasets (AmbigQA Star, CEBaB, etc.)
3. Creates PyTorch DataLoaders with different IDs (splits or ID fields)
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from huggingface_hub import login, whoami
from typing import Dict, List, Optional, Union
import warnings


def login_to_huggingface(token: Optional[str] = None) -> bool:
    """
    Login to Hugging Face Hub.
    
    Args:
        token: Optional Hugging Face token. If None, will try to use
               existing login or prompt via CLI.
    
    Returns:
        True if login successful, False otherwise
    """
    try:
        # Check if already logged in
        user_info = whoami()
        print(f"Already logged in as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception:
        pass
    
    # Try to login with token
    if token:
        try:
            login(token=token)
            print("Successfully logged in with provided token")
            return True
        except Exception as e:
            print(f"Failed to login with token: {e}")
            return False
    
    # Try to use token from environment
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Successfully logged in with environment token")
            return True
        except Exception as e:
            print(f"Failed to login with environment token: {e}")
    
    # Prompt for CLI login
    
    try:
        login()  # Interactive login
        print("Successfully logged in via CLI")
        return True
    except Exception as e:
        print(f"Login failed: {e}")
        print("\nPlease run: huggingface-cli login")
        return False


class AmbigQADataset(Dataset):
    """
    PyTorch Dataset wrapper for AmbigQA Star dataset.
    Handles different ID fields and splits.
    """
    
    def __init__(
        self,
        dataset_split,
        id_field: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
    ):
        """
        Args:
            dataset_split: HuggingFace dataset split (e.g., ds['train'])
            id_field: Name of the ID field. If None, uses index as ID.
            text_fields: List of text field names to include in samples
        """
        self.data = dataset_split
        self.id_field = id_field
        
        # Auto-detect text fields if not provided
        if text_fields is None:
            self.text_fields = [col for col in self.data.column_names 
                              if col not in ['id', 'idx', 'question_id']]
        else:
            self.text_fields = text_fields
        
        # Determine ID field
        if self.id_field is None:
            # Try to find common ID field names
            for possible_id in ['id', 'idx', 'question_id', 'example_id']:
                if possible_id in self.data.column_names:
                    self.id_field = possible_id
                    break
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get ID
        if self.id_field and self.id_field in sample:
            sample_id = sample[self.id_field]
        else:
            sample_id = idx
        
        # Extract text fields
        result = {'id': sample_id}
        for field in self.text_fields:
            if field in sample:
                result[field] = sample[field]
        
        # Include all other fields
        for key, value in sample.items():
            if key not in result and key != self.id_field:
                result[key] = value
        
        return result


class CEBaBDataset(Dataset):
    """
    PyTorch Dataset wrapper for CEBaB dataset.
    Handles all specified fields and creates concatenated ID.
    """
    
    def __init__(self, dataset_split):
        """
        Args:
            dataset_split: HuggingFace dataset split (e.g., ds['train'])
        """
        self.data = dataset_split
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Create concatenated ID: original_id_edit_id
        original_id = str(sample.get('original_id', ''))
        edit_id = str(sample.get('edit_id', ''))
        concatenated_id = f"{original_id}_{edit_id}" if original_id and edit_id else str(idx)
        
        # Build result with all specified fields
        result = {
            'id': concatenated_id,
            'original_id': original_id,
            'edit_id': edit_id,
            'is_original': sample.get('is_original', False),
            'edit_goal': sample.get('edit_goal'),
            'edit_type': sample.get('edit_type'),
            'edit_worker': sample.get('edit_worker'),
            'description': sample.get('description', ''),
            'review_majority': sample.get('review_majority'),
            'review_label_distribution': sample.get('review_label_distribution', {}),
            'review_workers': sample.get('review_workers', {}),
            'food_aspect_majority': sample.get('food_aspect_majority'),
            'ambiance_aspect_majority': sample.get('ambiance_aspect_majority'),
            'service_aspect_majority': sample.get('service_aspect_majority'),
            'noise_aspect_majority': sample.get('noise_aspect_majority'),
            'food_aspect_label_distribution': sample.get('food_aspect_label_distribution', {}),
            'ambiance_aspect_label_distribution': sample.get('ambiance_aspect_label_distribution', {}),
            'service_aspect_label_distribution': sample.get('service_aspect_label_distribution', {}),
            'noise_aspect_label_distribution': sample.get('noise_aspect_label_distribution', {}),
            'food_aspect_validation_workers': sample.get('food_aspect_validation_workers', {}),
            'ambiance_aspect_validation_workers': sample.get('ambiance_aspect_validation_workers', {}),
            'service_aspect_validation_workers': sample.get('service_aspect_validation_workers', {}),
            'noise_aspect_validation_workers': sample.get('noise_aspect_validation_workers', {}),
            'opentable_metadata': sample.get('opentable_metadata', {}),
        }
        
        # Include any additional fields that might be present
        for key, value in sample.items():
            if key not in result:
                result[key] = value
        
        return result


def create_cebab_dataloaders(
    dataset_name: str = "CEBaB/CEBaB",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    splits: Optional[List[str]] = None,
    **dataloader_kwargs
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for different splits of the CEBaB dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset (default: "CEBaB/CEBaB")
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        splits: List of splits to load (e.g., ['train', 'validation', 'test'])
                If None, loads all available splits
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    # Login to HuggingFace
    if not login_to_huggingface():
        raise RuntimeError("Failed to authenticate with Hugging Face. Please login first.")
    
    # Load dataset
    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nMake sure you have access to this dataset.")
        print("If it's a private dataset, ensure you're logged in and have access.")
        raise
    
    # Display dataset info
    for split_name, split_data in ds.items():
        if len(split_data) > 0:
            pass
    
    # Determine which splits to process
    if splits is None:
        splits = list(ds.keys())
    
    # Create DataLoaders for each split
    dataloaders = {}
    
    for split_name in splits:
        if split_name not in ds:
            warnings.warn(f"Split '{split_name}' not found in dataset. Skipping.")
            continue
        
        # Create CEBaB dataset wrapper
        dataset = CEBaBDataset(dataset_split=ds[split_name])
        
        # Determine shuffle setting (typically only for train)
        should_shuffle = shuffle and (split_name == 'train' or 'train' in split_name.lower())
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            **dataloader_kwargs
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders


def create_dataloaders(
    dataset_name: str = "ttomov/ambigqa_star",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    id_field: Optional[str] = None,
    splits: Optional[List[str]] = None,
    **dataloader_kwargs
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for different splits of the AmbigQA Star dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        id_field: Name of the ID field in the dataset
        splits: List of splits to load (e.g., ['train', 'validation', 'test'])
                If None, loads all available splits
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    # Login to HuggingFace
    if not login_to_huggingface():
        raise RuntimeError("Failed to authenticate with Hugging Face. Please login first.")
    
    # Load dataset
    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nMake sure you have access to this dataset.")
        print("If it's a private dataset, ensure you're logged in and have access.")
        raise
    
    # Display dataset info
    for split_name, split_data in ds.items():
        if len(split_data) > 0:
            pass
    
    # Determine which splits to process
    if splits is None:
        splits = list(ds.keys())
    
    # Create DataLoaders for each split
    dataloaders = {}
    
    for split_name in splits:
        if split_name not in ds:
            warnings.warn(f"Split '{split_name}' not found in dataset. Skipping.")
            continue
        
        # Create dataset wrapper
        dataset = AmbigQADataset(
            dataset_split=ds[split_name],
            id_field=id_field,
        )
        
        # Determine shuffle setting (typically only for train)
        should_shuffle = shuffle and (split_name == 'train' or 'train' in split_name.lower())
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            **dataloader_kwargs
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders


def inspect_dataset_sample(dataloader: DataLoader, num_samples: int = 1):
    """
    Inspect samples from a DataLoader.
    
    Args:
        dataloader: DataLoader to inspect
        num_samples: Number of batches to inspect
    """
    print("\n" + "=" * 70)
    print("Dataset Sample Inspection")
    print("=" * 70)
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        print(f"\nBatch {i + 1}:")
        print(f"  Batch size: {len(batch.get('id', batch.get(list(batch.keys())[0]))) if isinstance(batch, dict) else len(batch)}")
        
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: list of length {len(value)}")
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], str) and len(value[0]) < 100:
                            print(f"    First item: {value[0][:100]}...")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"  Batch type: {type(batch)}")
            print(f"  Batch content: {batch}")


# =============================================================================
# MAIN USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage - load all splits
    dataloaders = create_dataloaders(
        dataset_name="ttomov/ambigqa_star",
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    
    # Inspect a sample from each split
    for split_name, dataloader in dataloaders.items():
        inspect_dataset_sample(dataloader, num_samples=1)
    
    # Example 2: Load specific splits with custom ID field
    # Uncomment to use:
    # dataloaders_custom = create_dataloaders(
    #     dataset_name="ttomov/ambigqa_star",
    #     batch_size=32,
    #     shuffle=True,
    #     splits=['train', 'validation'],  # Only load these splits
    #     id_field='question_id',  # Specify ID field if known
    #     pin_memory=True,  # Additional DataLoader args
    # )
    
    # Example 3: CEBaB dataset
    # Uncomment to use:
    # from load_ambigqa_dataset import create_cebab_dataloaders
    # 
    # cebab_dataloaders = create_cebab_dataloaders(
    #     dataset_name="CEBaB/CEBaB",
    #     batch_size=32,
    #     shuffle=True,
    # )
    # 
    # train_loader = cebab_dataloaders['train']
    # for batch in train_loader:
    #     ids = batch['id']  # Format: "original_id_edit_id"
    #     original_ids = batch['original_id']
    #     edit_ids = batch['edit_id']
    #     descriptions = batch['description']
    #     is_original = batch['is_original']
    #     # ... your training code ...


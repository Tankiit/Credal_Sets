"""
MedMNIST Dataset Loading

Supports all MedMNIST variants at 28×28.
Distribution shift protocol: train on one dataset, test on related OOD dataset.

Shift pairs:
  - DermaMNIST (7 skin lesion classes) → train/test split has natural cross-source shift
  - PathMNIST (9 tissue types) → train from one center, test from another (built-in)
  - BloodMNIST (8 blood cell types) → clean controlled dataset, good ablation baseline
  - OrganAMNIST/OrganCMNIST/OrganSMNIST → same organs, different CT axes = natural shift

For the paper: PathMNIST is the strongest shift story (cross-institutional),
DermaMNIST is the medical AI hook, BloodMNIST is the clean ablation.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


# MedMNIST dataset info
MEDMNIST_INFO = {
    "dermamnist": {"n_classes": 7, "n_channels": 3, "task": "multi-class",
                   "description": "Dermatoscope images of skin lesions"},
    "pathmnist":  {"n_classes": 9, "n_channels": 3, "task": "multi-class",
                   "description": "Colon pathology tissue types (cross-center shift)"},
    "bloodmnist": {"n_classes": 8, "n_channels": 3, "task": "multi-class",
                   "description": "Blood cell microscopy"},
    "organamnist": {"n_classes": 11, "n_channels": 1, "task": "multi-class",
                    "description": "Abdominal CT - axial view"},
    "organcmnist": {"n_classes": 11, "n_channels": 1, "task": "multi-class",
                    "description": "Abdominal CT - coronal view"},
    "organsmnist": {"n_classes": 11, "n_channels": 1, "task": "multi-class",
                    "description": "Abdominal CT - sagittal view"},
    "tissuemnist": {"n_classes": 8, "n_channels": 1, "task": "multi-class",
                    "description": "Kidney cortex cell types"},
}

# Natural shift pairs for experiments
SHIFT_PAIRS = {
    # Train on axial CT, test on coronal CT (same organs, different viewpoint)
    "organ_axial_to_coronal": ("organamnist", "organcmnist"),
    # Train on axial CT, test on sagittal CT
    "organ_axial_to_sagittal": ("organamnist", "organsmnist"),
    # Train on coronal, test on sagittal
    "organ_coronal_to_sagittal": ("organcmnist", "organsmnist"),
}


class MedMNISTDataset(Dataset):
    """
    Wrapper around MedMNIST numpy arrays.
    
    Each item returns:
        - image: [C, 28, 28] float tensor, normalized to [0, 1]
        - label: integer class label
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.images = images  # [N, 28, 28, C] uint8
        self.labels = labels.squeeze().astype(np.int64)  # [N]
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # normalize to [0,1]
        
        # Handle channel dimension: [H, W, C] → [C, H, W]
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # [1, H, W] for grayscale
        else:
            img = img.transpose(2, 0, 1)  # [C, H, W]
        
        img = torch.from_numpy(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            "image": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_medmnist(
    dataset_name: str = "dermamnist",
    data_dir: str = "./data",
) -> Tuple[MedMNISTDataset, MedMNISTDataset, MedMNISTDataset]:
    """
    Load a MedMNIST dataset.
    
    Downloads via the medmnist package if not cached.
    Returns train, val, test datasets.
    """
    import medmnist
    from medmnist import INFO
    
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    
    print(f"Loading {dataset_name}: {info['description']}")
    print(f"  Classes: {info['n_classes']}, Channels: {info['n_channels']}, "
          f"Size: {info['n_samples']}")
    
    train_data = DataClass(split="train", download=True, root=data_dir)
    val_data = DataClass(split="val", download=True, root=data_dir)
    test_data = DataClass(split="test", download=True, root=data_dir)
    
    train_ds = MedMNISTDataset(train_data.imgs, train_data.labels)
    val_ds = MedMNISTDataset(val_data.imgs, val_data.labels)
    test_ds = MedMNISTDataset(test_data.imgs, test_data.labels)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds


def load_shift_pair(
    shift_name: str,
    data_dir: str = "./data",
) -> Dict[str, MedMNISTDataset]:
    """
    Load a distribution shift pair.
    
    Returns dict with:
      - train, val: from source dataset
      - test_id: in-distribution test (from source)
      - test_ood: out-of-distribution test (from target)
    """
    if shift_name in SHIFT_PAIRS:
        source_name, target_name = SHIFT_PAIRS[shift_name]
    else:
        raise ValueError(f"Unknown shift: {shift_name}. Options: {list(SHIFT_PAIRS.keys())}")
    
    print(f"\n=== Distribution Shift: {source_name} → {target_name} ===")
    
    src_train, src_val, src_test = load_medmnist(source_name, data_dir)
    _, _, tgt_test = load_medmnist(target_name, data_dir)
    
    return {
        "train": src_train,
        "val": src_val,
        "test_id": src_test,
        "test_ood": tgt_test,
        "source": source_name,
        "target": target_name,
    }


def create_corrupted_test(
    test_ds: MedMNISTDataset,
    corruption: str = "gaussian_noise",
    severity: float = 0.3,
) -> MedMNISTDataset:
    """
    Create a corrupted version of a test set for synthetic shift.
    
    Corruptions:
      - gaussian_noise: add N(0, σ²) noise
      - brightness: scale pixel values
      - contrast: reduce contrast toward mean
      - blur: simple box blur
    """
    images = test_ds.images.astype(np.float32) / 255.0
    
    if corruption == "gaussian_noise":
        noise = np.random.randn(*images.shape).astype(np.float32) * severity
        images = np.clip(images + noise, 0, 1)
    
    elif corruption == "brightness":
        images = np.clip(images * (1 + severity), 0, 1)
    
    elif corruption == "contrast":
        mean = images.mean(axis=(1, 2), keepdims=True)
        images = np.clip(mean + (images - mean) * (1 - severity), 0, 1)
    
    elif corruption == "blur":
        from scipy.ndimage import uniform_filter
        k = max(1, int(severity * 5))
        for i in range(len(images)):
            if images[i].ndim == 3:
                for c in range(images[i].shape[-1]):
                    images[i][:, :, c] = uniform_filter(images[i][:, :, c], size=k)
            else:
                images[i] = uniform_filter(images[i], size=k)
    
    images = (images * 255).astype(np.uint8)
    
    return MedMNISTDataset(images, test_ds.labels)


def create_vision_dataloaders(
    train_ds: MedMNISTDataset,
    val_ds: MedMNISTDataset,
    test_ds: MedMNISTDataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders optimized for GPU throughput."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader

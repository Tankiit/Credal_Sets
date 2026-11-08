"""Hugging Face image datasets behind the native concept-audit row contract."""
from pathlib import Path
import hashlib
import numbers
import numpy as np
from torch.utils.data import Dataset


DATASETS = {
    "cifar10": ("uoft-cs/cifar10", "img", "label"),
    "cifar100": ("uoft-cs/cifar100", "img", "fine_label"),
    "cub": ("bentrevett/caltech-ucsd-birds-200-2011", "image", "label"),
}


def _concept_vector(value):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Concepts must be a nonempty finite vector of targets in [0,1]")
    return result


class HFImageDataset(Dataset):
    """Returns (RGB PIL image, concept vector or None, integer task label).

    HF row order is preserved. External .npy concepts must match that exact order.
    """
    def __init__(self, dataset, *, image_column, label_column, concept_column=None,
                 concepts_file=None, metadata=None):
        if concept_column is not None and concepts_file is not None:
            raise ValueError("Choose concept_column or concepts_file, not both")
        required = [image_column, label_column] + ([concept_column] if concept_column else [])
        missing = set(required) - set(dataset.column_names)
        if missing:
            raise ValueError(f"Missing HF columns: {sorted(missing)}; available: {dataset.column_names}")
        if len(dataset) == 0:
            raise ValueError("Dataset split is empty")
        self.dataset = dataset
        self.image_column, self.label_column, self.concept_column = image_column, label_column, concept_column
        self.concepts = None
        self.metadata = dict(metadata or {})
        self.metadata.update({"provider": "huggingface", "fingerprint": getattr(dataset, "_fingerprint", None),
                              "image_column": image_column, "label_column": label_column,
                              "concept_column": concept_column})
        self.classes = getattr(dataset.features.get(label_column), "names", None)
        self.metadata["class_names"] = self.classes
        if concepts_file is not None:
            path = Path(concepts_file)
            self.concepts = np.load(path, allow_pickle=False)
            if self.concepts.ndim != 2 or self.concepts.shape[0] != len(dataset) or self.concepts.shape[1] == 0:
                raise ValueError("Concept file must have shape (number of rows in requested split, K)")
            if not np.isfinite(self.concepts).all() or np.any((self.concepts < 0) | (self.concepts > 1)):
                raise ValueError("Concept file targets must be finite and in [0,1]")
            self.metadata.update({"concepts_file": str(path), "concepts_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.has_concepts = self.concepts is not None or concept_column is not None
        self.num_concepts = None
        if self.concepts is not None:
            self.num_concepts = self.concepts.shape[1]
        elif concept_column is not None:
            self.num_concepts = len(_concept_vector(dataset.select_columns([concept_column])[0][concept_column]))
        self.metadata["num_concepts"] = self.num_concepts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row[self.image_column].convert("RGB")
        label = row[self.label_column]
        if not isinstance(label, numbers.Integral) or label < 0:
            raise ValueError("Task labels must be nonnegative integers; supply a ClassLabel or encoded label column")
        concepts = None
        if self.concepts is not None:
            concepts = self.concepts[index].astype(np.float32, copy=True)
        elif self.concept_column is not None:
            concepts = _concept_vector(row[self.concept_column])
            if len(concepts) != self.num_concepts:
                raise ValueError("Concept vector length must be constant across rows")
        return image, concepts, int(label)


def load_dataset(name, *, split="train", data_dir=None, cache_dir=None, revision=None, config=None,
                 image_column=None, label_column=None, concept_column=None,
                 concepts_file=None, require_concepts=False):
    """Load cifar10/cifar100/cub or a HF repo ID using datasets.load_dataset.

    split accepts HF slice expressions. data_dir reads local CIFAR batches into
    an in-memory HF dataset without downloads; otherwise use the Hub.
    No class labels are converted to concept annotations.
    """
    from datasets import load_dataset as hf_load_dataset

    key = name.lower().replace("-", "").replace("_", "")
    if key == "cub2002011":
        key = "cub"
    defaults = DATASETS.get(key)
    if defaults is None:
        defaults = next((spec for spec in DATASETS.values() if spec[0] == name), None)
    if defaults is None and "/" not in name:
        raise ValueError(f"Unknown dataset {name!r}; use {', '.join(DATASETS)} or a HF repository ID")
    repo, default_image, default_label = defaults or (name, "image", "label")
    if require_concepts and concept_column is None and concepts_file is None:
        raise ValueError("Concept audit requires --concept-column or --concepts-file; class labels are not concept annotations")
    kwargs = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    if config is not None:
        kwargs["name"] = config
    metadata = {"repository": repo, "split": split, "revision": revision, "config": config}
    if data_dir is not None:
        if repo not in (DATASETS["cifar10"][0], DATASETS["cifar100"][0]):
            raise ValueError("data_dir currently supports extracted CIFAR-10/100 Python batches")
        if revision is not None or config is not None:
            raise ValueError("revision/config apply to Hub datasets, not local CIFAR batches")
        from .local_cifar import load_local_cifar
        local_name = "cifar10" if repo == DATASETS["cifar10"][0] else "cifar100"
        dataset, local_metadata = load_local_cifar(local_name, data_dir, split)
        metadata.update(local_metadata)
        metadata["repository"] = None  # Do not claim local files came from the Hub.
    else:
        dataset = hf_load_dataset(repo, **kwargs)
    return HFImageDataset(dataset, image_column=image_column or default_image,
                          label_column=label_column or default_label,
                          concept_column=concept_column, concepts_file=concepts_file,
                          metadata=metadata)

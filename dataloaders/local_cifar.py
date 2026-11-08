"""Read existing CIFAR Python batches into an in-memory Hugging Face dataset."""
from pathlib import Path


def load_local_cifar(name, data_dir, split):
    """Use torchvision's checked batch reader, with downloads disabled.

    Slice before conversion to avoid encoding unused images. Returned HF columns
    match the corresponding Hub dataset, so downstream code is unchanged.
    """
    from datasets import Dataset, Features, Image, ClassLabel, ReadInstruction
    from torchvision.datasets import CIFAR10, CIFAR100

    factory = CIFAR10 if name == "cifar10" else CIFAR100
    folder = "cifar-10-batches-py" if name == "cifar10" else "cifar-100-python"
    root = Path(data_dir).expanduser().resolve()
    if root.name == folder:
        root = root.parent
    if not (root / folder).is_dir():
        raise FileNotFoundError(f"Expected extracted local dataset at {root / folder}")
    instructions = ReadInstruction.from_spec(split).to_absolute({"train": 50000, "test": 10000})
    if len(instructions) != 1:
        raise ValueError("Use a single train/test split or slice for local CIFAR")
    instruction = instructions[0]
    source = factory(root=str(root), train=instruction.splitname == "train", download=False)
    start, stop = instruction.from_ or 0, instruction.to
    indices = range(start, len(source) if stop is None else stop)
    label_column = "label" if name == "cifar10" else "fine_label"
    dataset = Dataset.from_dict(
        {"img": [source[i][0] for i in indices],
         label_column: [source.targets[i] for i in indices]},
        features=Features({"img": Image(), label_column: ClassLabel(names=source.classes)}),
    )
    return dataset, {"source": "local_cifar_batches", "data_dir": str(root),
                     "source_split": instruction.splitname, "source_row_start": start,
                     "source_row_stop": indices.stop}

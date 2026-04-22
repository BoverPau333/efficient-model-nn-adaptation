"""Dataset-related utilities shared across experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from src.experiments_config.config import SEED


def attach_subset_targets(subset, targets):
    """Attach a `.targets` array to a random_split subset."""
    subset.targets = np.array(targets)[subset.indices]
    return subset


def split_train_val(dataset, train_fraction: float = 0.8, seed: int = SEED):
    """Split a dataset into train/validation subsets and attach subset targets."""
    n_train = int(train_fraction * len(dataset))
    train_subset, val_subset = random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(seed),
    )
    attach_subset_targets(train_subset, dataset.targets)
    attach_subset_targets(val_subset, dataset.targets)
    return train_subset, val_subset


def extract_targets(dataset):
    """Extract targets from a dataset or subset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return np.array(dataset.dataset.targets)[dataset.indices]
    raise AttributeError("Cannot extract targets from dataset.")


def count_examples_per_class(dataset, classes: list) -> dict:
    """Return a dict with the number of examples per class."""
    targets = extract_targets(dataset)
    return {
        class_name: int((targets == class_idx).sum())
        for class_idx, class_name in enumerate(classes)
    }


class RemappedClassDataset(Dataset):
    """Dataset wrapper that filters indices and remaps labels to a compact range."""

    def __init__(self, dataset, indices, label_mapping: dict, classes: list):
        self.dataset = dataset
        self.indices = list(indices)
        self.label_mapping = {int(old): int(new) for old, new in label_mapping.items()}
        self.classes = list(classes)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        original_targets = extract_targets(dataset)
        self.targets = np.array(
            [self.label_mapping[int(original_targets[idx])] for idx in self.indices],
            dtype=int,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]
        return sample, self.label_mapping[int(label)]


def resolve_class_to_remove(classes: list, class_to_remove):
    """Resolve a class identifier expressed as index or class name."""
    if isinstance(class_to_remove, str):
        if class_to_remove not in classes:
            raise ValueError(
                f"Unknown class '{class_to_remove}'. Available classes: {classes}"
            )
        return classes.index(class_to_remove), class_to_remove

    class_idx = int(class_to_remove)
    if class_idx < 0 or class_idx >= len(classes):
        raise ValueError(
            f"Class index {class_idx} is out of range for {len(classes)} classes."
        )
    return class_idx, classes[class_idx]


def build_class_removal_metadata(classes: list, class_to_remove):
    """Build the metadata needed to remove one class consistently across splits."""
    removed_class_idx, removed_class_name = resolve_class_to_remove(classes, class_to_remove)
    remaining_classes = [
        class_name
        for class_idx, class_name in enumerate(classes)
        if class_idx != removed_class_idx
    ]

    if len(remaining_classes) < 2:
        raise ValueError(
            "Class removal would leave fewer than 2 classes, which is not a valid setup."
        )

    label_mapping = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            class_idx for class_idx in range(len(classes)) if class_idx != removed_class_idx
        )
    }

    return {
        "removed_class_idx": removed_class_idx,
        "removed_class_name": removed_class_name,
        "remaining_classes": remaining_classes,
        "label_mapping": label_mapping,
    }


def remove_class_and_remap(dataset, classes: list, class_to_remove):
    """Filter out one class from a dataset split and remap the remaining labels."""
    metadata = build_class_removal_metadata(classes, class_to_remove)
    targets = extract_targets(dataset)
    kept_indices = np.flatnonzero(targets != metadata["removed_class_idx"])
    filtered_dataset = RemappedClassDataset(
        dataset=dataset,
        indices=kept_indices,
        label_mapping=metadata["label_mapping"],
        classes=metadata["remaining_classes"],
    )
    return filtered_dataset, metadata

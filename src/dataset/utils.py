"""Dataset-related utilities shared across experiments."""

import numpy as np
import torch
from torch.utils.data import random_split

from src.config import SEED


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

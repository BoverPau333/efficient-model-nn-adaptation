"""Dataset reduction strategies for experiments."""

import numpy as np
from torch.utils.data import Subset

from src.config import SEED


def reduce_all_classes(dataset, fraction: float, seed: int = SEED) -> Subset:
    """Randomly keep the same fraction of samples from every class."""
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    kept = []

    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        n_keep = max(1, int(round(len(class_indices) * fraction)))
        chosen = rng.choice(class_indices, size=n_keep, replace=False)
        kept.extend(chosen.tolist())

    return Subset(dataset, kept)


def find_least_confused_class(cm: np.ndarray) -> int:
    """Return the class with the lowest off-diagonal confusion rate."""
    cm_float = cm.astype(float)
    row_sums = cm_float.sum(axis=1)
    off_diag_errors = row_sums - np.diag(cm_float)
    confusion_rates = np.divide(
        off_diag_errors,
        row_sums,
        out=np.ones_like(row_sums),
        where=row_sums > 0,
    )
    return int(np.argmin(confusion_rates))


def reduce_least_confused_class(dataset, cm: np.ndarray, fraction: float, seed: int = SEED) -> Subset:
    """Reduce only the class identified as least confused by the confusion matrix."""
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    target_class = find_least_confused_class(cm)
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    kept = []

    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        if class_idx == target_class:
            n_keep = max(1, int(round(len(class_indices) * fraction)))
            chosen = rng.choice(class_indices, size=n_keep, replace=False)
            kept.extend(chosen.tolist())
        else:
            kept.extend(class_indices.tolist())

    return Subset(dataset, kept)

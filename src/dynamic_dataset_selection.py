"""Seleccion dinamica de subconjuntos guiada por embeddings."""

import numpy as np
from torch.utils.data import Dataset, Subset

from src.dataset.utils import extract_targets


class RemappedSubset(Dataset):
    """Subset con remapeo opcional de etiquetas."""

    def __init__(self, dataset, indices, label_mapping=None, classes=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.label_mapping = None if label_mapping is None else {int(k): int(v) for k, v in label_mapping.items()}
        self.classes = list(classes) if classes is not None else getattr(dataset, "classes", None)
        original_targets = extract_targets(dataset)
        if self.label_mapping is None:
            self.targets = np.asarray([int(original_targets[idx]) for idx in self.indices], dtype=int)
        else:
            self.targets = np.asarray([self.label_mapping[int(original_targets[idx])] for idx in self.indices], dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]
        label = int(label)
        if self.label_mapping is not None:
            label = self.label_mapping[label]
        return sample, label


def sample_balanced_indices(dataset, samples_per_class: int, seed: int, excluded_classes=None):
    """Muestrea un numero equilibrado de ejemplos por clase."""
    excluded = set() if excluded_classes is None else {int(value) for value in excluded_classes}
    rng = np.random.default_rng(seed)
    targets = extract_targets(dataset)
    indices = []

    for class_idx in np.unique(targets):
        class_idx = int(class_idx)
        if class_idx in excluded:
            continue
        class_indices = np.flatnonzero(targets == class_idx)
        if len(class_indices) <= samples_per_class:
            selected = class_indices
        else:
            selected = rng.choice(class_indices, size=samples_per_class, replace=False)
        indices.extend(int(idx) for idx in selected.tolist())

    return sorted(indices)


def build_label_mapping_after_removal(num_original_classes: int, removed_class_idx: int):
    """Construye el remapeo compacto tras eliminar una clase."""
    return {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            class_idx for class_idx in range(num_original_classes) if class_idx != int(removed_class_idx)
        )
    }


def _select_closest(sample_ids, distances, num_samples):
    if len(sample_ids) == 0 or num_samples <= 0:
        return []
    order = np.argsort(distances)
    top_positions = order[: min(num_samples, len(order))]
    return [int(sample_ids[pos]) for pos in top_positions.tolist()]


def select_dynamic_subset(
    dataset,
    embeddings,
    labels,
    ids,
    modified_class_idx: int,
    neighbour_class_indices: list,
    class_centroids: dict,
    samples_per_modified_class: int,
    samples_per_neighbour_class: int,
    memory_samples_per_far_class: int,
    selection_strategy: str,
    seed: int,
    update_type: str,
):
    """Selecciona un subconjunto focalizado en la clase modificada y sus vecinas."""
    rng = np.random.default_rng(seed)
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    ids = np.asarray(ids)
    selected_ids = set()
    details = {
        "modified_class_samples": [],
        "neighbour_class_samples": {},
        "memory_class_samples": {},
    }

    neighbour_set = {int(item) for item in neighbour_class_indices}
    dataset_targets = extract_targets(dataset)
    far_classes = [
        int(class_idx)
        for class_idx in np.unique(dataset_targets)
        if class_idx not in neighbour_set and class_idx != int(modified_class_idx)
    ]

    modified_mask = labels == int(modified_class_idx)
    if modified_mask.any() and update_type != "remove":
        modified_embeddings = embeddings[modified_mask]
        modified_ids = ids[modified_mask]
        centroid = np.asarray(class_centroids[int(modified_class_idx)], dtype=float)
        distances = np.linalg.norm(modified_embeddings - centroid[None, :], axis=1)
        chosen = _select_closest(modified_ids, distances, samples_per_modified_class)
        selected_ids.update(chosen)
        details["modified_class_samples"] = chosen

    for neighbour_class_idx in neighbour_class_indices:
        neighbour_mask = labels == int(neighbour_class_idx)
        if not neighbour_mask.any():
            continue

        neighbour_embeddings = embeddings[neighbour_mask]
        neighbour_ids = ids[neighbour_mask]
        modified_centroid = np.asarray(class_centroids[int(modified_class_idx)], dtype=float)
        own_centroid = np.asarray(class_centroids[int(neighbour_class_idx)], dtype=float)
        distance_to_modified = np.linalg.norm(neighbour_embeddings - modified_centroid[None, :], axis=1)

        if selection_strategy == "frontier":
            distance_to_own = np.linalg.norm(neighbour_embeddings - own_centroid[None, :], axis=1)
            distances = distance_to_modified - distance_to_own
        else:
            distances = distance_to_modified

        chosen = _select_closest(neighbour_ids, distances, samples_per_neighbour_class)
        selected_ids.update(chosen)
        details["neighbour_class_samples"][str(neighbour_class_idx)] = chosen

    for far_class_idx in far_classes:
        if memory_samples_per_far_class <= 0:
            continue
        far_indices = np.flatnonzero(dataset_targets == far_class_idx)
        if len(far_indices) == 0:
            continue
        if len(far_indices) <= memory_samples_per_far_class:
            chosen_positions = far_indices
        else:
            chosen_positions = rng.choice(far_indices, size=memory_samples_per_far_class, replace=False)
        chosen_ids = [int(ids[pos]) for pos in np.asarray(chosen_positions).tolist()]
        selected_ids.update(chosen_ids)
        details["memory_class_samples"][str(far_class_idx)] = chosen_ids

    selected_indices = sorted(selected_ids)
    return Subset(dataset, selected_indices), details

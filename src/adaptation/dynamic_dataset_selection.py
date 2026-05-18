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


def _select_farthest_diverse(sample_ids, embeddings, num_samples, seed: int, anchor=None):
    """Selecciona ejemplos diversos mediante farthest-point sampling codicioso."""
    if len(sample_ids) == 0 or num_samples <= 0:
        return []

    sample_ids = np.asarray(sample_ids, dtype=int)
    embeddings = np.asarray(embeddings, dtype=float)
    limit = min(int(num_samples), len(sample_ids))
    if limit <= 0:
        return []

    if len(sample_ids) == 1:
        return [int(sample_ids[0])]

    if anchor is not None:
        anchor = np.asarray(anchor, dtype=float)
        start_idx = int(np.argmax(np.linalg.norm(embeddings - anchor[None, :], axis=1)))
    else:
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, len(sample_ids)))

    chosen_positions = [start_idx]
    min_distances = np.linalg.norm(embeddings - embeddings[start_idx][None, :], axis=1)
    min_distances[start_idx] = -np.inf

    while len(chosen_positions) < limit:
        next_idx = int(np.argmax(min_distances))
        if not np.isfinite(min_distances[next_idx]):
            break
        chosen_positions.append(next_idx)
        candidate_distances = np.linalg.norm(embeddings - embeddings[next_idx][None, :], axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)
        min_distances[chosen_positions] = -np.inf

    return [int(sample_ids[pos]) for pos in chosen_positions]


def _normalize_closeness(distances):
    """Convierte distancias en una cercania normalizada en [0, 1]."""
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return distances
    min_distance = float(np.min(distances))
    max_distance = float(np.max(distances))
    if np.isclose(max_distance, min_distance):
        return np.ones_like(distances, dtype=float)
    normalized = (distances - min_distance) / (max_distance - min_distance)
    return 1.0 - normalized


def _select_neighbour_by_composite_score(
    sample_ids,
    embeddings,
    modified_centroid,
    own_centroid,
    num_samples: int,
    alpha: float,
    beta: float,
    gamma: float,
):
    """Selecciona ejemplos de una clase vecina usando score compuesto y diversidad codiciosa."""
    if len(sample_ids) == 0 or num_samples <= 0:
        return [], []

    sample_ids = np.asarray(sample_ids, dtype=int)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    limit = min(int(num_samples), len(sample_ids))
    if limit <= 0:
        return [], []

    modified_centroid = np.asarray(modified_centroid, dtype=np.float32)
    own_centroid = np.asarray(own_centroid, dtype=np.float32)
    closeness_to_modified = _normalize_closeness(
        np.linalg.norm(embeddings - modified_centroid[None, :], axis=1)
    )
    closeness_to_own = _normalize_closeness(
        np.linalg.norm(embeddings - own_centroid[None, :], axis=1)
    )

    selected_positions = []
    score_rows = []
    selected_mask = np.zeros(len(sample_ids), dtype=bool)
    min_distances_to_selected = np.full(len(sample_ids), np.inf, dtype=np.float32)

    while len(selected_positions) < limit:
        remaining_positions = np.flatnonzero(~selected_mask)
        if remaining_positions.size == 0:
            break

        if not selected_positions or gamma <= 0.0:
            diversity = np.ones(remaining_positions.size, dtype=np.float32)
        else:
            diversity = _normalize_closeness(-min_distances_to_selected[remaining_positions])

        candidate_modified = closeness_to_modified[remaining_positions]
        candidate_own = closeness_to_own[remaining_positions]
        combined_scores = (alpha * candidate_modified) + (beta * candidate_own) + (gamma * diversity)
        best_local_idx = int(np.argmax(combined_scores))
        best_pos = int(remaining_positions[best_local_idx])
        selected_mask[best_pos] = True
        selected_positions.append(best_pos)
        score_rows.append(
            {
                "id": int(sample_ids[best_pos]),
                "score": float(combined_scores[best_local_idx]),
                "closeness_to_removed": float(closeness_to_modified[best_pos]),
                "closeness_to_own_class": float(closeness_to_own[best_pos]),
                "diversity": float(diversity[best_local_idx]),
            }
        )

        if len(selected_positions) < limit and gamma > 0.0:
            distances_to_new = np.linalg.norm(embeddings - embeddings[best_pos][None, :], axis=1)
            min_distances_to_selected = np.minimum(min_distances_to_selected, distances_to_new)

    return [int(sample_ids[pos]) for pos in selected_positions], score_rows


def _log_selection(progress_label: str | None, message: str):
    """Emite trazas de progreso para la fase de seleccion si se solicita."""
    if not progress_label:
        return
    print(f"[dynamic_selection] {progress_label} | {message}", flush=True)


def compute_selection_target_size(dataset_size: int, percentage: float) -> int:
    """Convierte un porcentaje en un numero de ejemplos a mantener."""
    if percentage <= 0 or percentage > 100:
        raise ValueError("--porc must be greater than 0 and at most 100.")
    if dataset_size <= 0:
        raise ValueError("dataset_size debe ser positivo.")
    return max(1, int(np.ceil(int(dataset_size) * (float(percentage) / 100.0))))


def _allocate_integer_budgets(total_budget: int, entries: list) -> dict:
    """Reparte un presupuesto entero respetando pesos y capacidad por entrada."""
    active_entries = [
        {
            "key": entry["key"],
            "weight": float(entry["weight"]),
            "capacity": int(entry["capacity"]),
        }
        for entry in entries
        if float(entry["weight"]) > 0.0 and int(entry["capacity"]) > 0
    ]
    if total_budget <= 0 or not active_entries:
        return {entry["key"]: 0 for entry in entries}

    total_budget = min(int(total_budget), sum(entry["capacity"] for entry in active_entries))
    total_weight = sum(entry["weight"] for entry in active_entries)
    exact_shares = {
        entry["key"]: (total_budget * entry["weight"] / total_weight)
        for entry in active_entries
    }
    allocations = {
        entry["key"]: min(int(np.floor(exact_shares[entry["key"]])), entry["capacity"])
        for entry in active_entries
    }
    remaining = total_budget - sum(allocations.values())

    while remaining > 0:
        candidates = [entry for entry in active_entries if allocations[entry["key"]] < entry["capacity"]]
        if not candidates:
            break
        best_entry = max(
            candidates,
            key=lambda entry: (
                exact_shares[entry["key"]] - allocations[entry["key"]],
                entry["weight"],
                entry["capacity"] - allocations[entry["key"]],
            ),
        )
        allocations[best_entry["key"]] += 1
        remaining -= 1

    return {
        entry["key"]: int(allocations.get(entry["key"], 0))
        for entry in entries
    }


def select_dynamic_subset(
    dataset,
    embeddings,
    labels,
    ids,
    modified_class_idx: int,
    neighbour_class_indices: list,
    class_centroids: dict,
    target_percentage: float,
    train_dataset_size: int,
    modified_class_weight: int,
    neighbour_class_weight: int,
    far_class_weight: int,
    selection_strategy: str,
    score_alpha: float,
    score_beta: float,
    score_gamma: float,
    seed: int,
    update_type: str,
    progress_label: str | None = None,
):
    """Selecciona un subconjunto focalizado en la clase modificada y sus vecinas."""
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    ids = np.asarray(ids, dtype=int)
    selected_ids = set()
    details = {
        "target_percentage": float(target_percentage),
        "target_num_samples": 0,
        "allocated_per_class": {},
        "removed_class_used_for_distances_only": update_type == "remove",
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
    target_num_samples = compute_selection_target_size(train_dataset_size, target_percentage)
    details["target_num_samples"] = int(target_num_samples)
    _log_selection(
        progress_label,
        f"target_num_samples={target_num_samples} | neighbours={list(map(int, neighbour_class_indices))} | far_classes={len(far_classes)}",
    )

    desired_far_per_class = max(int(far_class_weight), 0)
    far_entries = [
        {
            "key": int(class_idx),
            "weight": 1,
            "capacity": min(int(np.sum(labels == int(class_idx))), desired_far_per_class),
        }
        for class_idx in far_classes
        if desired_far_per_class > 0 and int(np.sum(labels == int(class_idx))) > 0
    ]
    far_total_budget = min(
        target_num_samples,
        sum(entry["capacity"] for entry in far_entries),
    )
    far_budgets = _allocate_integer_budgets(far_total_budget, far_entries)

    remaining_budget = max(target_num_samples - sum(far_budgets.values()), 0)
    neighbour_entries = [
        {
            "key": int(class_idx),
            "weight": int(neighbour_class_weight),
            "capacity": int(np.sum(labels == int(class_idx))),
        }
        for class_idx in neighbour_class_indices
        if int(np.sum(labels == int(class_idx))) > 0
    ]
    neighbour_budgets = _allocate_integer_budgets(remaining_budget, neighbour_entries)

    class_budgets = {}
    class_budgets.update(far_budgets)
    class_budgets.update(neighbour_budgets)
    details["allocated_per_class"] = {str(key): int(value) for key, value in class_budgets.items() if value > 0}
    _log_selection(
        progress_label,
        f"class budgets prepared | allocated_classes={len(details['allocated_per_class'])}",
    )

    modified_mask = labels == int(modified_class_idx)
    if modified_mask.any() and update_type != "remove":
        modified_embeddings = embeddings[modified_mask]
        modified_ids = ids[modified_mask]
        centroid = np.asarray(class_centroids[int(modified_class_idx)], dtype=float)
        distances = np.linalg.norm(modified_embeddings - centroid[None, :], axis=1)
        chosen = _select_closest(modified_ids, distances, class_budgets.get(int(modified_class_idx), 0))
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
            chosen = _select_closest(neighbour_ids, distances, class_budgets.get(int(neighbour_class_idx), 0))
            score_rows = []
            strategy_used = "frontier"
        elif selection_strategy == "nearest_to_modified":
            chosen = _select_closest(
                neighbour_ids,
                distance_to_modified,
                class_budgets.get(int(neighbour_class_idx), 0),
            )
            score_rows = []
            strategy_used = "nearest_to_modified"
        else:
            neighbour_budget = class_budgets.get(int(neighbour_class_idx), 0)
            chosen, score_rows = _select_neighbour_by_composite_score(
                sample_ids=neighbour_ids,
                embeddings=neighbour_embeddings,
                modified_centroid=modified_centroid,
                own_centroid=own_centroid,
                num_samples=neighbour_budget,
                alpha=score_alpha,
                beta=score_beta,
                gamma=score_gamma,
            )
            strategy_used = "composite_score"
        neighbour_budget = class_budgets.get(int(neighbour_class_idx), 0)
        selected_ids.update(chosen)
        _log_selection(
            progress_label,
            f"neighbour class={int(neighbour_class_idx)} | strategy={strategy_used} | budget={neighbour_budget} | selected={len(chosen)}",
        )
        details["neighbour_class_samples"][str(neighbour_class_idx)] = {
            "budget": int(neighbour_budget),
            "strategy": strategy_used,
            "weights": {
                "alpha": float(score_alpha),
                "beta": float(score_beta),
                "gamma": float(score_gamma),
            },
            "selected_ids": chosen,
            "scores": score_rows,
        }

    for far_class_idx in far_classes:
        budget = class_budgets.get(int(far_class_idx), 0)
        if budget <= 0:
            continue
        far_mask = labels == int(far_class_idx)
        if not far_mask.any():
            continue
        far_embeddings = embeddings[far_mask]
        far_ids = ids[far_mask]
        own_centroid = np.asarray(class_centroids[int(far_class_idx)], dtype=float)
        chosen = _select_farthest_diverse(
            sample_ids=far_ids,
            embeddings=far_embeddings,
            num_samples=budget,
            seed=seed + 1000 + int(far_class_idx),
            anchor=own_centroid,
        )
        selected_ids.update(chosen)
        _log_selection(
            progress_label,
            f"memory class={int(far_class_idx)} | budget={budget} | selected={len(chosen)}",
        )
        details["memory_class_samples"][str(far_class_idx)] = {
            "diversity": chosen,
            "budget": int(budget),
        }

    selected_indices = sorted(selected_ids)
    _log_selection(progress_label, f"selection completed | total_selected={len(selected_indices)}")
    return Subset(dataset, selected_indices), details

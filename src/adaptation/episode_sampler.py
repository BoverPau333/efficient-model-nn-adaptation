"""Muestreo determinista de episodios/support sets few-shot."""

from dataclasses import dataclass

import numpy as np

from src.dataset.utils import extract_targets


@dataclass(frozen=True)
class FewShotSupportSet:
    """Representa un support set few-shot por clase."""

    indices: list
    indices_by_class: dict
    shots_per_class: int
    num_classes: int


def sample_k_shot_support_set(dataset, shots_per_class: int, seed: int, class_indices: list | None = None):
    """Selecciona K ejemplos por clase de forma reproducible."""
    if shots_per_class <= 0:
        raise ValueError("shots_per_class debe ser mayor que 0")

    targets = np.asarray(extract_targets(dataset), dtype=int)
    available_classes = sorted(np.unique(targets).tolist()) if class_indices is None else [int(idx) for idx in class_indices]
    rng = np.random.default_rng(seed)

    indices = []
    indices_by_class = {}
    for class_idx in available_classes:
        class_positions = np.flatnonzero(targets == int(class_idx))
        if len(class_positions) < shots_per_class:
            raise ValueError(
                f"La clase {class_idx} solo tiene {len(class_positions)} ejemplos; "
                f"no se pueden muestrear {shots_per_class} shots."
            )
        chosen = rng.choice(class_positions, size=shots_per_class, replace=False)
        chosen = sorted(int(idx) for idx in chosen.tolist())
        indices.extend(chosen)
        indices_by_class[str(class_idx)] = chosen

    return FewShotSupportSet(
        indices=sorted(indices),
        indices_by_class=indices_by_class,
        shots_per_class=int(shots_per_class),
        num_classes=int(len(available_classes)),
    )

"""Utilidades compartidas para experimentos de eliminacion de clases."""

import numpy as np
import torch
from torch.utils.data import Subset

from src.experiments_config.class_removal_baseline_config import CLASSES_TO_REMOVE_BY_DATASET
from src.core.results_utils import parse_class_identifier


def format_percentage_slug(percentage: float) -> str:
    """Formatea un porcentaje para usarlo en rutas."""
    percentage_text = f"{float(percentage):g}".replace(".", "_")
    return f"porc_{percentage_text}"


def get_classes_to_remove(dataset_name: str, override_classes=None):
    """Resuelve la lista de clases a eliminar para un dataset."""
    if override_classes:
        parsed = [parse_class_identifier(value) for value in override_classes]
        if not parsed:
            raise ValueError("The override class list is empty.")
        return parsed

    configured = CLASSES_TO_REMOVE_BY_DATASET.get(dataset_name)
    if not configured:
        raise ValueError(
            f"No classes configured for dataset '{dataset_name}' in "
            "src/experiments_config/class_removal_baseline_config.py"
        )
    return [parse_class_identifier(value) for value in configured]


def select_training_subset(dataset, percentage: float, seed: int):
    """Mantiene un porcentaje determinista del split de entrenamiento."""
    if percentage <= 0 or percentage > 100:
        raise ValueError("--porc must be greater than 0 and at most 100.")

    if percentage == 100:
        return dataset

    total_examples = len(dataset)
    num_selected = max(1, int(np.ceil(total_examples * (percentage / 100.0))))
    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(total_examples, generator=generator)[:num_selected].tolist()
    return Subset(dataset, selected_indices)


def total_examples_from_split_counts(split_counts: dict, split_name: str = "train") -> int:
    """Devuelve el total de ejemplos de un split."""
    return int(sum(split_counts.get(split_name, {}).values()))

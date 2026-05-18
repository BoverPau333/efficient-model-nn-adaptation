"""Utilidades para adaptacion few-shot con Prototypical Networks."""

import math

import numpy as np

from src.core.distancias import distancia_coseno, distancia_euclidiana
from src.core.results_utils import (
    evaluate_classification_predictions,
    evaluate_prediction_confidence_from_probabilities,
)


def _distance_fn(metric: str):
    metric = metric.lower()
    if metric == "cosine":
        return distancia_coseno
    if metric == "euclidean":
        return distancia_euclidiana
    raise ValueError("metric debe ser 'cosine' o 'euclidean'")


def build_class_prototypes(embeddings, labels):
    """Construye un prototipo medio por clase."""
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings y labels deben tener la misma longitud")

    prototypes = {}
    for class_idx in sorted(np.unique(labels).tolist()):
        class_vectors = embeddings[labels == class_idx]
        if len(class_vectors) == 0:
            continue
        prototypes[int(class_idx)] = class_vectors.mean(axis=0)
    if not prototypes:
        raise ValueError("No se pudieron construir prototipos")
    return prototypes


def compute_distance_matrix_to_prototypes(query_embeddings, prototypes_by_class: dict, metric: str = "cosine"):
    """Calcula la matriz de distancias query-prototipo."""
    query_embeddings = np.asarray(query_embeddings, dtype=float)
    if query_embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D query embedding matrix, got shape {query_embeddings.shape}")

    ordered_classes = sorted(int(class_idx) for class_idx in prototypes_by_class)
    distance = _distance_fn(metric)
    matrix = np.zeros((len(query_embeddings), len(ordered_classes)), dtype=float)
    for row_idx, query_embedding in enumerate(query_embeddings):
        for col_idx, class_idx in enumerate(ordered_classes):
            matrix[row_idx, col_idx] = distance(query_embedding, prototypes_by_class[class_idx])
    return ordered_classes, matrix


def distances_to_probabilities(distance_matrix):
    """Convierte distancias en probabilidades via softmax sobre el negativo."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D distance matrix, got shape {distance_matrix.shape}")
    shifted = -distance_matrix
    shifted -= shifted.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    partition = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / partition


def compute_prototypical_loss(probabilities, true_labels, ordered_classes: list):
    """NLL media usando las probabilidades inducidas por los prototipos."""
    probabilities = np.asarray(probabilities, dtype=float)
    true_labels = np.asarray(true_labels, dtype=int)
    if len(probabilities) != len(true_labels):
        raise ValueError("probabilities y true_labels deben tener la misma longitud")

    class_to_position = {int(class_idx): idx for idx, class_idx in enumerate(ordered_classes)}
    nll_values = []
    for row_idx, label in enumerate(true_labels):
        label_position = class_to_position.get(int(label))
        if label_position is None:
            raise ValueError(f"La etiqueta {label} no existe entre los prototipos ordenados")
        prob = max(float(probabilities[row_idx, label_position]), 1e-12)
        nll_values.append(-math.log(prob))
    return float(np.mean(nll_values)) if nll_values else 0.0


def evaluate_prototypical_predictions(query_embeddings, query_labels, prototypes_by_class: dict, class_names: list, metric: str = "cosine"):
    """Clasifica queries usando prototipos y devuelve metricas en el formato del proyecto."""
    ordered_classes, distance_matrix = compute_distance_matrix_to_prototypes(
        query_embeddings,
        prototypes_by_class,
        metric=metric,
    )
    probabilities = distances_to_probabilities(distance_matrix)
    predicted_positions = np.argmin(distance_matrix, axis=1)
    predictions = np.asarray([ordered_classes[position] for position in predicted_positions], dtype=int)

    metrics = evaluate_classification_predictions(predictions, query_labels, class_names)
    metrics["distance_matrix"] = distance_matrix.tolist()
    metrics["probabilities"] = probabilities.tolist()
    metrics["prediction_confidence_mean"] = evaluate_prediction_confidence_from_probabilities(probabilities)
    metrics["loss"] = compute_prototypical_loss(probabilities, query_labels, ordered_classes)
    return metrics


def serialize_prototypes(prototypes_by_class: dict, class_names: list):
    """Serializa prototipos de forma legible para JSON."""
    rows = []
    for class_idx in sorted(prototypes_by_class):
        vector = np.asarray(prototypes_by_class[class_idx], dtype=float)
        rows.append(
            {
                "class_idx": int(class_idx),
                "class_name": class_names[int(class_idx)],
                "embedding_dim": int(vector.shape[0]),
                "vector": vector.tolist(),
                "l2_norm": float(np.linalg.norm(vector)),
            }
        )
    return rows

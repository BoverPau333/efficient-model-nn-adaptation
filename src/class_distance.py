"""Calculo de centroides y distancias entre clases."""

import numpy as np
from sklearn.metrics import pairwise_distances


def _normalize_rows(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("No se pueden normalizar vectores con norma cero")
    return vectors / norms


def compute_class_centroids(embeddings, labels, normalize: bool = False):
    """Calcula el centroide de cada clase."""
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    if normalize:
        embeddings = _normalize_rows(embeddings)

    centroids = []
    for class_idx in classes:
        centroid = embeddings[labels == class_idx].mean(axis=0)
        if normalize:
            centroid = _normalize_rows(centroid[None, :])[0]
        centroids.append(centroid)
    return classes, np.vstack(centroids)


def compute_distance_matrix(centroids, metric: str = "cosine"):
    """Construye la matriz de distancias entre centroides."""
    metric = metric.lower()
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric debe ser 'cosine' o 'euclidean'")
    return pairwise_distances(np.asarray(centroids, dtype=float), metric=metric)


def get_nearest_classes(classes, distance_matrix, modified_class_idx: int, k_neighbours: int):
    """Devuelve las k clases mas cercanas a la clase modificada."""
    classes = np.asarray(classes)
    if modified_class_idx not in classes:
        raise ValueError(f"La clase modificada {modified_class_idx} no aparece en los centroides")

    source_position = int(np.where(classes == modified_class_idx)[0][0])
    distances = distance_matrix[source_position]
    sorted_positions = np.argsort(distances)

    neighbours = []
    for position in sorted_positions:
        class_idx = int(classes[position])
        if class_idx == int(modified_class_idx):
            continue
        neighbours.append(
            {
                "class_idx": class_idx,
                "distance": float(distances[position]),
            }
        )
        if len(neighbours) >= k_neighbours:
            break
    return neighbours

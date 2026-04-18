"""Funciones de calculo de distancia entre vectores de embeddings."""

import numpy as np


def _as_numpy_vector(array_like) -> np.ndarray:
    """Convierte la entrada a un vector de numpy 1D."""
    vector = np.asarray(array_like, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {vector.shape}")
    return vector


def _normalizar_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza un vector a norma L2 unidad."""
    norma = np.linalg.norm(vector)
    if norma == 0.0:
        raise ValueError("No se puede normalizar un vector nulo")
    return vector / norma


def distancia_euclidiana(embedding_a, embedding_b) -> float:
    """Distancia euclidiana entre 2 vectores."""
    vector_a = _as_numpy_vector(embedding_a)
    vector_b = _as_numpy_vector(embedding_b)
    if vector_a.shape != vector_b.shape:
        raise ValueError(
            f"Tam diferentes, 1: {vector_a.shape} 2: {vector_b.shape}"
        )
    return float(np.linalg.norm(vector_a - vector_b))


def distancia_coseno(embedding_a, embedding_b) -> float:
    """Distancia coseno entre 2 vectores."""
    vector_a = _as_numpy_vector(embedding_a)
    vector_b = _as_numpy_vector(embedding_b)
    if vector_a.shape != vector_b.shape:
        raise ValueError(
            f"Tam diferentes, 1: {vector_a.shape} 2: {vector_b.shape}"
        )
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine distance is undefined for zero vectors")

    cosine_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return 1.0 - cosine_similarity


def distancia_euclidiana_normalizada(embedding_a, embedding_b) -> float:
    """Distancia euclidiana tras normalizar ambos vectores a norma unidad.

    Esta version elimina el efecto de la magnitud de los embeddings y permite
    compararla mejor con la distancia coseno.
    """
    vector_a = _as_numpy_vector(embedding_a)
    vector_b = _as_numpy_vector(embedding_b)
    if vector_a.shape != vector_b.shape:
        raise ValueError(
            f"Tam diferentes, 1: {vector_a.shape} 2: {vector_b.shape}"
        )

    vector_a = _normalizar_vector(vector_a)
    vector_b = _normalizar_vector(vector_b)
    return float(np.linalg.norm(vector_a - vector_b))

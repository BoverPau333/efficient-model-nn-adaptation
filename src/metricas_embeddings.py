"""Metricas para evaluar la calidad de un espacio de embeddings."""

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def _asegurar_embeddings_labels(embeddings, labels):
    """Convierte entradas a numpy y valida dimensiones basicas."""
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError(f"`embeddings` debe tener forma (n_muestras, n_features), no {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"`labels` debe ser un vector 1D, no {labels.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(
            f"`embeddings` y `labels` deben tener el mismo numero de muestras, no {len(embeddings)} y {len(labels)}"
        )
    if len(embeddings) < 2:
        raise ValueError("Se necesitan al menos 2 embeddings")

    return embeddings, labels


def _validar_metrica(metric: str) -> str:
    """Restringe las metricas a las dos distancias que queremos comparar."""
    metric = metric.lower()
    validas = {"euclidean", "cosine"}
    if metric not in validas:
        raise ValueError(f"Metrica no soportada: {metric}. Usa una de {sorted(validas)}")
    return metric


def _obtener_clases(labels):
    """Devuelve las clases unicas ordenadas."""
    return np.unique(labels)


def calcular_centroides(embeddings, labels):
    """Calcula el centroide de cada clase."""
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    clases = _obtener_clases(labels)

    centroides = np.vstack([
        embeddings[labels == clase].mean(axis=0)
        for clase in clases
    ])
    return clases, centroides


def ratio_intra_inter_clase(embeddings, labels, metric="euclidean"):
    """Calcula el ratio entre distancia intra-clase e inter-clase.

    Un valor mas bajo suele indicar un espacio mejor separado:
    - intra-clase baja: muestras de la misma clase mas compactas
    - inter-clase alta: clases mas alejadas entre si
    """
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)

    distancias = pairwise_distances(embeddings, metric=metric)
    misma_clase = labels[:, None] == labels[None, :]
    diagonal = np.eye(len(labels), dtype=bool)

    mascara_intra = misma_clase & ~diagonal
    mascara_inter = ~misma_clase

    intra = distancias[mascara_intra]
    inter = distancias[mascara_inter]

    if intra.size == 0:
        raise ValueError("No hay suficientes muestras por clase para calcular distancias intra-clase")
    if inter.size == 0:
        raise ValueError("Se necesitan al menos 2 clases para calcular distancias inter-clase")

    media_intra = float(np.mean(intra))
    media_inter = float(np.mean(inter))
    ratio = media_intra / media_inter if media_inter > 0 else np.inf

    return {
        "ratio_intra_inter": float(ratio),
        "distancia_media_intra": media_intra,
        "distancia_media_inter": media_inter,
    }


def margen_a_centroides(embeddings, labels, metric="euclidean"):
    """Calcula el margen de cada muestra frente al centroide correcto y su rival mas cercano.

    El margen se define como:
        distancia_al_centroide_rival - distancia_al_centroide_correcto

    Si el margen es positivo, la muestra esta mas cerca de su clase que de la rival.
    Si es negativo, es una muestra frontera o potencialmente mal situada.
    """
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)
    clases, centroides = calcular_centroides(embeddings, labels)

    distancias = pairwise_distances(embeddings, centroides, metric=metric)
    clase_a_indice = {clase: idx for idx, clase in enumerate(clases)}
    indices_correctos = np.array([clase_a_indice[label] for label in labels])

    dist_correcta = distancias[np.arange(len(labels)), indices_correctos]

    distancias_rivales = distancias.copy()
    distancias_rivales[np.arange(len(labels)), indices_correctos] = np.inf
    indice_rival = np.argmin(distancias_rivales, axis=1)
    dist_rival = distancias_rivales[np.arange(len(labels)), indice_rival]

    margenes = dist_rival - dist_correcta
    clases_rivales = clases[indice_rival]

    return {
        "margenes": margenes,
        "distancia_centroide_correcto": dist_correcta,
        "distancia_centroide_rival": dist_rival,
        "clase_rival_mas_cercana": clases_rivales,
        "margen_medio": float(np.mean(margenes)),
        "margen_mediano": float(np.median(margenes)),
        "margen_minimo": float(np.min(margenes)),
        "fraccion_margenes_negativos": float(np.mean(margenes < 0)),
    }


def knn_accuracy_embeddings(embeddings, labels, n_neighbors=5, metric="euclidean", cv_splits=5):
    """Evalua si el espacio organiza bien las clases usando kNN sobre embeddings.

    Se usa validacion cruzada estratificada para evitar medir sobre el mismo
    conjunto usado para entrenar el propio kNN.
    """
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)
    clases, conteos = np.unique(labels, return_counts=True)

    if len(clases) < 2:
        raise ValueError("Se necesitan al menos 2 clases para calcular kNN accuracy")

    minimo_por_clase = int(np.min(conteos))
    if minimo_por_clase < 2:
        raise ValueError("Cada clase necesita al menos 2 muestras para validacion cruzada")

    cv_splits = min(cv_splits, minimo_por_clase)
    if cv_splits < 2:
        raise ValueError("No hay suficientes muestras para validacion cruzada")

    n_neighbors = min(n_neighbors, len(embeddings) - len(clases))
    if n_neighbors < 1:
        n_neighbors = 1

    clasificador = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clasificador, embeddings, labels, cv=cv, scoring="accuracy")

    return {
        "knn_accuracy_media": float(np.mean(scores)),
        "knn_accuracy_std": float(np.std(scores)),
        "knn_scores_por_fold": scores,
        "n_neighbors": int(n_neighbors),
        "cv_splits": int(cv_splits),
    }


def silhouette_embeddings(embeddings, labels, metric="euclidean"):
    """Calcula silhouette score como medida global de separacion."""
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)

    if len(np.unique(labels)) < 2:
        raise ValueError("Se necesitan al menos 2 clases para silhouette score")

    valor = silhouette_score(embeddings, labels, metric=metric)
    return float(valor)


def matriz_distancias_centroides(embeddings, labels, metric="euclidean"):
    """Construye la matriz de distancias entre centroides de clase."""
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)
    clases, centroides = calcular_centroides(embeddings, labels)
    matriz = pairwise_distances(centroides, metric=metric)
    return clases, matriz


def correlacion_distancias_confusion(matriz_centroides, matriz_confusion):
    """Relaciona separacion geometrica entre clases y confusion del modelo.

    Se calcula la correlacion de Pearson entre:
    - confusion fuera de la diagonal
    - cercania entre centroides, expresada como distancia negativa

    Si la correlacion sale positiva y alta, significa que cuando dos clases
    estan mas cerca en embeddings, tienden a confundirse mas.
    """
    matriz_centroides = np.asarray(matriz_centroides, dtype=float)
    matriz_confusion = np.asarray(matriz_confusion, dtype=float)

    if matriz_centroides.shape != matriz_confusion.shape:
        raise ValueError(
            "La matriz de centroides y la de confusion deben tener la misma forma, "
            f"no {matriz_centroides.shape} y {matriz_confusion.shape}"
        )
    if matriz_centroides.ndim != 2 or matriz_centroides.shape[0] != matriz_centroides.shape[1]:
        raise ValueError("Las matrices deben ser cuadradas")

    mascara = ~np.eye(matriz_centroides.shape[0], dtype=bool)
    proximidad = -matriz_centroides[mascara]
    confusion = matriz_confusion[mascara]

    if np.allclose(proximidad, proximidad[0]) or np.allclose(confusion, confusion[0]):
        return float("nan")

    return float(np.corrcoef(proximidad, confusion)[0, 1])


def evaluar_metricas_embeddings(
    embeddings,
    labels,
    metric="euclidean",
    n_neighbors=5,
    cv_splits=5,
    matriz_confusion=None,
):
    """Calcula en bloque las metricas principales y de apoyo para embeddings."""
    embeddings, labels = _asegurar_embeddings_labels(embeddings, labels)
    metric = _validar_metrica(metric)

    resultado_ratio = ratio_intra_inter_clase(embeddings, labels, metric=metric)
    resultado_margen = margen_a_centroides(embeddings, labels, metric=metric)
    resultado_knn = knn_accuracy_embeddings(
        embeddings,
        labels,
        n_neighbors=n_neighbors,
        metric=metric,
        cv_splits=cv_splits,
    )
    silhouette = silhouette_embeddings(embeddings, labels, metric=metric)
    clases, dist_centroides = matriz_distancias_centroides(embeddings, labels, metric=metric)

    resultados = {
        "metric": metric,
        "ratio_intra_inter": resultado_ratio,
        "margen_centroides": resultado_margen,
        "knn_accuracy": resultado_knn,
        "silhouette_score": silhouette,
        "clases_centroides": clases,
        "matriz_distancias_centroides": dist_centroides,
    }

    if matriz_confusion is not None:
        resultados["correlacion_distancia_confusion"] = correlacion_distancias_confusion(
            dist_centroides,
            matriz_confusion,
        )

    return resultados

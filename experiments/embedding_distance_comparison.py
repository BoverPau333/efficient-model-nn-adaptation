"""Experimento para comparar metricas de embeddings con distancia euclidea y coseno."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, NUM_WORKERS, PLOTS_DIR, RESULTS_DIR
from src.dataset.loaders import DATASET_LOADERS
from src.embeddings import extraer_embeddings_y_logits
from src.metricas_embeddings import (
    evaluar_metricas_embeddings,
    margen_a_centroides,
)
from src.models import MODEL_BUILDERS
from src.training import finetune
from src.visualization import tensor_to_displayable_image


EMBEDDING_DISTANCE_METRICS = ["euclidean", "cosine"]
TOP_K_CLASES_SIMILARES = 5
NUM_EJEMPLOS_FRONTERA = 5


def guardar_csv(path, rows, fieldnames):
    """Guarda una lista de diccionarios en CSV."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV guardado -> {path}")


def calcular_confusion_desde_logits(logits, labels, num_classes):
    """Construye la matriz de confusion a partir de logits ya calculados."""
    preds = np.argmax(logits, axis=1)
    return sk_confusion_matrix(labels, preds, labels=list(range(num_classes)))


def ranking_clases_similares(matriz_centroides, classes):
    """Devuelve el ranking de clases mas parecidas para cada clase."""
    rankings = {}
    for idx, class_name in enumerate(classes):
        distancias = matriz_centroides[idx].copy()
        distancias[idx] = np.inf
        orden = np.argsort(distancias)
        rankings[class_name] = [classes[j] for j in orden]
    return rankings


def spearman_desde_ordenes(orden_a, orden_b):
    """Calcula Spearman entre dos rankings completos sin depender de scipy."""
    posiciones_b = {item: idx for idx, item in enumerate(orden_b)}
    n = len(orden_a)
    if n < 2:
        return float("nan")

    suma = 0.0
    for idx_a, item in enumerate(orden_a):
        d = idx_a - posiciones_b[item]
        suma += d * d

    return float(1.0 - (6.0 * suma) / (n * (n * n - 1)))


def estabilidad_ranking_clases(classes, matriz_euclidiana, matriz_coseno, top_k=TOP_K_CLASES_SIMILARES):
    """Compara cuan estable es el ranking de clases parecidas entre dos distancias."""
    ranking_e = ranking_clases_similares(matriz_euclidiana, classes)
    ranking_c = ranking_clases_similares(matriz_coseno, classes)

    rows = []
    spearmans = []
    overlaps = []

    for class_name in classes:
        orden_e = ranking_e[class_name]
        orden_c = ranking_c[class_name]
        top_e = set(orden_e[:top_k])
        top_c = set(orden_c[:top_k])
        overlap = len(top_e & top_c) / top_k if top_k > 0 else float("nan")
        spearman = spearman_desde_ordenes(orden_e, orden_c)

        rows.append(
            {
                "class_name": class_name,
                "spearman_ranking": spearman,
                "top_k_overlap": overlap,
                "top_euclidean": " | ".join(orden_e[:top_k]),
                "top_cosine": " | ".join(orden_c[:top_k]),
            }
        )
        spearmans.append(spearman)
        overlaps.append(overlap)

    return {
        "rows": rows,
        "spearman_medio": float(np.nanmean(spearmans)),
        "top_k_overlap_medio": float(np.nanmean(overlaps)),
        "ranking_euclidean": ranking_e,
        "ranking_cosine": ranking_c,
    }


def plot_matriz_centroides(matriz, classes, titulo, save_path):
    """Guarda un mapa de calor con la matriz de distancias entre centroides."""
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), max(5, len(classes) * 0.4)))
    image = ax.imshow(matriz, cmap="viridis")
    plt.colorbar(image, ax=ax, label="Distancia")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot guardado -> {save_path}")


def plot_ejemplos_frontera(dataset, indices, scores, labels, classes, title, save_path):
    """Visualiza ejemplos frontera ordenados por el score recibido."""
    if len(indices) == 0:
        return

    total = len(indices)
    fig, axes = plt.subplots(1, total, figsize=(3.3 * total, 3.8))
    if total == 1:
        axes = [axes]

    for ax, idx, score, label in zip(axes, indices, scores, labels):
        image, _ = dataset[int(idx)]
        ax.imshow(tensor_to_displayable_image(image))
        ax.set_title(f"{classes[int(label)]}\nscore={score:.4f}\nidx={int(idx)}", fontsize=9)
        ax.axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot guardado -> {save_path}")


def seleccionar_indices_frontera_por_margen(embeddings, labels, metric, top_n=NUM_EJEMPLOS_FRONTERA):
    """Selecciona los ejemplos con menor margen frente a centroides."""
    resultado = margen_a_centroides(embeddings, labels, metric=metric)
    margenes = resultado["margenes"]
    orden = np.argsort(margenes)
    indices = orden[: min(top_n, len(orden))]
    return {
        "indices": indices,
        "scores": margenes[indices],
        "labels": labels[indices],
    }


def seleccionar_indices_frontera_por_knn(embeddings_train, labels_train, embeddings_test, labels_test, metric, top_n=NUM_EJEMPLOS_FRONTERA):
    """Selecciona ejemplos frontera usando la diferencia de votos entre vecinos."""
    from sklearn.neighbors import NearestNeighbors

    vecinos = min(5, len(embeddings_train))
    modelo = NearestNeighbors(n_neighbors=vecinos, metric=metric)
    modelo.fit(embeddings_train)
    _, indices = modelo.kneighbors(embeddings_test)

    scores = []
    for idx_test, vecinos_idx in enumerate(indices):
        labels_vecinos = labels_train[vecinos_idx]
        coincidencias = labels_vecinos == labels_test[idx_test]
        score = float(np.mean(coincidencias))
        scores.append(score)

    scores = np.asarray(scores)
    orden = np.argsort(scores)
    indices_frontera = orden[: min(top_n, len(orden))]
    return {
        "indices": indices_frontera,
        "scores": scores[indices_frontera],
        "labels": labels_test[indices_frontera],
    }


def evaluar_dataset_y_modelo(dataset_name, loader_fn, model_name, model_builder):
    """Ejecuta el experimento completo para un dataset y una arquitectura."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name} | Arquitectura: {model_name}")
    print(f"{'=' * 80}")

    train_ds, val_ds, test_ds, classes = loader_fn()
    num_classes = len(classes)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Entrenando modelo...")
    model = model_builder(num_classes)
    finetune(model, train_loader, val_loader)

    print("Extrayendo embeddings de train y test...")
    train_output = extraer_embeddings_y_logits(model, train_loader)
    test_output = extraer_embeddings_y_logits(model, test_loader)
    cm = calcular_confusion_desde_logits(test_output["logits"], test_output["labels"], num_classes)

    resumen_rows = []
    ranking_rows = []

    resultados_por_metrica = {}
    matrices_centroides = {}

    for metric in EMBEDDING_DISTANCE_METRICS:
        print(f"Evaluando metricas con distancia {metric}...")
        resultados = evaluar_metricas_embeddings(
            test_output["embeddings"],
            test_output["labels"],
            metric=metric,
            matriz_confusion=cm,
        )
        resultados_por_metrica[metric] = resultados
        matrices_centroides[metric] = resultados["matriz_distancias_centroides"]

        resumen_rows.append(
            {
                "dataset": dataset_name,
                "architecture": model_name,
                "distance": metric,
                "knn_accuracy": resultados["knn_accuracy"]["knn_accuracy_media"],
                "knn_accuracy_std": resultados["knn_accuracy"]["knn_accuracy_std"],
                "ratio_intra_inter": resultados["ratio_intra_inter"]["ratio_intra_inter"],
                "distancia_media_intra": resultados["ratio_intra_inter"]["distancia_media_intra"],
                "distancia_media_inter": resultados["ratio_intra_inter"]["distancia_media_inter"],
                "silhouette": resultados["silhouette_score"],
                "correlacion_distancia_confusion": resultados.get("correlacion_distancia_confusion", float("nan")),
                "margen_medio": resultados["margen_centroides"]["margen_medio"],
                "fraccion_margenes_negativos": resultados["margen_centroides"]["fraccion_margenes_negativos"],
                "estabilidad_ranking_spearman_media": float("nan"),
                "estabilidad_ranking_topk_overlap_medio": float("nan"),
            }
        )

        plot_matriz_centroides(
            resultados["matriz_distancias_centroides"],
            resultados["clases_centroides"],
            f"{dataset_name} | {model_name} | Distancia entre centroides ({metric})",
            PLOTS_DIR / f"{dataset_name}_{model_name}_{metric}_centroides.png",
        )

        frontera_margen = seleccionar_indices_frontera_por_margen(
            test_output["embeddings"],
            test_output["labels"],
            metric=metric,
        )
        plot_ejemplos_frontera(
            test_ds,
            frontera_margen["indices"],
            frontera_margen["scores"],
            frontera_margen["labels"],
            classes,
            f"{dataset_name} | {model_name} | Frontera por margen ({metric})",
            PLOTS_DIR / f"{dataset_name}_{model_name}_{metric}_frontera_margen.png",
        )

        frontera_knn = seleccionar_indices_frontera_por_knn(
            train_output["embeddings"],
            train_output["labels"],
            test_output["embeddings"],
            test_output["labels"],
            metric=metric,
        )
        plot_ejemplos_frontera(
            test_ds,
            frontera_knn["indices"],
            frontera_knn["scores"],
            frontera_knn["labels"],
            classes,
            f"{dataset_name} | {model_name} | Frontera por kNN ({metric})",
            PLOTS_DIR / f"{dataset_name}_{model_name}_{metric}_frontera_knn.png",
        )

    estabilidad = estabilidad_ranking_clases(
        classes,
        matrices_centroides["euclidean"],
        matrices_centroides["cosine"],
    )

    for row in estabilidad["rows"]:
        ranking_rows.append(
            {
                "dataset": dataset_name,
                "architecture": model_name,
                **row,
            }
        )

    for row in resumen_rows:
        row["estabilidad_ranking_spearman_media"] = estabilidad["spearman_medio"]
        row["estabilidad_ranking_topk_overlap_medio"] = estabilidad["top_k_overlap_medio"]

    return resumen_rows, ranking_rows


def run_all():
    """Lanza el experimento para todos los datasets y arquitecturas disponibles."""
    all_summary_rows = []
    all_ranking_rows = []

    for dataset_name, loader_fn in DATASET_LOADERS.items():
        for model_name, model_builder in MODEL_BUILDERS.items():
            try:
                summary_rows, ranking_rows = evaluar_dataset_y_modelo(
                    dataset_name,
                    loader_fn,
                    model_name,
                    model_builder,
                )
                all_summary_rows.extend(summary_rows)
                all_ranking_rows.extend(ranking_rows)
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")

    if all_summary_rows:
        guardar_csv(
            RESULTS_DIR / "embedding_distance_comparison_summary.csv",
            all_summary_rows,
            list(all_summary_rows[0].keys()),
        )

    if all_ranking_rows:
        guardar_csv(
            RESULTS_DIR / "embedding_distance_comparison_rankings.csv",
            all_ranking_rows,
            list(all_ranking_rows[0].keys()),
        )

    print("\nExperimento de comparacion de distancias completado.")


if __name__ == "__main__":
    run_all()

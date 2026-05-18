"""Compara distancias sobre embeddings y logits y exporta CSVs sin generar plots."""

import csv

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader

from src.experiments_config.config import BATCH_SIZE, NUM_WORKERS, RESULTS_DIR
from src.dataset.loaders import DATASET_LOADERS
from src.core.embeddings import extraer_embeddings_y_logits
from src.metricas_embeddings import evaluar_metricas_embeddings
from src.models import MODEL_BUILDERS
from src.core.training import evaluate, finetune


DISTANCE_METRICS = [
    "cosine",
    "euclidean",
    "euclidean_normalized",
    "sqeuclidean_normalized",
]
REPRESENTATIONS = ("embeddings", "logits")


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


def _nombre_distancia(metric):
    """Devuelve un nombre legible para el CSV."""
    nombres = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "euclidean_normalized": "euclidean_l2_normalized",
        "sqeuclidean_normalized": "squared_euclidean_l2_normalized",
    }
    return nombres[metric]


def _extraer_fila_metricas(dataset_name, model_name, representation_name, metric, labels, features, cm):
    """Calcula una fila resumida con las metricas mas comparables."""
    resultados = evaluar_metricas_embeddings(
        features,
        labels,
        metric=metric,
        matriz_confusion=cm,
    )

    ratio = resultados["ratio_intra_inter"]
    knn = resultados["knn_accuracy"]
    margen = resultados["margen_centroides"]

    return {
        "dataset": dataset_name,
        "architecture": model_name,
        "representation": representation_name,
        "distance": _nombre_distancia(metric),
        "n_samples": int(len(labels)),
        "n_features": int(features.shape[1]),
        "knn_accuracy_mean": knn["knn_accuracy_media"],
        "knn_accuracy_std": knn["knn_accuracy_std"],
        "knn_neighbors": knn["n_neighbors"],
        "cv_splits": knn["cv_splits"],
        "ratio_intra_inter": ratio["ratio_intra_inter"],
        "mean_intra_class_distance": ratio["distancia_media_intra"],
        "mean_inter_class_distance": ratio["distancia_media_inter"],
        "silhouette_score": resultados["silhouette_score"],
        "distance_confusion_correlation": resultados.get("correlacion_distancia_confusion", float("nan")),
        "mean_centroid_margin": margen["margen_medio"],
        "median_centroid_margin": margen["margen_mediano"],
        "min_centroid_margin": margen["margen_minimo"],
        "negative_margin_fraction": margen["fraccion_margenes_negativos"],
    }


def _media_rangos(filas):
    """Agrega una puntuacion global por fila usando ranking por metrica."""
    if not filas:
        return filas

    reglas = {
        "knn_accuracy_mean": False,
        "ratio_intra_inter": True,
        "silhouette_score": False,
        "distance_confusion_correlation": False,
        "mean_centroid_margin": False,
        "negative_margin_fraction": True,
    }

    for columna, ascendente in reglas.items():
        valores = np.asarray([fila[columna] for fila in filas], dtype=float)

        if columna == "distance_confusion_correlation":
            valores = np.abs(valores)

        valores_para_ordenar = valores.copy()
        if np.all(np.isnan(valores_para_ordenar)):
            rangos = np.full(len(filas), np.nan, dtype=float)
        else:
            if ascendente:
                valores_para_ordenar = np.where(np.isnan(valores_para_ordenar), np.inf, valores_para_ordenar)
                orden = np.argsort(valores_para_ordenar)
            else:
                valores_para_ordenar = np.where(np.isnan(valores_para_ordenar), -np.inf, valores_para_ordenar)
                orden = np.argsort(-valores_para_ordenar)

            rangos = np.empty(len(filas), dtype=float)
            rangos[orden] = np.arange(1, len(filas) + 1, dtype=float)
            rangos[np.isnan(valores)] = np.nan

        for fila, rango in zip(filas, rangos):
            fila[f"rank_{columna}"] = rango

    rank_cols = [f"rank_{columna}" for columna in reglas]
    for fila in filas:
        fila["mean_rank"] = float(np.nanmean([fila[col] for col in rank_cols]))

    orden_global = np.argsort([fila["mean_rank"] for fila in filas])
    for pos, idx in enumerate(orden_global, start=1):
        filas[idx]["overall_rank"] = pos

    return filas


def _filas_comparacion(filas_resumen):
    """Genera un CSV derivado para comparar embeddings frente a logits."""
    filas = []
    agrupadas = {}
    for fila in filas_resumen:
        clave = (fila["dataset"], fila["architecture"], fila["distance"])
        agrupadas.setdefault(clave, {})[fila["representation"]] = fila

    for (dataset_name, model_name, distance), grupo in agrupadas.items():
        if set(grupo) != set(REPRESENTATIONS):
            continue

        fila_embeddings = grupo["embeddings"]
        fila_logits = grupo["logits"]
        knn_advantage = fila_embeddings["knn_accuracy_mean"] - fila_logits["knn_accuracy_mean"]
        silhouette_advantage = fila_embeddings["silhouette_score"] - fila_logits["silhouette_score"]
        ratio_advantage = fila_logits["ratio_intra_inter"] - fila_embeddings["ratio_intra_inter"]
        distance_confusion_corr_advantage = (
            abs(fila_embeddings["distance_confusion_correlation"]) - abs(fila_logits["distance_confusion_correlation"])
        )
        mean_centroid_margin_advantage = (
            fila_embeddings["mean_centroid_margin"] - fila_logits["mean_centroid_margin"]
        )
        negative_margin_fraction_advantage = (
            fila_logits["negative_margin_fraction"] - fila_embeddings["negative_margin_fraction"]
        )
        metric_advantages = [
            knn_advantage,
            silhouette_advantage,
            ratio_advantage,
            distance_confusion_corr_advantage,
            mean_centroid_margin_advantage,
            negative_margin_fraction_advantage,
        ]
        n_metrics_won_by_embeddings = int(sum(valor > 0 for valor in metric_advantages))
        n_metrics_won_by_logits = int(sum(valor < 0 for valor in metric_advantages))

        filas.append(
            {
                "dataset": dataset_name,
                "architecture": model_name,
                "distance": distance,
                "knn_accuracy_advantage_embeddings": knn_advantage,
                "silhouette_advantage_embeddings": silhouette_advantage,
                "ratio_intra_inter_advantage_embeddings": ratio_advantage,
                "distance_confusion_corr_abs_advantage_embeddings": distance_confusion_corr_advantage,
                "mean_centroid_margin_advantage_embeddings": mean_centroid_margin_advantage,
                "negative_margin_fraction_advantage_embeddings": negative_margin_fraction_advantage,
                "better_representation_by_mean_rank": min(
                    (fila_embeddings["representation"], fila_embeddings["mean_rank"]),
                    (fila_logits["representation"], fila_logits["mean_rank"]),
                    key=lambda item: item[1],
                )[0],
                "n_metrics_won_by_embeddings": n_metrics_won_by_embeddings,
                "n_metrics_won_by_logits": n_metrics_won_by_logits,
            }
        )

    return filas


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

    print("Extrayendo embeddings y logits...")
    test_output = extraer_embeddings_y_logits(model, test_loader)
    test_accuracy, _, cm = evaluate(model, test_loader, num_classes)
    cm_desde_logits = calcular_confusion_desde_logits(test_output["logits"], test_output["labels"], num_classes)

    if not np.array_equal(cm, cm_desde_logits):
        raise ValueError("La matriz de confusion desde evaluate y desde logits no coincide")

    filas_resumen = []
    for representation_name in REPRESENTATIONS:
        features = test_output[representation_name]
        print(f"Evaluando representacion {representation_name}...")

        filas_representacion = []
        for metric in DISTANCE_METRICS:
            print(f"  Distancia: {metric}")
            fila = _extraer_fila_metricas(
                dataset_name,
                model_name,
                representation_name,
                metric,
                test_output["labels"],
                features,
                cm,
            )
            fila["test_accuracy"] = test_accuracy
            filas_representacion.append(fila)

        _media_rangos(filas_representacion)
        filas_resumen.extend(filas_representacion)

    filas_comparacion = _filas_comparacion(filas_resumen)
    return filas_resumen, filas_comparacion


def run_all():
    """Lanza el experimento para todos los datasets y arquitecturas disponibles."""
    all_summary_rows = []
    all_comparison_rows = []

    for dataset_name, loader_fn in DATASET_LOADERS.items():
        for model_name, model_builder in MODEL_BUILDERS.items():
            try:
                summary_rows, comparison_rows = evaluar_dataset_y_modelo(
                    dataset_name,
                    loader_fn,
                    model_name,
                    model_builder,
                )
                all_summary_rows.extend(summary_rows)
                all_comparison_rows.extend(comparison_rows)
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")

    if all_summary_rows:
        guardar_csv(
            RESULTS_DIR / "logits_vs_embeddings_distance_summary.csv",
            all_summary_rows,
            list(all_summary_rows[0].keys()),
        )

    if all_comparison_rows:
        guardar_csv(
            RESULTS_DIR / "logits_vs_embeddings_distance_comparison.csv",
            all_comparison_rows,
            list(all_comparison_rows[0].keys()),
        )

    print("\nExperimento de comparacion de distancias completado.")


if __name__ == "__main__":
    run_all()

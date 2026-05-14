"""Comparacion agregada entre metodos de class removal."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.results_utils import load_json, save_json, write_csv


METHOD_SOURCES = {
    "baseline": "class_removal_baseline",
    "frozen_backbone_head": "class_removal_frozen_backbone_head",
    "finetuning": "class_removal_finetuning",
}

EXCLUDED_METHOD_VARIANTS = set()


METRIC_COLUMNS = [
    "tiempo_total_de_adaptacion",
    "accuracy_global",
    "accuracy_en_clases_restantes",
    "forgetting_u_olvido",
    "numero_de_parametros_entrenados_o_modificados",
    "numero_de_ejemplos_utilizados",
    "confianza_de_prediccion",
]


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _safe_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _method_variant(row: dict, source_name: str):
    if source_name == "baseline":
        return source_name

    train_percentage = row.get("train_percentage")
    training_mode = row.get("training_mode", source_name)
    if train_percentage in (None, "", "None"):
        return training_mode
    return f"{training_mode}_{float(train_percentage):g}%"


def normalize_summary_row(row: dict, source_name: str):
    """Lleva una fila de resumen a un esquema comun."""
    accuracy_global = row.get("accuracy_global", row.get("test_overall_accuracy"))
    accuracy_remaining = row.get("accuracy_en_clases_restantes", row.get("test_mean_per_class_accuracy"))
    total_time = row.get("tiempo_total_de_adaptacion", row.get("elapsed_seconds"))
    num_examples = row.get("numero_de_ejemplos_utilizados", row.get("num_examples_used_for_adaptation"))
    confidence = row.get("confianza_de_prediccion", row.get("prediction_confidence_mean"))
    trainable_params = row.get(
        "numero_de_parametros_entrenados_o_modificados",
        row.get("num_trainable_parameters"),
    )

    normalized = {
        "source_name": source_name,
        "method_variant": _method_variant(row, source_name),
        "dataset": row.get("dataset"),
        "model_name": row.get("model_name"),
        "removed_class": row.get("removed_class"),
        "status": row.get("status"),
        "train_percentage": _safe_float(row.get("train_percentage")) if "train_percentage" in row else (100.0 if source_name == "baseline" else None),
        "training_mode": row.get("training_mode", source_name),
        "backbone_mode": row.get("backbone_mode"),
        "trainable_scope": row.get("trainable_scope"),
        "best_epoch": _safe_int(row.get("best_epoch")),
        "epochs_ran": _safe_int(row.get("epochs_ran")),
        "best_val_loss": _safe_float(row.get("best_val_loss")),
        "best_val_accuracy": _safe_float(row.get("best_val_accuracy")),
        "tiempo_total_de_adaptacion": _safe_float(total_time),
        "accuracy_global": _safe_float(accuracy_global),
        "accuracy_en_clases_restantes": _safe_float(accuracy_remaining),
        "forgetting_u_olvido": _safe_float(row.get("forgetting_u_olvido")),
        "numero_de_ejemplos_utilizados": _safe_int(num_examples),
        "confianza_de_prediccion": _safe_float(confidence),
        "numero_de_parametros_entrenados_o_modificados": _safe_int(trainable_params),
        "memoria_adicional_requerida": _safe_float(row.get("memoria_adicional_requerida")),
    }
    return normalized


def load_all_method_rows(results_root: Path):
    """Carga y normaliza todas las ejecuciones reales de los tres metodos."""
    rows = []
    for source_name, folder_name in METHOD_SOURCES.items():
        method_dir = results_root / folder_name
        if not method_dir.exists():
            continue
        for metrics_path in sorted(method_dir.glob("**/final_metrics.json")):
            if "percentage_summaries" in str(metrics_path) or "method_comparison" in str(metrics_path):
                continue
            payload = load_json(metrics_path)
            normalized = normalize_summary_row(payload, source_name)
            rows.append(normalized)
    return rows


def filter_completed_rows(rows: list):
    """Mantiene solo filas completadas u omitidas por existir."""
    completed = []
    for row in rows:
        if row.get("method_variant") in EXCLUDED_METHOD_VARIANTS:
            continue
        status = row.get("status")
        if status is None:
            completed.append(row)
        elif status in {"completed", "skipped_existing"}:
            completed.append(row)
    return completed


def aggregate_rows(rows: list, group_keys: list):
    """Agrega medias y desviaciones por grupo."""
    grouped = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["num_runs"] = len(group_rows)

        for metric_name in METRIC_COLUMNS:
            values = [row[metric_name] for row in group_rows if row.get(metric_name) is not None]
            if not values:
                summary[f"{metric_name}_mean"] = None
                summary[f"{metric_name}_std"] = None
            else:
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_std"] = float(np.std(values))
        summary_rows.append(summary)

    return summary_rows


def add_relative_to_baseline(summary_rows: list, group_keys: list):
    """Anade deltas frente al baseline dentro del mismo dataset si existe."""
    baseline_lookup = {}
    for row in summary_rows:
        if row.get("source_name") == "baseline":
            key = tuple(row.get(group_key) for group_key in group_keys if group_key != "source_name")
            baseline_lookup[key] = row

    enriched = []
    for row in summary_rows:
        copied = dict(row)
        key = tuple(row.get(group_key) for group_key in group_keys if group_key != "source_name")
        baseline = baseline_lookup.get(key)
        if baseline is not None:
            for metric_name in ["tiempo_total_de_adaptacion", "accuracy_global", "forgetting_u_olvido"]:
                current = row.get(f"{metric_name}_mean")
                reference = baseline.get(f"{metric_name}_mean")
                copied[f"{metric_name}_delta_vs_baseline"] = (
                    None if current is None or reference is None else float(current) - float(reference)
                )
        enriched.append(copied)
    return enriched


def save_method_comparison_tables(output_dir: Path, all_rows: list, completed_rows: list):
    """Guarda tablas de comparacion a disco."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_runs_normalized.csv", all_rows)
    write_csv(output_dir / "completed_runs_normalized.csv", completed_rows)

    overall = aggregate_rows(completed_rows, ["source_name", "method_variant"])
    by_dataset = aggregate_rows(completed_rows, ["dataset", "source_name", "method_variant"])
    by_model = aggregate_rows(completed_rows, ["model_name", "source_name", "method_variant"])

    by_dataset = add_relative_to_baseline(by_dataset, ["dataset", "source_name", "method_variant"])

    write_csv(output_dir / "method_summary_overall.csv", overall)
    write_csv(output_dir / "method_summary_by_dataset.csv", by_dataset)
    write_csv(output_dir / "method_summary_by_model.csv", by_model)
    save_json(output_dir / "method_summary_overall.json", overall)
    save_json(output_dir / "method_summary_by_dataset.json", by_dataset)
    save_json(output_dir / "method_summary_by_model.json", by_model)

    return {
        "overall": overall,
        "by_dataset": by_dataset,
        "by_model": by_model,
    }


def _plot_metric_bar(ax, rows: list, metric_name: str, title: str, ylabel: str):
    labels = [row["method_variant"] for row in rows]
    means = np.array([row.get(f"{metric_name}_mean") or 0.0 for row in rows], dtype=float)
    stds = np.array([row.get(f"{metric_name}_std") or 0.0 for row in rows], dtype=float)

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_method_comparison(output_dir: Path, overall_rows: list):
    """Genera una figura agregada comparando metricas clave por metodo."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "method_comparison_overall.png"

    ordered_rows = sorted(
        overall_rows,
        key=lambda row: (
            0 if row["source_name"] == "baseline" else 1 if row["source_name"] == "frozen_backbone_head" else 2,
            row["method_variant"],
        ),
    )

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    plots = [
        ("tiempo_total_de_adaptacion", "Tiempo medio de adaptacion", "Seconds"),
        ("numero_de_parametros_entrenados_o_modificados", "Parametros entrenados", "Count"),
        ("accuracy_global", "Accuracy global media", "Accuracy"),
        ("accuracy_en_clases_restantes", "Accuracy en clases restantes", "Accuracy"),
        ("forgetting_u_olvido", "Forgetting medio", "Drop"),
        ("numero_de_ejemplos_utilizados", "Ejemplos usados", "Count"),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, plots):
        _plot_metric_bar(ax, ordered_rows, metric_name, title, ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path

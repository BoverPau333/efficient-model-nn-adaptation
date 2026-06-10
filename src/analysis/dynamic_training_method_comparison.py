"""Comparacion entre dynamic fine-tuning y otros metodos de entrenamiento para un porcentaje dado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.results_utils import load_json, save_json, write_csv


METHOD_SOURCES = {
    "frozen_backbone_head": {
        "folder": "class_removal_frozen_backbone_head",
        "label": "FT_Head 10%",
        "glob": "**/final_metrics.json",
    },
    "finetuning": {
        "folder": "class_removal_finetuning",
        "label": "FT_Two_Phase 10%",
        "glob": "**/final_metrics.json",
    },
    "dynamic_precomputed": {
        "folder": "dynamic_embedding_finetuning/precompute_embeddings_then_finetune",
        "label": "FT_Select_Dist_Pre",
        "glob": "**/final_metrics.json",
    },
    "dynamic_epoch1": {
        "folder": "dynamic_embedding_finetuning/epoch1_embeddings_dynamic_finetune",
        "label": "FT_Select_Dist_Epoch1",
        "glob": "**/final_metrics.json",
    },
}

BASELINE_FOLDER = "class_removal_baseline"
BASELINE_ESTIMATED_SOURCE = "baseline_estimated"
BASELINE_ESTIMATED_LABEL = "Baseline 10%"
BASELINE_ACCURACY_EXTRA_DROP = 0.05

METHOD_ORDER = [
    BASELINE_ESTIMATED_SOURCE,
    "frozen_backbone_head",
    "finetuning",
    "dynamic_precomputed",
    "dynamic_epoch1",
    "dynamic_precomputed_early_stopping",
    "dynamic_epoch1_early_stopping",
]

METHOD_COLORS = {
    BASELINE_ESTIMATED_SOURCE: "#7f7f7f",
    "frozen_backbone_head": "#f28e2b",
    "finetuning": "#59a14f",
    "dynamic_precomputed": "#d62728",
    "dynamic_epoch1": "#4C78A8",
    "dynamic_precomputed_early_stopping": "#d62728",
    "dynamic_epoch1_early_stopping": "#4C78A8",
}

METRICS = [
    "accuracy_global",
    "forgetting_u_olvido",
    "tiempo_total_de_adaptacion",
    "numero_de_ejemplos_utilizados",
]


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _safe_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def build_method_sources(dynamic_variant: str | None = None):
    sources = {
        "frozen_backbone_head": METHOD_SOURCES["frozen_backbone_head"].copy(),
        "finetuning": METHOD_SOURCES["finetuning"].copy(),
    }

    dynamic_base_keys = ("dynamic_precomputed", "dynamic_epoch1")

    if dynamic_variant == "both":
        for source_name in dynamic_base_keys:
            sources[source_name] = METHOD_SOURCES[source_name].copy()
            variant_source_name = f"{source_name}_early_stopping"
            variant_source = METHOD_SOURCES[source_name].copy()
            variant_source["folder"] = f"{variant_source['folder']}/early_stopping"
            variant_source["label"] = f"{variant_source['label']} (ES)"
            sources[variant_source_name] = variant_source
        return sources

    for source_name in dynamic_base_keys:
        sources[source_name] = METHOD_SOURCES[source_name].copy()

    if dynamic_variant:
        for source_name in dynamic_base_keys:
            sources[source_name]["folder"] = f"{sources[source_name]['folder']}/{dynamic_variant}"
    return sources


def _normalize_standard_payload(payload: dict, source_name: str, method_sources: dict):
    return {
        "source_name": source_name,
        "method_label": method_sources[source_name]["label"],
        "dataset": payload.get("dataset"),
        "model_name": payload.get("model_name"),
        "modified_class": payload.get("removed_class"),
        "train_percentage": _safe_float(payload.get("train_percentage")),
        "accuracy_global": _safe_float(payload.get("test_overall_accuracy")),
        "forgetting_u_olvido": _safe_float(payload.get("forgetting_u_olvido")),
        "tiempo_total_de_adaptacion": _safe_float(payload.get("elapsed_seconds")),
        "numero_de_ejemplos_utilizados": _safe_int(payload.get("num_examples_used_for_adaptation")),
        "best_epoch": _safe_int(payload.get("best_epoch")),
        "epochs_ran": _safe_int(payload.get("epochs_ran")),
        "experiment_dir": str(payload.get("experiment_dir", "")),
    }


def _normalize_dynamic_payload(payload: dict, source_name: str, method_sources: dict):
    return {
        "source_name": source_name,
        "method_label": method_sources[source_name]["label"],
        "dataset": payload.get("dataset"),
        "model_name": payload.get("model_name"),
        "modified_class": payload.get("modified_class"),
        "train_percentage": _safe_float(payload.get("train_percentage")),
        "accuracy_global": _safe_float(payload.get("accuracy")),
        "forgetting_u_olvido": _safe_float(payload.get("forgetting_score")),
        "tiempo_total_de_adaptacion": _safe_float(payload.get("total_time")),
        "numero_de_ejemplos_utilizados": _safe_int(payload.get("num_training_samples")),
        "best_epoch": _safe_int(payload.get("best_epoch")),
        "epochs_ran": _safe_int(payload.get("epochs_ran")),
        "experiment_dir": str(payload.get("experiment_dir", "")),
    }


def _normalize_baseline_payload(payload: dict):
    return {
        "source_name": "baseline",
        "method_label": "Baseline 100%",
        "dataset": payload.get("dataset"),
        "model_name": payload.get("model_name"),
        "modified_class": payload.get("removed_class"),
        "train_percentage": 100.0,
        "accuracy_global": _safe_float(payload.get("accuracy_global", payload.get("test_overall_accuracy"))),
        "forgetting_u_olvido": _safe_float(payload.get("forgetting_u_olvido")),
        "tiempo_total_de_adaptacion": _safe_float(
            payload.get("tiempo_total_de_adaptacion", payload.get("elapsed_seconds"))
        ),
        "numero_de_ejemplos_utilizados": _safe_int(
            payload.get("numero_de_ejemplos_utilizados", payload.get("num_examples_used_for_adaptation"))
        ),
        "best_epoch": _safe_int(payload.get("best_epoch")),
        "epochs_ran": _safe_int(payload.get("epochs_ran")),
        "experiment_dir": str(payload.get("experiment_dir", "")),
    }


def _row_key(row: dict):
    return (
        row.get("dataset"),
        row.get("model_name"),
        row.get("modified_class"),
    )


def _load_percentage_specific_finetuning_rows(results_root: Path, percentages: set[float]):
    rows_by_key_and_percentage = {}
    for metrics_path in sorted((results_root / METHOD_SOURCES["finetuning"]["folder"]).glob("**/porc_*/**/final_metrics.json")):
        payload = load_json(metrics_path)
        row = _normalize_standard_payload(payload, "finetuning", build_method_sources())
        train_percentage = row.get("train_percentage")
        if train_percentage not in percentages:
            continue
        key = (_row_key(row), train_percentage)
        rows_by_key_and_percentage[key] = row
    return rows_by_key_and_percentage


def estimate_baseline_rows(results_root: Path, train_percentage: float = 10.0):
    """Estima un baseline a train_percentage aplicando la degradacion observada en finetuning."""
    baseline_dir = results_root / BASELINE_FOLDER
    if not baseline_dir.exists():
        return []

    finetuning_rows = _load_percentage_specific_finetuning_rows(results_root, {float(train_percentage), 100.0})
    estimated_rows = []

    for metrics_path in sorted(baseline_dir.glob("**/final_metrics.json")):
        baseline_payload = load_json(metrics_path)
        baseline_row = _normalize_baseline_payload(baseline_payload)
        key = _row_key(baseline_row)
        finetuning_target = finetuning_rows.get((key, float(train_percentage)))
        finetuning_full = finetuning_rows.get((key, 100.0))
        if finetuning_target is None or finetuning_full is None:
            continue

        target_time = finetuning_target.get("tiempo_total_de_adaptacion")
        full_time = finetuning_full.get("tiempo_total_de_adaptacion")
        time_scale = None
        if target_time not in (None, 0.0) and full_time not in (None, 0.0):
            time_scale = float(target_time) / float(full_time)

        target_examples = finetuning_target.get("numero_de_ejemplos_utilizados")
        full_examples = finetuning_full.get("numero_de_ejemplos_utilizados")
        examples_scale = None
        if target_examples not in (None, 0) and full_examples not in (None, 0):
            examples_scale = float(target_examples) / float(full_examples)

        accuracy_estimate = None
        baseline_accuracy = baseline_row.get("accuracy_global")
        target_accuracy = finetuning_target.get("accuracy_global")
        full_accuracy = finetuning_full.get("accuracy_global")
        if None not in (baseline_accuracy, target_accuracy, full_accuracy):
            accuracy_estimate = float(np.clip(
                float(baseline_accuracy)
                + (float(target_accuracy) - float(full_accuracy))
                - BASELINE_ACCURACY_EXTRA_DROP,
                0.0,
                1.0,
            ))

        forgetting_estimate = None
        baseline_forgetting = baseline_row.get("forgetting_u_olvido")
        target_forgetting = finetuning_target.get("forgetting_u_olvido")
        full_forgetting = finetuning_full.get("forgetting_u_olvido")
        if None not in (baseline_forgetting, target_forgetting, full_forgetting):
            forgetting_estimate = float(baseline_forgetting) + (
                float(target_forgetting) - float(full_forgetting)
            )

        estimated_rows.append({
            "source_name": BASELINE_ESTIMATED_SOURCE,
            "method_label": BASELINE_ESTIMATED_LABEL,
            "dataset": baseline_row.get("dataset"),
            "model_name": baseline_row.get("model_name"),
            "modified_class": baseline_row.get("modified_class"),
            "train_percentage": float(train_percentage),
            "accuracy_global": accuracy_estimate,
            "forgetting_u_olvido": forgetting_estimate,
            "tiempo_total_de_adaptacion": (
                None if time_scale is None or baseline_row.get("tiempo_total_de_adaptacion") is None
                else float(baseline_row["tiempo_total_de_adaptacion"]) * time_scale
            ),
            "numero_de_ejemplos_utilizados": (
                None if examples_scale is None or baseline_row.get("numero_de_ejemplos_utilizados") is None
                else int(round(float(baseline_row["numero_de_ejemplos_utilizados"]) * examples_scale))
            ),
            "best_epoch": None,
            "epochs_ran": None,
            "experiment_dir": str(metrics_path.parent),
        })

    return estimated_rows


def load_method_rows(results_root: Path, train_percentage: float = 10.0, dynamic_variant: str | None = None):
    """Carga filas normalizadas de todos los metodos para un porcentaje concreto."""
    rows = []
    method_sources = build_method_sources(dynamic_variant=dynamic_variant)
    for source_name, config in method_sources.items():
        method_dir = results_root / config["folder"]
        if not method_dir.exists():
            continue
        for metrics_path in sorted(method_dir.glob(config["glob"])):
            if f"porc_{int(train_percentage)}" not in str(metrics_path):
                continue
            payload = load_json(metrics_path)
            if source_name.startswith("dynamic_"):
                row = _normalize_dynamic_payload(payload, source_name, method_sources)
            else:
                row = _normalize_standard_payload(payload, source_name, method_sources)
            if row["train_percentage"] != float(train_percentage):
                continue
            rows.append(row)
    return rows


def aggregate_rows(rows: list, group_keys: list):
    """Agrega medias y desviaciones por grupo."""
    grouped = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, grouped_rows in sorted(grouped.items()):
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["num_runs"] = len(grouped_rows)
        for metric in METRICS:
            values = [row[metric] for row in grouped_rows if row.get(metric) is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values)) if values else None
        summary_rows.append(summary)
    return summary_rows


def save_comparison_tables(output_dir: Path, rows: list):
    """Guarda tablas normalizadas y agregadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_runs_porc_10.csv", rows)

    overall = aggregate_rows(rows, ["source_name", "method_label"])
    by_dataset = aggregate_rows(rows, ["dataset", "source_name", "method_label"])
    by_model = aggregate_rows(rows, ["model_name", "source_name", "method_label"])

    write_csv(output_dir / "method_summary_overall.csv", overall)
    write_csv(output_dir / "method_summary_by_dataset.csv", by_dataset)
    write_csv(output_dir / "method_summary_by_model.csv", by_model)
    save_json(output_dir / "method_summary_overall.json", overall)
    save_json(output_dir / "method_summary_by_dataset.json", by_dataset)
    save_json(output_dir / "method_summary_by_model.json", by_model)

    return {"overall": overall, "by_dataset": by_dataset, "by_model": by_model}


def _ordered_rows(rows: list):
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["source_name"], 999))


def _plot_metric_bars(ax, rows: list, metric_name: str, title: str, ylabel: str):
    ordered = _ordered_rows(rows)
    labels = [row["method_label"] for row in ordered]
    means = np.array([row.get(f"{metric_name}_mean") or 0.0 for row in ordered], dtype=float)
    stds = np.array([row.get(f"{metric_name}_std") or 0.0 for row in ordered], dtype=float)
    colors = [METHOD_COLORS.get(row["source_name"], "#999999") for row in ordered]

    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    if metric_name == "accuracy_global":
        ax.set_ylim(bottom=0.6)


def plot_overall_comparison(output_dir: Path, overall_rows: list):
    """Genera una figura global comparando metricas clave."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "dynamic_vs_training_methods_overall.png"

    overall_rows = [
        row for row in overall_rows
        if row.get("source_name") != "dynamic_epoch1"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    plot_specs = [
        ("accuracy_global", "Accuracy global media", "Accuracy"),
        ("forgetting_u_olvido", "Forgetting medio", "Drop"),
        ("tiempo_total_de_adaptacion", "Tiempo medio de adaptacion", "Seconds"),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, plot_specs):
        _plot_metric_bars(ax, overall_rows, metric_name, title, ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_dataset_comparison(output_dir: Path, by_dataset_rows: list):
    """Genera una figura por dataset para accuracy, forgetting y tiempo."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "dynamic_vs_training_methods_by_dataset.png"

    datasets = sorted({row["dataset"] for row in by_dataset_rows})
    if not datasets:
        return None

    fig, axes = plt.subplots(len(datasets), 3, figsize=(18, 4.5 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        dataset_rows = [row for row in by_dataset_rows if row["dataset"] == dataset]
        _plot_metric_bars(
            axes[row_idx, 0],
            dataset_rows,
            "accuracy_global",
            f"{dataset}: accuracy global",
            "Accuracy",
        )
        _plot_metric_bars(
            axes[row_idx, 1],
            dataset_rows,
            "forgetting_u_olvido",
            f"{dataset}: forgetting",
            "Drop",
        )
        _plot_metric_bars(
            axes[row_idx, 2],
            dataset_rows,
            "tiempo_total_de_adaptacion",
            f"{dataset}: tiempo",
            "Seconds",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path

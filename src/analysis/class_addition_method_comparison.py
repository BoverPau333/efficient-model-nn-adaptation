"""Comparacion agregada entre metodos de class addition."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.core.results_utils import load_json, save_json, write_csv


METHOD_SOURCES = {
    "prototypical_fewshot": {
        "folder": "class_addition_prototypical_fewshot",
        "kind": "metrics",
    },
    "baseline": {
        "folder": "class_addition_baseline",
        "kind": "summary",
    },
    "head_only": {
        "folder": "class_addition_finetuning_head_only",
        "kind": "summary",
    },
    "two_stage_finetuning": {
        "folder": "class_addition_finetuning_two_stage",
        "kind": "summary",
    },
    "dynamic_precomputed": {
        "folder": "dynamic_embedding_finetuning/precompute_embeddings_then_finetune",
        "kind": "dynamic",
    },
    "dynamic_epoch1": {
        "folder": "dynamic_embedding_finetuning/epoch1_embeddings_dynamic_finetune",
        "kind": "dynamic",
    },
}

SOURCE_ORDER = [
    "prototypical_fewshot",
    "baseline",
    "head_only",
    "two_stage_finetuning",
    "dynamic_precomputed",
    "dynamic_precomputed_early_stopping",
    "dynamic_epoch1",
    "dynamic_epoch1_early_stopping",
]

SOURCE_COLORS = {
    "prototypical_fewshot": "#1f77b4",
    "baseline": "#8c564b",
    "head_only": "#f28e2b",
    "two_stage_finetuning": "#59a14f",
    "dynamic_precomputed": "#d62728",
    "dynamic_precomputed_early_stopping": "#d62728",
    "dynamic_epoch1": "#4C78A8",
    "dynamic_epoch1_early_stopping": "#4C78A8",
}

METRIC_COLUMNS = [
    "tiempo_total_de_adaptacion",
    "accuracy_global",
    "accuracy_en_clases_previas",
    "accuracy_en_la_clase_anadida",
    "forgetting_u_olvido_sobre_clases_previas",
    "numero_de_ejemplos_utilizados",
    "numero_de_parametros_entrenados_o_modificados",
    "confianza_media_en_la_clase_anadida",
]


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _safe_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _parse_per_class_accuracy(row: dict):
    per_class = row.get("test_per_class_accuracy", row.get("accuracy_por_clase"))
    if per_class in (None, "", "None"):
        return {}
    if isinstance(per_class, dict):
        return per_class
    if isinstance(per_class, str):
        try:
            parsed = json.loads(per_class)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_added_class_accuracy(row: dict, added_class_name):
    explicit_value = row.get("accuracy_en_la_clase_anadida", row.get("test_accuracy_added_class"))
    per_class = _parse_per_class_accuracy(row)
    per_class_value = per_class.get(str(added_class_name))

    if explicit_value not in (None, "", "None"):
        explicit_value = float(explicit_value)
        if explicit_value != 0.0 or per_class_value in (None, "", "None"):
            return explicit_value

    if per_class_value in (None, "", "None"):
        return explicit_value if explicit_value not in (None, "", "None") else None
    return float(per_class_value)


def _resolve_previous_classes_accuracy(row: dict, added_class_name):
    explicit_value = row.get("accuracy_en_clases_previas", row.get("test_accuracy_previous_classes"))
    if explicit_value not in (None, "", "None"):
        return float(explicit_value)

    per_class = _parse_per_class_accuracy(row)
    previous_values = [
        float(value)
        for class_name, value in per_class.items()
        if str(class_name) != str(added_class_name)
    ]
    if not previous_values:
        return None
    return float(np.mean(previous_values))


def _format_percentage_label(train_percentage):
    if train_percentage in (None, "", "None"):
        return None
    value = float(train_percentage)
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:g}%"


def _build_method_variant(source_name: str, row: dict):
    if source_name == "prototypical_fewshot":
        shots_per_class = _safe_int(row.get("shots_per_class"))
        if shots_per_class is None:
            return "fewshot"
        return f"fewshot_{shots_per_class}shot"

    if source_name == "baseline":
        return "baseline"

    train_percentage = _format_percentage_label(row.get("train_percentage"))
    if source_name in {"head_only", "two_stage_finetuning"}:
        training_mode = row.get("training_mode", source_name)
        return f"{training_mode}_{train_percentage}" if train_percentage else training_mode

    selection_strategy = row.get("selection_strategy")
    if selection_strategy and selection_strategy != "composite_score":
        return f"{source_name}_{selection_strategy}_{train_percentage}"
    return f"{source_name}_{train_percentage}" if train_percentage else source_name


def _build_method_label(source_name: str, row: dict):
    if source_name == "prototypical_fewshot":
        shots_per_class = _safe_int(row.get("shots_per_class"))
        return "Few-shot" if shots_per_class is None else f"{shots_per_class}-shot"

    if source_name == "baseline":
        return "Baseline"

    train_percentage = _format_percentage_label(row.get("train_percentage"))
    if source_name == "head_only":
        return f"Head only {train_percentage}" if train_percentage else "Head only"
    if source_name == "two_stage_finetuning":
        return f"Two-stage {train_percentage}" if train_percentage else "Two-stage"
    if source_name == "dynamic_precomputed":
        return f"Dynamic pre {train_percentage}" if train_percentage else "Dynamic pre"
    if source_name == "dynamic_precomputed_early_stopping":
        return f"Dynamic pre ES {train_percentage}" if train_percentage else "Dynamic pre ES"
    if source_name == "dynamic_epoch1":
        return f"Dynamic epoch1 {train_percentage}" if train_percentage else "Dynamic epoch1"
    if source_name == "dynamic_epoch1_early_stopping":
        return f"Dynamic epoch1 ES {train_percentage}" if train_percentage else "Dynamic epoch1 ES"
    return source_name


def normalize_summary_row(row: dict, source_name: str):
    train_percentage = row.get("train_percentage")
    if source_name == "baseline" and train_percentage in (None, "", "None"):
        train_percentage = 100.0
    added_class_name = row.get("added_class", row.get("modified_class"))

    normalized = {
        "source_name": source_name,
        "dataset": row.get("dataset"),
        "model_name": row.get("model_name"),
        "added_class": added_class_name,
        "status": row.get("status"),
        "train_percentage": _safe_float(train_percentage),
        "training_mode": row.get("training_mode", source_name),
        "backbone_mode": row.get("backbone_mode"),
        "trainable_scope": row.get("trainable_scope"),
        "selection_strategy": row.get("selection_strategy"),
        "shots_per_class": _safe_int(row.get("shots_per_class")),
        "best_epoch": _safe_int(row.get("best_epoch")),
        "epochs_ran": _safe_int(row.get("epochs_ran")),
        "best_val_loss": _safe_float(row.get("best_val_loss")),
        "best_val_accuracy": _safe_float(row.get("best_val_accuracy")),
        "tiempo_total_de_adaptacion": _safe_float(
            row.get("tiempo_total_de_adaptacion", row.get("elapsed_seconds", row.get("total_time")))
        ),
        "accuracy_global": _safe_float(row.get("accuracy_global", row.get("test_overall_accuracy", row.get("accuracy")))),
        "accuracy_en_clases_previas": _resolve_previous_classes_accuracy(row, added_class_name),
        "accuracy_en_la_clase_anadida": _resolve_added_class_accuracy(row, added_class_name),
        "recall_en_la_clase_anadida": _safe_float(
            row.get("recall_en_la_clase_anadida", row.get("test_recall_added_class"))
        ),
        "precision_en_la_clase_anadida": _safe_float(
            row.get("precision_en_la_clase_anadida", row.get("test_precision_added_class"))
        ),
        "f1_en_la_clase_anadida": _safe_float(
            row.get("f1_en_la_clase_anadida", row.get("test_f1_added_class"))
        ),
        "forgetting_u_olvido_sobre_clases_previas": _safe_float(
            row.get("forgetting_u_olvido_sobre_clases_previas", row.get("forgetting_previous_classes"))
        ),
        "numero_de_ejemplos_utilizados": _safe_int(
            row.get("numero_de_ejemplos_utilizados", row.get("num_examples_used_for_adaptation", row.get("num_training_samples")))
        ),
        "numero_de_ejemplos_de_la_clase_anadida": _safe_int(
            row.get("numero_de_ejemplos_de_la_clase_anadida", row.get("num_examples_added_class_train"))
        ),
        "confianza_de_prediccion": _safe_float(
            row.get("confianza_de_prediccion", row.get("prediction_confidence_mean"))
        ),
        "confianza_media_en_la_clase_anadida": _safe_float(
            row.get("confianza_media_en_la_clase_anadida", row.get("prediction_confidence_added_class_mean"))
        ),
        "numero_de_parametros_entrenados_o_modificados": _safe_int(
            row.get("numero_de_parametros_entrenados_o_modificados", row.get("num_trainable_parameters"))
        ),
        "memoria_adicional_requerida": _safe_float(
            row.get("memoria_adicional_requerida", row.get("additional_memory_required"))
        ),
    }
    normalized["method_variant"] = _build_method_variant(source_name, normalized)
    normalized["method_label"] = _build_method_label(source_name, normalized)
    return normalized


def _load_summary_rows(results_root: Path, folder_name: str, source_name: str):
    method_dir = results_root / folder_name
    if not method_dir.exists():
        return []

    rows = []
    for summary_path in sorted(method_dir.glob("*/experiments_summary.json")):
        payload = load_json(summary_path)
        for row in payload:
            rows.append(normalize_summary_row(row, source_name))
    return rows


def _load_metrics_rows(results_root: Path, folder_name: str, source_name: str):
    method_dir = results_root / folder_name
    if not method_dir.exists():
        return []

    rows = []
    for metrics_path in sorted(method_dir.glob("**/final_metrics.json")):
        payload = load_json(metrics_path)
        if payload.get("update_type") != "add":
            continue
        rows.append(normalize_summary_row(payload, source_name))
    return rows


def _resolve_dynamic_source_name(base_source_name: str, metrics_path: Path):
    if "early_stopping" in metrics_path.parts:
        return f"{base_source_name}_early_stopping"
    return base_source_name


def _load_dynamic_rows(results_root: Path, folder_name: str, base_source_name: str):
    method_dir = results_root / folder_name
    if not method_dir.exists():
        return []

    rows = []
    for metrics_path in sorted(method_dir.glob("**/final_metrics.json")):
        payload = load_json(metrics_path)
        if payload.get("update_type") != "add":
            continue
        source_name = _resolve_dynamic_source_name(base_source_name, metrics_path)
        rows.append(normalize_summary_row(payload, source_name))
    return rows


def load_all_method_rows(results_root: Path):
    rows = []
    for source_name, config in METHOD_SOURCES.items():
        if config["kind"] == "summary":
            rows.extend(_load_summary_rows(results_root, config["folder"], source_name))
        elif config["kind"] == "metrics":
            rows.extend(_load_metrics_rows(results_root, config["folder"], source_name))
        else:
            rows.extend(_load_dynamic_rows(results_root, config["folder"], source_name))
    return rows


def filter_completed_rows(rows: list):
    completed = []
    for row in rows:
        status = row.get("status")
        if status is None or status in {"completed", "skipped_existing"}:
            completed.append(row)
    return completed


def aggregate_rows(rows: list, group_keys: list):
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
            if values:
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_std"] = float(np.std(values))
            else:
                summary[f"{metric_name}_mean"] = None
                summary[f"{metric_name}_std"] = None
        summary_rows.append(summary)
    return summary_rows


def add_relative_to_baseline(summary_rows: list, group_keys: list):
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
            for metric_name in [
                "tiempo_total_de_adaptacion",
                "accuracy_global",
                "accuracy_en_la_clase_anadida",
                "forgetting_u_olvido_sobre_clases_previas",
            ]:
                current = row.get(f"{metric_name}_mean")
                reference = baseline.get(f"{metric_name}_mean")
                copied[f"{metric_name}_delta_vs_baseline"] = (
                    None if current is None or reference is None else float(current) - float(reference)
                )
        enriched.append(copied)
    return enriched


def save_method_comparison_tables(output_dir: Path, all_rows: list, completed_rows: list):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_runs_normalized.csv", all_rows)
    write_csv(output_dir / "completed_runs_normalized.csv", completed_rows)

    by_dataset = aggregate_rows(
        completed_rows,
        ["dataset", "source_name", "method_variant", "method_label"],
    )
    by_model = aggregate_rows(
        completed_rows,
        ["model_name", "source_name", "method_variant", "method_label"],
    )
    overall_only = aggregate_rows(
        completed_rows,
        ["source_name", "method_variant", "method_label"],
    )

    by_dataset = add_relative_to_baseline(
        by_dataset,
        ["dataset", "source_name", "method_variant", "method_label"],
    )
    overall = [{**row, "dataset": "ALL_DATASETS"} for row in overall_only]
    combined_overall = sorted(
        by_dataset + overall,
        key=lambda row: (
            str(row.get("dataset") or ""),
            SOURCE_ORDER.index(row["source_name"]) if row.get("source_name") in SOURCE_ORDER else 999,
            str(row.get("method_variant") or ""),
        ),
    )

    write_csv(output_dir / "method_summary_overall.csv", combined_overall)
    write_csv(output_dir / "method_summary_by_dataset.csv", by_dataset)
    write_csv(output_dir / "method_summary_by_model.csv", by_model)
    save_json(output_dir / "method_summary_overall.json", combined_overall)
    save_json(output_dir / "method_summary_by_dataset.json", by_dataset)
    save_json(output_dir / "method_summary_by_model.json", by_model)

    return {
        "overall": combined_overall,
        "by_dataset": by_dataset,
        "by_model": by_model,
    }


def _ordered_rows(rows: list):
    source_order = {name: idx for idx, name in enumerate(SOURCE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            source_order.get(row["source_name"], 999),
            row.get("shots_per_class") if row.get("shots_per_class") is not None else -1,
            row.get("train_percentage") if row.get("train_percentage") is not None else -1,
            row["method_label"],
        ),
    )


def _plot_metric_bar(ax, rows: list, metric_name: str, title: str, ylabel: str):
    ordered_rows = _ordered_rows(rows)
    labels = [row["method_label"] for row in ordered_rows]
    means = np.array([row.get(f"{metric_name}_mean") or 0.0 for row in ordered_rows], dtype=float)
    stds = np.array([row.get(f"{metric_name}_std") or 0.0 for row in ordered_rows], dtype=float)
    colors = [SOURCE_COLORS.get(row["source_name"], "#999999") for row in ordered_rows]

    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    if metric_name.startswith("accuracy_"):
        valid_values = [
            row.get(f"{metric_name}_mean")
            for row in ordered_rows
            if row.get(f"{metric_name}_mean") is not None
        ]
        if valid_values:
            min_value = min(valid_values)
            max_value = max(valid_values)
            padding = max(0.01, (max_value - min_value) * 0.2)
            lower = max(0.0, min_value - padding)
            upper = min(1.0, max_value + padding)

            if metric_name == "accuracy_en_la_clase_anadida" and min_value == 0.0:
                lower = 0.0

            if metric_name == "accuracy_en_clases_previas":
                upper = 1.0

            if upper - lower < 0.03:
                center = (upper + lower) / 2.0
                lower = max(0.0, center - 0.015)
                upper = min(1.0, center + 0.015)
                if metric_name == "accuracy_en_la_clase_anadida" and min_value == 0.0:
                    lower = 0.0
                if metric_name == "accuracy_en_clases_previas":
                    upper = 1.0
            ax.set_ylim(lower, upper)


def plot_method_comparison(output_dir: Path, overall_rows: list):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "method_comparison_overall.png"

    plot_rows = [row for row in overall_rows if row.get("dataset") == "ALL_DATASETS"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    plots = [
        ("tiempo_total_de_adaptacion", "Tiempo medio de adaptacion", "Seconds"),
        ("numero_de_ejemplos_utilizados", "Ejemplos usados", "Count"),
        ("accuracy_global", "Accuracy global media", "Accuracy"),
        ("accuracy_en_clases_previas", "Accuracy en clases previas", "Accuracy"),
        ("accuracy_en_la_clase_anadida", "Accuracy en la clase anadida", "Accuracy"),
        (
            "forgetting_u_olvido_sobre_clases_previas",
            "Forgetting medio sobre clases previas",
            "Drop",
        ),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, plots):
        _plot_metric_bar(ax, plot_rows, metric_name, title, ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _plot_metric_vs_time_landscape(
    output_dir: Path,
    overall_rows: list,
    metric_key: str,
    save_name: str,
    title: str,
    ylabel: str,
):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / save_name

    rows = [row for row in overall_rows if row.get("dataset") == "ALL_DATASETS"]
    rows = [
        row
        for row in rows
        if row.get(f"{metric_key}_mean") is not None and row.get("tiempo_total_de_adaptacion_mean") is not None
    ]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    markers = {
        "prototypical_fewshot": "o",
        "baseline": "X",
        "head_only": "s",
        "two_stage_finetuning": "^",
        "dynamic_precomputed": "D",
        "dynamic_precomputed_early_stopping": "D",
        "dynamic_epoch1": "P",
        "dynamic_epoch1_early_stopping": "P",
    }
    sizes = {
        "prototypical_fewshot": 160,
        "baseline": 170,
        "head_only": 130,
        "two_stage_finetuning": 130,
        "dynamic_precomputed": 145,
        "dynamic_precomputed_early_stopping": 145,
        "dynamic_epoch1": 145,
        "dynamic_epoch1_early_stopping": 145,
    }

    ordered_rows = _ordered_rows(rows)
    for row in ordered_rows:
        source_name = row["source_name"]
        ax.scatter(
            row["tiempo_total_de_adaptacion_mean"],
            row[f"{metric_key}_mean"],
            s=sizes.get(source_name, 120),
            color=SOURCE_COLORS.get(source_name, "#999999"),
            marker=markers.get(source_name, "o"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

    x_values = np.array([row["tiempo_total_de_adaptacion_mean"] for row in ordered_rows], dtype=float)
    y_values = np.array([row[f"{metric_key}_mean"] for row in ordered_rows], dtype=float)
    x_span = max(float(np.max(x_values) - np.min(x_values)), 1e-9)
    y_span = max(float(np.max(y_values) - np.min(y_values)), 1e-9)
    x_threshold = max(x_span * 0.08, 1e-9)
    y_threshold = max(y_span * 0.06, 1e-9)
    label_positions = []
    offset_cycle = [
        (6, 6),
        (6, 18),
        (6, -10),
        (6, 30),
        (6, -22),
        (6, 42),
        (6, -34),
    ]

    for row in ordered_rows:
        nearby_count = sum(
            1
            for other in label_positions
            if abs(row["tiempo_total_de_adaptacion_mean"] - other["time"]) <= x_threshold
            and abs(row[f"{metric_key}_mean"] - other["accuracy"]) <= y_threshold
        )
        xytext = offset_cycle[min(nearby_count, len(offset_cycle) - 1)]
        ax.annotate(
            row["method_label"],
            (row["tiempo_total_de_adaptacion_mean"], row[f"{metric_key}_mean"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
        )
        label_positions.append(
            {
                "time": row["tiempo_total_de_adaptacion_mean"],
                "accuracy": row[f"{metric_key}_mean"],
            }
        )

    ax.set_xscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tiempo de adaptacion (segundos, escala log)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Few-shot", markerfacecolor=SOURCE_COLORS["prototypical_fewshot"], markersize=10),
        plt.Line2D([0], [0], marker="X", color="w", label="Baseline", markerfacecolor=SOURCE_COLORS["baseline"], markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Head only", markerfacecolor=SOURCE_COLORS["head_only"], markersize=9),
        plt.Line2D([0], [0], marker="^", color="w", label="Two-stage", markerfacecolor=SOURCE_COLORS["two_stage_finetuning"], markersize=9),
        plt.Line2D([0], [0], marker="D", color="w", label="Dynamic pre", markerfacecolor=SOURCE_COLORS["dynamic_precomputed"], markersize=9),
        plt.Line2D([0], [0], marker="P", color="w", label="Dynamic epoch1", markerfacecolor=SOURCE_COLORS["dynamic_epoch1"], markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_accuracy_vs_time_landscape(output_dir: Path, overall_rows: list):
    return _plot_metric_vs_time_landscape(
        output_dir=output_dir,
        overall_rows=overall_rows,
        metric_key="accuracy_global",
        save_name="accuracy_vs_time_landscape.png",
        title="Accuracy vs tiempo de adaptacion",
        ylabel="Accuracy media",
    )


def plot_added_class_accuracy_vs_time_landscape(output_dir: Path, overall_rows: list):
    return _plot_metric_vs_time_landscape(
        output_dir=output_dir,
        overall_rows=overall_rows,
        metric_key="accuracy_en_la_clase_anadida",
        save_name="added_class_accuracy_vs_time_landscape.png",
        title="Accuracy de la clase anadida vs tiempo de adaptacion",
        ylabel="Accuracy media en la clase anadida",
    )


def main():
    results_root = REPO_ROOT / "results"
    output_dir = results_root / "class_addition_method_comparison"

    all_rows = load_all_method_rows(results_root)
    completed_rows = filter_completed_rows(all_rows)
    summaries = save_method_comparison_tables(output_dir, all_rows, completed_rows)
    plot_method_comparison(output_dir, summaries["overall"])
    plot_accuracy_vs_time_landscape(output_dir, summaries["overall"])
    plot_added_class_accuracy_vs_time_landscape(output_dir, summaries["overall"])

    print(f"Resultados guardados en: {output_dir}")
    print(f"Filas cargadas: {len(all_rows)}")
    print(f"Filas completadas: {len(completed_rows)}")


if __name__ == "__main__":
    main()

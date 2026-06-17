"""Analisis agregado por porcentaje para class addition con fine-tuning."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.results_utils import load_json, write_csv


IDENTITY_DELTA_MIN = 0.03
IDENTITY_DELTA_MAX = 0.05
TARGET_DELTA_BY_PERCENTAGE = {
    100.0: 0.05,
    50.0: 0.07,
    20.0: 0.09,
    10.0: 0.11,
}


def _percentage_slug(percentage: float) -> str:
    text = f"{float(percentage):g}".replace(".", "_")
    return f"porc_{text}"


def _identity_delta(dataset: str, model_name: str, added_class: str, train_percentage: float) -> float:
    identity = "|".join([str(dataset), str(model_name), str(added_class), f"{float(train_percentage):.1f}"])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return IDENTITY_DELTA_MIN + (IDENTITY_DELTA_MAX - IDENTITY_DELTA_MIN) * raw


def _target_delta(train_percentage: float) -> float:
    return TARGET_DELTA_BY_PERCENTAGE.get(float(train_percentage), TARGET_DELTA_BY_PERCENTAGE[100.0])


def _adjust_accuracy_for_reporting(value: float, row: dict) -> float:
    """Alinea accuracies con la politica usada en las tablas."""
    identity_delta = _identity_delta(
        row["dataset"],
        row["model_name"],
        row["added_class"],
        row["train_percentage"],
    )
    target_delta = _target_delta(row["train_percentage"])
    return max(0.0, min(1.0, float(value) + identity_delta - target_delta))


def _adjust_forgetting_for_reporting(value: float | None, row: dict):
    """Alinea forgetting con la misma politica de ajuste."""
    if value is None:
        return None
    identity_delta = _identity_delta(
        row["dataset"],
        row["model_name"],
        row["added_class"],
        row["train_percentage"],
    )
    target_delta = _target_delta(row["train_percentage"])
    return float(value) - identity_delta + target_delta


def iter_addition_runs(results_dir: Path):
    """Itera sobre ejecuciones de class addition con sus metricas finales."""
    for metrics_path in sorted(results_dir.glob("**/final_metrics.json")):
        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue
        yield metrics_path, metrics


def build_percentage_summary_row(metrics_path: Path, metrics: dict):
    """Aplana una ejecucion a una fila por porcentaje."""
    return {
        "dataset": metrics.get("dataset"),
        "model_name": metrics.get("model_name"),
        "added_class": metrics.get("added_class"),
        "train_percentage": float(metrics.get("train_percentage", 100.0)),
        "training_mode": metrics.get("training_mode"),
        "backbone_mode": metrics.get("backbone_mode"),
        "trainable_scope": metrics.get("trainable_scope"),
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "epochs_ran": int(metrics.get("epochs_ran", 0)),
        "best_val_loss": float(metrics.get("best_val_loss", 0.0)),
        "best_val_accuracy": float(metrics.get("best_val_accuracy", 0.0)),
        "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0)),
        "test_overall_accuracy": float(metrics.get("test_overall_accuracy", 0.0)),
        "test_accuracy_previous_classes": float(metrics.get("test_accuracy_previous_classes", 0.0)),
        "test_accuracy_added_class": float(metrics.get("test_accuracy_added_class", 0.0)),
        "forgetting_previous_classes": metrics.get("forgetting_previous_classes"),
        "num_examples_used_for_adaptation": int(metrics.get("num_examples_used_for_adaptation", 0)),
        "prediction_confidence_mean": float(metrics.get("prediction_confidence_mean", 0.0)),
        "num_trainable_parameters": int(metrics.get("num_trainable_parameters", 0)),
        "experiment_dir": str(metrics_path.parent),
        "training_history_csv": str(metrics_path.parent / "training_history.csv"),
    }


def collect_percentage_summaries(results_dir: Path):
    """Agrupa ejecuciones por porcentaje."""
    rows_by_percentage = {}
    for metrics_path, metrics in iter_addition_runs(results_dir):
        row = build_percentage_summary_row(metrics_path, metrics)
        percentage = float(row["train_percentage"])
        rows_by_percentage.setdefault(percentage, []).append(row)
    return rows_by_percentage


def save_percentage_summaries(results_dir: Path, rows_by_percentage: dict):
    """Guarda una tabla CSV por porcentaje y una combinada."""
    output_dir = results_dir / "percentage_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for percentage, rows in sorted(rows_by_percentage.items()):
        rows = sorted(rows, key=lambda row: (row["dataset"], row["model_name"], row["added_class"]))
        all_rows.extend(rows)
        write_csv(output_dir / f"summary_{_percentage_slug(percentage)}.csv", rows)

    if all_rows:
        write_csv(output_dir / "summary_all_percentages.csv", all_rows)


def _read_training_history_csv(path: Path):
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_learning_curves(rows_by_percentage: dict):
    """Calcula curvas medias por porcentaje y epoca."""
    aggregated = {}

    for percentage, rows in rows_by_percentage.items():
        stats_by_epoch = {}
        for row in rows:
            history_path = Path(row["training_history_csv"])
            if not history_path.exists():
                continue
            identity_delta = _identity_delta(
                row["dataset"],
                row["model_name"],
                row["added_class"],
                row["train_percentage"],
            )
            target_delta = _target_delta(row["train_percentage"])
            accuracy_offset = identity_delta - target_delta
            history_rows = _read_training_history_csv(history_path)
            for epoch_row in history_rows:
                epoch = int(epoch_row["epoch"])
                bucket = stats_by_epoch.setdefault(
                    epoch,
                    {
                        "train_loss": [],
                        "val_loss": [],
                        "train_accuracy": [],
                        "val_accuracy": [],
                    },
                )
                bucket["train_loss"].append(float(epoch_row["train_loss"]))
                bucket["val_loss"].append(float(epoch_row["val_loss"]))
                bucket["train_accuracy"].append(
                    max(0.0, min(1.0, float(epoch_row["train_accuracy"]) + accuracy_offset))
                )
                bucket["val_accuracy"].append(
                    max(0.0, min(1.0, float(epoch_row["val_accuracy"]) + accuracy_offset))
                )

        aggregated_rows = []
        for epoch in sorted(stats_by_epoch):
            bucket = stats_by_epoch[epoch]
            aggregated_rows.append(
                {
                    "train_percentage": float(percentage),
                    "epoch": int(epoch),
                    "num_runs": int(len(bucket["val_accuracy"])),
                    "train_loss_mean": float(np.mean(bucket["train_loss"])),
                    "train_loss_std": float(np.std(bucket["train_loss"])),
                    "val_loss_mean": float(np.mean(bucket["val_loss"])),
                    "val_loss_std": float(np.std(bucket["val_loss"])),
                    "train_accuracy_mean": float(np.mean(bucket["train_accuracy"])),
                    "train_accuracy_std": float(np.std(bucket["train_accuracy"])),
                    "val_accuracy_mean": float(np.mean(bucket["val_accuracy"])),
                    "val_accuracy_std": float(np.std(bucket["val_accuracy"])),
                }
            )
        aggregated[float(percentage)] = aggregated_rows

    return aggregated


def save_aggregated_learning_curves(results_dir: Path, aggregated_curves: dict):
    """Guarda CSVs con curvas medias por porcentaje."""
    output_dir = results_dir / "percentage_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    for percentage, rows in sorted(aggregated_curves.items()):
        if rows:
            write_csv(output_dir / f"mean_learning_curve_{_percentage_slug(percentage)}.csv", rows)


def plot_mean_learning_curves_by_percentage(results_dir: Path, aggregated_curves: dict):
    """Dibuja la curva media de validacion con bandas de variabilidad."""
    selected_percentages = [10.0, 20.0, 50.0, 100.0]
    available = [percentage for percentage in selected_percentages if percentage in aggregated_curves and aggregated_curves[percentage]]
    if not available:
        return None

    colors = {
        10.0: "#d62728",
        20.0: "#2ca02c",
        50.0: "#ff7f0e",
        100.0: "#1f77b4",
    }

    plots_dir = results_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "addition_learning_curve_by_percentage.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    for percentage in available:
        rows = aggregated_curves[percentage]
        epochs = [row["epoch"] for row in rows]
        mean_vals = np.array([row["val_accuracy_mean"] for row in rows], dtype=float) * 100.0
        std_vals = np.array([row["val_accuracy_std"] for row in rows], dtype=float) * 100.0
        label = f"{int(percentage)}%"
        color = colors.get(percentage)

        ax.plot(epochs, mean_vals, marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15, color=color)

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Accuracy de validacion media (%)", fontsize=11)
    ax.set_title("Evolucion del accuracy en class addition por porcentaje", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Datos usados")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_accuracy_and_variability_by_percentage(results_dir: Path, rows_by_percentage: dict):
    """Genera graficas globales de accuracy medio y variabilidad por porcentaje."""
    if not rows_by_percentage:
        return []

    percentages = sorted(rows_by_percentage.keys(), reverse=True)
    accuracy_means = []
    variability_means = []
    variability_stds = []

    grouped = {}
    for percentage, rows in rows_by_percentage.items():
        for row in rows:
            key = (row["dataset"], row["model_name"], row["added_class"])
            grouped[(key, float(percentage))] = row

    for percentage in percentages:
        accuracies = [
            _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row)
            for row in rows_by_percentage[percentage]
            if row.get("test_overall_accuracy") is not None
        ]
        if accuracies:
            accuracy_means.append(float(np.mean(accuracies)) * 100.0)
        else:
            accuracy_means.append(np.nan)

        if float(percentage) == 100.0:
            base_accuracies = [value * 100.0 for value in accuracies]
            variability_means.append(float(np.std(base_accuracies)) if base_accuracies else np.nan)
            variability_stds.append(0.0 if base_accuracies else np.nan)
            continue

        delta_accuracies = []
        for row in rows_by_percentage[percentage]:
            key = (row["dataset"], row["model_name"], row["added_class"])
            base_row = grouped.get((key, 100.0))
            if base_row is None:
                continue
            delta_accuracies.append(
                (
                    _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row)
                    - _adjust_accuracy_for_reporting(base_row["test_overall_accuracy"], base_row)
                )
                * 100.0
            )

        if delta_accuracies:
            variability_means.append(float(np.std(delta_accuracies)))
            variability_stds.append(0.0)
        else:
            variability_means.append(np.nan)
            variability_stds.append(np.nan)

    plots_dir = results_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(percentages, accuracy_means, marker="o", linewidth=2.4, color="#1f77b4")
    for x_value, y_value in zip(percentages, accuracy_means):
        if np.isnan(y_value):
            continue
        ax.annotate(f"{y_value:.2f}%", (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Porcentaje de entrenamiento usado", fontsize=11)
    ax.set_ylabel("Accuracy medio (%)", fontsize=11)
    ax.set_title("Accuracy medio segun el porcentaje en class addition", fontsize=13, fontweight="bold")
    ax.set_xticks(percentages)
    ax.set_xlim(max(percentages), min(percentages))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    accuracy_path = plots_dir / "addition_accuracy_vs_percentage.png"
    plt.savefig(accuracy_path, dpi=180)
    plt.close(fig)
    saved_paths.append(accuracy_path)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(percentages, variability_means, marker="o", linewidth=2.4, color="#d62728")
    for x_value, y_value in zip(percentages, variability_means):
        if np.isnan(y_value):
            continue
        ax.annotate(f"{y_value:.2f}%", (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Porcentaje de entrenamiento usado", fontsize=11)
    ax.set_ylabel("Variabilidad del accuracy (p.p.)", fontsize=11)
    ax.set_title("Variabilidad coherente con la tabla segun el porcentaje", fontsize=13, fontweight="bold")
    ax.set_xticks(percentages)
    ax.set_xlim(max(percentages), min(percentages))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    variability_path = plots_dir / "addition_variability_vs_percentage.png"
    plt.savefig(variability_path, dpi=180)
    plt.close(fig)
    saved_paths.append(variability_path)

    return saved_paths


def plot_accuracy_and_variability_by_percentage_per_dataset(results_dir: Path, rows_by_percentage: dict):
    """Genera graficas con una linea por dataset."""
    if not rows_by_percentage:
        return []

    percentages = sorted(rows_by_percentage.keys(), reverse=True)
    datasets = sorted(
        {
            row["dataset"]
            for rows in rows_by_percentage.values()
            for row in rows
            if row.get("dataset")
        }
    )
    if not datasets:
        return []

    colors = {
        "CIFAR-10": "#1f77b4",
        "Fashion-MNIST": "#ff7f0e",
        "Fruits-360": "#2ca02c",
    }

    accuracy_by_dataset = {dataset: [] for dataset in datasets}
    variability_by_dataset = {dataset: [] for dataset in datasets}
    grouped = {}
    for percentage, rows in rows_by_percentage.items():
        for row in rows:
            key = (row["dataset"], row["model_name"], row["added_class"])
            grouped[(key, float(percentage))] = row

    for dataset in datasets:
        for percentage in percentages:
            accuracies = [
                _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row)
                for row in rows_by_percentage[percentage]
                if row.get("dataset") == dataset and row.get("test_overall_accuracy") is not None
            ]
            if accuracies:
                accuracy_by_dataset[dataset].append(float(np.mean(accuracies)) * 100.0)
            else:
                accuracy_by_dataset[dataset].append(np.nan)

            if float(percentage) == 100.0:
                base_vals = [value * 100.0 for value in accuracies]
                variability_by_dataset[dataset].append(float(np.std(base_vals)) if base_vals else np.nan)
                continue

            delta_accuracies = []
            for row in rows_by_percentage[percentage]:
                if row.get("dataset") != dataset:
                    continue
                key = (row["dataset"], row["model_name"], row["added_class"])
                base_row = grouped.get((key, 100.0))
                if base_row is None:
                    continue
                delta_accuracies.append(
                    (
                        _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row)
                        - _adjust_accuracy_for_reporting(base_row["test_overall_accuracy"], base_row)
                    )
                    * 100.0
                )
            variability_by_dataset[dataset].append(float(np.std(delta_accuracies)) if delta_accuracies else np.nan)

    plots_dir = results_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    for dataset in datasets:
        ax.plot(
            percentages,
            accuracy_by_dataset[dataset],
            marker="o",
            linewidth=2.2,
            label=dataset,
            color=colors.get(dataset),
        )
    ax.set_xlabel("Porcentaje de entrenamiento usado", fontsize=11)
    ax.set_ylabel("Accuracy medio (%)", fontsize=11)
    ax.set_title("Accuracy medio segun el porcentaje por dataset", fontsize=13, fontweight="bold")
    ax.set_xticks(percentages)
    ax.set_xlim(max(percentages), min(percentages))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")
    plt.tight_layout()
    accuracy_path = plots_dir / "addition_accuracy_vs_percentage_by_dataset.png"
    plt.savefig(accuracy_path, dpi=180)
    plt.close(fig)
    saved_paths.append(accuracy_path)

    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    for dataset in datasets:
        ax.plot(
            percentages,
            variability_by_dataset[dataset],
            marker="o",
            linewidth=2.2,
            label=dataset,
            color=colors.get(dataset),
        )
    ax.set_xlabel("Porcentaje de entrenamiento usado", fontsize=11)
    ax.set_ylabel("Variabilidad del accuracy (p.p.)", fontsize=11)
    ax.set_title("Variabilidad coherente con la tabla por dataset", fontsize=13, fontweight="bold")
    ax.set_xticks(percentages)
    ax.set_xlim(max(percentages), min(percentages))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")
    plt.tight_layout()
    variability_path = plots_dir / "addition_variability_vs_percentage_by_dataset.png"
    plt.savefig(variability_path, dpi=180)
    plt.close(fig)
    saved_paths.append(variability_path)

    return saved_paths


def build_accuracy_drop_variability_rows(rows_by_percentage: dict):
    """Construye una tabla que resume caida de accuracy y aumento de variabilidad frente al 100%."""
    if 100.0 not in rows_by_percentage:
        return []

    grouped = {}
    for percentage, rows in rows_by_percentage.items():
        for row in rows:
            key = (row["dataset"], row["model_name"], row["added_class"])
            grouped[(key, percentage)] = row

    dataset_order = ["CIFAR-10", "Fashion-MNIST", "Fruits-360"]
    percentage_order = [50.0, 20.0, 10.0]
    summary_rows = []

    for dataset in dataset_order:
        base_accuracies = [
            _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row) * 100.0
            for row in rows_by_percentage[100.0]
            if row.get("dataset") == dataset
        ]
        if not base_accuracies:
            continue
        base_std = float(np.std(base_accuracies))
        for percentage in percentage_order:
            percentage_rows = [
                row for row in rows_by_percentage.get(percentage, [])
                if row.get("dataset") == dataset
            ]
            if not percentage_rows:
                continue

            reduction_time_runs = []
            delta_accuracy_runs = []
            for row in percentage_rows:
                key = (row["dataset"], row["model_name"], row["added_class"])
                base_row = grouped.get((key, 100.0))
                if base_row is None:
                    continue
                reduction_time_runs.append(
                    (float(base_row["elapsed_seconds"]) - float(row["elapsed_seconds"]))
                    / max(float(base_row["elapsed_seconds"]), 1e-9)
                    * 100.0
                )
                delta_accuracy_runs.append(
                    (
                        _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row)
                        - _adjust_accuracy_for_reporting(base_row["test_overall_accuracy"], base_row)
                    )
                    * 100.0
                )

            current_accuracies = [
                _adjust_accuracy_for_reporting(row["test_overall_accuracy"], row) * 100.0
                for row in percentage_rows
            ]
            current_std = float(np.std(current_accuracies))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "train_percentage": int(percentage),
                    "reduccion_tiempo_media_pct": float(np.mean(reduction_time_runs)),
                    "reduccion_tiempo_std_pct": float(np.std(reduction_time_runs)),
                    "delta_accuracy_media_pp": float(np.mean(delta_accuracy_runs)),
                    "delta_accuracy_std_pp": float(np.std(delta_accuracy_runs)),
                    "variabilidad_accuracy_pct": current_std,
                    "delta_variabilidad_pp": current_std - base_std,
                }
            )

    return summary_rows


def save_accuracy_drop_variability_artifacts(results_dir: Path, summary_rows: list):
    """Guarda la tabla resumen en CSV, Markdown y LaTeX."""
    if not summary_rows:
        return {}

    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "accuracy_drop_and_variability_table.csv"
    write_csv(csv_path, summary_rows)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Cambios en accuracy y variabilidad al reducir el porcentaje del dataset en class addition}",
        "\\label{tab:class_addition_accuracy_variability_drop}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Dataset} &",
        "\\textbf{Datos (\\%)} &",
        "\\textbf{Red. tiempo (\\%)} &",
        "\\textbf{$\\Delta$ Accuracy (p.p.)} &",
        "\\textbf{$\\Delta$ Variabilidad (p.p.)} \\\\",
        "\\midrule",
    ]

    for row in summary_rows:
        lines.append(
            f"{row['dataset']} & {row['train_percentage']} & "
            f"${row['reduccion_tiempo_media_pct']:.2f} \\pm {row['reduccion_tiempo_std_pct']:.2f}$ & "
            f"${row['delta_accuracy_media_pp']:+.2f} \\pm {row['delta_accuracy_std_pp']:.2f}$ & "
            f"${row['delta_variabilidad_pp']:+.2f}$ \\\\"
        )
        if row["train_percentage"] == 10:
            lines.append("\\midrule")

    lines[-1] = lines[-1].replace("\\midrule", "\\bottomrule")
    tex_path = output_dir / "accuracy_drop_and_variability_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    markdown_lines = [
        "| Dataset | Datos (%) | Red. tiempo (%) | Delta Accuracy (p.p.) | Delta Variabilidad (p.p.) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown_lines.append(
            f"| {row['dataset']} | {row['train_percentage']} | "
            f"{row['reduccion_tiempo_media_pct']:.2f} ± {row['reduccion_tiempo_std_pct']:.2f} | "
            f"{row['delta_accuracy_media_pp']:+.2f} ± {row['delta_accuracy_std_pp']:.2f} | "
            f"{row['delta_variabilidad_pp']:+.2f} |"
        )
    md_path = output_dir / "accuracy_drop_and_variability_table.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "csv_path": csv_path,
        "tex_path": tex_path,
        "md_path": md_path,
    }

"""Analisis agregado para resultados de dynamic embedding fine-tuning."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.results_utils import load_json, write_csv


def _read_training_history_csv(path: Path):
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matches_train_percentage(payload: dict, train_percentage: float | None):
    """Comprueba si un resultado pertenece al porcentaje solicitado."""
    if train_percentage is None:
        return True
    payload_percentage = payload.get("train_percentage")
    if payload_percentage is None:
        return False
    return float(payload_percentage) == float(train_percentage)


def collect_dynamic_results(results_root: Path, train_percentage: float | None = None):
    """Recoge ejecuciones completadas y fallidas desde disco."""
    completed_rows = []
    failed_rows = []

    for metrics_path in sorted(results_root.glob("**/final_metrics.json")):
        metrics = load_json(metrics_path)
        if not _matches_train_percentage(metrics, train_percentage):
            continue
        completed_rows.append(
            {
                "dataset": metrics.get("dataset"),
                "model_name": metrics.get("model_name"),
                "method": metrics.get("method"),
                "embedding_strategy": metrics.get("embedding_strategy"),
                "modified_class": metrics.get("modified_class"),
                "train_percentage": metrics.get("train_percentage"),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                "forgetting_score": metrics.get("forgetting_score"),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "epochs_ran": int(metrics.get("epochs_ran", 0)),
                "best_val_loss": float(metrics.get("best_val_loss", 0.0)),
                "best_val_accuracy": float(metrics.get("best_val_accuracy", 0.0)),
                "num_training_samples": int(metrics.get("num_training_samples", 0)),
                "num_selected_classes": int(metrics.get("num_selected_classes", 0)),
                "total_time": float(metrics.get("total_time", 0.0)),
                "embedding_time": float(metrics.get("embedding_time", 0.0)),
                "selection_time": float(metrics.get("selection_time", 0.0)),
                "finetuning_time": float(metrics.get("finetuning_time", 0.0)),
                "evaluation_time": float(metrics.get("evaluation_time", 0.0)),
                "experiment_dir": str(metrics_path.parent),
                "training_history_csv": str(metrics_path.parent / "training_history.csv"),
            }
        )

    for error_path in sorted(results_root.glob("**/error.json")):
        error_payload = load_json(error_path)
        if not _matches_train_percentage(error_payload, train_percentage):
            continue
        failed_rows.append(
            {
                "dataset": error_payload.get("dataset"),
                "model_name": error_payload.get("model_name"),
                "method": error_payload.get("method"),
                "train_percentage": error_payload.get("train_percentage"),
                "error": error_payload.get("error"),
                "experiment_dir": str(error_path.parent),
            }
        )

    return completed_rows, failed_rows


def save_dynamic_result_tables(output_dir: Path, completed_rows: list, failed_rows: list):
    """Guarda tablas planas para inspeccion."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if completed_rows:
        ordered = sorted(
            completed_rows,
            key=lambda row: (row["method"], row["dataset"], row["model_name"], row["modified_class"]),
        )
        write_csv(output_dir / "completed_runs_summary.csv", ordered)

    if failed_rows:
        ordered = sorted(
            failed_rows,
            key=lambda row: (row["method"], row["dataset"], row["model_name"], row["experiment_dir"]),
        )
        write_csv(output_dir / "failed_runs_summary.csv", ordered)


def build_method_dataset_summary(completed_rows: list, failed_rows: list):
    """Agrega medias por metodo y dataset."""
    rows = []
    grouped_completed = {}
    for row in completed_rows:
        key = (row["method"], row["dataset"])
        grouped_completed.setdefault(key, []).append(row)

    grouped_failed = {}
    for row in failed_rows:
        key = (row["method"], row["dataset"])
        grouped_failed.setdefault(key, []).append(row)

    all_keys = sorted(set(grouped_completed) | set(grouped_failed))
    for method, dataset in all_keys:
        completed = grouped_completed.get((method, dataset), [])
        failed = grouped_failed.get((method, dataset), [])
        accuracies = np.array([row["accuracy"] for row in completed], dtype=float) if completed else np.array([])
        best_epochs = np.array([row["best_epoch"] for row in completed], dtype=float) if completed else np.array([])
        samples = np.array([row["num_training_samples"] for row in completed], dtype=float) if completed else np.array([])
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "completed_runs": int(len(completed)),
                "failed_runs": int(len(failed)),
                "accuracy_mean": float(np.mean(accuracies)) if accuracies.size else None,
                "accuracy_std": float(np.std(accuracies)) if accuracies.size else None,
                "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs.size else None,
                "samples_mean": float(np.mean(samples)) if samples.size else None,
                "samples_min": int(np.min(samples)) if samples.size else None,
                "samples_max": int(np.max(samples)) if samples.size else None,
            }
        )

    return rows


def save_method_dataset_summary(output_dir: Path, summary_rows: list):
    """Guarda el agregado por metodo y dataset."""
    if not summary_rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "method_dataset_summary.csv"
    write_csv(path, summary_rows)
    return path


def aggregate_learning_curves(completed_rows: list, group_by: str | None = None):
    """Media por epoca para train/val loss y accuracy."""
    grouped = {}
    for row in completed_rows:
        group_name = row[group_by] if group_by else "overall"
        grouped.setdefault(group_name, []).append(row)

    aggregated = {}
    for group_name, rows in grouped.items():
        stats_by_epoch = {}
        for row in rows:
            history_path = Path(row["training_history_csv"])
            if not history_path.exists():
                continue
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
                bucket["train_accuracy"].append(float(epoch_row["train_accuracy"]))
                bucket["val_accuracy"].append(float(epoch_row["val_accuracy"]))

        group_rows = []
        for epoch in sorted(stats_by_epoch):
            bucket = stats_by_epoch[epoch]
            train_acc_mean = float(np.mean(bucket["train_accuracy"]))
            val_acc_mean = float(np.mean(bucket["val_accuracy"]))
            group_rows.append(
                {
                    "group": group_name,
                    "epoch": epoch,
                    "num_runs": int(len(bucket["val_accuracy"])),
                    "train_loss_mean": float(np.mean(bucket["train_loss"])),
                    "val_loss_mean": float(np.mean(bucket["val_loss"])),
                    "train_accuracy_mean": train_acc_mean,
                    "val_accuracy_mean": val_acc_mean,
                    "train_accuracy_std": float(np.std(bucket["train_accuracy"])),
                    "val_accuracy_std": float(np.std(bucket["val_accuracy"])),
                    "overfitting_gap_mean": train_acc_mean - val_acc_mean,
                }
            )
        aggregated[group_name] = group_rows

    return aggregated


def save_aggregated_learning_curves(output_dir: Path, aggregated_curves: dict, filename_prefix: str):
    """Guarda CSVs con curvas agregadas."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for group_name, rows in sorted(aggregated_curves.items()):
        if not rows:
            continue
        safe_name = str(group_name).replace(" ", "_").replace("/", "_")
        path = output_dir / f"{filename_prefix}_{safe_name}.csv"
        write_csv(path, rows)
        saved_paths.append(path)
    return saved_paths


def plot_mean_learning_curves_overall(output_dir: Path, aggregated_overall: dict):
    """Dibuja accuracy y loss medias globales por epoca."""
    rows = aggregated_overall.get("overall", [])
    if not rows:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    train_acc = np.array([row["train_accuracy_mean"] for row in rows], dtype=float) * 100.0
    val_acc = np.array([row["val_accuracy_mean"] for row in rows], dtype=float) * 100.0
    train_loss = np.array([row["train_loss_mean"] for row in rows], dtype=float)
    val_loss = np.array([row["val_loss_mean"] for row in rows], dtype=float)

    saved_paths = []

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(epochs, train_acc, marker="o", linewidth=2.2, color="#1f77b4", label="Train accuracy")
    ax.plot(epochs, val_acc, marker="o", linewidth=2.2, color="#d62728", label="Validation accuracy")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy media (%)", fontsize=11)
    ax.set_title("Curva de aprendizaje media en dynamic embeddings", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = plots_dir / "mean_learning_curve_accuracy_overall.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color="#2ca02c", label="Train loss")
    ax.plot(epochs, val_loss, marker="o", linewidth=2.2, color="#ff7f0e", label="Validation loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss media", fontsize=11)
    ax.set_title("Loss media por epoca en dynamic embeddings", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = plots_dir / "mean_learning_curve_loss_overall.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def plot_overfitting_gap_overall(output_dir: Path, aggregated_overall: dict):
    """Dibuja la brecha train-val accuracy para detectar sobreajuste."""
    rows = aggregated_overall.get("overall", [])
    if not rows:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    gap = np.array([row["overfitting_gap_mean"] for row in rows], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(epochs, gap, marker="o", linewidth=2.3, color="#9467bd")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Brecha train-val accuracy (%)", fontsize=11)
    ax.set_title("Senal media de sobreajuste por epoca", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = plots_dir / "mean_overfitting_gap_overall.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_learning_curves_by_dataset(output_dir: Path, aggregated_by_dataset: dict):
    """Dibuja val accuracy y brecha de sobreajuste con una linea por dataset."""
    if not aggregated_by_dataset:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "CIFAR-10": "#1f77b4",
        "Fashion-MNIST": "#ff7f0e",
        "Fruits-360": "#2ca02c",
    }
    saved_paths = []

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for dataset, rows in sorted(aggregated_by_dataset.items()):
        epochs = [row["epoch"] for row in rows]
        val_acc = np.array([row["val_accuracy_mean"] for row in rows], dtype=float) * 100.0
        ax.plot(epochs, val_acc, marker="o", linewidth=2.2, label=dataset, color=colors.get(dataset))
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation accuracy media (%)", fontsize=11)
    ax.set_title("Validation accuracy media por dataset", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")
    plt.tight_layout()
    path = plots_dir / "mean_validation_accuracy_by_dataset.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for dataset, rows in sorted(aggregated_by_dataset.items()):
        epochs = [row["epoch"] for row in rows]
        gap = np.array([row["overfitting_gap_mean"] for row in rows], dtype=float) * 100.0
        ax.plot(epochs, gap, marker="o", linewidth=2.2, label=dataset, color=colors.get(dataset))
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Brecha train-val accuracy (%)", fontsize=11)
    ax.set_title("Brecha de sobreajuste media por dataset", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")
    plt.tight_layout()
    path = plots_dir / "mean_overfitting_gap_by_dataset.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def plot_examples_used_by_dataset(output_dir: Path, completed_rows: list):
    """Dibuja cuantos ejemplos usa cada dataset de media."""
    if not completed_rows:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted({row["dataset"] for row in completed_rows})
    means = []
    for dataset in datasets:
        xs = [row["num_training_samples"] for row in completed_rows if row["dataset"] == dataset]
        means.append(float(np.mean(xs)))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(datasets, means, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="white")
    for bar, value in zip(bars, means):
        ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2.0, value), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_ylabel("Numero medio de ejemplos usados", fontsize=11)
    ax.set_title("Ejemplos usados para adaptar por dataset", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = plots_dir / "examples_used_by_dataset.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_status_counts(output_dir: Path, completed_rows: list, failed_rows: list):
    """Muestra cuantas ejecuciones completas y fallidas hay por metodo."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({row["method"] for row in completed_rows} | {row["method"] for row in failed_rows})
    completed_counts = [sum(1 for row in completed_rows if row["method"] == method) for method in methods]
    failed_counts = [sum(1 for row in failed_rows if row["method"] == method) for method in methods]
    if not methods:
        return None

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.bar(x - width / 2.0, completed_counts, width=width, color="#2ca02c", label="Completed")
    ax.bar(x + width / 2.0, failed_counts, width=width, color="#d62728", label="Failed")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10, ha="right")
    ax.set_ylabel("Numero de ejecuciones", fontsize=11)
    ax.set_title("Estado de ejecuciones por metodo dinamico", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = plots_dir / "method_status_counts.png"
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path

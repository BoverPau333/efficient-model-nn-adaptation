"""Plot all baseline training-loss curves and highlight early-stopping selections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOTS = [
    REPO_ROOT / "results" / "class_addition_baseline",
    REPO_ROOT / "results" / "class_removal_baseline",
]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "plots" / "baseline_training_error_curves.png"

DATASET_COLORS = {
    "CIFAR_10": "#D55E00",
    "Fashion_MNIST": "#0072B2",
    "Fruits_360": "#009E73",
    "Paintings": "#CC79A7",
}

TASK_TITLES = {
    "class_addition_baseline": "Baseline de adicion",
    "class_removal_baseline": "Baseline de eliminacion",
}

MODEL_ORDER = ["ResNet18", "MobileNetV3_Small", "EfficientNet_B0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-panel figure with all baseline training-loss curves."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        action="append",
        default=None,
        help="Optional baseline result root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--task",
        choices=["class_addition_baseline", "class_removal_baseline"],
        action="append",
        default=None,
        help="Optional baseline task filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--single-panel",
        action="store_true",
        help="Plot all selected runs together in a single panel.",
    )
    return parser.parse_args()


def load_history(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if not row.get("epoch") or not row.get("train_loss"):
                continue
            rows.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]) if row.get("val_loss") else None,
                }
            )
    return rows


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r") as handle:
        return json.load(handle)


def discover_runs(baseline_roots: list[Path], allowed_tasks: set[str] | None = None) -> list[dict]:
    runs = []
    for baseline_root in baseline_roots:
        if not baseline_root.exists():
            continue

        task_name = baseline_root.name
        if allowed_tasks and task_name not in allowed_tasks:
            continue
        for history_path in sorted(baseline_root.glob("*/*/*/training_history.csv")):
            metrics_path = history_path.with_name("final_metrics.json")
            if not metrics_path.exists():
                continue

            dataset_name = history_path.parts[-4]
            model_name = history_path.parts[-3]
            run_name = history_path.parts[-2]
            history = load_history(history_path)
            if not history:
                continue

            metrics = load_metrics(metrics_path)
            best_epoch = int(metrics["best_epoch"])
            best_row = next((row for row in history if row["epoch"] == best_epoch), history[-1])

            runs.append(
                {
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "dataset_label": metrics.get("dataset", dataset_name.replace("_", "-")),
                    "model_name": model_name,
                    "run_name": run_name,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_row["train_loss"],
                    "best_val_loss": float(metrics["best_val_loss"]),
                    "epochs_ran": int(metrics["epochs_ran"]),
                }
            )
    return runs


def make_title(task_name: str, model_name: str) -> str:
    task_label = TASK_TITLES.get(task_name, task_name.replace("_", " "))
    model_label = model_name.replace("_", "-")
    return f"{task_label}\n{model_label}"


def plot_runs(runs: list[dict], output_path: Path) -> None:
    tasks = [task for task in TASK_TITLES if any(run["task_name"] == task for run in runs)]
    models = [model for model in MODEL_ORDER if any(run["model_name"] == model for run in runs)]

    fig, axes = plt.subplots(
        len(tasks),
        len(models),
        figsize=(6 * max(len(models), 1), 4.3 * max(len(tasks), 1)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.patch.set_facecolor("#F8F5F0")

    legend_handles = [
        Line2D([0], [0], color=color, lw=3, label=dataset.replace("_", "-"))
        for dataset, color in DATASET_COLORS.items()
        if any(run["dataset_name"] == dataset for run in runs)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=10,
            label="Modelo elegido por early stopping",
        )
    )

    for row_idx, task_name in enumerate(tasks):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#FFFDF8")

            panel_runs = [
                run for run in runs if run["task_name"] == task_name and run["model_name"] == model_name
            ]

            if not panel_runs:
                ax.axis("off")
                continue

            for run in panel_runs:
                color = DATASET_COLORS.get(run["dataset_name"], "#666666")
                epochs = [row["epoch"] for row in run["history"]]
                train_loss = [row["train_loss"] for row in run["history"]]
                ax.plot(
                    epochs,
                    train_loss,
                    color=color,
                    linewidth=1.8,
                    alpha=0.45,
                )
                ax.scatter(
                    [run["best_epoch"]],
                    [run["best_train_loss"]],
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    marker="*",
                    s=130,
                    zorder=5,
                )
                ax.annotate(
                    run["run_name"],
                    (run["best_epoch"], run["best_train_loss"]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=7,
                    color=color,
                    alpha=0.95,
                )

            max_epoch = max(run["epochs_ran"] for run in panel_runs)
            ax.set_xlim(1, max_epoch)
            ax.grid(True, alpha=0.18, linestyle="--")
            ax.set_title(make_title(task_name, model_name), fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoca", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Error de entrenamiento (train_loss)", fontsize=10)

    fig.suptitle(
        "Curvas de error del baseline\nColor por dataset y estrella en la epoca elegida por early stopping",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(len(legend_handles), 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_runs_single_panel(runs: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#F8F5F0")
    ax.set_facecolor("#FFFDF8")

    legend_handles = [
        Line2D([0], [0], color=color, lw=3, label=dataset.replace("_", "-"))
        for dataset, color in DATASET_COLORS.items()
        if any(run["dataset_name"] == dataset for run in runs)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=10,
            label="Modelo elegido por early stopping",
        )
    )

    for run in runs:
        color = DATASET_COLORS.get(run["dataset_name"], "#666666")
        epochs = [row["epoch"] for row in run["history"]]
        train_loss = [row["train_loss"] for row in run["history"]]
        label = f"{run['model_name']} | {run['run_name']}"
        ax.plot(epochs, train_loss, color=color, linewidth=1.7, alpha=0.42)
        ax.scatter(
            [run["best_epoch"]],
            [run["best_train_loss"]],
            color=color,
            edgecolor="black",
            linewidth=0.6,
            marker="*",
            s=130,
            zorder=5,
        )
        ax.annotate(
            label,
            (run["best_epoch"], run["best_train_loss"]),
            textcoords="offset points",
            xytext=(5, -8),
            fontsize=6.5,
            color=color,
            alpha=0.9,
        )

    max_epoch = max(run["epochs_ran"] for run in runs)
    ax.set_xlim(1, max_epoch)
    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Error de entrenamiento (train_loss)", fontsize=11)
    ax.set_title(
        "Class removal baseline\nTodas las ejecuciones y modelos en una sola grafica",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.18, linestyle="--")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(len(legend_handles), 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_roots = args.baseline_root or DEFAULT_BASELINE_ROOTS
    allowed_tasks = set(args.task) if args.task else None
    runs = discover_runs(baseline_roots, allowed_tasks=allowed_tasks)
    if not runs:
        raise SystemExit("No se encontraron ejecuciones baseline con training_history.csv.")

    if args.single_panel:
        plot_runs_single_panel(runs, args.output)
    else:
        plot_runs(runs, args.output)
    print(f"Figura guardada en: {args.output}")
    print(f"Ejecuciones representadas: {len(runs)}")


if __name__ == "__main__":
    main()

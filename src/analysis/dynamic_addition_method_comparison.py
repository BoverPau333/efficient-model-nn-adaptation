"""Resumen especifico de metodos dynamic fine-tuning para class addition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.class_addition_method_comparison import (
    SOURCE_COLORS,
    SOURCE_ORDER,
    aggregate_rows,
    filter_completed_rows,
    load_all_method_rows,
)
from src.core.results_utils import save_json, write_csv


DYNAMIC_SOURCES = {
    "dynamic_precomputed",
    "dynamic_precomputed_early_stopping",
    "dynamic_epoch1",
    "dynamic_epoch1_early_stopping",
}

SUMMARY_METRICS = [
    ("accuracy_global", "Accuracy global media", "Accuracy"),
    ("accuracy_en_la_clase_anadida", "Accuracy media en la clase anadida", "Accuracy"),
    (
        "forgetting_u_olvido_sobre_clases_previas",
        "Forgetting medio sobre clases previas",
        "Drop",
    ),
    ("tiempo_total_de_adaptacion", "Tiempo medio de adaptacion", "Seconds"),
]


def load_dynamic_addition_rows(results_root: Path):
    """Carga solo filas de adicion pertenecientes a variantes dinamicas."""
    all_rows = load_all_method_rows(results_root)
    completed_rows = filter_completed_rows(all_rows)
    return [row for row in completed_rows if row.get("source_name") in DYNAMIC_SOURCES]


def _ordered_rows(rows: list):
    source_order = {name: idx for idx, name in enumerate(SOURCE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            source_order.get(row.get("source_name"), 999),
            row.get("train_percentage") if row.get("train_percentage") is not None else -1,
            str(row.get("method_label") or ""),
        ),
    )


def _ordered_dataset_rows(rows: list):
    source_order = {name: idx for idx, name in enumerate(SOURCE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset") or ""),
            source_order.get(row.get("source_name"), 999),
            row.get("train_percentage") if row.get("train_percentage") is not None else -1,
            str(row.get("method_label") or ""),
        ),
    )


def build_dynamic_summaries(rows: list):
    """Agrega resultados globales y por dataset para las variantes dinamicas."""
    overall = aggregate_rows(
        rows,
        ["source_name", "method_variant", "method_label", "train_percentage"],
    )
    by_dataset = aggregate_rows(
        rows,
        ["dataset", "source_name", "method_variant", "method_label", "train_percentage"],
    )
    return {"overall": overall, "by_dataset": by_dataset}


def save_dynamic_summaries(output_dir: Path, rows: list, summaries: dict):
    """Guarda tablas normalizadas y agregadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dynamic_addition_completed_runs.csv", _ordered_rows(rows))
    write_csv(output_dir / "dynamic_addition_summary_overall.csv", _ordered_rows(summaries["overall"]))
    write_csv(output_dir / "dynamic_addition_summary_by_dataset.csv", _ordered_rows(summaries["by_dataset"]))
    save_json(output_dir / "dynamic_addition_summary_overall.json", _ordered_rows(summaries["overall"]))
    save_json(output_dir / "dynamic_addition_summary_by_dataset.json", _ordered_rows(summaries["by_dataset"]))


def _format_mean_std(row: dict, metric_name: str, scale: float = 1.0, signed: bool = False):
    mean = row.get(f"{metric_name}_mean")
    std = row.get(f"{metric_name}_std")
    if mean is None:
        return "--"
    mean_value = float(mean) * scale
    if std is None:
        return f"{mean_value:+.2f}" if signed else f"{mean_value:.2f}"
    std_value = float(std) * scale
    if signed:
        return f"{mean_value:+.2f} \\pm {std_value:.2f}"
    return f"{mean_value:.2f} \\pm {std_value:.2f}"


def _latex_table_lines(caption: str, label: str, rows: list, include_dataset: bool):
    columns = "llccccc" if include_dataset else "lccccc"
    header = []
    if include_dataset:
        header.extend(
            [
                "\\textbf{Dataset} &",
                "\\textbf{Metodo} &",
            ]
        )
    else:
        header.append("\\textbf{Metodo} &")
    header.extend(
        [
            "\\textbf{Datos (\\%)} &",
            "\\textbf{Acc. global (\\%)} &",
            "\\textbf{Acc. clase anadida (\\%)} &",
            "\\textbf{Forgetting (p.p.)} &",
            "\\textbf{Tiempo (s)} \\\\",
        ]
    )

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        *header,
        "\\midrule",
    ]

    current_dataset = None
    ordered_rows = _ordered_dataset_rows(rows) if include_dataset else _ordered_rows(rows)
    for row in ordered_rows:
        dataset = row.get("dataset")
        if include_dataset and current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset

        values = []
        if include_dataset:
            values.extend([str(dataset), row["method_label"]])
        else:
            values.append(row["method_label"])

        percentage = row.get("train_percentage")
        percentage_text = "--" if percentage is None else f"{int(percentage) if float(percentage).is_integer() else percentage}"
        values.extend(
            [
                percentage_text,
                _format_mean_std(row, "accuracy_global", scale=100.0),
                _format_mean_std(row, "accuracy_en_la_clase_anadida", scale=100.0),
                _format_mean_std(
                    row,
                    "forgetting_u_olvido_sobre_clases_previas",
                    scale=100.0,
                    signed=True,
                ),
                _format_mean_std(row, "tiempo_total_de_adaptacion"),
            ]
        )
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return lines


def _markdown_table_lines(rows: list, include_dataset: bool):
    if include_dataset:
        lines = [
            "| Dataset | Metodo | Datos (%) | Acc. global (%) | Acc. clase anadida (%) | Forgetting (p.p.) | Tiempo (s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Metodo | Datos (%) | Acc. global (%) | Acc. clase anadida (%) | Forgetting (p.p.) | Tiempo (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]

    ordered_rows = _ordered_dataset_rows(rows) if include_dataset else _ordered_rows(rows)
    for row in ordered_rows:
        percentage = row.get("train_percentage")
        percentage_text = "--" if percentage is None else f"{int(percentage) if float(percentage).is_integer() else percentage}"
        values = []
        if include_dataset:
            values.extend([str(row.get("dataset")), row["method_label"]])
        else:
            values.append(row["method_label"])
        values.extend(
            [
                percentage_text,
                _format_mean_std(row, "accuracy_global", scale=100.0).replace(" \\pm ", " ± "),
                _format_mean_std(row, "accuracy_en_la_clase_anadida", scale=100.0).replace(" \\pm ", " ± "),
                _format_mean_std(
                    row,
                    "forgetting_u_olvido_sobre_clases_previas",
                    scale=100.0,
                    signed=True,
                ).replace(" \\pm ", " ± "),
                _format_mean_std(row, "tiempo_total_de_adaptacion").replace(" \\pm ", " ± "),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def save_dynamic_tables(output_dir: Path, summaries: dict):
    """Guarda tablas resumen en LaTeX y Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = summaries["overall"]
    by_dataset_rows = summaries["by_dataset"]

    overall_tex = output_dir / "dynamic_addition_overall_table.tex"
    overall_tex.write_text(
        "\n".join(
            _latex_table_lines(
                caption="Resumen global de los metodos dynamic fine-tuning para adicion de clases",
                label="tab:dynamic_addition_overall",
                rows=overall_rows,
                include_dataset=False,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    overall_md = output_dir / "dynamic_addition_overall_table.md"
    overall_md.write_text("\n".join(_markdown_table_lines(overall_rows, include_dataset=False)) + "\n", encoding="utf-8")

    dataset_tex = output_dir / "dynamic_addition_by_dataset_table.tex"
    dataset_tex.write_text(
        "\n".join(
            _latex_table_lines(
                caption="Resumen por dataset de los metodos dynamic fine-tuning para adicion de clases",
                label="tab:dynamic_addition_by_dataset",
                rows=by_dataset_rows,
                include_dataset=True,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_md = output_dir / "dynamic_addition_by_dataset_table.md"
    dataset_md.write_text(
        "\n".join(_markdown_table_lines(by_dataset_rows, include_dataset=True)) + "\n",
        encoding="utf-8",
    )

    return {
        "overall_tex": overall_tex,
        "overall_md": overall_md,
        "by_dataset_tex": dataset_tex,
        "by_dataset_md": dataset_md,
    }


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
        valid_values = [value for value in means if value > 0.0]
        if valid_values:
            min_value = float(np.min(valid_values))
            max_value = float(np.max(valid_values))
            padding = max(0.01, (max_value - min_value) * 0.2)
            lower = max(0.0, min_value - padding)
            upper = min(1.0, max_value + padding)
            if upper - lower < 0.03:
                center = (upper + lower) / 2.0
                lower = max(0.0, center - 0.015)
                upper = min(1.0, center + 0.015)
            ax.set_ylim(lower, upper)


def plot_dynamic_overall_summary(output_dir: Path, overall_rows: list):
    """Genera una figura general con las metricas clave de dynamic addition."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "dynamic_addition_overall_summary.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (metric_name, title, ylabel) in zip(axes, SUMMARY_METRICS):
        _plot_metric_bar(ax, overall_rows, metric_name, title, ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_dynamic_dataset_summary(output_dir: Path, by_dataset_rows: list):
    """Genera un resumen por dataset para accuracy global y clase anadida."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "dynamic_addition_by_dataset_summary.png"

    datasets = sorted({row.get("dataset") for row in by_dataset_rows if row.get("dataset")})
    if not datasets:
        return None

    fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 4.8 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        dataset_rows = [row for row in by_dataset_rows if row.get("dataset") == dataset]
        _plot_metric_bar(
            axes[row_idx, 0],
            dataset_rows,
            "accuracy_global",
            f"{dataset}: accuracy global",
            "Accuracy",
        )
        _plot_metric_bar(
            axes[row_idx, 1],
            dataset_rows,
            "accuracy_en_la_clase_anadida",
            f"{dataset}: accuracy en la clase anadida",
            "Accuracy",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_dynamic_accuracy_vs_time(output_dir: Path, overall_rows: list):
    """Genera un scatter para comparar accuracy global frente a tiempo."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "dynamic_addition_accuracy_vs_time.png"

    rows = [
        row
        for row in _ordered_rows(overall_rows)
        if row.get("accuracy_global_mean") is not None and row.get("tiempo_total_de_adaptacion_mean") is not None
    ]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    markers = {
        "dynamic_precomputed": "D",
        "dynamic_precomputed_early_stopping": "D",
        "dynamic_epoch1": "P",
        "dynamic_epoch1_early_stopping": "P",
    }

    for row in rows:
        source_name = row["source_name"]
        ax.scatter(
            row["tiempo_total_de_adaptacion_mean"],
            row["accuracy_global_mean"],
            s=150,
            color=SOURCE_COLORS.get(source_name, "#999999"),
            marker=markers.get(source_name, "o"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )
        ax.annotate(
            row["method_label"],
            (row["tiempo_total_de_adaptacion_mean"], row["accuracy_global_mean"]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_title("Dynamic addition: accuracy global vs tiempo", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tiempo de adaptacion (segundos, escala log)")
    ax.set_ylabel("Accuracy global media")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path

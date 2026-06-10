"""Estudio de clases conflictivas vs poco conflictivas en adicion y eliminacion."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.core.results_utils import save_json, write_csv


RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "conflictive_class_study"

CONFLICTIVE_CLASS_MAP = {
    "CIFAR-10": {
        "conflictiva": {"cat", "dog"},
        "poco_conflictiva": {"truck", "deer"},
    },
    "Fashion-MNIST": {
        "conflictiva": {"Shirt", "Coat"},
        "poco_conflictiva": {"Bag", "Sneaker"},
    },
    "Fruits-360": {
        "conflictiva": {"Pear 5", "Tomato 8"},
        "poco_conflictiva": {"Banana 4", "Cucumber 1"},
    },
}

CONFLICT_GROUP_ORDER = ["conflictiva", "poco_conflictiva"]
CONFLICT_GROUP_LABELS = {
    "conflictiva": "Clases conflictivas",
    "poco_conflictiva": "Clases poco conflictivas",
}
CONFLICT_GROUP_COLORS = {
    "conflictiva": "#BC4749",
    "poco_conflictiva": "#4C956C",
}

ADDITION_METRICS = [
    ("tiempo_total_de_adaptacion", "Tiempo de adaptacion", "Segundos"),
    ("accuracy_global", "Accuracy global", "Accuracy"),
    ("accuracy_en_clases_previas", "Accuracy en clases previas", "Accuracy"),
    ("accuracy_en_la_clase_anadida", "Accuracy en la clase anadida", "Accuracy"),
    (
        "forgetting_u_olvido_sobre_clases_previas",
        "Forgetting sobre clases previas",
        "Drop",
    ),
]

REMOVAL_METRICS = [
    ("tiempo_total_de_adaptacion", "Tiempo de adaptacion", "Segundos"),
    ("accuracy_global", "Accuracy global", "Accuracy"),
    ("forgetting_u_olvido", "Forgetting", "Drop"),
]

SOURCE_ORDER = {
    "baseline": 0,
    "prototypical_fewshot": 1,
    "frozen_backbone_head": 2,
    "head_only": 2,
    "finetuning": 3,
    "two_stage_finetuning": 3,
    "dynamic_precomputed": 4,
    "dynamic_precomputed_early_stopping": 5,
    "dynamic_epoch1": 6,
    "dynamic_epoch1_early_stopping": 7,
}

SOURCE_LABELS = {
    "baseline": "Baseline",
    "prototypical_fewshot": "Few-shot",
    "frozen_backbone_head": "Frozen backbone + head",
    "head_only": "Head only",
    "finetuning": "Fine-tuning",
    "two_stage_finetuning": "Two-stage fine-tuning",
    "dynamic_precomputed": "Dynamic precomputed",
    "dynamic_precomputed_early_stopping": "Dynamic precomputed ES",
    "dynamic_epoch1": "Dynamic epoch1",
    "dynamic_epoch1_early_stopping": "Dynamic epoch1 ES",
}


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _safe_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _load_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _classify_conflict_group(dataset: str, class_name: str):
    if dataset not in CONFLICTIVE_CLASS_MAP or class_name in (None, "", "None"):
        return None
    for group_name, classes in CONFLICTIVE_CLASS_MAP[dataset].items():
        if str(class_name) in classes:
            return group_name
    return None


def _extract_shots(method_variant: str):
    if method_variant in (None, "", "None"):
        return None
    match = re.search(r"fewshot_(\d+)shot", str(method_variant))
    if not match:
        return None
    return int(match.group(1))


def _humanize_token(token: str):
    parts = [part for part in str(token).replace("_", " ").split() if part]
    if not parts:
        return str(token)
    return " ".join(part.upper() if part.isupper() else part.capitalize() for part in parts)


def _build_removal_method_label(row: dict):
    source_name = row.get("source_name")
    train_percentage = _safe_float(row.get("train_percentage"))
    method_variant = str(row.get("method_variant") or source_name or "")

    if source_name == "baseline":
        return "Baseline"

    if source_name == "prototypical_fewshot":
        shots = _extract_shots(method_variant)
        suffix = method_variant.split("_")[-1] if "_" in method_variant else ""
        if shots is None:
            return "Few-shot"
        if suffix and suffix not in {"fewshot", f"{shots}shot"}:
            return f"{shots}-shot {suffix}"
        return f"{shots}-shot"

    if source_name in {"dynamic_precomputed", "dynamic_epoch1"}:
        parts = method_variant.split("_")
        strategy = parts[-2] if len(parts) >= 2 and parts[-1].endswith("%") else parts[-1]
        if train_percentage is None:
            return _humanize_token(strategy)
        return f"{_humanize_token(strategy)} {train_percentage:g}%"

    if source_name == "frozen_backbone_head" and method_variant == "frozen_backbone_head":
        return "100%"

    if source_name == "finetuning" and method_variant == "two_stage_finetuning":
        return "100%"

    if train_percentage is not None:
        return f"{train_percentage:g}%"
    return method_variant


def _normalize_addition_rows():
    rows = _load_csv_rows(RESULTS_DIR / "class_addition_method_comparison" / "completed_runs_normalized.csv")
    normalized = []
    for row in rows:
        conflict_group = _classify_conflict_group(row.get("dataset"), row.get("added_class"))
        if conflict_group is None:
            continue
        normalized.append(
            {
                "task": "addition",
                "dataset": row.get("dataset"),
                "source_name": row.get("source_name"),
                "method_variant": row.get("method_variant"),
                "method_label": row.get("method_label") or row.get("method_variant"),
                "train_percentage": _safe_float(row.get("train_percentage")),
                "shots_per_class": _safe_int(row.get("shots_per_class")),
                "target_class": row.get("added_class"),
                "conflict_group": conflict_group,
                "conflict_group_label": CONFLICT_GROUP_LABELS[conflict_group],
                "tiempo_total_de_adaptacion": _safe_float(row.get("tiempo_total_de_adaptacion")),
                "accuracy_global": _safe_float(row.get("accuracy_global")),
                "accuracy_en_clases_previas": _safe_float(row.get("accuracy_en_clases_previas")),
                "accuracy_en_la_clase_anadida": _safe_float(row.get("accuracy_en_la_clase_anadida")),
                "forgetting_u_olvido_sobre_clases_previas": _safe_float(
                    row.get("forgetting_u_olvido_sobre_clases_previas")
                ),
            }
        )
    return normalized


def _normalize_removal_rows():
    rows = _load_csv_rows(RESULTS_DIR / "class_removal_method_comparison" / "completed_runs_normalized.csv")
    normalized = []
    for row in rows:
        if row.get("source_name") == "finetuning" and row.get("method_variant") == "head_only_100%":
            continue
        conflict_group = _classify_conflict_group(row.get("dataset"), row.get("removed_class"))
        if conflict_group is None:
            continue
        normalized.append(
            {
                "task": "removal",
                "dataset": row.get("dataset"),
                "source_name": row.get("source_name"),
                "method_variant": row.get("method_variant"),
                "method_label": _build_removal_method_label(row),
                "train_percentage": _safe_float(row.get("train_percentage")),
                "shots_per_class": _extract_shots(row.get("method_variant")),
                "target_class": row.get("removed_class"),
                "conflict_group": conflict_group,
                "conflict_group_label": CONFLICT_GROUP_LABELS[conflict_group],
                "tiempo_total_de_adaptacion": _safe_float(row.get("tiempo_total_de_adaptacion")),
                "accuracy_global": _safe_float(row.get("accuracy_global")),
                "accuracy_en_clases_restantes": _safe_float(row.get("accuracy_en_clases_restantes")),
                "forgetting_u_olvido": _safe_float(row.get("forgetting_u_olvido")),
            }
        )
    return normalized


def _aggregate_rows(rows: list[dict], metric_specs: list[tuple[str, str, str]], group_keys: list[str] | None = None):
    grouped = {}
    if group_keys is None:
        group_keys = [
            "task",
            "dataset",
            "source_name",
            "method_variant",
            "method_label",
            "conflict_group",
            "conflict_group_label",
            "train_percentage",
            "shots_per_class",
        ]

    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    metric_names = [spec[0] for spec in metric_specs]
    for key, group_rows in sorted(grouped.items()):
        summary = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        summary["num_runs"] = len(group_rows)
        summary["target_classes"] = sorted({str(row["target_class"]) for row in group_rows})
        for metric_name in metric_names:
            values = [row[metric_name] for row in group_rows if row.get(metric_name) is not None]
            summary[f"{metric_name}_mean"] = None if not values else float(np.mean(values))
            summary[f"{metric_name}_std"] = None if not values else float(np.std(values))
        summary_rows.append(summary)
    return summary_rows


def _variant_sort_key(row: dict):
    source_rank = SOURCE_ORDER.get(str(row.get("source_name")), 999)
    shots = row.get("shots_per_class")
    train_percentage = row.get("train_percentage")
    if shots is not None:
        return (source_rank, 0, int(shots), str(row.get("method_label") or ""))
    if train_percentage is not None:
        return (source_rank, 1, float(train_percentage), str(row.get("method_label") or ""))
    return (source_rank, 2, math.inf, str(row.get("method_label") or ""))


def _set_informative_ylim(ax, metric_name: str, means: list[float], stds: list[float]):
    valid_pairs = [
        (float(mean), float(std))
        for mean, std in zip(means, stds)
        if mean is not None
    ]
    if not valid_pairs:
        return

    lower_candidates = [mean - std for mean, std in valid_pairs]
    upper_candidates = [mean + std for mean, std in valid_pairs]
    min_value = min(lower_candidates)
    max_value = max(upper_candidates)

    if metric_name.startswith("accuracy_"):
        padding = max(0.01, (max_value - min_value) * 0.22)
        lower = max(0.0, min_value - padding)
        upper = min(1.0, max_value + padding)
        if upper - lower < 0.04:
            center = (upper + lower) / 2.0
            lower = max(0.0, center - 0.02)
            upper = min(1.0, center + 0.02)
        ax.set_ylim(lower, upper)
        return

    padding = max(0.02, (max_value - min_value) * 0.18)
    lower = min_value - padding
    upper = max_value + padding
    if lower <= 0 <= upper and min_value >= 0:
        lower = max(0.0, lower)
    if upper - lower < 0.04:
        center = (upper + lower) / 2.0
        lower = center - 0.02
        upper = center + 0.02
    ax.set_ylim(lower, upper)


def _resolve_subplot_grid(metric_specs: list[tuple[str, str, str]]):
    n_metrics = len(metric_specs)
    if n_metrics == 3:
        return 1, 3
    ncols = 2 if n_metrics <= 4 else 3
    nrows = int(np.ceil(n_metrics / ncols))
    return nrows, ncols


def _plot_method_dataset_comparison(
    *,
    task_name: str,
    dataset_name: str,
    source_name: str,
    rows: list[dict],
    metric_specs: list[tuple[str, str, str]],
    output_dir: Path,
):
    if not rows:
        return None

    grouped_by_variant = {}
    for row in rows:
        grouped_by_variant.setdefault(row["method_variant"], []).append(row)

    ordered_variants = sorted(
        grouped_by_variant,
        key=lambda variant: _variant_sort_key(grouped_by_variant[variant][0]),
    )
    method_labels = [grouped_by_variant[variant][0]["method_label"] for variant in ordered_variants]

    n_metrics = len(metric_specs)
    nrows, ncols = _resolve_subplot_grid(metric_specs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(ordered_variants), dtype=float)
    width = 0.36

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        all_means = []
        all_stds = []
        for idx, group_name in enumerate(CONFLICT_GROUP_ORDER):
            group_rows = []
            for variant in ordered_variants:
                match = next(
                    (row for row in grouped_by_variant[variant] if row["conflict_group"] == group_name),
                    None,
                )
                group_rows.append(match)

            offset = (-width / 2.0) if idx == 0 else (width / 2.0)
            means = [None if row is None or row.get(f"{metric_name}_mean") is None else row[f"{metric_name}_mean"] for row in group_rows]
            stds = [0.0 if row is None or row.get(f"{metric_name}_std") is None else row[f"{metric_name}_std"] for row in group_rows]
            all_means.extend(means)
            all_stds.extend(stds)
            ax.bar(
                x + offset,
                [0.0 if value is None else value for value in means],
                width=width,
                yerr=stds,
                capsize=4,
                color=CONFLICT_GROUP_COLORS[group_name],
                edgecolor="white",
                label=CONFLICT_GROUP_LABELS[group_name],
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        _set_informative_ylim(ax, metric_name, all_means, all_stds)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"{dataset_name} | {SOURCE_LABELS.get(source_name, source_name)} | {task_name}",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    save_dir = output_dir / "plots" / task_name / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{source_name}_conflictive_vs_non_conflictive.png"
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _plot_operation_overall_comparison(
    *,
    task_name: str,
    rows: list[dict],
    metric_specs: list[tuple[str, str, str]],
    output_dir: Path,
    save_dirname: str,
    save_filename: str,
    title_prefix: str,
):
    if not rows:
        return None

    grouped_by_variant = {}
    for row in rows:
        grouped_by_variant.setdefault(row["method_variant"], []).append(row)

    ordered_variants = sorted(
        grouped_by_variant,
        key=lambda variant: _variant_sort_key(grouped_by_variant[variant][0]),
    )
    method_labels = [grouped_by_variant[variant][0]["method_label"] for variant in ordered_variants]

    n_metrics = len(metric_specs)
    nrows, ncols = _resolve_subplot_grid(metric_specs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(ordered_variants), dtype=float)
    width = 0.36

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        all_means = []
        all_stds = []
        for idx, group_name in enumerate(CONFLICT_GROUP_ORDER):
            group_rows = []
            for variant in ordered_variants:
                match = next(
                    (row for row in grouped_by_variant[variant] if row["conflict_group"] == group_name),
                    None,
                )
                group_rows.append(match)

            offset = (-width / 2.0) if idx == 0 else (width / 2.0)
            means = [
                None if row is None or row.get(f"{metric_name}_mean") is None else row[f"{metric_name}_mean"]
                for row in group_rows
            ]
            stds = [
                0.0 if row is None or row.get(f"{metric_name}_std") is None else row[f"{metric_name}_std"]
                for row in group_rows
            ]
            all_means.extend(means)
            all_stds.extend(stds)
            ax.bar(
                x + offset,
                [0.0 if value is None else value for value in means],
                width=width,
                yerr=stds,
                capsize=4,
                color=CONFLICT_GROUP_COLORS[group_name],
                edgecolor="white",
                label=CONFLICT_GROUP_LABELS[group_name],
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        _set_informative_ylim(ax, metric_name, all_means, all_stds)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.suptitle(f"{title_prefix} | {task_name}", fontsize=14, fontweight="bold", y=0.992)
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    save_dir = output_dir / "plots" / task_name / save_dirname
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _generate_plots(task_name: str, summary_rows: list[dict], metric_specs: list[tuple[str, str, str]], output_dir: Path):
    generated = []
    grouped = {}
    for row in summary_rows:
        key = (row["dataset"], row["source_name"])
        grouped.setdefault(key, []).append(row)

    for (dataset_name, source_name), rows in sorted(grouped.items(), key=lambda item: (item[0][0], SOURCE_ORDER.get(item[0][1], 999), item[0][1])):
        path = _plot_method_dataset_comparison(
            task_name=task_name,
            dataset_name=dataset_name,
            source_name=source_name,
            rows=rows,
            metric_specs=metric_specs,
            output_dir=output_dir,
        )
        if path is not None:
            generated.append(str(path))
    return generated


def _generate_cross_dataset_method_plots(task_name: str, summary_rows: list[dict], metric_specs: list[tuple[str, str, str]], output_dir: Path):
    generated = []
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["source_name"], []).append(row)

    for source_name, rows in sorted(grouped.items(), key=lambda item: (SOURCE_ORDER.get(item[0], 999), item[0])):
        path = _plot_operation_overall_comparison(
            task_name=task_name,
            rows=rows,
            metric_specs=metric_specs,
            output_dir=output_dir,
            save_dirname="ALL_DATASETS_BY_METHOD",
            save_filename=f"{source_name}_mean_across_datasets_conflictive_vs_non_conflictive.png",
            title_prefix=f"Media sobre todos los datasets | {SOURCE_LABELS.get(source_name, source_name)}",
        )
        if path is not None:
            generated.append(str(path))
    return generated


def _generate_operation_overall_plot(task_name: str, summary_rows: list[dict], metric_specs: list[tuple[str, str, str]], output_dir: Path):
    path = _plot_operation_overall_comparison(
        task_name=task_name,
        rows=summary_rows,
        metric_specs=metric_specs,
        output_dir=output_dir,
        save_dirname="ALL_METHODS",
        save_filename="all_methods_mean_across_datasets_conflictive_vs_non_conflictive.png",
        title_prefix="Media global de todos los metodos",
    )
    return None if path is None else str(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    addition_rows = _normalize_addition_rows()
    removal_rows = _normalize_removal_rows()

    write_csv(OUTPUT_DIR / "addition_rows_with_conflict_group.csv", addition_rows)
    write_csv(OUTPUT_DIR / "removal_rows_with_conflict_group.csv", removal_rows)

    addition_summary = _aggregate_rows(addition_rows, ADDITION_METRICS)
    removal_summary = _aggregate_rows(removal_rows, REMOVAL_METRICS)
    addition_summary_all_datasets = _aggregate_rows(
        addition_rows,
        ADDITION_METRICS,
        group_keys=[
            "task",
            "source_name",
            "method_variant",
            "method_label",
            "conflict_group",
            "conflict_group_label",
            "train_percentage",
            "shots_per_class",
        ],
    )
    removal_summary_all_datasets = _aggregate_rows(
        removal_rows,
        REMOVAL_METRICS,
        group_keys=[
            "task",
            "source_name",
            "method_variant",
            "method_label",
            "conflict_group",
            "conflict_group_label",
            "train_percentage",
            "shots_per_class",
        ],
    )

    write_csv(OUTPUT_DIR / "addition_summary_by_dataset_method.csv", addition_summary)
    write_csv(OUTPUT_DIR / "removal_summary_by_dataset_method.csv", removal_summary)
    write_csv(OUTPUT_DIR / "addition_summary_all_datasets_by_method.csv", addition_summary_all_datasets)
    write_csv(OUTPUT_DIR / "removal_summary_all_datasets_by_method.csv", removal_summary_all_datasets)
    save_json(OUTPUT_DIR / "addition_summary_by_dataset_method.json", addition_summary)
    save_json(OUTPUT_DIR / "removal_summary_by_dataset_method.json", removal_summary)
    save_json(OUTPUT_DIR / "addition_summary_all_datasets_by_method.json", addition_summary_all_datasets)
    save_json(OUTPUT_DIR / "removal_summary_all_datasets_by_method.json", removal_summary_all_datasets)

    generated_plots = {
        "addition": _generate_plots("addition", addition_summary, ADDITION_METRICS, OUTPUT_DIR),
        "removal": _generate_plots("removal", removal_summary, REMOVAL_METRICS, OUTPUT_DIR),
        "addition_all_datasets_by_method": _generate_cross_dataset_method_plots(
            "addition",
            addition_summary_all_datasets,
            ADDITION_METRICS,
            OUTPUT_DIR,
        ),
        "removal_all_datasets_by_method": _generate_cross_dataset_method_plots(
            "removal",
            removal_summary_all_datasets,
            REMOVAL_METRICS,
            OUTPUT_DIR,
        ),
        "addition_all_methods": _generate_operation_overall_plot(
            "addition",
            addition_summary_all_datasets,
            ADDITION_METRICS,
            OUTPUT_DIR,
        ),
        "removal_all_methods": _generate_operation_overall_plot(
            "removal",
            removal_summary_all_datasets,
            REMOVAL_METRICS,
            OUTPUT_DIR,
        ),
    }
    save_json(OUTPUT_DIR / "generated_plots.json", generated_plots)

    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print(f"Filas de adicion analizadas: {len(addition_rows)}")
    print(f"Filas de eliminacion analizadas: {len(removal_rows)}")
    print(f"Graficas de adicion: {len(generated_plots['addition'])}")
    print(f"Graficas de eliminacion: {len(generated_plots['removal'])}")
    print(f"Graficas media por metodo (adicion): {len(generated_plots['addition_all_datasets_by_method'])}")
    print(f"Graficas media por metodo (eliminacion): {len(generated_plots['removal_all_datasets_by_method'])}")


if __name__ == "__main__":
    main()

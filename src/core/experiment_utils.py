"""Utilidades compartidas para scripts de experimentos."""

from pathlib import Path

from src.core.results_utils import load_json, save_json, slugify, write_csv
from src.dataset.utils import count_examples_per_class


def aggregate_split_counts(train_ds, val_ds, test_ds, classes: list):
    """Cuenta ejemplos por split con el mismo formato en todos los scripts."""
    return {
        "train": count_examples_per_class(train_ds, classes),
        "val": count_examples_per_class(val_ds, classes),
        "test": count_examples_per_class(test_ds, classes),
    }


def save_experiment_artifacts(experiment_dir: Path, training_result: dict, metrics_payload: dict):
    """Guarda historial y metricas finales de una ejecucion."""
    save_json(experiment_dir / "training_history.json", training_result["history"])
    write_csv(experiment_dir / "training_history.csv", training_result["history"])
    save_json(experiment_dir / "final_metrics.json", metrics_payload)


def load_summary_from_metrics(metrics_path: Path, fallback_builder, *, status: str):
    """Recupera el resumen guardado o lo recompone desde metricas."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = fallback_builder(existing_metrics)
    summary["status"] = status
    return summary


def collect_dataset_summary_rows(
    dataset_output_dir: Path,
    *,
    artifact_prefix: str,
    load_completed_summary,
    build_error_summary,
    build_row_key,
):
    """Reconstruye el resumen de un dataset leyendo artefactos persistidos."""
    rows_by_key = {}

    for metrics_path in sorted(dataset_output_dir.rglob("final_metrics.json")):
        if not metrics_path.parent.name.startswith(artifact_prefix):
            continue
        summary = load_completed_summary(metrics_path)
        rows_by_key[build_row_key(summary)] = summary

    for error_path in sorted(dataset_output_dir.rglob("error.json")):
        if not error_path.parent.name.startswith(artifact_prefix):
            continue
        error_payload = load_json(error_path)
        key = build_row_key(error_payload)
        if key in rows_by_key:
            continue
        rows_by_key[key] = build_error_summary(error_payload)

    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: tuple(_normalize_sort_value(value) for value in item),
        )
    ]


def rebuild_summary_files(
    base_output_dir: Path,
    dataset_names: list,
    *,
    collect_dataset_rows,
    aggregate_json_name: str = "all_experiments_summary.json",
    aggregate_csv_name: str = "all_experiments_summary.csv",
):
    """Reescribe los resumenes por dataset y el agregado global."""
    all_summary_rows = []
    requested_dataset_dirs = {base_output_dir / slugify(dataset_name) for dataset_name in dataset_names}
    existing_dataset_dirs = (
        {path for path in base_output_dir.iterdir() if path.is_dir()}
        if base_output_dir.exists()
        else set()
    )

    for dataset_output_dir in sorted(requested_dataset_dirs | existing_dataset_dirs):
        if not dataset_output_dir.exists():
            continue
        summary_rows = collect_dataset_rows(dataset_output_dir)
        save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
        write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
        all_summary_rows.extend(summary_rows)

    save_json(base_output_dir / aggregate_json_name, all_summary_rows)
    write_csv(base_output_dir / aggregate_csv_name, all_summary_rows)
    return all_summary_rows


def load_reference_metrics(reference_dir: Path, dataset_name: str, model_name: str, *, run_hint: str):
    """Carga metricas de referencia y valida la accuracy por clase."""
    metrics_path = reference_dir / slugify(dataset_name) / slugify(model_name) / "final_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Missing full-training ImageNet reference metrics at "
            f"'{metrics_path}'. Run {run_hint} first or pass --reference-dir with the correct location."
        )

    metrics_payload = load_json(metrics_path)
    per_class_accuracy = metrics_payload.get("test_per_class_accuracy")
    if not isinstance(per_class_accuracy, dict) or not per_class_accuracy:
        raise ValueError(
            f"Reference metrics at '{metrics_path}' do not contain a valid "
            "'test_per_class_accuracy' mapping."
        )
    return metrics_payload, metrics_path


def _normalize_sort_value(value):
    """Normaliza valores mixtos para ordenar filas de forma estable."""
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value or ""))

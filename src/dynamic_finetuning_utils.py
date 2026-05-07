"""Utilidades compartidas para experimentos de fine-tuning dinamico."""

import argparse
import time
import traceback
from pathlib import Path

from src.dataset.utils import remove_class_and_remap, resolve_class_to_remove
from src.experiments_config.config import BATCH_SIZE, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.results_utils import (
    build_dynamic_summary_row,
    compute_forgetting_from_reference,
    count_trainable_parameters,
    evaluate_prediction_confidence,
    load_json,
    maybe_load_baseline_summary_row,
    parse_class_identifier,
    save_json,
    slugify,
    write_csv,
)


DEFAULT_DYNAMIC_REFERENCE_DIR = RESULTS_DIR / "full_training_reference_imagenet"
DEFAULT_DYNAMIC_BASELINE_DIR = RESULTS_DIR / "class_removal_baseline"


def prepare_update_datasets(train_ds, val_ds, test_ds, classes: list, modified_class, update_type: str):
    """Prepara los splits activos segun el tipo de actualizacion."""
    update_type = update_type.lower()
    modified_class_idx, modified_class_name = resolve_class_to_remove(classes, modified_class)

    if update_type == "remove":
        filtered_train, metadata = remove_class_and_remap(train_ds, classes, modified_class)
        filtered_val, _ = remove_class_and_remap(val_ds, classes, modified_class)
        filtered_test, _ = remove_class_and_remap(test_ds, classes, modified_class)
        return {
            "train_active": filtered_train,
            "val_active": filtered_val,
            "test_active": filtered_test,
            "distance_train_dataset": train_ds,
            "distance_classes": list(classes),
            "active_classes": metadata["remaining_classes"],
            "modified_class_idx_original": int(metadata["removed_class_idx"]),
            "modified_class_name": metadata["removed_class_name"],
            "label_mapping_after_removal": metadata["label_mapping"],
            "final_num_classes": len(metadata["remaining_classes"]),
        }

    if update_type == "add":
        return {
            "train_active": train_ds,
            "val_active": val_ds,
            "test_active": test_ds,
            "distance_train_dataset": train_ds,
            "distance_classes": list(classes),
            "active_classes": list(classes),
            "modified_class_idx_original": int(modified_class_idx),
            "modified_class_name": modified_class_name,
            "label_mapping_after_removal": None,
            "final_num_classes": len(classes),
        }

    raise ValueError("update_type debe ser 'add' o 'remove'")
def build_dynamic_arg_parser(description: str, default_output_dir: Path, dataset_choices):
    """Construye el parser comun para los experimentos dinamicos."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, choices=sorted(dataset_choices))
    parser.add_argument("--models", nargs="*", default=["MobileNetV3-Small"], help="Modelos a ejecutar.")
    parser.add_argument("--modified-class", required=True, help="Clase modificada: indice o nombre.")
    parser.add_argument("--update-type", choices=["add", "remove"], default="remove")
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--initial-samples-per-class", type=int, default=8)
    parser.add_argument("--samples-per-modified-class", type=int, default=64)
    parser.add_argument("--samples-per-neighbour-class", type=int, default=32)
    parser.add_argument("--memory-samples-per-far-class", type=int, default=4)
    parser.add_argument("--selection-strategy", choices=["frontier", "nearest_to_modified"], default="frontier")
    parser.add_argument("--embedding-representation", choices=["embeddings", "logits"], default="embeddings")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--reference-dir", default=str(DEFAULT_DYNAMIC_REFERENCE_DIR))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_DYNAMIC_BASELINE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_experiment_dir(base_output_dir: Path, dataset_name: str, model_name: str, modified_class, update_type: str):
    """Crea la ruta del experimento dinamico."""
    return (
        base_output_dir
        / slugify(dataset_name)
        / slugify(model_name)
        / f"{update_type}_{slugify(modified_class)}"
    )


def load_existing_dynamic_summary(metrics_path: Path):
    """Carga el resumen existente y lo marca como omitido."""
    summary = load_json(metrics_path).get("summary", {})
    summary["status"] = "skipped_existing"
    return summary


def timestamp_slug():
    """Timestamp compacto para artefactos auxiliares."""
    return time.strftime("%Y%m%d_%H%M%S")


def save_run_artifacts(
    experiment_dir: Path,
    history_rows: list,
    metrics_payload: dict,
    selected_neighbours_payload: dict,
    config_payload=None,
):
    """Guarda artefactos estandar del experimento."""
    save_json(experiment_dir / "training_history.json", history_rows)
    write_csv(experiment_dir / "training_history.csv", history_rows)
    save_json(experiment_dir / "final_metrics.json", metrics_payload)
    if config_payload is not None:
        save_json(experiment_dir / "run_config.json", config_payload)
    save_json(experiment_dir / "selected_neighbours.json", selected_neighbours_payload)
    method_slug = selected_neighbours_payload.get("method", "method")
    save_json(
        experiment_dir / f"selected_neighbours_{method_slug}_{timestamp_slug()}.json",
        selected_neighbours_payload,
    )


def rebuild_comparison_summary(comparison_root: Path):
    """Reconstruye una tabla agregada combinando las ejecuciones de ambos metodos."""
    all_rows = []
    for summary_path in sorted(comparison_root.glob("*/experiments_summary.json")):
        rows = load_json(summary_path)
        if isinstance(rows, list):
            all_rows.extend(rows)

    if all_rows:
        save_json(comparison_root / "comparison_summary.json", all_rows)
        write_csv(comparison_root / "comparison_summary.csv", all_rows)


def build_dynamic_run_config(args, dataset_name: str, model_name: str, method_name: str, embedding_strategy: str):
    """Serializa la configuracion esencial del experimento."""
    config = {
        "dataset": dataset_name,
        "model_name": model_name,
        "method": method_name,
        "embedding_strategy": embedding_strategy,
        "modified_class": str(args.modified_class),
        "update_type": args.update_type,
        "k_neighbours": int(args.k_neighbours),
        "distance_metric": args.distance_metric,
        "selection_strategy": args.selection_strategy,
        "embedding_representation": args.embedding_representation,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "initial_samples_per_class": int(args.initial_samples_per_class),
        "samples_per_modified_class": int(args.samples_per_modified_class),
        "samples_per_neighbour_class": int(args.samples_per_neighbour_class),
        "memory_samples_per_far_class": int(args.memory_samples_per_far_class),
        "reference_dir": str(args.reference_dir),
        "baseline_dir": str(args.baseline_dir),
    }
    return config


def finalize_dynamic_experiment(
    *,
    dataset_name: str,
    model_name: str,
    args,
    method_name: str,
    embedding_strategy: str,
    experiment_dir: Path,
    model,
    setup: dict,
    classes: list,
    reference_metrics: dict,
    checkpoint_path: Path,
    reference_metrics_path: Path,
    neighbours: list,
    history_rows: list,
    selected_train,
    test_loader,
    evaluation_metrics: dict,
    timing: dict,
    training_summary: dict,
    selection_details: dict,
):
    """Construye el payload final, guarda artefactos y devuelve la fila resumen."""
    forgetting_score = compute_forgetting_from_reference(
        reference_metrics.get("test_per_class_accuracy"),
        evaluation_metrics["per_class_accuracy"],
    )
    baseline_row = None
    if args.update_type == "remove":
        baseline_row = maybe_load_baseline_summary_row(
            baseline_dir=Path(args.baseline_dir),
            dataset_name=dataset_name,
            model_name=model_name,
            modified_class_name=setup["modified_class_name"],
        )

    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "method": method_name,
        "embedding_strategy": embedding_strategy,
        "update_type": args.update_type,
        "modified_class": setup["modified_class_name"],
        "modified_class_idx_original": int(setup["modified_class_idx_original"]),
        "distance_metric": args.distance_metric,
        "selection_strategy": args.selection_strategy,
        "embedding_representation": args.embedding_representation,
        "k_neighbours": int(args.k_neighbours),
        "neighbour_classes": [
            {
                "class_idx": int(item["class_idx"]),
                "class_name": classes[int(item["class_idx"])],
                "distance": float(item["distance"]),
            }
            for item in neighbours
        ],
        "selected_neighbour_class_names": [classes[int(item["class_idx"])] for item in neighbours],
        "reference_checkpoint_path": str(checkpoint_path),
        "reference_metrics_path": str(reference_metrics_path),
        "final_num_classes": int(setup["final_num_classes"]),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(training_summary["epochs_ran"]),
        "best_epoch": int(training_summary["best_epoch"]),
        "best_val_loss": float(training_summary["best_val_loss"]),
        "best_val_accuracy": float(training_summary["best_val_accuracy"]),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "initial_samples_per_class": int(args.initial_samples_per_class),
        "samples_per_modified_class": int(args.samples_per_modified_class),
        "samples_per_neighbour_class": int(args.samples_per_neighbour_class),
        "memory_samples_per_far_class": int(args.memory_samples_per_far_class),
        "total_time": float(timing["total_time"]),
        "embedding_time": float(timing["embedding_time"]),
        "selection_time": float(timing["selection_time"]),
        "finetuning_time": float(timing["finetuning_time"]),
        "evaluation_time": float(timing["evaluation_time"]),
        "accuracy": float(evaluation_metrics["accuracy"]),
        "f1_macro": float(evaluation_metrics["f1_macro"]),
        "f1_weighted": float(evaluation_metrics["f1_weighted"]),
        "mean_per_class_accuracy": float(evaluation_metrics["mean_per_class_accuracy"]),
        "test_per_class_accuracy": evaluation_metrics["per_class_accuracy"],
        "forgetting_score": None if forgetting_score is None else float(forgetting_score),
        "prediction_confidence_mean": float(evaluate_prediction_confidence(model, test_loader)),
        "num_trainable_parameters": int(count_trainable_parameters(model)),
        "num_training_samples": int(len(selected_train)),
        "num_selected_classes": int(len(set(int(label) for label in selected_train.targets.tolist()))),
        "additional_memory_required": 0.0,
        "baseline_accuracy": None if baseline_row is None else baseline_row.get("accuracy_global"),
        "baseline_time": None if baseline_row is None else baseline_row.get("tiempo_total_de_adaptacion"),
        "accuracy_delta_vs_baseline": None
        if baseline_row is None
        else float(evaluation_metrics["accuracy"]) - float(baseline_row["accuracy_global"]),
        "time_delta_vs_baseline": None
        if baseline_row is None
        else float(timing["total_time"]) - float(baseline_row["tiempo_total_de_adaptacion"]),
    }
    metrics_payload["summary"] = build_dynamic_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        update_type=args.update_type,
        status="completed",
        metrics_payload=metrics_payload,
    )

    save_run_artifacts(
        experiment_dir=experiment_dir,
        history_rows=history_rows,
        metrics_payload=metrics_payload,
        selected_neighbours_payload={
            "dataset": dataset_name,
            "model_name": model_name,
            "method": method_name,
            "modified_class": setup["modified_class_name"],
            "update_type": args.update_type,
            "selected_neighbours": metrics_payload["neighbour_classes"],
            "selection_details": selection_details,
        },
        config_payload=build_dynamic_run_config(
            args=args,
            dataset_name=dataset_name,
            model_name=model_name,
            method_name=method_name,
            embedding_strategy=embedding_strategy,
        ),
    )
    return metrics_payload["summary"]


def build_failed_dynamic_summary(args, model_name: str, method_name: str, embedding_strategy: str, error_message: str):
    """Construye una fila de error consistente."""
    return {
        "dataset": args.dataset,
        "model_name": model_name,
        "status": "failed",
        "update_type": args.update_type,
        "method": method_name,
        "embedding_strategy": embedding_strategy,
        "modified_class": str(args.modified_class),
        "error": error_message,
    }


def run_dynamic_experiment_suite(args, method_name: str, embedding_strategy: str, run_single_experiment):
    """Ejecuta el bucle principal compartido por los scripts dinamicos."""
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name in args.models:
        try:
            summary_rows.append(run_single_experiment(args.dataset, model_name, args, base_output_dir))
        except Exception as exc:
            summary_rows.append(
                build_failed_dynamic_summary(
                    args=args,
                    model_name=model_name,
                    method_name=method_name,
                    embedding_strategy=embedding_strategy,
                    error_message=str(exc),
                )
            )
            error_dir = build_experiment_dir(
                base_output_dir=base_output_dir,
                dataset_name=args.dataset,
                model_name=model_name,
                modified_class=args.modified_class,
                update_type=args.update_type,
            )
            error_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                error_dir / "error.json",
                {
                    "dataset": args.dataset,
                    "model_name": model_name,
                    "method": method_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    save_json(base_output_dir / "experiments_summary.json", summary_rows)
    write_csv(base_output_dir / "experiments_summary.csv", summary_rows)
    rebuild_comparison_summary(base_output_dir.parent)
    return base_output_dir

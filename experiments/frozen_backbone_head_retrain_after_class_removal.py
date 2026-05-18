"""Baseline-style class-removal experiment with frozen ImageNet backbones."""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from src.adaptation.class_removal_experiment_utils import get_classes_to_remove, total_examples_from_split_counts
from src.experiments_config.class_removal_baseline_config import (
    DEFAULT_DATASET,
)
from src.experiments_config.config import BATCH_SIZE, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class, remove_class_and_remap
from src.metrics_elimination import METRICAS_ELIMINACION
from src.models import IMAGENET_FROZEN_HEAD_MODEL_BUILDERS
from src.core.results_utils import (
    build_loader,
    count_trainable_parameters,
    evaluate_prediction_confidence,
    load_json,
    save_json,
    set_seed,
    slugify,
    write_csv,
)
from src.core.training import evaluate, train_with_early_stopping


DEFAULT_MAX_EPOCHS = 5
DEFAULT_REFERENCE_OUTPUT_DIR = RESULTS_DIR / "full_training_reference_imagenet"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "class_removal_frozen_backbone_head"


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the class-removal baseline with ImageNet backbones frozen "
            "and only the classification head trainable."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASET_LOADERS),
        help="Dataset to run the experiment on.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the experiment on every registered dataset in one execution.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(IMAGENET_FROZEN_HEAD_MODEL_BUILDERS),
        default=list(IMAGENET_FROZEN_HEAD_MODEL_BUILDERS.keys()),
        help="Subset of model names to run. Default: all registered frozen-head builders.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional override for classes to remove. Use class names or integer indices.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Number of head-only training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate used during retraining.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for train/validation/test dataloaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory for checkpoints, logs and summaries.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_OUTPUT_DIR),
        help=(
            "Directory containing the full-class ImageNet reference runs used "
            "to compute forgetting on the remaining classes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if final metrics already exist.",
    )
    return parser.parse_args()


def load_reference_metrics(reference_dir: Path, dataset_name: str, model_name: str):
    """Load the full-class ImageNet reference metrics for one dataset/model pair."""
    metrics_path = reference_dir / slugify(dataset_name) / slugify(model_name) / "final_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Missing full-training ImageNet reference metrics at "
            f"'{metrics_path}'. Run experiments/full_training_reference_imagenet.py first "
            "or pass --reference-dir with the correct location."
        )

    metrics_payload = load_json(metrics_path)
    per_class_accuracy = metrics_payload.get("test_per_class_accuracy")
    if not isinstance(per_class_accuracy, dict) or not per_class_accuracy:
        raise ValueError(
            f"Reference metrics at '{metrics_path}' do not contain a valid "
            "'test_per_class_accuracy' mapping."
        )
    return metrics_payload, metrics_path


def compute_forgetting_from_reference(reference_per_class_accuracy: dict, current_per_class_accuracy: dict) -> float:
    """Average how much the remaining classes worsen versus the full-class reference."""
    remaining_classes = list(current_per_class_accuracy)
    if not remaining_classes:
        raise ValueError("Cannot compute forgetting without remaining classes.")

    missing_classes = [
        class_name for class_name in remaining_classes if class_name not in reference_per_class_accuracy
    ]
    if missing_classes:
        raise ValueError(
            "The ImageNet reference is missing remaining classes required for forgetting: "
            f"{missing_classes}"
        )

    degradations = [
        float(reference_per_class_accuracy[class_name]) - float(current_per_class_accuracy[class_name])
        for class_name in remaining_classes
    ]
    return float(np.mean(degradations))


def build_summary_row(
    dataset_name: str,
    model_name: str,
    removed_class_name,
    final_num_classes: int,
    status: str,
    metrics_payload=None,
    error_message=None,
):
    """Flatten the most useful fields for the aggregate CSV/JSON summary."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": str(removed_class_name),
        "final_num_classes": int(final_num_classes),
        "initialization": "imagenet",
        "backbone_mode": "frozen",
        "trainable_scope": "head_only",
        "status": status,
    }

    if metrics_payload is not None:
        row.update(
            {
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
                "tiempo_total_de_adaptacion": float(metrics_payload["elapsed_seconds"]),
                "accuracy_global": float(metrics_payload["test_overall_accuracy"]),
                "accuracy_en_clases_restantes": float(metrics_payload["test_mean_per_class_accuracy"]),
                "forgetting_u_olvido": float(metrics_payload["forgetting_u_olvido"]),
                "numero_de_ejemplos_utilizados": int(
                    metrics_payload["num_examples_used_for_adaptation"]
                ),
                "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
                "numero_de_parametros_entrenados_o_modificados": int(
                    metrics_payload["num_trainable_parameters"]
                ),
                "memoria_adicional_requerida": float(metrics_payload["additional_memory_required"]),
            }
        )

    if error_message is not None:
        row["error"] = error_message

    return row


def save_experiment_artifacts(experiment_dir: Path, training_result: dict, metrics_payload: dict):
    """Persist logs and final metrics."""
    save_json(experiment_dir / "training_history.json", training_result["history"])
    write_csv(experiment_dir / "training_history.csv", training_result["history"])
    save_json(experiment_dir / "final_metrics.json", metrics_payload)


def load_existing_summary(metrics_path: Path):
    """Load the flattened summary from an existing finished experiment."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = build_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            removed_class_name=existing_metrics["removed_class"],
            final_num_classes=existing_metrics["final_num_classes"],
            status="completed",
            metrics_payload=existing_metrics,
        )
    summary["status"] = "skipped_existing"
    return summary


def aggregate_counts(train_ds, val_ds, test_ds, classes: list):
    """Count examples per class for the filtered splits."""
    return {
        "train": count_examples_per_class(train_ds, classes),
        "val": count_examples_per_class(val_ds, classes),
        "test": count_examples_per_class(test_ds, classes),
    }


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    model_builder,
    removed_class_name,
    filtered_train,
    filtered_val,
    filtered_test,
    filtered_classes: list,
    args,
    dataset_output_dir: Path,
):
    """Train and evaluate one frozen-backbone/head-only class-removal experiment."""
    experiment_dir = (
        dataset_output_dir
        / slugify(model_name)
        / f"removed_{slugify(removed_class_name)}"
    )
    metrics_path = experiment_dir / "final_metrics.json"
    error_path = experiment_dir / "error.json"

    if not args.overwrite and metrics_path.exists():
        print(f"[SKIP] {dataset_name} | {model_name} | remove={removed_class_name}")
        return load_existing_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    reference_metrics, reference_metrics_path = load_reference_metrics(
        reference_dir=Path(args.reference_dir),
        dataset_name=dataset_name,
        model_name=model_name,
    )

    set_seed(args.seed)
    train_loader = build_loader(filtered_train, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(filtered_val, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(filtered_test, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(
        f"Dataset: {dataset_name} | Model: {model_name} | Removed class: {removed_class_name}"
    )
    print("Initialization: ImageNet | Backbone: frozen | Trainable scope: head only")
    print(f"{'=' * 90}")

    num_classes = len(filtered_classes)
    model = model_builder(num_classes)
    num_trainable_parameters = count_trainable_parameters(model)

    t0 = time.time()
    training_result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=None,
        checkpoint_path=None,
        verbose=True,
    )
    elapsed = time.time() - t0

    test_overall_accuracy, per_class_accuracy, confusion_matrix = evaluate(
        model,
        test_loader,
        num_classes,
    )

    split_counts = aggregate_counts(filtered_train, filtered_val, filtered_test, filtered_classes)
    prediction_confidence_mean = evaluate_prediction_confidence(model, test_loader)
    test_per_class_accuracy = {
        class_name: float(per_class_accuracy[class_idx])
        for class_idx, class_name in enumerate(filtered_classes)
    }
    forgetting_value = compute_forgetting_from_reference(
        reference_per_class_accuracy=reference_metrics["test_per_class_accuracy"],
        current_per_class_accuracy=test_per_class_accuracy,
    )
    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": str(removed_class_name),
        "final_num_classes": int(num_classes),
        "initialization": "imagenet",
        "backbone_mode": "frozen",
        "trainable_scope": "head_only",
        "selection_metric": "validation_loss",
        "max_epochs": int(args.epochs),
        "patience": None,
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "best_epoch": int(training_result["best_epoch"]),
        "epochs_ran": int(training_result["epochs_ran"]),
        "best_val_loss": float(training_result["best_val_loss"]),
        "best_val_accuracy": float(training_result["best_val_accuracy"]),
        "elapsed_seconds": float(elapsed),
        "test_overall_accuracy": float(test_overall_accuracy),
        "test_mean_per_class_accuracy": float(np.mean(per_class_accuracy)),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(filtered_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": total_examples_from_split_counts(split_counts, "train"),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "num_trainable_parameters": int(num_trainable_parameters),
        "additional_memory_required": 0.0,
        "forgetting_u_olvido": float(forgetting_value),
        "forgetting_reference_source": "full_training_reference_imagenet",
        "forgetting_reference_metrics_path": str(reference_metrics_path),
        "metricas_eliminacion": [metrica.nombre for metrica in METRICAS_ELIMINACION],
        "stores_model_checkpoint": False,
    }
    summary_row = build_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        removed_class_name=removed_class_name,
        final_num_classes=num_classes,
        status="completed",
        metrics_payload=metrics_payload,
    )

    save_experiment_artifacts(experiment_dir, training_result, metrics_payload)
    return summary_row


def run_all_experiments(args):
    """Run the complete frozen-backbone/head-only class-removal experiment."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {sorted(DATASET_LOADERS)}")

    selected_models = {
        model_name: IMAGENET_FROZEN_HEAD_MODEL_BUILDERS[model_name]
        for model_name in args.models
    }
    all_summary_rows = []
    base_output_dir = Path(args.output_dir)

    print(f"Datasets selected: {dataset_names}")
    print(f"Models selected: {list(selected_models)}")
    print("Initialization: ImageNet")
    print("Backbone mode: frozen")
    print("Trainable scope: head only")
    print(f"Training epochs: {args.epochs}")
    print("Selection metric for best checkpoint: validation loss")

    for dataset_name in dataset_names:
        print(f"\nRunning dataset: {dataset_name}")
        dataset_output_dir = base_output_dir / slugify(dataset_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []

        try:
            train_ds, val_ds, test_ds, original_classes = DATASET_LOADERS[dataset_name]()
            classes_to_remove = get_classes_to_remove(dataset_name, args.classes)
            print(f"Classes to remove: {classes_to_remove}")
        except Exception as exc:
            error_message = str(exc)
            print(f"[ERROR] Could not initialize dataset '{dataset_name}': {error_message}")
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "initialization": "imagenet",
                    "backbone_mode": "frozen",
                    "trainable_scope": "head_only",
                    "status": "failed_dataset_setup",
                    "error": error_message,
                }
            )
            save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
            write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
            all_summary_rows.extend(summary_rows)
            continue

        for class_to_remove in classes_to_remove:
            try:
                filtered_train, metadata = remove_class_and_remap(train_ds, original_classes, class_to_remove)
                filtered_val, _ = remove_class_and_remap(val_ds, original_classes, class_to_remove)
                filtered_test, _ = remove_class_and_remap(test_ds, original_classes, class_to_remove)
            except Exception as exc:
                error_message = str(exc)
                print(f"[ERROR] Could not prepare class removal '{class_to_remove}': {error_message}")
                for model_name in selected_models:
                    summary_rows.append(
                        build_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            removed_class_name=class_to_remove,
                            final_num_classes=max(len(original_classes) - 1, 0),
                            status="failed",
                            error_message=error_message,
                        )
                    )
                save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
                write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
                continue

            removed_class_name = metadata["removed_class_name"]
            filtered_classes = metadata["remaining_classes"]

            for model_name, model_builder in selected_models.items():
                try:
                    summary_row = run_single_experiment(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        model_builder=model_builder,
                        removed_class_name=removed_class_name,
                        filtered_train=filtered_train,
                        filtered_val=filtered_val,
                        filtered_test=filtered_test,
                        filtered_classes=filtered_classes,
                        args=args,
                        dataset_output_dir=dataset_output_dir,
                    )
                    summary_rows.append(summary_row)
                except Exception as exc:
                    error_message = str(exc)
                    experiment_dir = (
                        dataset_output_dir
                        / slugify(model_name)
                        / f"removed_{slugify(removed_class_name)}"
                    )
                    save_json(
                        experiment_dir / "error.json",
                        {
                            "dataset": dataset_name,
                            "model_name": model_name,
                            "removed_class": removed_class_name,
                            "initialization": "imagenet",
                            "backbone_mode": "frozen",
                            "trainable_scope": "head_only",
                            "error": error_message,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"[ERROR] {dataset_name} | {model_name} | remove={removed_class_name}: {error_message}")
                    summary_rows.append(
                        build_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            removed_class_name=removed_class_name,
                            final_num_classes=len(filtered_classes),
                            status="failed",
                            error_message=error_message,
                        )
                    )

                save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
                write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)

        save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
        write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
        all_summary_rows.extend(summary_rows)

    if all_summary_rows:
        save_json(base_output_dir / "experiments_summary.json", all_summary_rows)
        write_csv(base_output_dir / "experiments_summary.csv", all_summary_rows)

    print("\nFrozen-backbone head-only experiment completed.")
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    run_all_experiments(parse_args())

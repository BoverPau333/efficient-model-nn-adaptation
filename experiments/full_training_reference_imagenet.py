"""Entrena referencias completas con pesos ImageNet para comparar forgetting."""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from src.adaptation.class_removal_experiment_utils import total_examples_from_split_counts
from src.core.experiment_utils import (
    aggregate_split_counts,
    load_summary_from_metrics,
    save_experiment_artifacts,
)
from src.experiments_config.config import BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.dataset.loaders import DATASET_LOADERS
from src.metrics_elimination import METRICAS_ELIMINACION
from src.models import IMAGENET_MODEL_BUILDERS
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


DEFAULT_OUTPUT_DIR = RESULTS_DIR / "full_training_reference_imagenet"


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train full-class reference models from ImageNet initialization and "
            "store checkpoints plus metrics for later forgetness comparisons."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_LOADERS),
        default=None,
        help="Optional single dataset to run. If omitted, use --all-datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the reference training for every registered dataset.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(IMAGENET_MODEL_BUILDERS),
        default=list(IMAGENET_MODEL_BUILDERS.keys()),
        help="Subset of model names to train. Default: all registered ImageNet builders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Maximum number of training epochs. Default uses shared config.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional early stopping patience. If omitted, training runs for all epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate used during training.",
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
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if final metrics already exist.",
    )
    args = parser.parse_args()
    if not args.all_datasets and args.dataset is None:
        parser.error("Use --dataset <name> or --all-datasets.")
    return args


def derive_summary_metrics(metrics_payload=None):
    """Project reference-training outputs onto the baseline summary schema."""
    if metrics_payload is None:
        return {}

    epochs_ran = int(metrics_payload["epochs_ran"])
    elapsed_seconds = float(metrics_payload["elapsed_seconds"])
    return {
        "tiempo_total_de_adaptacion": elapsed_seconds,
        "tiempo_por_epoca": elapsed_seconds / max(epochs_ran, 1),
        "accuracy_global": float(metrics_payload["test_overall_accuracy"]),
        "accuracy_en_clases_restantes": float(metrics_payload["test_mean_per_class_accuracy"]),
        "forgetting_u_olvido": None,
        "numero_de_ejemplos_utilizados": int(metrics_payload["num_examples_used_for_adaptation"]),
        "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
        "numero_de_parametros_entrenados_o_modificados": int(
            metrics_payload["num_trainable_parameters"]
        ),
        "memoria_adicional_requerida": float(metrics_payload["additional_memory_required"]),
    }


def build_summary_row(dataset_name: str, model_name: str, status: str, metrics_payload=None, error_message=None):
    """Flatten the most useful fields for the aggregate CSV/JSON summary."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": "__all__",
        "final_num_classes": None,
        "status": status,
    }

    if metrics_payload is not None:
        derived_metrics = derive_summary_metrics(metrics_payload)
        row.update(
            {
                "removed_class": "__all__",
                "final_num_classes": int(metrics_payload["total_num_classes"]),
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
            }
        )
        row.update(derived_metrics)

    if error_message is not None:
        row["error"] = error_message

    return row


def load_existing_summary(metrics_path: Path):
    """Carga un resumen previo sin repetir la reconstruccion."""
    return load_summary_from_metrics(
        metrics_path,
        lambda existing_metrics: build_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            status="completed",
            metrics_payload=existing_metrics,
        ),
        status="skipped_existing",
    )


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    model_builder,
    train_ds,
    val_ds,
    test_ds,
    classes: list,
    args,
    dataset_output_dir: Path,
):
    """Train and evaluate one full-dataset reference model."""
    experiment_dir = dataset_output_dir / slugify(model_name)
    metrics_path = experiment_dir / "final_metrics.json"
    checkpoint_path = experiment_dir / "best_model.pt"
    error_path = experiment_dir / "error.json"

    if not args.overwrite and metrics_path.exists():
        print(f"[SKIP] {dataset_name} | {model_name}")
        return load_existing_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    set_seed(args.seed)
    train_loader = build_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Initialization: ImageNet")
    print(f"{'=' * 90}")

    model = model_builder(len(classes))
    num_trainable_parameters = count_trainable_parameters(model)

    t0 = time.time()
    training_result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    elapsed = time.time() - t0

    test_overall_accuracy, per_class_accuracy, confusion_matrix = evaluate(
        model,
        test_loader,
        len(classes),
    )

    split_counts = aggregate_split_counts(train_ds, val_ds, test_ds, classes)
    prediction_confidence_mean = evaluate_prediction_confidence(model, test_loader)
    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "training_scope": "all_classes",
        "initialization": "imagenet",
        "selection_metric": "validation_loss",
        "max_epochs": int(args.epochs),
        "patience": None if args.patience is None else int(args.patience),
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "total_num_classes": int(len(classes)),
        "best_epoch": int(training_result["best_epoch"]),
        "epochs_ran": int(training_result["epochs_ran"]),
        "best_val_loss": float(training_result["best_val_loss"]),
        "best_val_accuracy": float(training_result["best_val_accuracy"]),
        "elapsed_seconds": float(elapsed),
        "test_overall_accuracy": float(test_overall_accuracy),
        "test_mean_per_class_accuracy": float(np.mean(per_class_accuracy)),
        "test_per_class_accuracy": {
            class_name: float(per_class_accuracy[class_idx])
            for class_idx, class_name in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": total_examples_from_split_counts(split_counts, "train"),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "num_trainable_parameters": int(num_trainable_parameters),
        "additional_memory_required": 0.0,
        "forgetting_u_olvido": None,
        "metricas_eliminacion": [metrica.nombre for metrica in METRICAS_ELIMINACION],
        "stores_model_checkpoint": True,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_payload["summary"] = build_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        status="completed",
        metrics_payload=metrics_payload,
    )

    save_experiment_artifacts(experiment_dir, training_result, metrics_payload)
    return metrics_payload["summary"]


def run_all_experiments(args):
    """Run the full-class ImageNet-initialized reference training."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    selected_models = {model_name: IMAGENET_MODEL_BUILDERS[model_name] for model_name in args.models}
    all_summary_rows = []
    base_output_dir = Path(args.output_dir)

    print(f"Datasets selected: {dataset_names}")
    print(f"Models selected: {list(selected_models)}")
    print("Training scope: all classes")
    print("Initialization: ImageNet")
    print("Selection metric for best checkpoint: validation loss")

    for dataset_name in dataset_names:
        print(f"\nRunning dataset: {dataset_name}")
        dataset_output_dir = base_output_dir / slugify(dataset_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []

        try:
            train_ds, val_ds, test_ds, classes = DATASET_LOADERS[dataset_name]()
        except Exception as exc:
            error_message = str(exc)
            print(f"[ERROR] Could not initialize dataset '{dataset_name}': {error_message}")
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "training_scope": "all_classes",
                    "initialization": "imagenet",
                    "status": "failed_dataset_setup",
                    "error": error_message,
                }
            )
            save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
            write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
            all_summary_rows.extend(summary_rows)
            continue

        for model_name, model_builder in selected_models.items():
            try:
                summary_row = run_single_experiment(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    model_builder=model_builder,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                    classes=classes,
                    args=args,
                    dataset_output_dir=dataset_output_dir,
                )
                summary_rows.append(summary_row)
            except Exception as exc:
                error_message = str(exc)
                experiment_dir = dataset_output_dir / slugify(model_name)
                save_json(
                    experiment_dir / "error.json",
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "training_scope": "all_classes",
                        "initialization": "imagenet",
                        "error": error_message,
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"[ERROR] {dataset_name} | {model_name}: {error_message}")
                summary_rows.append(
                    build_summary_row(
                        dataset_name=dataset_name,
                        model_name=model_name,
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

    print("\nReference training completed.")
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    run_all_experiments(parse_args())

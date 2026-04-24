"""Baseline: retrain from scratch after removing one class from a dataset."""

import argparse
import csv
import json
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.experiments_config.class_removal_baseline_config import (
    CLASSES_TO_REMOVE_BY_DATASET,
    DEFAULT_DATASET,
)
from src.experiments_config.config import BATCH_SIZE, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class, remove_class_and_remap
from src.metrics_elimination import METRICAS_ELIMINACION
from src.models import MODEL_BUILDERS
from src.training import evaluate, train_with_early_stopping


DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 5


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain each project model after removing configured classes from a dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASET_LOADERS),
        help="Dataset to run the baseline on.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the baseline on every registered dataset in one execution.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_BUILDERS),
        default=list(MODEL_BUILDERS.keys()),
        help="Subset of model names to run. Default: all registered models.",
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
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience measured in epochs without validation-loss improvement.",
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
        default=str(RESULTS_DIR / "class_removal_baseline"),
        help="Base output directory for checkpoints, logs and summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if final metrics already exist.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def slugify(value) -> str:
    """Create filesystem-friendly names."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "item"


def parse_class_identifier(raw_value):
    """Parse a class identifier as int when possible, otherwise keep it as text."""
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    return raw_value


def get_classes_to_remove(dataset_name: str, override_classes=None):
    """Resolve the class list to use for a given dataset."""
    if override_classes:
        parsed = [parse_class_identifier(value) for value in override_classes]
        if not parsed:
            raise ValueError("The override class list is empty.")
        return parsed

    configured = CLASSES_TO_REMOVE_BY_DATASET.get(dataset_name)
    if not configured:
        raise ValueError(
            f"No classes configured for dataset '{dataset_name}' in "
            "src/experiments_config/class_removal_baseline_config.py"
        )
    return [parse_class_identifier(value) for value in configured]


def save_json(path: Path, payload):
    """Write JSON payloads with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list):
    """Write rows to CSV, preserving first-seen column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    """Construct a dataloader with a deterministic generator."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def count_trainable_parameters(model) -> int:
    """Count how many parameters are updated during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_prediction_confidence(model, loader) -> float:
    """Compute the mean max-softmax confidence over a dataloader."""
    model.eval()
    confidences = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(next(model.parameters()).device)
            probs = F.softmax(model(imgs), dim=1)
            batch_confidence = probs.max(dim=1).values
            confidences.extend(batch_confidence.cpu().numpy().tolist())

    if not confidences:
        return 0.0
    return float(np.mean(confidences))


def total_examples_from_split_counts(split_counts: dict, split_name: str = "train") -> int:
    """Return the total number of examples used for a given split."""
    return int(sum(split_counts.get(split_name, {}).values()))


def derive_summary_metrics(metrics_payload=None):
    """Project experiment outputs onto the elimination-metrics summary schema."""
    if metrics_payload is None:
        return {}

    per_class_accuracy = metrics_payload.get("test_per_class_accuracy", {})
    forgetting_value = metrics_payload.get("forgetting_u_olvido")
    metrics = {
        "tiempo_total_de_adaptacion": float(metrics_payload["elapsed_seconds"]),
        "accuracy_global": float(metrics_payload["test_overall_accuracy"]),
        "accuracy_por_clase": json.dumps(per_class_accuracy, ensure_ascii=True, sort_keys=True),
        "accuracy_en_clases_restantes": float(metrics_payload["test_mean_per_class_accuracy"]),
        "forgetting_u_olvido": None if forgetting_value is None else float(forgetting_value),
        "numero_de_ejemplos_utilizados": int(
            metrics_payload["num_examples_used_for_adaptation"]
        ),
        "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
        "numero_de_parametros_entrenados_o_modificados": int(
            metrics_payload["num_trainable_parameters"]
        ),
        "memoria_adicional_requerida": float(metrics_payload["additional_memory_required"]),
    }
    metrics.update(
        {
            "Tiempo total de adaptacion": metrics["tiempo_total_de_adaptacion"],
            "Accuracy global": metrics["accuracy_global"],
            "Accuracy por clase": metrics["accuracy_por_clase"],
            "Accuracy en clases restantes": metrics["accuracy_en_clases_restantes"],
            "Forgetting u olvido": metrics["forgetting_u_olvido"],
            "Numero de ejemplos utilizados": metrics["numero_de_ejemplos_utilizados"],
            "Confianza de prediccion": metrics["confianza_de_prediccion"],
            "Numero de parametros entrenados o modificados": (
                metrics["numero_de_parametros_entrenados_o_modificados"]
            ),
            "Memoria adicional requerida": metrics["memoria_adicional_requerida"],
        }
    )
    return metrics


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
        "status": status,
    }

    if metrics_payload is not None:
        derived_metrics = derive_summary_metrics(metrics_payload)
        row.update(
            {
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
                "test_overall_accuracy": float(metrics_payload["test_overall_accuracy"]),
                "test_mean_per_class_accuracy": float(metrics_payload["test_mean_per_class_accuracy"]),
                "elapsed_seconds": float(metrics_payload["elapsed_seconds"]),
            }
        )
        row.update(derived_metrics)

    if error_message is not None:
        row["error"] = error_message

    return row


def save_experiment_artifacts(experiment_dir: Path, training_result: dict, metrics_payload: dict):
    """Persist checkpoint metadata, logs and final metrics."""
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
    """Train and evaluate one model/class-removal combination."""
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

    set_seed(args.seed)
    train_loader = build_loader(filtered_train, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(filtered_val, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(filtered_test, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Removed class: {removed_class_name}")
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
        patience=args.patience,
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
    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": str(removed_class_name),
        "final_num_classes": int(num_classes),
        "selection_metric": "validation_loss",
        "max_epochs": int(args.epochs),
        "patience": int(args.patience),
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
        "test_per_class_accuracy": {
            class_name: float(per_class_accuracy[class_idx])
            for class_idx, class_name in enumerate(filtered_classes)
        },
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(filtered_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": total_examples_from_split_counts(split_counts, "train"),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "num_trainable_parameters": int(num_trainable_parameters),
        "additional_memory_required": 0.0,
        "forgetting_u_olvido": None,
        "metricas_eliminacion": [metrica.nombre for metrica in METRICAS_ELIMINACION],
        "stores_model_checkpoint": False,
    }
    metrics_payload["summary"] = build_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        removed_class_name=removed_class_name,
        final_num_classes=num_classes,
        status="completed",
        metrics_payload=metrics_payload,
    )

    save_experiment_artifacts(experiment_dir, training_result, metrics_payload)
    return metrics_payload["summary"]


def run_all_experiments(args):
    """Run the complete class-removal baseline for one or all datasets."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {sorted(DATASET_LOADERS)}")

    selected_models = {model_name: MODEL_BUILDERS[model_name] for model_name in args.models}
    all_summary_rows = []
    base_output_dir = Path(args.output_dir)

    print(f"Datasets selected: {dataset_names}")
    print(f"Models selected: {list(selected_models)}")
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

    print("\nBaseline completed.")
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    run_all_experiments(parse_args())

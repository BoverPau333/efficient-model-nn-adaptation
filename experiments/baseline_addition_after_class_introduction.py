"""Baseline de adicion: referencia de 9 clases y reentrenamiento desde cero a 10 clases."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptation.class_addition_experiment_utils import (
    build_addition_metrics_payload,
    build_addition_summary_row,
    get_classes_to_add,
    zero_model_parameters,
)
from src.adaptation.class_removal_experiment_utils import total_examples_from_split_counts
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
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class, remove_class_and_remap, resolve_class_to_remove
from src.experiments_config.class_to_add import DEFAULT_DATASET
from src.experiments_config.config import BATCH_SIZE, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.models import IMAGENET_MODEL_BUILDERS, MODEL_BUILDERS


DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 5
DEFAULT_REFERENCE_OUTPUT_DIR = RESULTS_DIR / "full_training_reference_add"
DEFAULT_BASELINE_OUTPUT_DIR = RESULTS_DIR / "class_addition_baseline"


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a 9-class ImageNet-initialized reference model and then retrain "
            "from scratch to reintroduce the held-out class."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASET_LOADERS),
        help="Dataset to run the addition baseline on.",
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
        help="Optional override for classes to add later. Use class names or integer indices.",
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
        help="Learning rate used during the baseline retraining stage.",
    )
    parser.add_argument(
        "--reference-lr",
        type=float,
        default=LR,
        help="Learning rate used during the 9-class ImageNet reference stage.",
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
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_OUTPUT_DIR),
        help="Base output directory for the 9-class ImageNet references.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_BASELINE_OUTPUT_DIR),
        help="Base output directory for the class-addition baseline results.",
    )
    parser.add_argument(
        "--zero-init",
        action="store_true",
        help=(
            "If set, zero every parameter before baseline retraining. By default "
            "the baseline retrains from standard random initialization."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if artifacts already exist.",
    )
    return parser.parse_args()


def aggregate_counts(train_ds, val_ds, test_ds, classes: list):
    """Count examples per class for each split."""
    return {
        "train": count_examples_per_class(train_ds, classes),
        "val": count_examples_per_class(val_ds, classes),
        "test": count_examples_per_class(test_ds, classes),
    }


def save_phase_artifacts(experiment_dir: Path, training_result: dict, metrics_payload: dict):
    """Persist logs and final metrics for one phase."""
    save_json(experiment_dir / "training_history.json", training_result["history"])
    write_csv(experiment_dir / "training_history.csv", training_result["history"])
    save_json(experiment_dir / "final_metrics.json", metrics_payload)


def load_existing_summary(metrics_path: Path):
    """Load the flattened summary from an existing finished experiment."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = build_addition_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            added_class_name=existing_metrics["added_class"],
            final_num_classes=existing_metrics["final_num_classes"],
            status="completed",
            metrics_payload=existing_metrics,
        )
    summary["status"] = "skipped_existing"
    return summary


def load_completed_summary(metrics_path: Path):
    """Load the flattened summary from a finished experiment without altering its status."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = build_addition_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            added_class_name=existing_metrics["added_class"],
            final_num_classes=existing_metrics["final_num_classes"],
            status="completed",
            metrics_payload=existing_metrics,
        )
    return summary


def collect_dataset_summary_rows(dataset_output_dir: Path):
    """Rebuild one dataset summary by scanning persisted experiment artifacts."""
    rows_by_key = {}

    for metrics_path in sorted(dataset_output_dir.rglob("final_metrics.json")):
        if not metrics_path.parent.name.startswith("added_"):
            continue
        summary = load_completed_summary(metrics_path)
        key = (summary.get("model_name"), summary.get("added_class"))
        rows_by_key[key] = summary

    for error_path in sorted(dataset_output_dir.rglob("error.json")):
        if not error_path.parent.name.startswith("added_"):
            continue
        error_payload = load_json(error_path)
        key = (error_payload.get("model_name"), error_payload.get("added_class"))
        if key in rows_by_key:
            continue
        rows_by_key[key] = build_addition_summary_row(
            dataset_name=error_payload["dataset"],
            model_name=error_payload["model_name"],
            added_class_name=error_payload["added_class"],
            final_num_classes=error_payload.get("final_num_classes", 0),
            status="failed",
            error_message=error_payload.get("error"),
        )

    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (
                str(item[0] or ""),
                str(item[1] or ""),
            ),
        )
    ]


def rebuild_summary_files(base_output_dir: Path, dataset_names: list):
    """Recompute dataset and global summaries from persisted experiment outputs."""
    all_summary_rows = []
    requested_dataset_dirs = {base_output_dir / slugify(dataset_name) for dataset_name in dataset_names}
    existing_dataset_dirs = {
        path
        for path in base_output_dir.iterdir()
        if path.is_dir()
    } if base_output_dir.exists() else set()

    for dataset_output_dir in sorted(requested_dataset_dirs | existing_dataset_dirs):
        if not dataset_output_dir.exists():
            continue
        summary_rows = collect_dataset_summary_rows(dataset_output_dir)
        save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
        write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
        all_summary_rows.extend(summary_rows)

    save_json(base_output_dir / "all_experiments_summary.json", all_summary_rows)
    write_csv(base_output_dir / "all_experiments_summary.csv", all_summary_rows)
    return all_summary_rows


def train_nine_class_reference(
    dataset_name: str,
    model_name: str,
    added_class,
    original_classes: list,
    train_ds,
    val_ds,
    test_ds,
    args,
):
    """Train or reuse the 9-class ImageNet reference that omits the future added class."""
    reference_root = Path(args.reference_dir) / slugify(dataset_name) / slugify(model_name) / slugify(added_class)
    metrics_path = reference_root / "final_metrics.json"
    checkpoint_path = reference_root / "modelo_base_9_clases.pth"
    error_path = reference_root / "error.json"

    filtered_train, metadata = remove_class_and_remap(train_ds, original_classes, added_class)
    filtered_val, _ = remove_class_and_remap(val_ds, original_classes, added_class)
    filtered_test, _ = remove_class_and_remap(test_ds, original_classes, added_class)
    filtered_classes = metadata["remaining_classes"]
    added_class_name = metadata["removed_class_name"]

    if not args.overwrite and metrics_path.exists() and checkpoint_path.exists():
        existing_metrics = load_json(metrics_path)
        return existing_metrics, checkpoint_path, filtered_classes, added_class_name

    reference_root.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    set_seed(args.seed)
    train_loader = build_loader(filtered_train, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(filtered_val, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(filtered_test, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Omitted class for reference: {added_class_name}")
    print(f"{'=' * 90}")

    model = IMAGENET_MODEL_BUILDERS[model_name](len(filtered_classes))
    num_trainable_parameters = count_trainable_parameters(model)

    t0 = time.time()
    training_result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.reference_lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    elapsed = time.time() - t0

    test_overall_accuracy, per_class_accuracy, confusion_matrix = evaluate(
        model,
        test_loader,
        len(filtered_classes),
    )
    split_counts = aggregate_counts(filtered_train, filtered_val, filtered_test, filtered_classes)
    prediction_confidence_mean = evaluate_prediction_confidence(model, test_loader)
    test_per_class_accuracy = {
        class_name: float(per_class_accuracy[class_idx])
        for class_idx, class_name in enumerate(filtered_classes)
    }

    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "omitted_for_later_addition_class": str(added_class_name),
        "final_num_classes": int(len(filtered_classes)),
        "training_scope": "all_classes_except_future_addition",
        "initialization": "imagenet",
        "selection_metric": "validation_loss",
        "max_epochs": int(args.epochs),
        "patience": int(args.patience),
        "learning_rate": float(args.reference_lr),
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
        "test_per_class_accuracy": test_per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(filtered_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": total_examples_from_split_counts(split_counts, "train"),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "num_trainable_parameters": int(num_trainable_parameters),
        "additional_memory_required": 0.0,
        "stores_model_checkpoint": True,
        "checkpoint_path": str(checkpoint_path),
    }

    save_phase_artifacts(reference_root, training_result, metrics_payload)
    return metrics_payload, checkpoint_path, filtered_classes, added_class_name


def run_single_addition_baseline(
    dataset_name: str,
    model_name: str,
    added_class,
    original_classes: list,
    train_ds,
    val_ds,
    test_ds,
    args,
    dataset_output_dir: Path,
):
    """Run the baseline retraining stage that reintroduces the omitted class."""
    reference_metrics, reference_checkpoint_path, previous_classes, added_class_name = train_nine_class_reference(
        dataset_name=dataset_name,
        model_name=model_name,
        added_class=added_class,
        original_classes=original_classes,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        args=args,
    )

    experiment_dir = dataset_output_dir / slugify(model_name) / f"added_{slugify(added_class_name)}"
    metrics_path = experiment_dir / "final_metrics.json"
    error_path = experiment_dir / "error.json"

    if not args.overwrite and metrics_path.exists():
        print(f"[SKIP] {dataset_name} | {model_name} | add={added_class_name}")
        return load_existing_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    set_seed(args.seed)
    train_loader = build_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Added class: {added_class_name}")
    print(f"{'=' * 90}")

    model = MODEL_BUILDERS[model_name](len(original_classes))
    if args.zero_init:
        zero_model_parameters(model)
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
        len(original_classes),
    )
    split_counts = aggregate_counts(train_ds, val_ds, test_ds, original_classes)
    prediction_confidence_mean = evaluate_prediction_confidence(model, test_loader)
    metrics_payload = build_addition_metrics_payload(
        dataset_name=dataset_name,
        model_name=model_name,
        added_class_name=added_class_name,
        original_classes=original_classes,
        train_ds=train_ds,
        test_ds=test_ds,
        args=args,
        training_result=training_result,
        elapsed=elapsed,
        test_overall_accuracy=test_overall_accuracy,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion_matrix,
        split_counts=split_counts,
        prediction_confidence_mean=prediction_confidence_mean,
        num_trainable_parameters=num_trainable_parameters,
        model=model,
        build_loader_fn=build_loader,
        reference_metrics=reference_metrics,
        reference_metrics_path=Path(args.reference_dir) / slugify(dataset_name) / slugify(model_name) / slugify(added_class_name) / "final_metrics.json",
        reference_checkpoint_path=reference_checkpoint_path,
    )

    save_phase_artifacts(experiment_dir, training_result, metrics_payload)
    return metrics_payload["summary"]


def run_all_experiments(args):
    """Run the complete class-addition baseline for one or all datasets."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {sorted(DATASET_LOADERS)}")

    selected_models = {model_name: MODEL_BUILDERS[model_name] for model_name in args.models}
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
            classes_to_add = get_classes_to_add(dataset_name, args.classes)
            print(f"Classes to add later: {classes_to_add}")
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
            continue

        for class_to_add in classes_to_add:
            try:
                _, added_class_name = resolve_class_to_remove(original_classes, class_to_add)
            except Exception as exc:
                error_message = str(exc)
                print(f"[ERROR] Could not prepare class addition '{class_to_add}': {error_message}")
                for model_name in selected_models:
                    summary_rows.append(
                        build_addition_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            added_class_name=class_to_add,
                            final_num_classes=len(original_classes),
                            status="failed",
                            error_message=error_message,
                        )
                    )
                save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
                write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
                continue

            for model_name in selected_models:
                try:
                    summary_row = run_single_addition_baseline(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        added_class=added_class_name,
                        original_classes=original_classes,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        test_ds=test_ds,
                        args=args,
                        dataset_output_dir=dataset_output_dir,
                    )
                    summary_rows.append(summary_row)
                except Exception as exc:
                    error_message = str(exc)
                    experiment_dir = (
                        dataset_output_dir
                        / slugify(model_name)
                        / f"added_{slugify(added_class_name)}"
                    )
                    save_json(
                        experiment_dir / "error.json",
                        {
                            "dataset": dataset_name,
                            "model_name": model_name,
                            "added_class": added_class_name,
                            "error": error_message,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"[ERROR] {dataset_name} | {model_name} | add={added_class_name}: {error_message}")
                    summary_rows.append(
                        build_addition_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            added_class_name=added_class_name,
                            final_num_classes=len(original_classes),
                            status="failed",
                            error_message=error_message,
                        )
                    )

                save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
                write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)

    return rebuild_summary_files(base_output_dir, dataset_names)


def main():
    args = parse_args()
    run_all_experiments(args)


if __name__ == "__main__":
    main()

"""Class-addition experiment initialized from a 9-class reference checkpoint."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptation.class_addition_experiment_utils import (
    build_addition_finetuning_summary_row,
    compute_previous_class_forgetting,
    f1_from_confusion_matrix,
    get_classes_to_add,
    load_addition_reference_artifacts,
    precision_from_confusion_matrix,
    prediction_confidence_for_single_class,
)
from src.adaptation.class_removal_experiment_utils import (
    format_percentage_slug,
    select_training_subset,
    total_examples_from_split_counts,
)
from src.adaptation.finetuning_schedule_utils import (
    resolve_finetuning_training_setup,
    run_finetuning_schedule,
)
from src.core.results_utils import (
    add_output_class,
    build_loader,
    count_trainable_parameters,
    evaluate_prediction_confidence,
    freeze_backbone_keep_head_trainable,
    load_json,
    save_json,
    set_seed,
    slugify,
    write_csv,
)
from src.core.training import evaluate
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class, remove_class_and_remap, resolve_class_to_remove
from src.experiments_config.class_to_add import DEFAULT_DATASET
from src.experiments_config.config import BATCH_SIZE, DEVICE, LR, NUM_WORKERS, RESULTS_DIR, SEED
from src.metrics_addition import METRICAS_ADICION
from src.models import MODEL_BUILDERS


DEFAULT_HEAD_ONLY_EPOCHS = 5
DEFAULT_FINETUNING_HEAD_EPOCHS = 5
DEFAULT_FINETUNING_UNFROZEN_EPOCHS = 10
DEFAULT_REFERENCE_OUTPUT_DIR = RESULTS_DIR / "full_training_reference_add"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "class_addition_finetuning"
DEFAULT_DATASET_PERCENTAGE = 100.0


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reintroduce a held-out class by loading a 9-class reference checkpoint, "
            "expanding the classification head and fine-tuning on a percentage of the "
            "full training split."
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
        choices=sorted(MODEL_BUILDERS),
        default=list(MODEL_BUILDERS.keys()),
        help="Subset of model names to run. Default: all registered model builders.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional override for classes to add. Use class names or integer indices.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_HEAD_ONLY_EPOCHS,
        help="Number of training epochs when using head-only retraining.",
    )
    parser.add_argument(
        "--two-stage-finetuning",
        action="store_true",
        help=(
            "Run two-stage fine-tuning: first train 5 epochs with the backbone frozen "
            "and then 10 epochs with the whole model unfrozen."
        ),
    )
    parser.add_argument(
        "--frozen-epochs",
        type=int,
        default=DEFAULT_FINETUNING_HEAD_EPOCHS,
        help="Head-only epochs to run before unfreezing the full model in two-stage fine-tuning.",
    )
    parser.add_argument(
        "--unfrozen-epochs",
        type=int,
        default=DEFAULT_FINETUNING_UNFROZEN_EPOCHS,
        help="Full-model epochs to run after unfreezing in two-stage fine-tuning.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate used during fine-tuning.",
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
        help="Directory containing the 9-class reference checkpoints used as initialization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if final metrics already exist.",
    )
    parser.add_argument(
        "--porc",
        type=float,
        default=DEFAULT_DATASET_PERCENTAGE,
        help="Percentage of the full training split to use for adaptation (0, 100].",
    )
    return parser.parse_args()


def aggregate_counts(train_ds, val_ds, test_ds, classes: list):
    """Count examples per class for each split."""
    return {
        "train": count_examples_per_class(train_ds, classes),
        "val": count_examples_per_class(val_ds, classes),
        "test": count_examples_per_class(test_ds, classes),
    }


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
        summary = build_addition_finetuning_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            added_class_name=existing_metrics["added_class"],
            final_num_classes=existing_metrics["final_num_classes"],
            status="completed",
            training_setup={
                "mode_label": existing_metrics.get("training_mode", "head_only"),
                "backbone_mode": existing_metrics.get("backbone_mode", "frozen"),
                "trainable_scope": existing_metrics.get("trainable_scope", "head_only"),
                "train_percentage": existing_metrics.get("train_percentage", 100.0),
            },
            metrics_payload=existing_metrics,
        )
    summary["status"] = "skipped_existing"
    return summary


def load_completed_summary(metrics_path: Path):
    """Load the flattened summary from a finished experiment without altering its status."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = build_addition_finetuning_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            added_class_name=existing_metrics["added_class"],
            final_num_classes=existing_metrics["final_num_classes"],
            status="completed",
            training_setup={
                "mode_label": existing_metrics.get("training_mode", "head_only"),
                "backbone_mode": existing_metrics.get("backbone_mode", "frozen"),
                "trainable_scope": existing_metrics.get("trainable_scope", "head_only"),
                "train_percentage": existing_metrics.get("train_percentage", 100.0),
            },
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
        key = (
            summary.get("model_name"),
            summary.get("training_mode"),
            summary.get("train_percentage"),
            summary.get("added_class"),
        )
        rows_by_key[key] = summary

    for error_path in sorted(dataset_output_dir.rglob("error.json")):
        if not error_path.parent.name.startswith("added_"):
            continue
        error_payload = load_json(error_path)
        key = (
            error_payload.get("model_name"),
            error_payload.get("training_mode"),
            error_payload.get("train_percentage"),
            error_payload.get("added_class"),
        )
        if key in rows_by_key:
            continue
        rows_by_key[key] = build_addition_finetuning_summary_row(
            dataset_name=error_payload["dataset"],
            model_name=error_payload["model_name"],
            added_class_name=error_payload["added_class"],
            final_num_classes=error_payload.get("final_num_classes", 0),
            status="failed",
            training_setup={
                "mode_label": error_payload.get("training_mode", "head_only"),
                "backbone_mode": error_payload.get("backbone_mode", "frozen"),
                "trainable_scope": error_payload.get("trainable_scope", "head_only"),
                "train_percentage": error_payload.get("train_percentage", 100.0),
            },
            error_message=error_payload.get("error"),
        )

    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (
                str(item[0] or ""),
                str(item[1] or ""),
                float(item[2] or 0.0),
                str(item[3] or ""),
            ),
        )
    ]


def rebuild_summary_files(base_output_dir: Path, dataset_names: list):
    """Recompute dataset and global summaries from persisted experiment outputs."""
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
        summary_rows = collect_dataset_summary_rows(dataset_output_dir)
        save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
        write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
        all_summary_rows.extend(summary_rows)

    save_json(base_output_dir / "all_experiments_summary.json", all_summary_rows)
    write_csv(base_output_dir / "all_experiments_summary.csv", all_summary_rows)
    return all_summary_rows


def initialize_model_from_reference(
    *,
    model_name: str,
    reference_checkpoint_path: Path,
    filtered_num_classes: int,
    added_class_idx: int,
):
    """Load a 9-class checkpoint, expand its head and freeze the backbone."""
    model = MODEL_BUILDERS[model_name](filtered_num_classes)
    checkpoint = torch.load(reference_checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    add_output_class(model, added_class_idx)
    freeze_backbone_keep_head_trainable(model)
    return model


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    added_class_name,
    added_class_idx: int,
    train_ds,
    val_ds,
    test_ds,
    original_classes: list,
    filtered_classes: list,
    args,
    dataset_output_dir: Path,
):
    """Train and evaluate one class-addition fine-tuning experiment."""
    training_setup = resolve_finetuning_training_setup(
        two_stage_finetuning=args.two_stage_finetuning,
        train_percentage=args.porc,
        head_only_epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        unfrozen_epochs=args.unfrozen_epochs,
    )
    experiment_dir = (
        dataset_output_dir
        / slugify(model_name)
        / training_setup["mode_dirname"]
        / format_percentage_slug(args.porc)
        / f"added_{slugify(added_class_name)}"
    )
    metrics_path = experiment_dir / "final_metrics.json"
    error_path = experiment_dir / "error.json"

    if not args.overwrite and metrics_path.exists():
        print(f"[SKIP] {dataset_name} | {model_name} | add={added_class_name} | mode={training_setup['mode_label']}")
        return load_existing_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    reference_metrics, reference_metrics_path, reference_checkpoint_path = load_addition_reference_artifacts(
        reference_dir=Path(args.reference_dir),
        dataset_name=dataset_name,
        model_name=model_name,
        added_class_name=added_class_name,
    )

    set_seed(args.seed)
    sampled_train = select_training_subset(train_ds, args.porc, args.seed)
    train_loader = build_loader(sampled_train, args.batch_size, True, args.num_workers, args.seed)
    val_loader = build_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)
    test_loader = build_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    print(f"\n{'=' * 90}")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Added class: {added_class_name}")
    print(
        "Initialization: 9-class reference checkpoint | "
        f"Backbone: {training_setup['backbone_mode']} | "
        f"Trainable scope: {training_setup['trainable_scope']}"
    )
    print(f"Training schedule: {training_setup['description']}")
    print(f"Training split used: {args.porc:g}% ({len(sampled_train)}/{len(train_ds)} examples)")
    print(f"{'=' * 90}")

    model = initialize_model_from_reference(
        model_name=model_name,
        reference_checkpoint_path=reference_checkpoint_path,
        filtered_num_classes=len(filtered_classes),
        added_class_idx=added_class_idx,
    )
    num_trainable_parameters_before = count_trainable_parameters(model)

    t0 = time.time()
    training_result = run_finetuning_schedule(
        model,
        train_loader,
        val_loader,
        two_stage_finetuning=args.two_stage_finetuning,
        head_only_epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        unfrozen_epochs=args.unfrozen_epochs,
        lr=args.lr,
        verbose=True,
    )
    elapsed = time.time() - t0
    num_trainable_parameters_after = count_trainable_parameters(model)

    test_overall_accuracy, per_class_accuracy, confusion_matrix = evaluate(
        model,
        test_loader,
        len(original_classes),
    )
    split_counts = aggregate_counts(sampled_train, val_ds, test_ds, original_classes)
    prediction_confidence_mean = evaluate_prediction_confidence(model, test_loader)
    test_per_class_accuracy = {
        class_name: float(per_class_accuracy[class_idx])
        for class_idx, class_name in enumerate(original_classes)
    }
    current_previous_class_accuracy = {
        class_name: accuracy
        for class_name, accuracy in test_per_class_accuracy.items()
        if class_name != str(added_class_name)
    }
    forgetting_previous_classes = compute_previous_class_forgetting(
        reference_per_class_accuracy=reference_metrics["test_per_class_accuracy"],
        current_per_class_accuracy=test_per_class_accuracy,
        added_class_name=added_class_name,
    )
    prediction_confidence_added_class_mean = float(
        prediction_confidence_for_single_class(
            model=model,
            dataset=test_ds,
            class_idx=added_class_idx,
            build_loader_fn=build_loader,
            args=args,
        )
    )

    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "added_class": str(added_class_name),
        "final_num_classes": int(len(original_classes)),
        "initialization": "reference_without_added_class",
        "training_mode": training_setup["mode_label"],
        "backbone_mode": training_setup["backbone_mode"],
        "trainable_scope": training_setup["trainable_scope"],
        "train_percentage": float(args.porc),
        "selection_metric": "validation_loss",
        "max_epochs": int(training_setup["max_epochs"]),
        "head_only_epochs": int(training_setup["head_epochs"]),
        "full_model_epochs": int(training_setup["full_model_epochs"]),
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
        "test_per_class_accuracy": test_per_class_accuracy,
        "test_mean_per_class_accuracy": float(np.mean(per_class_accuracy)),
        "test_accuracy_previous_classes": float(np.mean(list(current_previous_class_accuracy.values()))),
        "test_accuracy_added_class": float(test_per_class_accuracy[str(added_class_name)]),
        "test_precision_added_class": precision_from_confusion_matrix(confusion_matrix, added_class_idx),
        "test_recall_added_class": float(per_class_accuracy[added_class_idx]),
        "test_f1_added_class": f1_from_confusion_matrix(confusion_matrix, added_class_idx),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(original_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": total_examples_from_split_counts(split_counts, "train"),
        "num_examples_added_class_train": int(split_counts["train"].get(str(added_class_name), 0)),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "prediction_confidence_added_class_mean": float(prediction_confidence_added_class_mean),
        "num_trainable_parameters": int(num_trainable_parameters_after),
        "num_trainable_parameters_before_training": int(num_trainable_parameters_before),
        "additional_memory_required": 0.0,
        "full_train_examples": int(len(train_ds)),
        "forgetting_previous_classes": None
        if forgetting_previous_classes is None
        else float(forgetting_previous_classes),
        "reference_9_class_metrics_path": str(reference_metrics_path),
        "reference_9_class_checkpoint_path": str(reference_checkpoint_path),
        "reference_9_class_source": "full_training_reference_add",
        "metricas_adicion": [metrica.nombre for metrica in METRICAS_ADICION],
        "stores_model_checkpoint": False,
    }
    summary_row = build_addition_finetuning_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        added_class_name=added_class_name,
        final_num_classes=len(original_classes),
        status="completed",
        training_setup=training_setup,
        metrics_payload=metrics_payload,
    )
    metrics_payload["summary"] = summary_row

    save_experiment_artifacts(experiment_dir, training_result, metrics_payload)
    return summary_row


def run_all_experiments(args):
    """Run the complete class-addition fine-tuning experiment."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {sorted(DATASET_LOADERS)}")

    selected_models = {
        model_name: MODEL_BUILDERS[model_name]
        for model_name in args.models
    }
    base_output_dir = Path(args.output_dir)
    training_setup = resolve_finetuning_training_setup(
        two_stage_finetuning=args.two_stage_finetuning,
        train_percentage=args.porc,
        head_only_epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        unfrozen_epochs=args.unfrozen_epochs,
    )

    print(f"Datasets selected: {dataset_names}")
    print(f"Models selected: {list(selected_models)}")
    print("Initialization: 9-class reference checkpoint")
    print(f"Training mode: {training_setup['mode_label']}")
    print(f"Backbone mode: {training_setup['backbone_mode']}")
    print(f"Trainable scope: {training_setup['trainable_scope']}")
    print(f"Training schedule: {training_setup['description']}")
    print(f"Training split percentage: {args.porc:g}%")
    print("Selection metric for best checkpoint: validation loss")

    for dataset_name in dataset_names:
        print(f"\nRunning dataset: {dataset_name}")
        dataset_output_dir = base_output_dir / slugify(dataset_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []

        try:
            train_ds, val_ds, test_ds, original_classes = DATASET_LOADERS[dataset_name]()
            classes_to_add = get_classes_to_add(dataset_name, args.classes)
            print(f"Classes to add: {classes_to_add}")
        except Exception as exc:
            error_message = str(exc)
            print(f"[ERROR] Could not initialize dataset '{dataset_name}': {error_message}")
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "initialization": "reference_without_added_class",
                    "training_mode": training_setup["mode_label"],
                    "backbone_mode": training_setup["backbone_mode"],
                    "trainable_scope": training_setup["trainable_scope"],
                    "train_percentage": float(args.porc),
                    "status": "failed_dataset_setup",
                    "error": error_message,
                }
            )
            save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
            write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
            continue

        for class_to_add in classes_to_add:
            try:
                added_class_idx, added_class_name = resolve_class_to_remove(original_classes, class_to_add)
                _, metadata = remove_class_and_remap(train_ds, original_classes, added_class_name)
                filtered_classes = metadata["remaining_classes"]
            except Exception as exc:
                error_message = str(exc)
                print(f"[ERROR] Could not prepare class addition '{class_to_add}': {error_message}")
                for model_name in selected_models:
                    summary_rows.append(
                        build_addition_finetuning_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            added_class_name=class_to_add,
                            final_num_classes=len(original_classes),
                            status="failed",
                            training_setup=training_setup,
                            error_message=error_message,
                        )
                    )
                save_json(dataset_output_dir / "experiments_summary.json", summary_rows)
                write_csv(dataset_output_dir / "experiments_summary.csv", summary_rows)
                continue

            for model_name in selected_models:
                try:
                    summary_row = run_single_experiment(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        added_class_name=added_class_name,
                        added_class_idx=added_class_idx,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        test_ds=test_ds,
                        original_classes=original_classes,
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
                        / training_setup["mode_dirname"]
                        / format_percentage_slug(args.porc)
                        / f"added_{slugify(added_class_name)}"
                    )
                    save_json(
                        experiment_dir / "error.json",
                        {
                            "dataset": dataset_name,
                            "model_name": model_name,
                            "added_class": added_class_name,
                            "final_num_classes": len(original_classes),
                            "initialization": "reference_without_added_class",
                            "training_mode": training_setup["mode_label"],
                            "backbone_mode": training_setup["backbone_mode"],
                            "trainable_scope": training_setup["trainable_scope"],
                            "train_percentage": float(args.porc),
                            "error": error_message,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"[ERROR] {dataset_name} | {model_name} | add={added_class_name}: {error_message}")
                    summary_rows.append(
                        build_addition_finetuning_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            added_class_name=added_class_name,
                            final_num_classes=len(original_classes),
                            status="failed",
                            training_setup=training_setup,
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

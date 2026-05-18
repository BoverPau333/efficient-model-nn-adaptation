"""Adaptacion few-shot con Prototypical Networks tras modificar el conjunto de clases."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptation.class_removal_experiment_utils import get_classes_to_remove
from src.dataset.loaders import DATASET_LOADERS
from src.dataset.utils import count_examples_per_class, remove_class_and_remap
from src.core.embedding_utils import IndexedDataset, extract_embeddings
from src.adaptation.episode_sampler import sample_k_shot_support_set
from src.experiments_config.class_removal_baseline_config import DEFAULT_DATASET
from src.experiments_config.config import BATCH_SIZE, NUM_WORKERS, RESULTS_DIR, SEED
from src.metrics_elimination import METRICAS_ELIMINACION
from src.models import IMAGENET_MODEL_BUILDERS
from src.adaptation.prototypical_utils import (
    build_class_prototypes,
    evaluate_prototypical_predictions,
    serialize_prototypes,
)
from src.core.results_utils import (
    build_loader,
    compute_forgetting_from_reference,
    load_json,
    load_reference_model,
    parse_class_identifier,
    save_json,
    set_seed,
    slugify,
    write_csv,
)


DEFAULT_REFERENCE_OUTPUT_DIR = RESULTS_DIR / "full_training_reference_imagenet"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "class_removal_prototypical_fewshot"


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a pretrained ImageNet-initialized model after a class-set change "
            "using few-shot Prototypical Networks over frozen embeddings."
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
        choices=sorted(IMAGENET_MODEL_BUILDERS),
        default=list(IMAGENET_MODEL_BUILDERS.keys()),
        help="Subset of model names to run. Default: all registered ImageNet builders.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Classes to modify. For remove, accepts names or integer indices.",
    )
    parser.add_argument(
        "--update-type",
        choices=["remove", "add"],
        default="remove",
        help="Class-set modification to evaluate. 'add' is scaffolded but not yet supported by local datasets.",
    )
    parser.add_argument(
        "--shots-per-class",
        type=int,
        default=5,
        help="Number of support examples per active class.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine"],
        default="cosine",
        help="Distance used for prototype classification. Reuses src.core.distancias.distancia_coseno.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for support/validation/test embedding extraction.",
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
        help="Base output directory for metrics, support sets and summaries.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_OUTPUT_DIR),
        help="Directory containing the full-class ImageNet reference runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if final metrics already exist.",
    )
    return parser.parse_args()


def build_experiment_dir(dataset_output_dir: Path, model_name: str, modified_class_name, shots_per_class: int, update_type: str):
    """Create the per-run output path."""
    return (
        dataset_output_dir
        / slugify(model_name)
        / f"shots_{int(shots_per_class)}"
        / f"{update_type}_{slugify(modified_class_name)}"
    )


def aggregate_counts(train_ds, val_ds, test_ds, classes: list):
    """Count examples per class for the active splits."""
    return {
        "train": count_examples_per_class(train_ds, classes),
        "val": count_examples_per_class(val_ds, classes),
        "test": count_examples_per_class(test_ds, classes),
    }


def build_history_row(shots_per_class: int, support_size: int, val_metrics: dict):
    """Represent the one-shot prototype adaptation as a single history row."""
    return [
        {
            "epoch": 1,
            "phase": "prototype_adaptation",
            "shots_per_class": int(shots_per_class),
            "support_examples": int(support_size),
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
    ]


def build_summary_row(
    dataset_name: str,
    model_name: str,
    modified_class_name,
    final_num_classes: int,
    update_type: str,
    status: str,
    metrics_payload=None,
    error_message=None,
):
    """Flatten the run into the project summary schema."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": str(modified_class_name) if update_type == "remove" else "__none__",
        "final_num_classes": int(final_num_classes),
        "initialization": "imagenet",
        "backbone_mode": "frozen",
        "trainable_scope": "prototype_only",
        "update_type": update_type,
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
                "forgetting_u_olvido": None
                if metrics_payload["forgetting_u_olvido"] is None
                else float(metrics_payload["forgetting_u_olvido"]),
                "numero_de_ejemplos_utilizados": int(metrics_payload["num_examples_used_for_adaptation"]),
                "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
                "numero_de_parametros_entrenados_o_modificados": int(metrics_payload["num_trainable_parameters"]),
                "memoria_adicional_requerida": float(metrics_payload["additional_memory_required"]),
                "shots_per_class": int(metrics_payload["shots_per_class"]),
                "distance_metric": metrics_payload["distance_metric"],
                "method": metrics_payload["method"],
            }
        )

    if error_message is not None:
        row["error"] = error_message

    return row


def save_experiment_artifacts(
    experiment_dir: Path,
    history_rows: list,
    metrics_payload: dict,
    run_config: dict,
    prototype_rows: list,
    support_payload: dict,
):
    """Persist metrics, support metadata and prototypes."""
    save_json(experiment_dir / "training_history.json", history_rows)
    write_csv(experiment_dir / "training_history.csv", history_rows)
    save_json(experiment_dir / "final_metrics.json", metrics_payload)
    save_json(experiment_dir / "run_config.json", run_config)
    save_json(experiment_dir / "prototypes.json", prototype_rows)
    save_json(experiment_dir / "support_set.json", support_payload)


def load_existing_summary(metrics_path: Path):
    """Load the flattened summary from an existing finished experiment."""
    existing_metrics = load_json(metrics_path)
    summary = existing_metrics.get("summary")
    if summary is None:
        summary = build_summary_row(
            dataset_name=existing_metrics["dataset"],
            model_name=existing_metrics["model_name"],
            modified_class_name=existing_metrics.get("removed_class", existing_metrics.get("modified_class")),
            final_num_classes=existing_metrics["final_num_classes"],
            update_type=existing_metrics.get("update_type", "remove"),
            status="completed",
            metrics_payload=existing_metrics,
        )
    summary["status"] = "skipped_existing"
    return summary


def prepare_modified_datasets(dataset_name: str, train_ds, val_ds, test_ds, original_classes: list, class_identifier, update_type: str):
    """Prepare active datasets after the class-set change."""
    if update_type == "remove":
        filtered_train, metadata = remove_class_and_remap(train_ds, original_classes, class_identifier)
        filtered_val, _ = remove_class_and_remap(val_ds, original_classes, class_identifier)
        filtered_test, _ = remove_class_and_remap(test_ds, original_classes, class_identifier)
        return {
            "train": filtered_train,
            "val": filtered_val,
            "test": filtered_test,
            "classes": metadata["remaining_classes"],
            "modified_class_name": metadata["removed_class_name"],
            "modified_class_idx_original": int(metadata["removed_class_idx"]),
        }

    parsed_class = parse_class_identifier(class_identifier)
    raise NotImplementedError(
        f"update_type='add' todavia no esta soportado para '{dataset_name}'. "
        f"El codigo queda preparado, pero falta definir la fuente de datos para la nueva clase '{parsed_class}'."
    )


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    modified_setup: dict,
    args,
    dataset_output_dir: Path,
):
    """Run one Prototypical Networks adaptation experiment."""
    modified_class_name = modified_setup["modified_class_name"]
    active_train = modified_setup["train"]
    active_val = modified_setup["val"]
    active_test = modified_setup["test"]
    active_classes = modified_setup["classes"]

    experiment_dir = build_experiment_dir(
        dataset_output_dir=dataset_output_dir,
        model_name=model_name,
        modified_class_name=modified_class_name,
        shots_per_class=args.shots_per_class,
        update_type=args.update_type,
    )
    metrics_path = experiment_dir / "final_metrics.json"
    error_path = experiment_dir / "error.json"

    if not args.overwrite and metrics_path.exists():
        print(f"[SKIP] {dataset_name} | {model_name} | {args.update_type}={modified_class_name}")
        return load_existing_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if error_path.exists():
        error_path.unlink()

    set_seed(args.seed)
    total_start = time.time()

    original_num_classes = len(active_classes) + (1 if args.update_type == "remove" else 0)
    model, reference_metrics, checkpoint_path, reference_metrics_path = load_reference_model(
        reference_dir=Path(args.reference_dir),
        dataset_name=dataset_name,
        model_name=model_name,
        num_classes=original_num_classes,
    )

    support_selection = sample_k_shot_support_set(
        dataset=active_train,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
        class_indices=list(range(len(active_classes))),
    )
    support_dataset = Subset(active_train, support_selection.indices)

    support_loader = build_loader(
        IndexedDataset(support_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_loader(
        IndexedDataset(active_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = build_loader(
        IndexedDataset(active_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"\n{'=' * 90}")
    print(
        f"Dataset: {dataset_name} | Model: {model_name} | "
        f"{args.update_type}={modified_class_name} | shots={args.shots_per_class}"
    )
    print(f"{'=' * 90}")

    support_embedding_start = time.time()
    support_extracted = extract_embeddings(model, support_loader, representation="embeddings", use_grad=False)
    support_embedding_time = time.time() - support_embedding_start

    prototype_start = time.time()
    prototypes_by_class = build_class_prototypes(
        support_extracted["vectors"],
        support_extracted["labels"],
    )
    prototype_build_time = time.time() - prototype_start

    val_embedding_start = time.time()
    val_extracted = extract_embeddings(model, val_loader, representation="embeddings", use_grad=False)
    val_embedding_time = time.time() - val_embedding_start
    val_metrics = evaluate_prototypical_predictions(
        query_embeddings=val_extracted["vectors"],
        query_labels=val_extracted["labels"],
        prototypes_by_class=prototypes_by_class,
        class_names=active_classes,
        metric=args.distance_metric,
    )

    test_embedding_start = time.time()
    test_extracted = extract_embeddings(model, test_loader, representation="embeddings", use_grad=False)
    test_embedding_time = time.time() - test_embedding_start
    evaluation_start = time.time()
    test_metrics = evaluate_prototypical_predictions(
        query_embeddings=test_extracted["vectors"],
        query_labels=test_extracted["labels"],
        prototypes_by_class=prototypes_by_class,
        class_names=active_classes,
        metric=args.distance_metric,
    )
    evaluation_time = time.time() - evaluation_start

    split_counts = aggregate_counts(active_train, active_val, active_test, active_classes)
    forgetting_value = compute_forgetting_from_reference(
        reference_per_class_accuracy=reference_metrics.get("test_per_class_accuracy"),
        current_per_class_accuracy=test_metrics["per_class_accuracy"],
    )

    prototype_rows = serialize_prototypes(prototypes_by_class, active_classes)
    prototype_memory = sum(
        np.asarray(prototypes_by_class[class_idx], dtype=float).nbytes
        for class_idx in prototypes_by_class
    )
    elapsed = time.time() - total_start

    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "method": "prototypical_fewshot",
        "embedding_strategy": "reference_backbone_embeddings",
        "initialization": "imagenet",
        "backbone_mode": "frozen",
        "trainable_scope": "prototype_only",
        "update_type": args.update_type,
        "removed_class": str(modified_class_name) if args.update_type == "remove" else "__none__",
        "modified_class": str(modified_class_name),
        "modified_class_idx_original": modified_setup.get("modified_class_idx_original"),
        "final_num_classes": int(len(active_classes)),
        "shots_per_class": int(args.shots_per_class),
        "distance_metric": args.distance_metric,
        "selection_metric": "prototype_validation_loss",
        "best_epoch": 1,
        "epochs_ran": 1,
        "best_val_loss": float(val_metrics["loss"]),
        "best_val_accuracy": float(val_metrics["accuracy"]),
        "elapsed_seconds": float(elapsed),
        "support_embedding_time": float(support_embedding_time),
        "prototype_build_time": float(prototype_build_time),
        "validation_embedding_time": float(val_embedding_time),
        "test_embedding_time": float(test_embedding_time),
        "evaluation_time": float(evaluation_time),
        "test_overall_accuracy": float(test_metrics["accuracy"]),
        "test_mean_per_class_accuracy": float(test_metrics["mean_per_class_accuracy"]),
        "test_per_class_accuracy": test_metrics["per_class_accuracy"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "class_names": list(active_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": int(len(support_selection.indices)),
        "support_examples_per_class": {active_classes[int(k)]: len(v) for k, v in support_selection.indices_by_class.items()},
        "prediction_confidence_mean": float(test_metrics["prediction_confidence_mean"]),
        "num_trainable_parameters": 0,
        "additional_memory_required": float(prototype_memory),
        "forgetting_u_olvido": None if forgetting_value is None else float(forgetting_value),
        "forgetting_reference_source": "full_training_reference_imagenet",
        "forgetting_reference_metrics_path": str(reference_metrics_path),
        "reference_checkpoint_path": str(checkpoint_path),
        "metricas_eliminacion": [metrica.nombre for metrica in METRICAS_ELIMINACION],
        "stores_model_checkpoint": False,
        "prototype_embedding_dim": int(np.asarray(next(iter(prototypes_by_class.values()))).shape[0]),
    }
    metrics_payload["summary"] = build_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        modified_class_name=modified_class_name,
        final_num_classes=len(active_classes),
        update_type=args.update_type,
        status="completed",
        metrics_payload=metrics_payload,
    )

    save_experiment_artifacts(
        experiment_dir=experiment_dir,
        history_rows=build_history_row(args.shots_per_class, len(support_selection.indices), val_metrics),
        metrics_payload=metrics_payload,
        run_config={
            "dataset": dataset_name,
            "model_name": model_name,
            "method": "prototypical_fewshot",
            "update_type": args.update_type,
            "modified_class": str(modified_class_name),
            "shots_per_class": int(args.shots_per_class),
            "distance_metric": args.distance_metric,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "reference_dir": str(args.reference_dir),
        },
        prototype_rows=prototype_rows,
        support_payload={
            "dataset": dataset_name,
            "model_name": model_name,
            "update_type": args.update_type,
            "modified_class": str(modified_class_name),
            "shots_per_class": int(args.shots_per_class),
            "distance_metric": args.distance_metric,
            "support_indices": support_selection.indices,
            "support_indices_by_class": support_selection.indices_by_class,
            "support_class_names": {
                active_classes[int(class_idx)]: indices
                for class_idx, indices in support_selection.indices_by_class.items()
            },
        },
    )
    return metrics_payload["summary"]


def run_all_experiments(args):
    """Run the complete prototypical few-shot suite for one or all datasets."""
    dataset_names = sorted(DATASET_LOADERS) if args.all_datasets else [args.dataset]
    selected_models = list(args.models)
    all_summary_rows = []
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets selected: {dataset_names}")
    print(f"Models selected: {selected_models}")
    print(f"Update type: {args.update_type}")
    print(f"Shots per class: {args.shots_per_class}")
    print(f"Distance metric: {args.distance_metric}")

    for dataset_name in dataset_names:
        print(f"\nRunning dataset: {dataset_name}")
        dataset_output_dir = base_output_dir / slugify(dataset_name)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []

        try:
            train_ds, val_ds, test_ds, original_classes = DATASET_LOADERS[dataset_name]()
            class_identifiers = get_classes_to_remove(dataset_name, args.classes) if args.update_type == "remove" else (
                [parse_class_identifier(value) for value in args.classes] if args.classes else []
            )
            if not class_identifiers:
                raise ValueError("No classes provided for the requested update_type.")
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

        for class_identifier in class_identifiers:
            try:
                modified_setup = prepare_modified_datasets(
                    dataset_name=dataset_name,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                    original_classes=original_classes,
                    class_identifier=class_identifier,
                    update_type=args.update_type,
                )
            except Exception as exc:
                error_message = str(exc)
                print(f"[ERROR] Could not prepare modification '{class_identifier}': {error_message}")
                for model_name in selected_models:
                    summary_rows.append(
                        build_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            modified_class_name=class_identifier,
                            final_num_classes=max(len(original_classes) - 1, 0),
                            update_type=args.update_type,
                            status="failed",
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
                        modified_setup=modified_setup,
                        args=args,
                        dataset_output_dir=dataset_output_dir,
                    )
                    summary_rows.append(summary_row)
                except Exception as exc:
                    error_message = str(exc)
                    experiment_dir = build_experiment_dir(
                        dataset_output_dir=dataset_output_dir,
                        model_name=model_name,
                        modified_class_name=modified_setup["modified_class_name"],
                        shots_per_class=args.shots_per_class,
                        update_type=args.update_type,
                    )
                    experiment_dir.mkdir(parents=True, exist_ok=True)
                    save_json(
                        experiment_dir / "error.json",
                        {
                            "dataset": dataset_name,
                            "model_name": model_name,
                            "modified_class": modified_setup["modified_class_name"],
                            "update_type": args.update_type,
                            "shots_per_class": int(args.shots_per_class),
                            "error": error_message,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(
                        f"[ERROR] {dataset_name} | {model_name} | "
                        f"{args.update_type}={modified_setup['modified_class_name']}: {error_message}"
                    )
                    summary_rows.append(
                        build_summary_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            modified_class_name=modified_setup["modified_class_name"],
                            final_num_classes=len(modified_setup["classes"]),
                            update_type=args.update_type,
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


def main():
    args = parse_args()
    run_all_experiments(args)


if __name__ == "__main__":
    main()

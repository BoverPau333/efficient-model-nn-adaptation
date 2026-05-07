"""Fine-tuning dinamico con embeddings precomputados antes de adaptar el modelo."""

import sys
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.class_distance import compute_class_centroids, compute_distance_matrix, get_nearest_classes
from src.dataset.loaders import DATASET_LOADERS
from src.dynamic_dataset_selection import RemappedSubset, select_dynamic_subset
from src.dynamic_finetuning_utils import (
    build_dynamic_arg_parser,
    build_experiment_dir,
    finalize_dynamic_experiment,
    load_existing_dynamic_summary,
    parse_class_identifier,
    prepare_update_datasets,
    run_dynamic_experiment_suite,
)
from src.embedding_utils import IndexedDataset, extract_embeddings
from src.experiments_config.config import RESULTS_DIR
from src.results_utils import (
    build_loader,
    evaluate_classification_metrics,
    load_reference_model,
    remove_output_class,
    set_seed,
)
from src.training import train_with_early_stopping


METHOD_NAME = "precompute_embeddings_then_finetune"
EMBEDDING_STRATEGY = "precomputed_before_finetuning"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "dynamic_embedding_finetuning" / METHOD_NAME


def parse_args():
    parser = build_dynamic_arg_parser(
        description="Fine-tuning dinamico con embeddings precomputados.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
        dataset_choices=DATASET_LOADERS,
    )
    return parser.parse_args()


def run_single_experiment(dataset_name: str, model_name: str, args, base_output_dir: Path):
    """Ejecuta una configuracion concreta."""
    train_ds, val_ds, test_ds, classes = DATASET_LOADERS[dataset_name]()
    modified_class = parse_class_identifier(args.modified_class)
    setup = prepare_update_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        classes=classes,
        modified_class=modified_class,
        update_type=args.update_type,
    )

    experiment_dir = build_experiment_dir(
        base_output_dir=base_output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        modified_class=args.modified_class,
        update_type=args.update_type,
    )
    metrics_path = experiment_dir / "final_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        return load_existing_dynamic_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    total_start = perf_counter()
    model, reference_metrics, checkpoint_path, reference_metrics_path = load_reference_model(
        reference_dir=Path(args.reference_dir),
        dataset_name=dataset_name,
        model_name=model_name,
        num_classes=len(classes),
    )
    if args.update_type == "remove":
        remove_output_class(model, setup["modified_class_idx_original"])

    distance_loader = build_loader(
        IndexedDataset(setup["distance_train_dataset"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    embedding_start = perf_counter()
    extracted = extract_embeddings(
        model,
        distance_loader,
        representation=args.embedding_representation,
        use_grad=False,
    )
    embedding_time = perf_counter() - embedding_start

    selection_start = perf_counter()
    normalize_centroids = args.distance_metric == "cosine"
    centroid_classes, centroids = compute_class_centroids(
        extracted["vectors"],
        extracted["labels"],
        normalize=normalize_centroids,
    )
    distance_matrix = compute_distance_matrix(centroids, metric=args.distance_metric)
    neighbours = get_nearest_classes(
        centroid_classes,
        distance_matrix,
        modified_class_idx=setup["modified_class_idx_original"],
        k_neighbours=args.k_neighbours,
    )
    neighbour_class_indices = [item["class_idx"] for item in neighbours]
    centroid_map = {int(class_idx): centroids[pos] for pos, class_idx in enumerate(centroid_classes)}
    selected_subset, selection_details = select_dynamic_subset(
        dataset=setup["distance_train_dataset"],
        embeddings=extracted["vectors"],
        labels=extracted["labels"],
        ids=extracted["ids"],
        modified_class_idx=setup["modified_class_idx_original"],
        neighbour_class_indices=neighbour_class_indices,
        class_centroids=centroid_map,
        samples_per_modified_class=args.samples_per_modified_class,
        samples_per_neighbour_class=args.samples_per_neighbour_class,
        memory_samples_per_far_class=args.memory_samples_per_far_class,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
        update_type=args.update_type,
    )
    selection_time = perf_counter() - selection_start

    if args.update_type == "remove":
        selected_train = RemappedSubset(
            setup["distance_train_dataset"],
            selected_subset.indices,
            label_mapping=setup["label_mapping_after_removal"],
            classes=setup["active_classes"],
        )
    else:
        selected_train = RemappedSubset(
            setup["distance_train_dataset"],
            selected_subset.indices,
            label_mapping=None,
            classes=setup["active_classes"],
        )

    train_loader = build_loader(
        selected_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_loader(
        setup["val_active"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = build_loader(
        setup["test_active"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    finetuning_start = perf_counter()
    training_result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=None,
        checkpoint_path=None,
        verbose=True,
    )
    finetuning_time = perf_counter() - finetuning_start

    evaluation_start = perf_counter()
    evaluation_metrics = evaluate_classification_metrics(model, test_loader, setup["active_classes"])
    evaluation_time = perf_counter() - evaluation_start
    total_time = perf_counter() - total_start

    return finalize_dynamic_experiment(
        dataset_name=dataset_name,
        model_name=model_name,
        args=args,
        method_name=METHOD_NAME,
        embedding_strategy=EMBEDDING_STRATEGY,
        experiment_dir=experiment_dir,
        model=model,
        setup=setup,
        classes=classes,
        reference_metrics=reference_metrics,
        checkpoint_path=checkpoint_path,
        reference_metrics_path=reference_metrics_path,
        neighbours=neighbours,
        history_rows=training_result["history"],
        selected_train=selected_train,
        test_loader=test_loader,
        evaluation_metrics=evaluation_metrics,
        timing={
            "total_time": total_time,
            "embedding_time": embedding_time,
            "selection_time": selection_time,
            "finetuning_time": finetuning_time,
            "evaluation_time": evaluation_time,
        },
        training_summary={
            "epochs_ran": training_result["epochs_ran"],
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "best_val_accuracy": training_result["best_val_accuracy"],
        },
        selection_details=selection_details,
    )


def main():
    args = parse_args()
    base_output_dir = run_dynamic_experiment_suite(
        args=args,
        method_name=METHOD_NAME,
        embedding_strategy=EMBEDDING_STRATEGY,
        run_single_experiment=run_single_experiment,
    )
    print(f"Resultados guardados en: {base_output_dir}")


if __name__ == "__main__":
    main()

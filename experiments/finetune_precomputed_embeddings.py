"""Fine-tuning dinamico con embeddings precomputados antes de adaptar el modelo."""

import copy
import sys
from pathlib import Path
from time import perf_counter

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.class_distance import compute_class_centroids, compute_distance_matrix, get_nearest_classes
from src.dataset.loaders import DATASET_LOADERS
from src.adaptation.dynamic_dataset_selection import RemappedSubset, select_dynamic_subset
from src.adaptation.dynamic_finetuning_utils import (
    build_dynamic_arg_parser,
    build_experiment_dir,
    finalize_dynamic_experiment,
    load_existing_dynamic_summary,
    parse_class_identifier,
    prepare_update_datasets,
    run_dynamic_experiment_suite,
)
from src.core.embedding_utils import IndexedDataset, extract_embeddings
from src.experiments_config.config import RESULTS_DIR
from src.core.results_utils import (
    build_loader,
    evaluate_classification_metrics,
    freeze_backbone_keep_head_trainable,
    load_reference_model,
    remove_output_class,
    set_seed,
)
from src.core.training import train_with_early_stopping


METHOD_NAME = "precompute_embeddings_then_finetune"
EMBEDDING_STRATEGY = "precomputed_before_finetuning"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "dynamic_embedding_finetuning" / METHOD_NAME


def log_progress(dataset_name: str, model_name: str, modified_class, message: str):
    """Imprime una traza con contexto para seguir el avance del experimento."""
    print(
        f"[{METHOD_NAME}] dataset={dataset_name} | model={model_name} | modified_class={modified_class} | {message}",
        flush=True,
    )


def parse_args():
    parser = build_dynamic_arg_parser(
        description="Fine-tuning dinamico con embeddings precomputados.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
        dataset_choices=DATASET_LOADERS,
    )
    return parser.parse_args()


def run_single_experiment(dataset_name: str, model_name: str, args, base_output_dir: Path):
    """Ejecuta una configuracion concreta."""
    log_progress(dataset_name, model_name, args.modified_class, "starting experiment")
    train_ds, val_ds, test_ds, classes = DATASET_LOADERS[dataset_name]()
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"datasets loaded | train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}",
    )
    modified_class = parse_class_identifier(args.modified_class)
    setup = prepare_update_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        classes=classes,
        modified_class=modified_class,
        update_type=args.update_type,
    )
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"active split prepared | active_train={len(setup['train_active'])} | final_classes={len(setup['active_classes'])}",
    )

    experiment_dir = build_experiment_dir(
        base_output_dir=base_output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        modified_class=args.modified_class,
        update_type=args.update_type,
        train_percentage=args.porc,
    )
    metrics_path = experiment_dir / "final_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        log_progress(dataset_name, model_name, args.modified_class, f"skipping existing run at {experiment_dir}")
        return load_existing_dynamic_summary(metrics_path)

    experiment_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    total_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "loading reference model")
    model, reference_metrics, checkpoint_path, reference_metrics_path = load_reference_model(
        reference_dir=Path(args.reference_dir),
        dataset_name=dataset_name,
        model_name=model_name,
        num_classes=len(classes),
    )
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"reference model loaded | checkpoint={checkpoint_path}",
    )
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    student_class_indices = None
    teacher_class_indices = None
    if args.update_type == "remove":
        remove_output_class(model, setup["modified_class_idx_original"])
        teacher_class_indices = [
            idx for idx in range(len(classes)) if idx != int(setup["modified_class_idx_original"])
        ]
        student_class_indices = list(range(len(setup["active_classes"])))
    else:
        common_num_classes = min(len(classes), len(setup["active_classes"]))
        teacher_class_indices = list(range(common_num_classes))
        student_class_indices = list(range(common_num_classes))
    freeze_backbone_keep_head_trainable(model)
    log_progress(dataset_name, model_name, args.modified_class, "backbone frozen, head ready for fine-tuning")

    distance_loader = build_loader(
        IndexedDataset(setup["distance_train_dataset"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"distance loader built | samples={len(setup['distance_train_dataset'])} | batch_size={args.batch_size}",
    )
    embedding_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting embedding extraction")
    extracted = extract_embeddings(
        model,
        distance_loader,
        representation=args.embedding_representation,
        use_grad=False,
    )
    embedding_time = perf_counter() - embedding_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"embedding extraction completed in {embedding_time:.1f}s | vectors={len(extracted['labels'])}",
    )

    selection_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting dynamic subset selection")
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
        target_percentage=args.porc,
        train_dataset_size=len(setup["train_active"]),
        modified_class_weight=args.samples_per_modified_class,
        neighbour_class_weight=args.samples_per_neighbour_class,
        far_class_weight=args.memory_samples_per_far_class,
        selection_strategy=args.selection_strategy,
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        seed=args.seed,
        update_type=args.update_type,
        progress_label=f"{METHOD_NAME} | dataset={dataset_name} | model={model_name} | modified_class={args.modified_class}",
    )
    selection_time = perf_counter() - selection_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"dynamic subset selection completed in {selection_time:.1f}s | selected={len(selected_subset.indices)}",
    )

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
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"train/val/test loaders ready | selected_train={len(selected_train)} | val={len(setup['val_active'])} | test={len(setup['test_active'])}",
    )

    finetuning_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting fine-tuning")
    training_result = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        checkpoint_path=None,
        teacher_model=teacher_model,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        student_class_indices=student_class_indices,
        teacher_class_indices=teacher_class_indices,
        verbose=True,
    )
    finetuning_time = perf_counter() - finetuning_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"fine-tuning completed in {finetuning_time:.1f}s | epochs_ran={training_result['epochs_ran']} | best_epoch={training_result['best_epoch']}",
    )

    evaluation_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting final evaluation")
    evaluation_metrics = evaluate_classification_metrics(model, test_loader, setup["active_classes"])
    evaluation_time = perf_counter() - evaluation_start
    total_time = perf_counter() - total_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"evaluation completed in {evaluation_time:.1f}s | accuracy={evaluation_metrics['accuracy']:.4f} | total_time={total_time:.1f}s",
    )

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

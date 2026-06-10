"""Fine-tuning dinamico capturando embeddings durante la primera epoca."""

import copy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.class_distance import compute_class_centroids, compute_distance_matrix, get_nearest_classes
from src.dataset.loaders import DATASET_LOADERS
from src.adaptation.dynamic_dataset_selection import (
    RemappedSubset,
    select_dynamic_subset,
)
from src.adaptation.dynamic_finetuning_utils import (
    build_dynamic_arg_parser,
    build_experiment_dir,
    finalize_dynamic_experiment,
    load_dynamic_reference_model,
    load_existing_dynamic_summary,
    parse_class_identifier,
    prepare_update_datasets,
    run_dynamic_experiment_suite,
)
from src.core.embedding_utils import IndexedDataset, extract_embeddings
from src.experiments_config.config import DEVICE, RESULTS_DIR
from src.core.results_utils import (
    build_loader,
    evaluate_classification_metrics,
    freeze_backbone_keep_head_trainable,
    remove_output_class,
    set_seed,
)
from src.core.training import compute_distillation_loss, train_with_early_stopping


METHOD_NAME = "epoch1_embeddings_dynamic_finetune"
EMBEDDING_STRATEGY = "captured_during_first_epoch"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "dynamic_embedding_finetuning" / METHOD_NAME


def log_progress(dataset_name: str, model_name: str, modified_class, message: str):
    """Imprime una traza con contexto para seguir el avance del experimento."""
    print(
        f"[{METHOD_NAME}] dataset={dataset_name} | model={model_name} | modified_class={modified_class} | {message}",
        flush=True,
    )


def parse_args():
    parser = build_dynamic_arg_parser(
        description="Fine-tuning dinamico guiado por embeddings de la epoca 1.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
        dataset_choices=DATASET_LOADERS,
    )
    return parser.parse_args()


def resolve_original_ids_from_subset(subset_indices, captured_ids):
    """Convierte ids locales de un subset en ids del dataset padre, si hace falta."""
    subset_indices = np.asarray(subset_indices, dtype=int)
    captured_ids = np.asarray(captured_ids, dtype=int)
    if captured_ids.size == 0:
        return captured_ids

    if captured_ids.min() >= 0 and captured_ids.max() < len(subset_indices):
        return subset_indices[captured_ids]
    return captured_ids


def evaluate_loader_loss(model, loader):
    """Evalua loss y accuracy sobre un loader."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            total_loss += loss_fn(logits, labels).item()
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "val_accuracy": total_correct / max(total_examples, 1),
    }


def train_first_epoch_with_capture(
    model,
    loader,
    learning_rate: float,
    representation: str,
    teacher_model=None,
    distillation_weight: float = 0.0,
    distillation_temperature: float = 1.0,
    student_class_indices=None,
    teacher_class_indices=None,
):
    """Ejecuta una primera epoca capturando embeddings del forward pass."""
    if representation not in {"embeddings", "logits"}:
        raise ValueError("representation debe ser 'embeddings' o 'logits'")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_distill_loss = 0.0
    running_correct = 0
    running_examples = 0
    captured_vectors = []
    captured_labels = []
    captured_ids = []

    for imgs, labels, ids in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        embeddings, logits = model.forward_embeddings_and_logits(imgs)
        selected_vectors = embeddings if representation == "embeddings" else logits
        ce_loss = loss_fn(logits, labels)
        distill_loss = torch.zeros((), device=DEVICE)
        if teacher_model is not None and distillation_weight > 0.0:
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            distill_loss = compute_distillation_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                temperature=distillation_temperature,
                student_class_indices=student_class_indices,
                teacher_class_indices=teacher_class_indices,
            )
        loss = ce_loss + (distillation_weight * distill_loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_distill_loss += float(distill_loss.item())
        running_correct += int((logits.argmax(dim=1) == labels).sum().item())
        running_examples += int(labels.size(0))

        captured_vectors.append(selected_vectors.detach().cpu().numpy())
        captured_labels.append(labels.detach().cpu().numpy())
        captured_ids.append(np.asarray(ids))

    return {
        "train_loss": running_loss / max(len(loader), 1),
        "train_ce_loss": running_ce_loss / max(len(loader), 1),
        "train_distill_loss": running_distill_loss / max(len(loader), 1),
        "train_accuracy": running_correct / max(running_examples, 1),
        "vectors": np.concatenate(captured_vectors, axis=0),
        "labels": np.concatenate(captured_labels, axis=0),
        "ids": np.concatenate(captured_ids, axis=0),
    }


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
    inverse_label_mapping = None
    if setup["label_mapping_after_removal"] is not None:
        inverse_label_mapping = {int(new): int(old) for old, new in setup["label_mapping_after_removal"].items()}
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

    model, reference_metrics, checkpoint_path, reference_metrics_path = load_dynamic_reference_model(
        dataset_name=dataset_name,
        model_name=model_name,
        args=args,
        setup=setup,
        classes=classes,
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
        teacher_class_indices = [
            idx for idx in range(len(classes)) if idx != int(setup["modified_class_idx_original"])
        ]
        student_class_indices = list(teacher_class_indices)
    freeze_backbone_keep_head_trainable(model)
    log_progress(dataset_name, model_name, args.modified_class, "backbone frozen, head ready for fine-tuning")

    initial_train_dataset = setup["train_active"]
    initial_train_loader = build_loader(
        IndexedDataset(initial_train_dataset),
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
        (
            "initial train loader ready | using_full_train_for_epoch1=yes "
            f"| selected_train={len(initial_train_dataset)} | val={len(setup['val_active'])} | test={len(setup['test_active'])}"
        ),
    )

    first_epoch_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting first epoch with capture")
    first_epoch_result = train_first_epoch_with_capture(
        model,
        initial_train_loader,
        learning_rate=args.learning_rate,
        representation=args.embedding_representation,
        teacher_model=teacher_model,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        student_class_indices=student_class_indices,
        teacher_class_indices=teacher_class_indices,
    )
    first_epoch_time = perf_counter() - first_epoch_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"first epoch with capture completed in {first_epoch_time:.1f}s | captured={len(first_epoch_result['labels'])}",
    )
    validation_after_first = evaluate_loader_loss(model, val_loader)
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"validation after first epoch | val_loss={validation_after_first['val_loss']:.4f} | val_acc={validation_after_first['val_accuracy']:.4f}",
    )
    history_rows = [
        {
            "epoch": 1,
            "phase": "full_train_epoch1",
            "train_loss": float(first_epoch_result["train_loss"]),
            "train_ce_loss": float(first_epoch_result["train_ce_loss"]),
            "train_distill_loss": float(first_epoch_result["train_distill_loss"]),
            "train_accuracy": float(first_epoch_result["train_accuracy"]),
            "val_loss": float(validation_after_first["val_loss"]),
            "val_accuracy": float(validation_after_first["val_accuracy"]),
        }
    ]
    selection_start = perf_counter()
    log_progress(dataset_name, model_name, args.modified_class, "starting focused subset rebuild from captured embeddings")

    captured_embeddings = first_epoch_result["vectors"]
    captured_labels = first_epoch_result["labels"]
    captured_ids = first_epoch_result["ids"]

    if args.update_type == "remove":
        original_ids = resolve_original_ids_from_subset(initial_train_dataset.indices, captured_ids)
        original_labels = np.asarray([inverse_label_mapping[int(label)] for label in captured_labels], dtype=int)

        removed_only_indices = np.flatnonzero(np.asarray(train_ds.targets) == setup["modified_class_idx_original"]).tolist()
        if removed_only_indices:
            removed_only_dataset = torch.utils.data.Subset(train_ds, removed_only_indices)
            removed_loader = build_loader(
                IndexedDataset(removed_only_dataset),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            supplement_start = perf_counter()
            removed_embeddings = extract_embeddings(
                model,
                removed_loader,
                representation=args.embedding_representation,
                use_grad=False,
            )
            embedding_time += perf_counter() - supplement_start
            log_progress(
                dataset_name,
                model_name,
                args.modified_class,
                f"removed-class embedding supplement completed in {perf_counter() - supplement_start:.1f}s | extra_vectors={len(removed_embeddings['labels'])}",
            )
            original_ids = np.concatenate([original_ids, np.asarray([removed_only_indices[int(idx)] for idx in removed_embeddings["ids"]], dtype=int)])
            original_labels = np.concatenate([original_labels, removed_embeddings["labels"]])
            captured_embeddings = np.concatenate([captured_embeddings, removed_embeddings["vectors"]], axis=0)
    else:
        original_ids = captured_ids
        original_labels = captured_labels

    centroid_classes, centroids = compute_class_centroids(
        captured_embeddings,
        original_labels,
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
        dataset=train_ds,
        embeddings=captured_embeddings,
        labels=original_labels,
        ids=original_ids,
        modified_class_idx=setup["modified_class_idx_original"],
        neighbour_class_indices=neighbour_class_indices,
        class_centroids=centroid_map,
        target_percentage=args.porc,
        train_dataset_size=len(setup["train_active"]),
        modified_class_weight=args.samples_per_modified_class,
        modified_class_fraction=args.modified_class_fraction,
        neighbour_class_weight=args.samples_per_neighbour_class,
        far_class_weight=args.memory_samples_per_far_class,
        selection_strategy=args.selection_strategy,
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        seed=args.seed,
        update_type=args.update_type,
        progress_label=f"{METHOD_NAME} focused | dataset={dataset_name} | model={model_name} | modified_class={args.modified_class}",
    )
    selection_time = perf_counter() - selection_start
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"focused subset rebuild completed | selection_time={selection_time:.1f}s | selected={len(selected_subset.indices)}",
    )
    selection_details["initial_selection"] = {
        "used_full_train_for_epoch1": True,
        "num_examples": int(len(initial_train_dataset)),
    }

    if args.update_type == "remove":
        selected_train = RemappedSubset(
            train_ds,
            selected_subset.indices,
            label_mapping=setup["label_mapping_after_removal"],
            classes=setup["active_classes"],
        )
    else:
        selected_train = RemappedSubset(
            train_ds,
            selected_subset.indices,
            label_mapping=None,
            classes=setup["active_classes"],
        )

    remaining_epochs = max(args.epochs - 1, 0)
    finetuning_start = perf_counter()
    if remaining_epochs > 0:
        log_progress(
            dataset_name,
            model_name,
            args.modified_class,
            f"starting focused fine-tuning for remaining_epochs={remaining_epochs}",
        )
        focused_train_loader = build_loader(
            selected_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        remaining_result = train_with_early_stopping(
            model,
            focused_train_loader,
            val_loader,
            epochs=remaining_epochs,
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
        log_progress(
            dataset_name,
            model_name,
            args.modified_class,
            f"focused fine-tuning completed | epochs_ran={remaining_result['epochs_ran']} | best_epoch={remaining_result['best_epoch']}",
        )
        for epoch_info in remaining_result["history"]:
            history_rows.append(
                {
                    **epoch_info,
                    "epoch": int(epoch_info["epoch"]) + 1,
                    "phase": "focused_finetuning",
                }
            )
        best_epoch = 1
        best_val_loss = float(validation_after_first["val_loss"])
        best_val_accuracy = float(validation_after_first["val_accuracy"])
        if float(remaining_result["best_val_loss"]) < best_val_loss:
            best_epoch = int(remaining_result["best_epoch"]) + 1
            best_val_loss = float(remaining_result["best_val_loss"])
            best_val_accuracy = float(remaining_result["best_val_accuracy"])
        epochs_ran = 1 + int(remaining_result["epochs_ran"])
    else:
        best_epoch = 1
        best_val_loss = float(validation_after_first["val_loss"])
        best_val_accuracy = float(validation_after_first["val_accuracy"])
        epochs_ran = 1
    finetuning_time = perf_counter() - finetuning_start + first_epoch_time
    log_progress(
        dataset_name,
        model_name,
        args.modified_class,
        f"training pipeline completed in {finetuning_time:.1f}s | epochs_ran={epochs_ran} | best_epoch={best_epoch}",
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
        history_rows=history_rows,
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
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
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

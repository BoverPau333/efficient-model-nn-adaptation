"""Fine-tuning dinamico capturando embeddings durante la primera epoca."""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.class_distance import compute_class_centroids, compute_distance_matrix, get_nearest_classes
from src.dataset.loaders import DATASET_LOADERS
from src.dynamic_dataset_selection import (
    RemappedSubset,
    sample_balanced_indices,
    select_dynamic_subset,
)
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
from src.experiments_config.config import DEVICE, RESULTS_DIR
from src.results_utils import (
    build_loader,
    evaluate_classification_metrics,
    load_reference_model,
    remove_output_class,
    set_seed,
)
from src.training import train_with_early_stopping


METHOD_NAME = "epoch1_embeddings_dynamic_finetune"
EMBEDDING_STRATEGY = "captured_during_first_epoch"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "dynamic_embedding_finetuning" / METHOD_NAME


def parse_args():
    parser = build_dynamic_arg_parser(
        description="Fine-tuning dinamico guiado por embeddings de la epoca 1.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
        dataset_choices=DATASET_LOADERS,
    )
    return parser.parse_args()


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


def train_first_epoch_with_capture(model, loader, learning_rate: float, representation: str):
    """Ejecuta una primera epoca capturando embeddings del forward pass."""
    if representation not in {"embeddings", "logits"}:
        raise ValueError("representation debe ser 'embeddings' o 'logits'")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model.train()
    running_loss = 0.0
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
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += int((logits.argmax(dim=1) == labels).sum().item())
        running_examples += int(labels.size(0))

        captured_vectors.append(selected_vectors.detach().cpu().numpy())
        captured_labels.append(labels.detach().cpu().numpy())
        captured_ids.append(np.asarray(ids))

    return {
        "train_loss": running_loss / max(len(loader), 1),
        "train_accuracy": running_correct / max(running_examples, 1),
        "vectors": np.concatenate(captured_vectors, axis=0),
        "labels": np.concatenate(captured_labels, axis=0),
        "ids": np.concatenate(captured_ids, axis=0),
    }


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
    inverse_label_mapping = None
    if setup["label_mapping_after_removal"] is not None:
        inverse_label_mapping = {int(new): int(old) for old, new in setup["label_mapping_after_removal"].items()}

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

    initial_indices = sample_balanced_indices(
        setup["train_active"],
        samples_per_class=args.initial_samples_per_class,
        seed=args.seed,
    )
    if args.update_type == "remove":
        initial_train_subset = RemappedSubset(
            train_ds,
            [setup["train_active"].indices[idx] for idx in initial_indices],
            label_mapping=setup["label_mapping_after_removal"],
            classes=setup["active_classes"],
        )
    else:
        initial_train_subset = RemappedSubset(
            setup["train_active"],
            initial_indices,
            label_mapping=None,
            classes=setup["active_classes"],
        )

    initial_train_loader = build_loader(
        IndexedDataset(initial_train_subset),
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

    first_epoch_start = perf_counter()
    first_epoch_result = train_first_epoch_with_capture(
        model,
        initial_train_loader,
        learning_rate=args.learning_rate,
        representation=args.embedding_representation,
    )
    first_epoch_time = perf_counter() - first_epoch_start
    validation_after_first = evaluate_loader_loss(model, val_loader)
    history_rows = [
        {
            "epoch": 1,
            "phase": "initial_balanced_epoch",
            "train_loss": float(first_epoch_result["train_loss"]),
            "train_accuracy": float(first_epoch_result["train_accuracy"]),
            "val_loss": float(validation_after_first["val_loss"]),
            "val_accuracy": float(validation_after_first["val_accuracy"]),
        }
    ]

    embedding_time = float(first_epoch_time)
    selection_start = perf_counter()

    captured_embeddings = first_epoch_result["vectors"]
    captured_labels = first_epoch_result["labels"]
    captured_ids = first_epoch_result["ids"]

    if args.update_type == "remove":
        original_ids = np.asarray([initial_train_subset.indices[int(idx)] for idx in captured_ids], dtype=int)
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
            original_ids = np.concatenate([original_ids, np.asarray([removed_only_indices[int(idx)] for idx in removed_embeddings["ids"]], dtype=int)])
            original_labels = np.concatenate([original_labels, removed_embeddings["labels"]])
            captured_embeddings = np.concatenate([captured_embeddings, removed_embeddings["vectors"]], axis=0)
    else:
        original_ids = captured_ids
        original_labels = captured_labels

    normalize_centroids = args.distance_metric == "cosine"
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

    candidate_original_indices = []
    original_targets = np.asarray(train_ds.targets)
    for class_idx in neighbour_class_indices:
        candidate_original_indices.extend(np.flatnonzero(original_targets == int(class_idx)).tolist())
    if args.update_type == "add":
        candidate_original_indices.extend(
            np.flatnonzero(original_targets == int(setup["modified_class_idx_original"])).tolist()
        )
    candidate_original_indices = sorted(set(candidate_original_indices))
    candidate_dataset = torch.utils.data.Subset(train_ds, candidate_original_indices)
    candidate_loader = build_loader(
        IndexedDataset(candidate_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    candidate_embedding_start = perf_counter()
    candidate_extracted = extract_embeddings(
        model,
        candidate_loader,
        representation=args.embedding_representation,
        use_grad=False,
    )
    embedding_time += perf_counter() - candidate_embedding_start

    centroid_map = {int(class_idx): centroids[pos] for pos, class_idx in enumerate(centroid_classes)}
    selected_subset, selection_details = select_dynamic_subset(
        dataset=train_ds,
        embeddings=candidate_extracted["vectors"],
        labels=candidate_extracted["labels"],
        ids=np.asarray([candidate_original_indices[int(idx)] for idx in candidate_extracted["ids"]], dtype=int),
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
            patience=None,
            checkpoint_path=None,
            verbose=True,
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

"""Utilidades compartidas para experimentos de adicion de clases."""

import json
from pathlib import Path

import numpy as np
import torch

from src.core.results_utils import (
    evaluate_prediction_confidence,
    load_json,
    parse_class_identifier,
    slugify,
)
from src.experiments_config.class_to_add import CLASSES_TO_ADD_BY_DATASET
from src.metrics_addition import METRICAS_ADICION


def get_classes_to_add(dataset_name: str, override_classes=None):
    """Resuelve la lista de clases a anadir para un dataset."""
    if override_classes:
        parsed = [parse_class_identifier(value) for value in override_classes]
        if not parsed:
            raise ValueError("The override class list is empty.")
        return parsed

    configured = CLASSES_TO_ADD_BY_DATASET.get(dataset_name)
    if not configured:
        raise ValueError(
            f"No classes configured for dataset '{dataset_name}' in "
            "src/experiments_config/class_to_add.py"
        )
    return [parse_class_identifier(value) for value in configured]


def zero_model_parameters(model):
    """Pone a cero todos los pesos y sesgos del modelo."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def serialize_per_class_accuracy(per_class_accuracy):
    """Serializa accuracy por clase para tablas planas."""
    if isinstance(per_class_accuracy, str):
        return per_class_accuracy
    return json.dumps(per_class_accuracy, ensure_ascii=True, sort_keys=True)


def compute_previous_class_forgetting(
    reference_per_class_accuracy: dict,
    current_per_class_accuracy: dict,
    added_class_name,
):
    """Promedia la degradacion en las clases originales, excluyendo la anadida."""
    previous_classes = [
        class_name
        for class_name in current_per_class_accuracy
        if class_name != str(added_class_name)
    ]
    if not previous_classes:
        return None

    degradations = []
    for class_name in previous_classes:
        if class_name not in reference_per_class_accuracy:
            return None
        degradations.append(
            float(reference_per_class_accuracy[class_name]) - float(current_per_class_accuracy[class_name])
        )
    return float(np.mean(degradations))


def precision_from_confusion_matrix(confusion_matrix, class_idx: int) -> float:
    """Calcula la precision para una clase concreta desde la matriz de confusion."""
    true_positives = float(confusion_matrix[class_idx, class_idx])
    predicted_positive = float(confusion_matrix[:, class_idx].sum())
    if predicted_positive == 0.0:
        return 0.0
    return true_positives / predicted_positive


def f1_from_confusion_matrix(confusion_matrix, class_idx: int) -> float:
    """Calcula el F1 para una clase concreta desde la matriz de confusion."""
    precision = precision_from_confusion_matrix(confusion_matrix, class_idx)
    true_positives = float(confusion_matrix[class_idx, class_idx])
    actual_positive = float(confusion_matrix[class_idx, :].sum())
    recall = 0.0 if actual_positive == 0.0 else true_positives / actual_positive
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def prediction_confidence_for_single_class(model, dataset, class_idx: int, build_loader_fn, args) -> float:
    """Calcula la confianza media max-softmax sobre una sola clase objetivo."""
    target_indices = np.flatnonzero(np.asarray(dataset.targets) == int(class_idx))
    if len(target_indices) == 0:
        return 0.0

    subset = torch.utils.data.Subset(dataset, target_indices.tolist())
    subset.targets = np.asarray(dataset.targets)[target_indices]
    loader = build_loader_fn(subset, args.batch_size, False, args.num_workers, args.seed)
    return evaluate_prediction_confidence(model, loader)


def derive_addition_summary_metrics(metrics_payload=None):
    """Proyecta un payload de adicion al esquema plano de metricas resumen."""
    if metrics_payload is None:
        return {}

    per_class_accuracy = metrics_payload.get("test_per_class_accuracy", {})
    metrics = {
        "tiempo_total_de_adaptacion": float(metrics_payload["elapsed_seconds"]),
        "accuracy_global": float(metrics_payload["test_overall_accuracy"]),
        "accuracy_por_clase": serialize_per_class_accuracy(per_class_accuracy),
        "accuracy_en_clases_previas": float(metrics_payload["test_accuracy_previous_classes"]),
        "accuracy_en_la_clase_anadida": float(metrics_payload["test_accuracy_added_class"]),
        "recall_en_la_clase_anadida": float(metrics_payload["test_recall_added_class"]),
        "precision_en_la_clase_anadida": float(metrics_payload["test_precision_added_class"]),
        "f1_en_la_clase_anadida": float(metrics_payload["test_f1_added_class"]),
        "forgetting_u_olvido_sobre_clases_previas": metrics_payload["forgetting_previous_classes"],
        "numero_de_ejemplos_utilizados": int(metrics_payload["num_examples_used_for_adaptation"]),
        "numero_de_ejemplos_de_la_clase_anadida": int(metrics_payload["num_examples_added_class_train"]),
        "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
        "confianza_media_en_la_clase_anadida": float(metrics_payload["prediction_confidence_added_class_mean"]),
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
            "Accuracy en clases previas": metrics["accuracy_en_clases_previas"],
            "Accuracy en la clase anadida": metrics["accuracy_en_la_clase_anadida"],
            "Recall en la clase anadida": metrics["recall_en_la_clase_anadida"],
            "Precision en la clase anadida": metrics["precision_en_la_clase_anadida"],
            "F1 de la clase anadida": metrics["f1_en_la_clase_anadida"],
            "Forgetting u olvido sobre clases previas": (
                metrics["forgetting_u_olvido_sobre_clases_previas"]
            ),
            "Numero de ejemplos utilizados": metrics["numero_de_ejemplos_utilizados"],
            "Numero de ejemplos de la clase anadida": (
                metrics["numero_de_ejemplos_de_la_clase_anadida"]
            ),
            "Confianza de prediccion": metrics["confianza_de_prediccion"],
            "Confianza media en la clase anadida": (
                metrics["confianza_media_en_la_clase_anadida"]
            ),
            "Numero de parametros entrenados o modificados": (
                metrics["numero_de_parametros_entrenados_o_modificados"]
            ),
            "Memoria adicional requerida": metrics["memoria_adicional_requerida"],
        }
    )
    return metrics


def build_addition_summary_row(
    dataset_name: str,
    model_name: str,
    added_class_name,
    final_num_classes: int,
    status: str,
    metrics_payload=None,
    error_message=None,
):
    """Aplana los campos mas utiles de un experimento de adicion."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "added_class": str(added_class_name),
        "final_num_classes": int(final_num_classes),
        "status": status,
    }

    if metrics_payload is not None:
        row.update(
            {
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
                "test_overall_accuracy": float(metrics_payload["test_overall_accuracy"]),
                "elapsed_seconds": float(metrics_payload["elapsed_seconds"]),
            }
        )
        row.update(derive_addition_summary_metrics(metrics_payload))

    if error_message is not None:
        row["error"] = error_message

    return row


def build_addition_finetuning_summary_row(
    dataset_name: str,
    model_name: str,
    added_class_name,
    final_num_classes: int,
    status: str,
    training_setup: dict,
    metrics_payload=None,
    error_message=None,
):
    """Aplana un experimento de adicion por fine-tuning con su metadata de entrenamiento."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "added_class": str(added_class_name),
        "final_num_classes": int(final_num_classes),
        "initialization": "reference_without_added_class",
        "training_mode": training_setup["mode_label"],
        "backbone_mode": training_setup["backbone_mode"],
        "trainable_scope": training_setup["trainable_scope"],
        "train_percentage": float(training_setup["train_percentage"]),
        "status": status,
    }

    if metrics_payload is not None:
        row.update(
            {
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
                "test_overall_accuracy": float(metrics_payload["test_overall_accuracy"]),
                "elapsed_seconds": float(metrics_payload["elapsed_seconds"]),
            }
        )
        row.update(derive_addition_summary_metrics(metrics_payload))

    if error_message is not None:
        row["error"] = error_message

    return row


def load_addition_reference_artifacts(reference_dir: Path, dataset_name: str, model_name: str, added_class_name):
    """Carga metricas y checkpoint de la referencia de 9 clases para una clase anadida futura."""
    experiment_dir = (
        Path(reference_dir)
        / slugify(dataset_name)
        / slugify(model_name)
        / slugify(added_class_name)
    )
    metrics_path = experiment_dir / "final_metrics.json"
    checkpoint_path = experiment_dir / "modelo_base_9_clases.pth"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Missing 9-class reference metrics at "
            f"'{metrics_path}'. Run experiments/baseline_addition_after_class_introduction.py "
            "or pass --reference-dir with the correct location."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing 9-class reference checkpoint at "
            f"'{checkpoint_path}'. Run experiments/baseline_addition_after_class_introduction.py "
            "or pass --reference-dir with the correct location."
        )

    metrics_payload = load_json(metrics_path)
    per_class_accuracy = metrics_payload.get("test_per_class_accuracy")
    if not isinstance(per_class_accuracy, dict) or not per_class_accuracy:
        raise ValueError(
            f"Reference metrics at '{metrics_path}' do not contain a valid "
            "'test_per_class_accuracy' mapping."
        )
    return metrics_payload, metrics_path, checkpoint_path


def build_addition_metrics_payload(
    *,
    dataset_name: str,
    model_name: str,
    added_class_name,
    original_classes: list,
    train_ds,
    test_ds,
    args,
    training_result: dict,
    elapsed: float,
    test_overall_accuracy: float,
    per_class_accuracy,
    confusion_matrix,
    split_counts: dict,
    prediction_confidence_mean: float,
    num_trainable_parameters: int,
    model,
    build_loader_fn,
    reference_metrics: dict,
    reference_metrics_path,
    reference_checkpoint_path,
):
    """Construye el payload final del baseline de adicion con metricas reutilizables."""
    test_per_class_accuracy = {
        class_name: float(per_class_accuracy[class_idx])
        for class_idx, class_name in enumerate(original_classes)
    }
    added_class_idx = original_classes.index(added_class_name)
    added_class_mask_train = np.asarray(train_ds.targets) == added_class_idx
    added_class_mask_test = np.asarray(test_ds.targets) == added_class_idx

    current_previous_class_accuracy = {
        class_name: accuracy
        for class_name, accuracy in test_per_class_accuracy.items()
        if class_name != added_class_name
    }
    forgetting_previous_classes = compute_previous_class_forgetting(
        reference_per_class_accuracy=reference_metrics["test_per_class_accuracy"],
        current_per_class_accuracy=test_per_class_accuracy,
        added_class_name=added_class_name,
    )

    prediction_confidence_added_class_mean = 0.0
    if added_class_mask_test.any():
        prediction_confidence_added_class_mean = float(
            prediction_confidence_for_single_class(
                model=model,
                dataset=test_ds,
                class_idx=added_class_idx,
                build_loader_fn=build_loader_fn,
                args=args,
            )
        )

    metrics_payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "added_class": str(added_class_name),
        "final_num_classes": int(len(original_classes)),
        "selection_metric": "validation_loss",
        "max_epochs": int(args.epochs),
        "patience": int(args.patience),
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "initialization": "zero" if args.zero_init else "scratch",
        "best_epoch": int(training_result["best_epoch"]),
        "epochs_ran": int(training_result["epochs_ran"]),
        "best_val_loss": float(training_result["best_val_loss"]),
        "best_val_accuracy": float(training_result["best_val_accuracy"]),
        "elapsed_seconds": float(elapsed),
        "test_overall_accuracy": float(test_overall_accuracy),
        "test_per_class_accuracy": test_per_class_accuracy,
        "test_mean_per_class_accuracy": float(np.mean(per_class_accuracy)),
        "test_accuracy_previous_classes": float(np.mean(list(current_previous_class_accuracy.values()))),
        "test_accuracy_added_class": float(test_per_class_accuracy[added_class_name]),
        "test_precision_added_class": precision_from_confusion_matrix(confusion_matrix, added_class_idx),
        "test_recall_added_class": float(per_class_accuracy[added_class_idx]),
        "test_f1_added_class": f1_from_confusion_matrix(confusion_matrix, added_class_idx),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": list(original_classes),
        "examples_per_split": split_counts,
        "num_examples_used_for_adaptation": int(sum(split_counts.get("train", {}).values())),
        "num_examples_added_class_train": int(added_class_mask_train.sum()),
        "prediction_confidence_mean": float(prediction_confidence_mean),
        "prediction_confidence_added_class_mean": float(prediction_confidence_added_class_mean),
        "num_trainable_parameters": int(num_trainable_parameters),
        "additional_memory_required": 0.0,
        "forgetting_previous_classes": None if forgetting_previous_classes is None else float(forgetting_previous_classes),
        "reference_9_class_metrics_path": str(reference_metrics_path),
        "reference_9_class_checkpoint_path": str(reference_checkpoint_path),
        "metricas_adicion": [metrica.nombre for metrica in METRICAS_ADICION],
        "stores_model_checkpoint": False,
    }
    metrics_payload["summary"] = build_addition_summary_row(
        dataset_name=dataset_name,
        model_name=model_name,
        added_class_name=added_class_name,
        final_num_classes=len(original_classes),
        status="completed",
        metrics_payload=metrics_payload,
    )
    return metrics_payload

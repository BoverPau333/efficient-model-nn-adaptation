"""Utilidades compartidas para resultados y configuracion de experimentos."""

import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.experiments_config.config import DEVICE
from src.models import IMAGENET_MODEL_BUILDERS


def set_seed(seed: int):
    """Fija las semillas principales para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def slugify(value) -> str:
    """Crea nombres seguros para filesystem."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "item"


def parse_class_identifier(raw_value):
    """Interpreta un identificador de clase como int cuando sea posible."""
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    return raw_value


def save_json(path: Path, payload):
    """Guarda un JSON estable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path):
    """Carga un JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list):
    """Escribe filas en CSV respetando el orden de aparicion de columnas."""
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
    """Construye un DataLoader determinista."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def count_trainable_parameters(model) -> int:
    """Cuenta los parametros entrenables."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_prediction_confidence(model, loader) -> float:
    """Calcula la confianza media max-softmax."""
    model.eval()
    confidences = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(next(model.parameters()).device)
            probs = F.softmax(model(imgs), dim=1)
            confidences.extend(probs.max(dim=1).values.cpu().numpy().tolist())

    if not confidences:
        return 0.0
    return float(np.mean(confidences))


def compute_forgetting_from_reference(reference_per_class_accuracy: dict, current_per_class_accuracy: dict):
    """Calcula el forgetting medio frente al entrenamiento de referencia."""
    if reference_per_class_accuracy is None or current_per_class_accuracy is None:
        return None

    remaining_classes = list(current_per_class_accuracy)
    if not remaining_classes:
        return None

    missing_classes = [
        class_name for class_name in remaining_classes if class_name not in reference_per_class_accuracy
    ]
    if missing_classes:
        return None

    degradations = [
        float(reference_per_class_accuracy[class_name]) - float(current_per_class_accuracy[class_name])
        for class_name in remaining_classes
    ]
    return float(np.mean(degradations))


def evaluate_classification_metrics(model, loader, class_names: list):
    """Evalua accuracy, F1 y accuracy por clase."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    num_classes = len(class_names)
    per_class_accuracy = {}
    for class_idx, class_name in enumerate(class_names):
        mask = all_labels == class_idx
        per_class_accuracy[class_name] = float((all_preds[mask] == all_labels[mask]).mean()) if mask.any() else 0.0

    accuracy = float((all_preds == all_labels).mean()) if len(all_labels) else 0.0
    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "per_class_accuracy": per_class_accuracy,
        "labels": all_labels.tolist(),
        "predictions": all_preds.tolist(),
        "mean_per_class_accuracy": float(np.mean(list(per_class_accuracy.values()))) if per_class_accuracy else 0.0,
    }


def _get_linear_head(model):
    """Localiza la ultima capa lineal usada como cabecera."""
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        return "fc", model.fc
    if hasattr(model, "classifier"):
        for idx in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[idx]
            if isinstance(layer, torch.nn.Linear):
                return ("classifier", idx), layer
    raise AttributeError("No se pudo localizar la cabecera lineal del modelo")


def remove_output_class(model, removed_class_idx: int):
    """Reduce la cabecera del modelo eliminando una salida concreta."""
    head_ref, head = _get_linear_head(model)
    new_head = torch.nn.Linear(head.in_features, head.out_features - 1).to(head.weight.device)

    kept_indices = [idx for idx in range(head.out_features) if idx != int(removed_class_idx)]
    with torch.no_grad():
        new_head.weight.copy_(head.weight[kept_indices])
        new_head.bias.copy_(head.bias[kept_indices])

    if head_ref == "fc":
        model.fc = new_head
    else:
        _, idx = head_ref
        model.classifier[idx] = new_head
    return model


def freeze_backbone_keep_head_trainable(model):
    """Congela el cuerpo del modelo y deja entrenable solo la cabecera lineal final."""
    head_ref, head = _get_linear_head(model)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if head_ref == "fc":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model


def load_reference_model(reference_dir: Path, dataset_name: str, model_name: str, num_classes: int):
    """Carga el modelo de referencia entrenado y sus metricas."""
    experiment_dir = reference_dir / slugify(dataset_name) / slugify(model_name)
    metrics_path = experiment_dir / "final_metrics.json"
    checkpoint_path = experiment_dir / "best_model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe '{metrics_path}'")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe '{checkpoint_path}'")

    metrics = load_json(metrics_path)
    model = IMAGENET_MODEL_BUILDERS[model_name](num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    return model, metrics, checkpoint_path, metrics_path


def maybe_load_baseline_summary_row(
    baseline_dir: Path,
    dataset_name: str,
    model_name: str,
    modified_class_name,
):
    """Intenta recuperar una fila del baseline de eliminacion, si existe."""
    summary_path = baseline_dir / slugify(dataset_name) / "experiments_summary.json"
    if not summary_path.exists():
        return None

    rows = load_json(summary_path)
    for row in rows:
        if (
            row.get("dataset") == dataset_name
            and row.get("model_name") == model_name
            and str(row.get("removed_class")) == str(modified_class_name)
            and row.get("status") in {"completed", "skipped_existing"}
        ):
            return row
    return None


def derive_dynamic_summary_metrics(metrics_payload: dict):
    """Proyecta el resultado al esquema de resumen del proyecto y anade nuevas columnas."""
    per_class_accuracy = metrics_payload.get("test_per_class_accuracy", {})
    serialized_per_class_accuracy = json.dumps(per_class_accuracy, ensure_ascii=True, sort_keys=True)

    metrics = {
        "tiempo_total_de_adaptacion": float(metrics_payload["total_time"]),
        "accuracy_global": float(metrics_payload["accuracy"]),
        "accuracy_por_clase": serialized_per_class_accuracy,
        "accuracy_en_clases_restantes": float(metrics_payload["mean_per_class_accuracy"]),
        "forgetting_u_olvido": metrics_payload.get("forgetting_score"),
        "numero_de_ejemplos_utilizados": int(metrics_payload["num_training_samples"]),
        "confianza_de_prediccion": float(metrics_payload["prediction_confidence_mean"]),
        "numero_de_parametros_entrenados_o_modificados": int(metrics_payload["num_trainable_parameters"]),
        "memoria_adicional_requerida": float(metrics_payload.get("additional_memory_required", 0.0)),
        "method": metrics_payload["method"],
        "embedding_strategy": metrics_payload["embedding_strategy"],
        "total_time": float(metrics_payload["total_time"]),
        "embedding_time": float(metrics_payload["embedding_time"]),
        "selection_time": float(metrics_payload["selection_time"]),
        "finetuning_time": float(metrics_payload["finetuning_time"]),
        "evaluation_time": float(metrics_payload["evaluation_time"]),
        "num_training_samples": int(metrics_payload["num_training_samples"]),
        "num_selected_classes": int(metrics_payload["num_selected_classes"]),
        "modified_class": str(metrics_payload["modified_class"]),
        "k_neighbours": int(metrics_payload["k_neighbours"]),
        "accuracy": float(metrics_payload["accuracy"]),
        "f1_macro": float(metrics_payload["f1_macro"]),
        "f1_weighted": float(metrics_payload["f1_weighted"]),
        "forgetting_score": metrics_payload.get("forgetting_score"),
    }
    return metrics


def build_dynamic_summary_row(
    dataset_name: str,
    model_name: str,
    update_type: str,
    status: str,
    metrics_payload=None,
    error_message=None,
):
    """Aplana un resultado de fine-tuning dinamico al formato de tablas existente."""
    row = {
        "dataset": dataset_name,
        "model_name": model_name,
        "removed_class": str(metrics_payload["modified_class"]) if metrics_payload is not None and update_type == "remove" else "__none__",
        "final_num_classes": None if metrics_payload is None else int(metrics_payload["final_num_classes"]),
        "status": status,
    }

    if metrics_payload is not None:
        row.update(
            {
                "best_epoch": int(metrics_payload["best_epoch"]),
                "epochs_ran": int(metrics_payload["epochs_ran"]),
                "best_val_loss": float(metrics_payload["best_val_loss"]),
                "best_val_accuracy": float(metrics_payload["best_val_accuracy"]),
                "update_type": update_type,
                "distance_metric": metrics_payload["distance_metric"],
                "selection_strategy": metrics_payload["selection_strategy"],
            }
        )
        row.update(derive_dynamic_summary_metrics(metrics_payload))

    if error_message is not None:
        row["error"] = error_message

    return row

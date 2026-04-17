"""Utilidades para extraer embeddings y predicciones de un modelo."""

import numpy as np
import torch

from src.config import DEVICE


def extraer_embeddings_y_logits(model, loader):
    """Extrae embeddings, logits y etiquetas de un dataloader.

    El modelo debe exponer un metodo `forward_embeddings_and_logits(imgs)`
    que devuelva una tupla `(embeddings, logits)`.
    """
    if not hasattr(model, "forward_embeddings_and_logits"):
        raise AttributeError(
            "El modelo no define `forward_embeddings_and_logits`, necesario para extraer embeddings"
        )

    model.eval()

    todos_embeddings = []
    todos_logits = []
    todas_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            embeddings, logits = model.forward_embeddings_and_logits(imgs)

            todos_embeddings.append(embeddings.cpu().numpy())
            todos_logits.append(logits.cpu().numpy())
            todas_labels.append(labels.cpu().numpy())

    return {
        "embeddings": np.concatenate(todos_embeddings, axis=0),
        "logits": np.concatenate(todos_logits, axis=0),
        "labels": np.concatenate(todas_labels, axis=0),
    }

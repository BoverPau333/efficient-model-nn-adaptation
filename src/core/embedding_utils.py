"""Utilidades para extraer embeddings con indices de muestra."""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.experiments_config.config import DEVICE


class IndexedDataset(Dataset):
    """Wrapper que anade el indice original al ejemplo devuelto por el dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = getattr(dataset, "targets", None)
        self.classes = getattr(dataset, "classes", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        return sample, label, idx


def extract_embeddings(model, loader, representation: str = "embeddings", use_grad: bool = False):
    """Extrae representaciones, labels e indices de un dataloader."""
    if not hasattr(model, "forward_embeddings_and_logits"):
        raise AttributeError("El modelo debe implementar `forward_embeddings_and_logits`")
    if representation not in {"embeddings", "logits"}:
        raise ValueError("representation debe ser 'embeddings' o 'logits'")

    model.eval()
    all_vectors = []
    all_labels = []
    all_ids = []

    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        for imgs, labels, ids in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            embeddings, logits = model.forward_embeddings_and_logits(imgs)
            vectors = embeddings if representation == "embeddings" else logits

            all_vectors.append(vectors.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_ids.append(np.asarray(ids))

    if not all_vectors:
        raise ValueError("El dataloader no contiene ejemplos")

    return {
        "vectors": np.concatenate(all_vectors, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "ids": np.concatenate(all_ids, axis=0),
        "representation": representation,
    }

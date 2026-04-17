"""Training and evaluation helpers."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.config import DEVICE, EPOCHS, LR


def finetune(model, trainloader, valloader, epochs=EPOCHS, lr=LR):
    """Fine-tune trainable parameters and keep the best validation weights."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    best_val = float("inf")
    best_weights = None
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train = 0.0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        train_losses.append(running_train / len(trainloader))

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in valloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                running_val += loss_fn(model(imgs), labels).item()

        avg_val = running_val / len(valloader)
        val_losses.append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return train_losses, val_losses, best_epoch


def evaluate(model, loader, num_classes: int):
    """Return overall accuracy, per-class accuracy and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall = float((all_preds == all_labels).mean())
    cm = sk_confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    per_class = np.zeros(num_classes)
    for class_idx in range(num_classes):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            per_class[class_idx] = (all_preds[mask] == all_labels[mask]).mean()

    return overall, per_class, cm

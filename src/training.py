"""Training and evaluation helpers."""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.experiments_config.config import DEVICE, EPOCHS, LR


def train_with_early_stopping(
    model,
    trainloader,
    valloader,
    epochs=EPOCHS,
    lr=LR,
    patience=None,
    checkpoint_path=None,
    verbose=False,
):
    """Train a model and restore the best weights according to validation loss."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        running_train_examples = 0

        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_train_examples += int(labels.size(0))

        avg_train_loss = running_train_loss / len(trainloader)
        train_accuracy = running_train_correct / max(running_train_examples, 1)

        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        running_val_examples = 0
        with torch.no_grad():
            for imgs, labels in valloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                running_val_loss += loss_fn(logits, labels).item()
                running_val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                running_val_examples += int(labels.size(0))

        avg_val_loss = running_val_loss / len(valloader)
        val_accuracy = running_val_correct / max(running_val_examples, 1)
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(avg_val_loss),
            "val_accuracy": float(val_accuracy),
        }
        history.append(epoch_info)

        if verbose:
            print(
                f"    Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | train_acc={train_accuracy:.4f} | "
                f"val_loss={avg_val_loss:.4f} | val_acc={val_accuracy:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": best_weights,
                        "best_val_loss": float(best_val_loss),
                        "best_val_accuracy": float(best_val_accuracy),
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch + 1} (patience={patience}).")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "epochs_ran": len(history),
    }


def finetune(model, trainloader, valloader, epochs=EPOCHS, lr=LR):
    """Fine-tune trainable parameters and keep the best validation weights."""
    result = train_with_early_stopping(
        model,
        trainloader,
        valloader,
        epochs=epochs,
        lr=lr,
        patience=None,
        checkpoint_path=None,
        verbose=False,
    )
    train_losses = [epoch["train_loss"] for epoch in result["history"]]
    val_losses = [epoch["val_loss"] for epoch in result["history"]]
    return train_losses, val_losses, result["best_epoch"]


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

"""
reduce_and_evaluate.py
======================
Studies how classification accuracy degrades when training examples are reduced.

Two reduction strategies are implemented:

1. reduce_all_classes(dataset, fraction)
   Randomly removes (1 - fraction) × 100 % of examples from EVERY class.

2. reduce_least_confused_class(dataset, classes, confusion_matrix, fraction)
   Removes examples only from the class that has the LOWEST off-diagonal
   confusion rate — i.e. the class the model has "most confidently learned"
   and that therefore may need fewer examples.

Both functions return a Subset of the original dataset.

The main experiment function  run_reduction_study()  iterates over a list of
retention fractions, fine-tunes only the classification head (frozen backbone)
from scratch each time, and records per-class and overall accuracy.

Model used: MobileNetV3-Small  (lightest, best accuracy in prior experiments)

Usage
-----
    python reduce_and_evaluate.py

Outputs
-------
    results/reduction_all_classes.csv
    results/reduction_least_confused.csv
    (plots are saved to results/plots/)
"""

import os
import copy
import time
import csv

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from similarity_visualization import visualize_top_confused_pairs

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = "./datasets/data"
RESULTS_DIR  = "./results"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
EPOCHS       = 5
LR           = 1e-3
BATCH_SIZE   = 32
NUM_WORKERS  = 2
SEED         = 42

# Fractions of training data to KEEP  (1.0 = full dataset, 0.1 = 10 % kept)
RETENTION_FRACTIONS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Model builder  –  MobileNetV3-Small, frozen backbone, new head
# ─────────────────────────────────────────────────────────────────────────────
def build_mobilenet(num_classes: int) -> nn.Module:
    """
    Build a MobileNetV3-Small with an ImageNet-pretrained backbone (frozen)
    and a fresh classification head for `num_classes` outputs.
    Each call returns a freshly initialised head, so every experiment starts
    from the same pre-trained features but an untrained classifier.
    """
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False                       # freeze backbone
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, num_classes)   # new trainable head
    return m.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Reduction strategy 1 – uniform random reduction across ALL classes
# ─────────────────────────────────────────────────────────────────────────────
def reduce_all_classes(dataset, fraction: float, seed: int = SEED) -> Subset:
    """
    Randomly keep `fraction` of samples from every class.

    Parameters
    ----------
    dataset  : a dataset with a `.targets` attribute (list / array of int labels)
    fraction : float in (0, 1] — proportion of examples to KEEP per class
    seed     : random seed for reproducibility

    Returns
    -------
    torch.utils.data.Subset of `dataset`
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng     = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    kept    = []

    for cls_idx in np.unique(targets):
        idx_cls = np.where(targets == cls_idx)[0]
        n_keep  = max(1, int(round(len(idx_cls) * fraction)))
        chosen  = rng.choice(idx_cls, size=n_keep, replace=False)
        kept.extend(chosen.tolist())

    return Subset(dataset, kept)


# ─────────────────────────────────────────────────────────────────────────────
# Reduction strategy 2 – targeted reduction of the least-confused class
# ─────────────────────────────────────────────────────────────────────────────
def find_least_confused_class(cm: np.ndarray) -> int:
    """
    Given a confusion matrix `cm` (rows = true, cols = predicted),
    return the class index with the lowest total off-diagonal confusion rate.

    "Least confused" = the class that is most reliably distinguished from all
    others, i.e. the one where the model has learned the most and may therefore
    need fewer training examples.
    """
    cm_float = cm.astype(float)
    row_sums  = cm_float.sum(axis=1)

    # Off-diagonal confusion rate per class
    off_diag_errors = row_sums - np.diag(cm_float)
    confusion_rates = np.divide(
        off_diag_errors, row_sums,
        out=np.ones_like(row_sums),
        where=row_sums > 0
    )
    return int(np.argmin(confusion_rates))


def reduce_least_confused_class(dataset, cm: np.ndarray,
                                 fraction: float, seed: int = SEED) -> Subset:
    """
    Reduce examples only from the class with the lowest confusion rate.

    The rest of the classes keep all their examples.

    Parameters
    ----------
    dataset  : dataset with `.targets`
    cm       : confusion matrix from a prior full-data evaluation (numpy array)
    fraction : proportion of examples to KEEP for the least-confused class
    seed     : random seed

    Returns
    -------
    torch.utils.data.Subset
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    target_cls = find_least_confused_class(cm)
    rng        = np.random.default_rng(seed)
    targets    = np.array(dataset.targets)
    kept       = []

    for cls_idx in np.unique(targets):
        idx_cls = np.where(targets == cls_idx)[0]
        if cls_idx == target_cls:
            n_keep = max(1, int(round(len(idx_cls) * fraction)))
            chosen = rng.choice(idx_cls, size=n_keep, replace=False)
            kept.extend(chosen.tolist())
        else:
            kept.extend(idx_cls.tolist())

    return Subset(dataset, kept)


# ─────────────────────────────────────────────────────────────────────────────
# Class example counter
# ─────────────────────────────────────────────────────────────────────────────
def count_examples_per_class(dataset, classes: list) -> dict:
    """
    Return a dict {class_name: count} for a dataset or Subset.
    """
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "indices"):
        base    = dataset.dataset
        targets = np.array(base.targets)[dataset.indices]
    else:
        raise AttributeError("Cannot extract targets.")

    counts = {}
    for i, name in enumerate(classes):
        counts[name] = int((targets == i).sum())
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (fine-tuning only the head, from scratch each call)
# ─────────────────────────────────────────────────────────────────────────────
def finetune(model, trainloader, valloader, epochs=EPOCHS, lr=LR):
    """
    Fine-tune only the trainable parameters (classification head) for
    `epochs` epochs.  Keeps the best weights by validation loss.

    Returns train_losses, val_losses, best_epoch (1-based).
    """
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    best_val  = float("inf")
    best_wts  = None
    best_ep   = 0
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_tr = running / len(trainloader)
        train_losses.append(avg_tr)

        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for imgs, labels in valloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                running_v += loss_fn(model(imgs), labels).item()
        avg_val = running_v / len(valloader)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_ep  = ep + 1
            best_wts = copy.deepcopy(model.state_dict())

    if best_wts:
        model.load_state_dict(best_wts)

    return train_losses, val_losses, best_ep


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  →  overall accuracy + per-class accuracy + confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, num_classes: int):
    """
    Returns
    -------
    overall_acc : float
    per_class_acc : np.ndarray  shape (num_classes,)
    cm : np.ndarray  shape (num_classes, num_classes)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall = (all_preds == all_labels).mean()
    cm      = sk_confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    per_class = np.zeros(num_classes)
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[i] = (all_preds[mask] == all_labels[mask]).mean()

    return float(overall), per_class, cm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders  (return raw train/val/test splits + classes)
# ─────────────────────────────────────────────────────────────────────────────
def load_cifar10():
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    n_tr = int(0.8 * len(full))
    tr, va = random_split(full, [n_tr, len(full) - n_tr],
                          generator=torch.Generator().manual_seed(SEED))
    # Attach targets to Subsets for easy access
    tr.targets = np.array(full.targets)[tr.indices]
    va.targets = np.array(full.targets)[va.indices]
    return tr, va, test, full.classes


def load_fashion_mnist():
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    n_tr = int(0.8 * len(full))
    tr, va = random_split(full, [n_tr, len(full) - n_tr],
                          generator=torch.Generator().manual_seed(SEED))
    tr.targets = np.array(full.targets.numpy())[tr.indices]
    va.targets = np.array(full.targets.numpy())[va.indices]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return tr, va, test, classes


FRUITS_SELECTED = [
    "apple_red_1", "apple_golden_1", "apple_granny_smith_1",
    "apple_pink_lady_1", "apple_red_delicios_1", "apple_crimson_snow_1",
    "Pear 1", "Pear 3", "Pear 5", "Pear 6", "Pear 7",
    "Tomato 1", "Tomato 5", "Tomato 7", "Tomato 8", "Tomato 9", "Tomato Maroon 2",
    "Blackberry 1", "Raspberry 2", "Strawberry 2",
    "Orange 3", "Banana 4",
    "Zucchini Green 1", "Cucumber 1", "Cucumber 5", "Avocado Green 1",
    "Nut 3", "Almonds 1",
]


class FilteredImageFolder(ImageFolder):
    def __init__(self, root, selected_classes, transform=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = {c: i for i, c in enumerate(selected_classes)}
        self.samples = [
            (path, self.class_to_idx[os.path.basename(os.path.dirname(path))])
            for path, _ in self.imgs
            if os.path.basename(os.path.dirname(path)) in self.class_to_idx
        ]
        self.targets = [s[1] for s in self.samples]
        self.classes = selected_classes


def load_fruits360():
    fruits_path = os.path.join(DATA_DIR, "fruits", "fruits-360_original-size",
                               "fruits-360-original-size")
    if not os.path.isdir(fruits_path):
        raise FileNotFoundError(
            "Fruits-360 not found. Run:\n"
            "  kaggle datasets download moltean/fruits -p data\n"
            "  unzip -q -o data/fruits.zip -d data/fruits"
        )
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
        transforms.RandomGrayscale(p=0.15),
        transforms.GaussianBlur(5, sigma=(0.5, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr_ds = FilteredImageFolder(os.path.join(fruits_path, "Training"),   FRUITS_SELECTED, train_tf)
    va_ds = FilteredImageFolder(os.path.join(fruits_path, "Validation"), FRUITS_SELECTED, test_tf)
    te_ds = FilteredImageFolder(os.path.join(fruits_path, "Test"),       FRUITS_SELECTED, test_tf)
    return tr_ds, va_ds, te_ds, FRUITS_SELECTED


def load_paintings():
    dataset_path = os.path.join(DATA_DIR, "painting", "dataset", "dataset_updated")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            "Paintings not found. Run:\n"
            "  kaggle datasets download -d thedownhill/art-images-drawings-painting-sculpture-engraving -p data\n"
            "  unzip -q -o data/art-images-drawings-painting-sculpture-engraving.zip -d data/painting"
        )
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class SafeImageFolder(ImageFolder):
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            try:   sample = self.loader(path)
            except Exception: return self.__getitem__((idx + 1) % len(self))
            if self.transform: sample = self.transform(sample)
            return sample, target

    full = SafeImageFolder(os.path.join(dataset_path, "training_set"),   transform=train_tf)
    test = SafeImageFolder(os.path.join(dataset_path, "validation_set"), transform=test_tf)
    n_va = int(0.2 * len(full))
    tr, va = random_split(full, [len(full) - n_va, n_va],
                          generator=torch.Generator().manual_seed(SEED))
    tr.targets = np.array(full.targets)[tr.indices]
    va.targets = np.array(full.targets)[va.indices]
    return tr, va, test, full.classes


DATASET_LOADERS = {
    "CIFAR-10":       load_cifar10,
    "Fashion-MNIST":  load_fashion_mnist,
    "Paintings":      load_paintings,
    "Fruits-360":     load_fruits360,
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def _write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_overall_accuracy_vs_fraction(results_by_dataset: dict, strategy_label: str,
                                       save_path: str = None):
    """
    Line plot: x = retention fraction, y = overall test accuracy.
    One line per dataset.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results_by_dataset)))

    for (ds_name, rows), color in zip(results_by_dataset.items(), colors):
        fracs = [r["fraction"] for r in rows]
        accs  = [r["overall_test_acc"] for r in rows]
        ax.plot(fracs, accs, marker="o", linewidth=2, color=color, label=ds_name)
        for f, a in zip(fracs, accs):
            ax.annotate(f"{a:.2f}", (f, a), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=color)

    ax.set_xlabel("Retention fraction (1.0 = full training set)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title(f"Overall Accuracy vs. Training Data Size\n({strategy_label})",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot → {save_path}")
    plt.show()
    plt.close(fig)


def plot_per_class_accuracy_heatmap(results: list, classes: list,
                                     ds_name: str, strategy_label: str,
                                     save_path: str = None):
    """
    Heatmap: rows = class, columns = retention fraction.
    Cell colour = per-class test accuracy.
    """
    fracs       = [r["fraction"] for r in results]
    n_cls       = len(classes)
    data        = np.zeros((n_cls, len(fracs)))

    for j, row in enumerate(results):
        for i in range(n_cls):
            key = f"per_class_acc_{i}"
            data[i, j] = row.get(key, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(fracs) * 1.2), max(5, n_cls * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Test Accuracy")

    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=9)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Retention fraction", fontsize=11)
    ax.set_title(f"{ds_name} – Per-class Accuracy vs. Data Size\n({strategy_label})",
                 fontsize=12, fontweight="bold")

    for i in range(n_cls):
        for j in range(len(fracs)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if data[i, j] < 0.5 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot → {save_path}")
    plt.show()
    plt.close(fig)


def plot_class_examples_bar(counts: dict, ds_name: str, save_path: str = None):
    """
    Horizontal bar chart showing number of training examples per class
    (at 100 % retention).
    """
    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(classes)))

    fig, ax = plt.subplots(figsize=(8, max(4, len(classes) * 0.35)))
    bars = ax.barh(classes, values, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    ax.set_xlabel("Number of training examples", fontsize=10)
    ax.set_title(f"{ds_name} – Training examples per class (100 % retention)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot → {save_path}")
    plt.show()
    plt.close(fig)


def plot_least_confused_reduction(results: list, classes: list,
                                   least_confused_cls: str,
                                   ds_name: str, save_path: str = None):
    """
    Dual-panel plot for the targeted reduction experiment:
      Left  – overall accuracy vs. fraction kept for the least-confused class
      Right – per-class accuracy at 100 % vs. at minimum fraction
    """
    fracs        = [r["fraction"] for r in results]
    overall_accs = [r["overall_test_acc"] for r in results]
    n_cls        = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall accuracy curve
    ax = axes[0]
    ax.plot(fracs, overall_accs, marker="o", linewidth=2.5,
            color="#E53935", label="Overall test acc")
    for f, a in zip(fracs, overall_accs):
        ax.annotate(f"{a:.3f}", (f, a), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#E53935")
    ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Fraction kept for\n least-confused class", fontsize=10)
    ax.set_ylabel("Overall Test Accuracy", fontsize=10)
    ax.set_title(
        f"{ds_name}\nOverall accuracy vs. fraction kept\nfor '{least_confused_cls}'",
        fontsize=10, fontweight="bold")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: per-class comparison (100 % vs. minimum fraction)
    ax2    = axes[1]
    full   = results[0]   # fraction = 1.0
    sparse = results[-1]  # smallest fraction
    acc_full   = [full.get(f"per_class_acc_{i}", 0) for i in range(n_cls)]
    acc_sparse = [sparse.get(f"per_class_acc_{i}", 0) for i in range(n_cls)]
    x = np.arange(n_cls)
    w = 0.38
    bars1 = ax2.bar(x - w/2, acc_full,   w, color="#42A5F5", label="100% data",       edgecolor="white")
    bars2 = ax2.bar(x + w/2, acc_sparse, w, color="#EF9A9A",
                    label=f"{int(sparse['fraction']*100)}% (least-confused class)", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Per-class Test Accuracy", fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(f"{ds_name}\nPer-class accuracy: full vs. reduced\nfor '{least_confused_cls}'",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot → {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runners
# ─────────────────────────────────────────────────────────────────────────────
def run_reduction_all_classes(ds_name: str, tr_full, va_split, te_split,
                               classes: list, fractions=RETENTION_FRACTIONS):
    """
    Strategy 1: uniformly reduce all classes.

    For each retention fraction:
      - subsample training data
      - build a fresh MobileNetV3-Small head
      - fine-tune for EPOCHS epochs
      - evaluate on the fixed test set

    Returns list of result dicts (one per fraction).
    """
    num_classes = len(classes)
    rows = []

    # Build fixed val/test loaders (never reduced)
    va_loader = DataLoader(va_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n{'─'*60}")
    print(f"  [{ds_name}] Strategy: Reduce ALL classes")
    print(f"{'─'*60}")

    # Count examples at full size
    full_counts = count_examples_per_class(tr_full, classes)
    plot_class_examples_bar(
        full_counts, ds_name,
        save_path=os.path.join(PLOTS_DIR, f"{ds_name.replace(' ','_')}_examples_per_class.png")
    )

    for frac in fractions:
        print(f"\n  Fraction kept: {frac:.0%}")

        subset    = reduce_all_classes(tr_full, fraction=frac)
        tr_loader = DataLoader(subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        counts = count_examples_per_class(subset, classes)
        total  = sum(counts.values())
        print(f"  Total training examples: {total}  "
              f"(per class min={min(counts.values())} max={max(counts.values())})")

        model = build_mobilenet(num_classes)
        t0    = time.time()
        finetune(model, tr_loader, va_loader)
        elapsed = time.time() - t0

        overall, per_cls, cm = evaluate(model, te_loader, num_classes)
        print(f"  Test acc: {overall:.4f}  time: {elapsed:.1f}s")

        row = {
            "dataset":          ds_name,
            "fraction":         frac,
            "total_train":      total,
            "overall_test_acc": round(overall, 6),
            "elapsed_seconds":  round(elapsed, 2),
        }
        for i, name in enumerate(classes):
            row[f"per_class_acc_{i}"] = round(float(per_cls[i]), 6)
            row[f"n_train_{name}"]    = counts[name]
        rows.append(row)
        del model

    return rows


def run_reduction_least_confused(ds_name: str, tr_full, va_split, te_split,
                                  classes: list, fractions=RETENTION_FRACTIONS):
    """
    Strategy 2: reduce only the least-confused class.

    First runs a full-data evaluation to identify the least-confused class,
    then iterates over retention fractions applied only to that class.

    Returns list of result dicts (one per fraction) + the least-confused class name.
    """
    num_classes = len(classes)
    rows = []

    va_loader = DataLoader(va_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te_split, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    tr_loader_full = DataLoader(tr_full, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print(f"\n{'─'*60}")
    print(f"  [{ds_name}] Strategy: Reduce LEAST-CONFUSED class")
    print(f"{'─'*60}")

    # ── Step 1: train once on full data to get a confusion matrix ──────────
    print("  Training on full data to identify least-confused class...")
    model_full = build_mobilenet(num_classes)
    finetune(model_full, tr_loader_full, va_loader)
    _, _, cm_full = evaluate(model_full, te_loader, num_classes)
    del model_full

    lc_idx  = find_least_confused_class(cm_full)
    lc_name = classes[lc_idx]
    print(f"  Least-confused class: '{lc_name}' (index {lc_idx})")

    # ── Step 2: iterate over fractions ────────────────────────────────────
    for frac in fractions:
        print(f"\n  Fraction kept for '{lc_name}': {frac:.0%}")

        subset    = reduce_least_confused_class(tr_full, cm_full, fraction=frac)
        tr_loader = DataLoader(subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        counts = count_examples_per_class(subset, classes)
        total  = sum(counts.values())
        print(f"  '{lc_name}' examples: {counts[lc_name]}  |  "
              f"total training: {total}")

        model = build_mobilenet(num_classes)
        t0    = time.time()
        finetune(model, tr_loader, va_loader)
        elapsed = time.time() - t0

        overall, per_cls, cm = evaluate(model, te_loader, num_classes)
        
        if frac == 1.0:
            visualize_top_confused_pairs(
                model=model,
                test_loader=te_loader,
                test_dataset=te_split,
                classes=classes,
                cm=cm,
                device=DEVICE,
                top_k=5,
                num_samples=5,
                save_dir=PLOTS_DIR,
                ds_name=ds_name,
            )
        
        
        print(f"  Test acc: {overall:.4f}  time: {elapsed:.1f}s")

        row = {
            "dataset":              ds_name,
            "least_confused_class": lc_name,
            "fraction":             frac,
            "n_lc_train":           counts[lc_name],
            "total_train":          total,
            "overall_test_acc":     round(overall, 6),
            "elapsed_seconds":      round(elapsed, 2),
        }
        for i, name in enumerate(classes):
            row[f"per_class_acc_{i}"] = round(float(per_cls[i]), 6)
        rows.append(row)
        del model

    return rows, lc_name, cm_full
    
    



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    all_rows_all     = []   # all-classes reduction
    all_rows_lc      = []   # least-confused reduction
    results_by_ds_all = {}  # for the cross-dataset plot

    for ds_name, loader_fn in DATASET_LOADERS.items():
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        try:
            tr, va, te, classes = loader_fn()
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # ── Strategy 1: all classes ──────────────────────────────────────
        rows_all = run_reduction_all_classes(ds_name, tr, va, te, classes)
        results_by_ds_all[ds_name] = rows_all
        all_rows_all.extend(rows_all)

        plot_per_class_accuracy_heatmap(
            rows_all, classes, ds_name,
            strategy_label="All-class uniform reduction",
            save_path=os.path.join(
                PLOTS_DIR,
                f"{ds_name.replace(' ','_')}_all_classes_heatmap.png"
            )
        )

        # ── Strategy 2: least-confused class ────────────────────────────
        rows_lc, lc_name, _ = run_reduction_least_confused(ds_name, tr, va, te, classes)
        all_rows_lc.extend(rows_lc)

        plot_least_confused_reduction(
            rows_lc, classes, lc_name, ds_name,
            save_path=os.path.join(
                PLOTS_DIR,
                f"{ds_name.replace(' ','_')}_least_confused.png"
            )
        )

    # ── Cross-dataset overview plot ───────────────────────────────────────
    if results_by_ds_all:
        plot_overall_accuracy_vs_fraction(
            results_by_ds_all,
            strategy_label="All-class uniform reduction – MobileNetV3-Small",
            save_path=os.path.join(PLOTS_DIR, "all_datasets_overall_accuracy.png")
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────
    if all_rows_all:
        fields_all = list(all_rows_all[0].keys())
        _write_csv(
            os.path.join(RESULTS_DIR, "reduction_all_classes.csv"),
            all_rows_all, fields_all
        )

    if all_rows_lc:
        fields_lc = list(all_rows_lc[0].keys())
        _write_csv(
            os.path.join(RESULTS_DIR, "reduction_least_confused.csv"),
            all_rows_lc, fields_lc
        )

    print("\nAll done.")


if __name__ == "__main__":
    run_all()

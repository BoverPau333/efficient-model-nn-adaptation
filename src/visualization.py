"""Visualization helpers shared by experiments."""

import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments_config.config import SEED


def write_csv(path: str, rows: list, fieldnames: list):
    """Write experiment rows to a CSV file."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV -> {path}")


def plot_overall_accuracy_vs_fraction(results_by_dataset: dict, strategy_label: str, save_path: str = None):
    """Plot overall test accuracy against the retained training fraction."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_dataset)))

    for (dataset_name, rows), color in zip(results_by_dataset.items(), colors):
        fractions = [row["fraction"] for row in rows]
        accuracies = [row["overall_test_acc"] for row in rows]
        ax.plot(fractions, accuracies, marker="o", linewidth=2, color=color, label=dataset_name)
        for fraction, accuracy in zip(fractions, accuracies):
            ax.annotate(
                f"{accuracy:.2f}",
                (fraction, accuracy),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Retention fraction (1.0 = full training set)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title(f"Overall Accuracy vs. Training Data Size\n({strategy_label})", fontsize=13, fontweight="bold")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def plot_per_class_accuracy_heatmap(results: list, classes: list, ds_name: str, strategy_label: str, save_path: str = None):
    """Plot a per-class accuracy heatmap across retention fractions."""
    fractions = [row["fraction"] for row in results]
    data = np.zeros((len(classes), len(fractions)))

    for col_idx, row in enumerate(results):
        for class_idx in range(len(classes)):
            data[class_idx, col_idx] = row.get(f"per_class_acc_{class_idx}", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(fractions) * 1.2), max(5, len(classes) * 0.4)))
    image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(image, ax=ax, label="Test Accuracy")

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{int(fraction * 100)}%" for fraction in fractions], fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Retention fraction", fontsize=11)
    ax.set_title(f"{ds_name} - Per-class Accuracy vs. Data Size\n({strategy_label})", fontsize=12, fontweight="bold")

    for row_idx in range(len(classes)):
        for col_idx in range(len(fractions)):
            ax.text(
                col_idx,
                row_idx,
                f"{data[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if data[row_idx, col_idx] < 0.5 else "black",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def plot_class_examples_bar(counts: dict, ds_name: str, save_path: str = None):
    """Plot the number of training samples per class."""
    classes = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(classes)))

    fig, ax = plt.subplots(figsize=(8, max(4, len(classes) * 0.35)))
    bars = ax.barh(classes, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    ax.set_xlabel("Number of training examples", fontsize=10)
    ax.set_title(f"{ds_name} - Training examples per class (100 % retention)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def plot_least_confused_reduction(results: list, classes: list, least_confused_cls: str, ds_name: str, save_path: str = None):
    """Plot the targeted reduction experiment summary."""
    fractions = [row["fraction"] for row in results]
    overall_accuracies = [row["overall_test_acc"] for row in results]
    n_classes = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(fractions, overall_accuracies, marker="o", linewidth=2.5, color="#E53935", label="Overall test acc")
    for fraction, accuracy in zip(fractions, overall_accuracies):
        ax.annotate(f"{accuracy:.3f}", (fraction, accuracy), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#E53935")
    ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Fraction kept for\n least-confused class", fontsize=10)
    ax.set_ylabel("Overall Test Accuracy", fontsize=10)
    ax.set_title(f"{ds_name}\nOverall accuracy vs. fraction kept\nfor '{least_confused_cls}'", fontsize=10, fontweight="bold")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    full = results[0]
    sparse = results[-1]
    acc_full = [full.get(f"per_class_acc_{i}", 0) for i in range(n_classes)]
    acc_sparse = [sparse.get(f"per_class_acc_{i}", 0) for i in range(n_classes)]
    x = np.arange(n_classes)
    width = 0.38
    ax2.bar(x - width / 2, acc_full, width, color="#42A5F5", label="100% data", edgecolor="white")
    ax2.bar(
        x + width / 2,
        acc_sparse,
        width,
        color="#EF9A9A",
        label=f"{int(sparse['fraction'] * 100)}% (least-confused class)",
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Per-class Test Accuracy", fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(f"{ds_name}\nPer-class accuracy: full vs. reduced\nfor '{least_confused_cls}'", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def get_top_confused_pairs(cm: np.ndarray, classes: list, top_k: int = 5):
    """Return the top-k most mutually confused class pairs."""
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            pairs.append((int(cm[i, j] + cm[j, i]), i, j))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return pairs[:top_k]


def get_predictions_with_indices(model, loader, device):
    """Collect predictions, labels and dataset indices in loader order."""
    model.eval()
    all_preds = []
    all_labels = []
    all_indices = []
    idx_counter = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            batch_size = len(labels)
            indices = list(range(idx_counter, idx_counter + batch_size))
            idx_counter += batch_size
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend(indices)

    return np.array(all_preds), np.array(all_labels), np.array(all_indices)


def get_confused_samples(preds, labels, indices, class_a, class_b):
    """Return sample indices for the two confusion directions."""
    a_to_b = []
    b_to_a = []

    for pred, true, idx in zip(preds, labels, indices):
        if true == class_a and pred == class_b:
            a_to_b.append(idx)
        elif true == class_b and pred == class_a:
            b_to_a.append(idx)

    return a_to_b, b_to_a


def tensor_to_displayable_image(img_tensor):
    """Convert a normalized tensor into a numpy image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = img_tensor.detach().cpu()
    if image.shape[0] == 3:
        image = image * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def build_confusion_plot_name(ds_name: str, class_a_name: str, class_b_name: str, rank: int = None):
    """Build a filesystem-friendly filename for a confusion plot."""
    rank_prefix = f"{rank:02d}_" if rank is not None else ""
    safe_name = f"{rank_prefix}{ds_name}_{class_a_name}_vs_{class_b_name}"
    safe_name = safe_name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}_confusion.png"


def plot_confusions(dataset, indices_a_to_b, indices_b_to_a, classes, class_a, class_b, num_samples=5, save_path=None):
    """Plot misclassified examples for a confused class pair."""
    if len(indices_a_to_b) == 0 and len(indices_b_to_a) == 0:
        print(f"No mutual errors found between '{classes[class_a]}' and '{classes[class_b]}'.")
        return

    rng = random.Random(SEED)
    sample_a_to_b = rng.sample(indices_a_to_b, min(num_samples, len(indices_a_to_b)))
    sample_b_to_a = rng.sample(indices_b_to_a, min(num_samples, len(indices_b_to_a)))
    total_cols = max(len(sample_a_to_b), len(sample_b_to_a), 1)

    fig, axes = plt.subplots(2, total_cols, figsize=(3.2 * total_cols, 6.5))
    if total_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    row_titles = [
        f"{classes[class_a]} -> {classes[class_b]}",
        f"{classes[class_b]} -> {classes[class_a]}",
    ]

    for col in range(total_cols):
        ax = axes[0, col]
        if col < len(sample_a_to_b):
            idx = sample_a_to_b[col]
            img, _ = dataset[idx]
            ax.imshow(tensor_to_displayable_image(img))
            ax.set_title(f"True: {classes[class_a]}\nPred: {classes[class_b]}\nIdx: {idx}", fontsize=9)
        ax.axis("off")

        ax = axes[1, col]
        if col < len(sample_b_to_a):
            idx = sample_b_to_a[col]
            img, _ = dataset[idx]
            ax.imshow(tensor_to_displayable_image(img))
            ax.set_title(f"True: {classes[class_b]}\nPred: {classes[class_a]}\nIdx: {idx}", fontsize=9)
        ax.axis("off")

    axes[0, 0].set_ylabel(row_titles[0], fontsize=11)
    axes[1, 0].set_ylabel(row_titles[1], fontsize=11)

    plt.suptitle(f"Mutual confusion analysis: {classes[class_a]} <-> {classes[class_b]}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def visualize_confusion_pair(model, test_loader, test_dataset, classes, class_a_name, class_b_name, device, num_samples=5, save_dir=None, ds_name=None, rank=None):
    """Visualize bidirectional confusion for one pair of classes."""
    class_a = classes.index(class_a_name)
    class_b = classes.index(class_b_name)
    preds, labels, indices = get_predictions_with_indices(model, test_loader, device)
    a_to_b, b_to_a = get_confused_samples(preds, labels, indices, class_a, class_b)

    print(f"\nConfusion pair: {class_a_name} <-> {class_b_name}")
    print(f"  {class_a_name} -> {class_b_name}: {len(a_to_b)} samples")
    print(f"  {class_b_name} -> {class_a_name}: {len(b_to_a)} samples")
    print(f"  Mutual confusion score: {len(a_to_b) + len(b_to_a)}")

    save_path = None
    if save_dir and ds_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, build_confusion_plot_name(ds_name, class_a_name, class_b_name, rank=rank))

    plot_confusions(test_dataset, a_to_b, b_to_a, classes, class_a, class_b, num_samples=num_samples, save_path=save_path)


def visualize_top_confused_pairs(model, test_loader, test_dataset, classes, cm, device, top_k=5, num_samples=5, save_dir=None, ds_name=None):
    """Find and visualize the top-k most mutually confused class pairs."""
    top_pairs = get_top_confused_pairs(cm, classes, top_k=top_k)

    print("\nTop confused pairs:")
    for rank, (score, i, j) in enumerate(top_pairs, start=1):
        print(f"  {rank}. {classes[i]} <-> {classes[j]}  | mutual confusion = {score}")

    for rank, (_, i, j) in enumerate(top_pairs, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{rank}/{len(top_pairs)}] Visualizing: {classes[i]} <-> {classes[j]}")
        print(f"{'=' * 70}")
        visualize_confusion_pair(
            model=model,
            test_loader=test_loader,
            test_dataset=test_dataset,
            classes=classes,
            class_a_name=classes[i],
            class_b_name=classes[j],
            device=device,
            num_samples=num_samples,
            save_dir=save_dir,
            ds_name=ds_name,
            rank=rank,
        )

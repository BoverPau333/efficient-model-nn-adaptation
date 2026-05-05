import os
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./datasets/data"
RESULTS_DIR = "./results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
NUM_WORKERS = 2

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_mobilenet(num_classes: int) -> nn.Module:
    """
    Build a MobileNetV3-Small with a frozen pretrained backbone and a fresh head.
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def finetune(model, trainloader, valloader, epochs=EPOCHS, lr=LR):
    """
    Fine-tune only the classifier head and keep the best weights by val loss.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_val = float("inf")
    best_weights = None

    for _ in range(epochs):
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in valloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                running_val += loss_fn(model(imgs), labels).item()

        avg_val = running_val / len(valloader)
        if avg_val < best_val:
            best_val = avg_val
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)


def evaluate(model, loader, num_classes: int):
    """
    Return overall accuracy, per-class accuracy, and confusion matrix.
    """
    model.eval()
    all_preds, all_labels = [], []
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
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[i] = (all_preds[mask] == all_labels[mask]).mean()

    return overall, per_class, cm


def load_cifar10():
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    n_tr = int(0.8 * len(full))
    tr, va = random_split(
        full,
        [n_tr, len(full) - n_tr],
        generator=torch.Generator().manual_seed(SEED),
    )
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
    full = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    n_tr = int(0.8 * len(full))
    tr, va = random_split(
        full,
        [n_tr, len(full) - n_tr],
        generator=torch.Generator().manual_seed(SEED),
    )
    tr.targets = np.array(full.targets.numpy())[tr.indices]
    va.targets = np.array(full.targets.numpy())[va.indices]
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
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
        self.targets = [sample[1] for sample in self.samples]
        self.classes = selected_classes


def load_fruits360():
    fruits_path = os.path.join(
        DATA_DIR,
        "fruits",
        "fruits-360_original-size",
        "fruits-360-original-size",
    )
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
    tr_ds = FilteredImageFolder(os.path.join(fruits_path, "Training"), FRUITS_SELECTED, train_tf)
    va_ds = FilteredImageFolder(os.path.join(fruits_path, "Validation"), FRUITS_SELECTED, test_tf)
    te_ds = FilteredImageFolder(os.path.join(fruits_path, "Test"), FRUITS_SELECTED, test_tf)
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
            try:
                sample = self.loader(path)
            except Exception:
                return self.__getitem__((idx + 1) % len(self))
            if self.transform:
                sample = self.transform(sample)
            return sample, target

    full = SafeImageFolder(os.path.join(dataset_path, "training_set"), transform=train_tf)
    test = SafeImageFolder(os.path.join(dataset_path, "validation_set"), transform=test_tf)
    n_va = int(0.2 * len(full))
    tr, va = random_split(
        full,
        [len(full) - n_va, n_va],
        generator=torch.Generator().manual_seed(SEED),
    )
    tr.targets = np.array(full.targets)[tr.indices]
    va.targets = np.array(full.targets)[va.indices]
    return tr, va, test, full.classes


DATASET_LOADERS = {
    "CIFAR-10": load_cifar10,
    "Fashion-MNIST": load_fashion_mnist,
    "Paintings": load_paintings,
    "Fruits-360": load_fruits360,
}


def get_top_confused_pairs(cm: np.ndarray, classes: list, top_k: int = 5):
    """
    Return the top-k most mutually confused class pairs.

    For each unordered pair (i, j), score = cm[i, j] + cm[j, i].
    Pairs are sorted in descending order of mutual confusion.
    """
    pairs = []
    num_classes = len(classes)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mutual_confusion = int(cm[i, j] + cm[j, i])
            pairs.append((mutual_confusion, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_k]


def get_predictions_with_indices(model, loader, device):
    """
    Collect predictions, labels, and dataset indices in loader order.
    Works correctly when loader is created with shuffle=False.
    """
    model.eval()
    all_preds, all_labels, all_indices = [], [], []
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
    """
    Return indices for:
      - true class_a predicted as class_b
      - true class_b predicted as class_a
    """
    a_to_b = []
    b_to_a = []

    for pred, true, idx in zip(preds, labels, indices):
        if true == class_a and pred == class_b:
            a_to_b.append(idx)
        elif true == class_b and pred == class_a:
            b_to_a.append(idx)

    return a_to_b, b_to_a


def _tensor_to_displayable_image(img_tensor):
    """
    Convert normalized tensor to displayable numpy image.
    Assumes ImageNet normalization used in the training script.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.detach().cpu()

    if img.shape[0] == 3:
        img = img * std + mean

    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _build_confusion_plot_name(ds_name: str, class_a_name: str, class_b_name: str, rank: int = None):
    rank_prefix = f"{rank:02d}_" if rank is not None else ""
    safe_name = f"{rank_prefix}{ds_name}_{class_a_name}_vs_{class_b_name}"
    safe_name = safe_name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}_confusion.png"


def plot_confusions(
    dataset,
    indices_a_to_b,
    indices_b_to_a,
    classes,
    class_a,
    class_b,
    num_samples=5,
    save_path=None,
):
    """
    Plot misclassified examples in two rows:
      Row 1: class_a -> class_b
      Row 2: class_b -> class_a
    """
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
            ax.imshow(_tensor_to_displayable_image(img))
            ax.set_title(
                f"True: {classes[class_a]}\nPred: {classes[class_b]}\nIdx: {idx}",
                fontsize=9,
            )
        ax.axis("off")

        ax = axes[1, col]
        if col < len(sample_b_to_a):
            idx = sample_b_to_a[col]
            img, _ = dataset[idx]
            ax.imshow(_tensor_to_displayable_image(img))
            ax.set_title(
                f"True: {classes[class_b]}\nPred: {classes[class_a]}\nIdx: {idx}",
                fontsize=9,
            )
        ax.axis("off")

    axes[0, 0].set_ylabel(row_titles[0], fontsize=11)
    axes[1, 0].set_ylabel(row_titles[1], fontsize=11)

    plt.suptitle(
        f"Mutual confusion analysis: {classes[class_a]} <-> {classes[class_b]}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot -> {save_path}")
    plt.show()
    plt.close(fig)


def visualize_confusion_pair(
    model,
    test_loader,
    test_dataset,
    classes,
    class_a_name,
    class_b_name,
    device,
    num_samples=5,
    save_dir=None,
    ds_name=None,
    rank=None,
):
    """
    Visualize bidirectional confusion for one pair of classes.
    """
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
        save_path = os.path.join(
            save_dir,
            _build_confusion_plot_name(ds_name, class_a_name, class_b_name, rank=rank),
        )

    plot_confusions(
        test_dataset,
        a_to_b,
        b_to_a,
        classes,
        class_a,
        class_b,
        num_samples=num_samples,
        save_path=save_path,
    )


def visualize_top_confused_pairs(
    model,
    test_loader,
    test_dataset,
    classes,
    cm,
    device,
    top_k=5,
    num_samples=5,
    save_dir=None,
    ds_name=None,
):
    """
    Find the top-k most mutually confused class pairs from the confusion matrix
    and visualize each pair.
    """
    top_pairs = get_top_confused_pairs(cm, classes, top_k=top_k)

    print("\nTop confused pairs:")
    for rank, (score, i, j) in enumerate(top_pairs, start=1):
        print(f"  {rank}. {classes[i]} <-> {classes[j]}  | mutual confusion = {score}")

    for rank, (score, i, j) in enumerate(top_pairs, start=1):
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


def run_similarity_visualization_for_dataset(ds_name: str, loader_fn, top_k=5, num_samples=5):
    """
    Train one model on full data for a dataset and visualize its top confused pairs.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {ds_name}")
    print(f"{'=' * 60}")

    tr, va, te, classes = loader_fn()
    num_classes = len(classes)

    tr_loader = DataLoader(tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_loader = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(te, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training MobileNetV3-Small on full {ds_name} dataset...")
    model = build_mobilenet(num_classes)
    finetune(model, tr_loader, va_loader)

    overall, per_class, cm = evaluate(model, te_loader, num_classes)
    print(f"Test accuracy: {overall:.4f}")
    print(f"Mean per-class accuracy: {per_class.mean():.4f}")

    visualize_top_confused_pairs(
        model=model,
        test_loader=te_loader,
        test_dataset=te,
        classes=classes,
        cm=cm,
        device=DEVICE,
        top_k=top_k,
        num_samples=num_samples,
        save_dir=PLOTS_DIR,
        ds_name=ds_name,
    )


def run_all_similarity_visualizations(top_k=5, num_samples=5):
    """
    Train and visualize confusion pairs for every configured dataset.
    """
    print(f"Device: {DEVICE}")

    for ds_name, loader_fn in DATASET_LOADERS.items():
        try:
            run_similarity_visualization_for_dataset(
                ds_name,
                loader_fn,
                top_k=top_k,
                num_samples=num_samples,
            )
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")

    print("\nSimilarity visualization run complete.")


if __name__ == "__main__":
    run_all_similarity_visualizations()

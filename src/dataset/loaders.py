"""Dataset loaders used by the experiments."""

import os

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from src.experiments_config.config import DATA_DIR
from src.dataset.utils import split_train_val


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
    """ImageFolder variant that keeps only a selected set of classes."""

    def __init__(self, root, selected_classes, transform=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}
        self.samples = [
            (path, self.class_to_idx[os.path.basename(os.path.dirname(path))])
            for path, _ in self.imgs
            if os.path.basename(os.path.dirname(path)) in self.class_to_idx
        ]
        self.targets = [sample[1] for sample in self.samples]
        self.classes = selected_classes


class SafeImageFolder(ImageFolder):
    """ImageFolder that skips unreadable images at access time."""

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            sample = self.loader(path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            sample = self.transform(sample)
        return sample, target


def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    train, val = split_train_val(full)
    return train, val, test, full.classes


def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    full.targets = full.targets.numpy()
    train, val = split_train_val(full)
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    return train, val, test, classes


def load_fruits360():
    fruits_path = DATA_DIR / "fruits" / "fruits-360_original-size" / "fruits-360-original-size"
    if not fruits_path.is_dir():
        raise FileNotFoundError(
            "Fruits-360 not found. Run:\n"
            "  kaggle datasets download moltean/fruits -p data\n"
            "  unzip -q -o data/fruits.zip -d data/fruits"
        )

    train_transform = transforms.Compose([
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
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train = FilteredImageFolder(fruits_path / "Training", FRUITS_SELECTED, train_transform)
    val = FilteredImageFolder(fruits_path / "Validation", FRUITS_SELECTED, eval_transform)
    test = FilteredImageFolder(fruits_path / "Test", FRUITS_SELECTED, eval_transform)
    return train, val, test, FRUITS_SELECTED


def load_paintings():
    dataset_path = DATA_DIR / "painting" / "dataset" / "dataset_updated"
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            "Paintings not found. Run:\n"
            "  kaggle datasets download -d thedownhill/art-images-drawings-painting-sculpture-engraving -p data\n"
            "  unzip -q -o data/art-images-drawings-painting-sculpture-engraving.zip -d data/painting"
        )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = SafeImageFolder(dataset_path / "training_set", transform=train_transform)
    test = SafeImageFolder(dataset_path / "validation_set", transform=eval_transform)
    train, val = split_train_val(full)
    return train, val, test, full.classes


DATASET_LOADERS = {
    "CIFAR-10": load_cifar10,
    "Fashion-MNIST": load_fashion_mnist,
    "Paintings": load_paintings,
    "Fruits-360": load_fruits360,
}

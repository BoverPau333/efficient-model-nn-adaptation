"""Editable configuration for the class-removal retraining baseline."""

DEFAULT_DATASET = "Fruits-360"

# Edit this dictionary to choose which classes should be removed for each dataset.
# Values can be class names (preferred) or integer class indices.
CLASSES_TO_REMOVE_BY_DATASET = {
    "CIFAR-10": ["cat", "dog", "truck"],
    "Fashion-MNIST": ["Shirt", "Coat", "Bag"],
    "Fruits-360": ["Pear 5", "Tomato 8", "Banana 4"],
}


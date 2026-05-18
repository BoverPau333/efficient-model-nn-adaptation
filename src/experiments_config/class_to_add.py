"""Clases a anadir durante el experimento."""

DEFAULT_DATASET = "Fruits-360"

# Pueden ser el nombre de la clase o el indice de esta.
CLASSES_TO_ADD_BY_DATASET = {
    "CIFAR-10": ["cat", "dog", "truck", "deer"],
    "Fashion-MNIST": ["Shirt", "Coat", "Bag", "Sneaker"],
    "Fruits-360": ["Pear 5", "Tomato 8", "Banana 4", "Cucumber 1"],
}

"""Clases a elimanar durante el experimento"""

DEFAULT_DATASET = "Fruits-360"

# Pueden ser le nombre de la clase o el indice de esta 
CLASSES_TO_REMOVE_BY_DATASET = {
    "CIFAR-10": ["cat", "dog", "truck","deer"],
    "Fashion-MNIST": ["Shirt", "Coat", "Bag", "Sneaker"],
    "Fruits-360": ["Pear 5", "Tomato 8", "Banana 4"],
}


"""Reusable model builders."""

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.config import DEVICE


def build_mobilenet(num_classes: int) -> nn.Module:
    """Build a MobileNetV3-Small with a frozen backbone and fresh head."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

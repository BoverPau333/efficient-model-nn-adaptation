"""Constructores reutilizables de modelos."""

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

from src.config import DEVICE


def _congelar_backbone(model):
    """Congela todos los parametros del modelo."""
    for param in model.parameters():
        param.requires_grad = False


def _forward_embeddings_logits_mobilenet(model, imgs):
    """Extrae embeddings y logits para MobileNetV3-Small."""
    features = model.features(imgs)
    pooled = model.avgpool(features)
    flattened = torch.flatten(pooled, 1)
    embeddings = model.classifier[:3](flattened)
    logits = model.classifier[3:](embeddings)
    return embeddings, logits


def _forward_embeddings_logits_resnet(model, imgs):
    """Extrae embeddings y logits para ResNet18."""
    x = model.conv1(imgs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    embeddings = torch.flatten(x, 1)
    logits = model.fc(embeddings)
    return embeddings, logits


def _forward_embeddings_logits_efficientnet(model, imgs):
    """Extrae embeddings y logits para EfficientNet-B0."""
    features = model.features(imgs)
    pooled = model.avgpool(features)
    flattened = torch.flatten(pooled, 1)
    embeddings = model.classifier[:1](flattened)
    logits = model.classifier[1:](embeddings)
    return embeddings, logits


def build_mobilenet(num_classes: int) -> nn.Module:
    """Construye un MobileNetV3-Small con backbone congelado y cabeza nueva."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    _congelar_backbone(model)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.forward_embeddings_and_logits = lambda imgs: _forward_embeddings_logits_mobilenet(model, imgs)
    model.embedding_dim = in_features
    return model.to(DEVICE)


def build_resnet18(num_classes: int) -> nn.Module:
    """Construye un ResNet18 con backbone congelado y cabeza nueva."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    _congelar_backbone(model)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.forward_embeddings_and_logits = lambda imgs: _forward_embeddings_logits_resnet(model, imgs)
    model.embedding_dim = in_features
    return model.to(DEVICE)


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    """Construye un EfficientNet-B0 con backbone congelado y cabeza nueva."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    _congelar_backbone(model)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.forward_embeddings_and_logits = lambda imgs: _forward_embeddings_logits_efficientnet(model, imgs)
    model.embedding_dim = in_features
    return model.to(DEVICE)


MODEL_BUILDERS = {
    "MobileNetV3-Small": build_mobilenet,
    "ResNet18": build_resnet18,
    "EfficientNet-B0": build_efficientnet_b0,
}

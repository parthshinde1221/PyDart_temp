from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


# ----------------------------
# Demo / built-in models
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 16, 5, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DemoResNet50(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ----------------------------
# Default synthetic dataloader
# ----------------------------
def create_synthetic_dataloader(
    batch_size: int,
    num_samples: int,
    input_size: Tuple[int, int, int],
    num_classes: int = 10,
) -> DataLoader:
    inputs = torch.randn(num_samples, *input_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size)


# ----------------------------
# ModelSpec
# ----------------------------
@dataclass
class ModelSpec:
    """
    Minimal workload wrapper for PyDart.

    Supports:
    - built-in/demo models via `builder`
    - user-provided custom dataloader via `dataloader_fn`
    - synthetic dataloader fallback using input_size/batch_size/num_samples
    """

    name: str
    builder: Callable[[], nn.Module]
    input_size: Tuple[int, int, int]
    batch_size: int = 10
    num_samples: int = 100
    num_classes: int = 10
    dataloader_fn: Optional[Callable[[], DataLoader]] = None

    def build_model(self) -> nn.Module:
        return self.builder()

    def build_dataloader(self) -> DataLoader:
        if self.dataloader_fn is not None:
            return self.dataloader_fn()

        return create_synthetic_dataloader(
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            input_size=self.input_size,
            num_classes=self.num_classes,
        )

    def build_input_batch(self) -> torch.Tensor:
        dataloader = self.build_dataloader()
        inputs, _ = next(iter(dataloader))
        return inputs


# ----------------------------
# Built-in registry
# ----------------------------
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "light_cnn": ModelSpec(
        name="light_cnn",
        builder=lambda: SimpleCNN(num_classes=10),
        input_size=(3, 28, 28),
        batch_size=10,
        num_samples=100,
        num_classes=10,
    ),
    "heavy_resnet50": ModelSpec(
        name="heavy_resnet50",
        builder=lambda: DemoResNet50(num_classes=10),
        input_size=(3, 224, 224),
        batch_size=10,
        num_samples=100,
        num_classes=10,
    ),
}

HEAVY_MODEL_KEYS = ["heavy_resnet50"]
LIGHT_MODEL_KEYS = ["light_cnn"]

def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model spec '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def list_model_specs() -> list[str]:
    return list(MODEL_REGISTRY.keys())


# custom_spec = ModelSpec(
#     name="my_model",
#     builder=lambda: my_model,
#     input_size=(3, 224, 224),
#     dataloader_fn=my_dataloader_fn,
# )
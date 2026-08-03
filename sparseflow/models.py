"""Small offline reference model for mechanism verification."""

from __future__ import annotations

import torch
import torch.nn as nn


INPUT_SHAPE = (1, 3, 16, 16)


def _conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    ]


class ReferenceCNN(nn.Module):
    """Sequential Conv-BN model supported by the V0/V1 pruning pass."""

    identifier = "reference_cnn_v1"

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(3, 8),
            nn.MaxPool2d(2),
            *_conv_block(8, 16),
            nn.MaxPool2d(2),
            *_conv_block(16, 24),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(24, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def build_reference_model(seed: int = 1234, num_classes: int = 4) -> ReferenceCNN:
    torch.manual_seed(seed)
    return ReferenceCNN(num_classes=num_classes).eval()

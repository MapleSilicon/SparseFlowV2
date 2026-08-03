"""Deterministic offline reference models for SparseFlow evidence."""

from __future__ import annotations

import torch
import torch.nn as nn


INPUT_SHAPE = (1, 3, 16, 16)
RESNET18_INPUT_SHAPE = (1, 3, 64, 64)


def _conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    ]


class ReferenceCNN(nn.Module):
    """Sequential Conv-BN model supported by the SparseFlow V0 pruning pass."""

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


class BasicBlock(nn.Module):
    """Standard two-convolution ResNet BasicBlock with an optional projection."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs

        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        output = output + identity
        return self.relu(output)


class ResNet18Reference(nn.Module):
    """Local ResNet-18 reference used only for V1 Gate 1 analysis."""

    identifier = "resnet18-reference"
    architecture_identifier = "resnet18_basicblock_reference_v1"

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self._in_channels = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_stage(64, blocks=2)
        self.layer2 = self._make_stage(128, blocks=2, stride=2)
        self.layer3 = self._make_stage(256, blocks=2, stride=2)
        self.layer4 = self._make_stage(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self._initialize_parameters()

    def _make_stage(
        self,
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        output_channels = channels * BasicBlock.expansion
        downsample = None
        if stride != 1 or self._in_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._in_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        layers = [
            BasicBlock(
                self._in_channels,
                channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self._in_channels = output_channels
        layers.extend(
            BasicBlock(self._in_channels, channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        return self.fc(output)


def build_resnet18_reference(
    seed: int = 1234,
    num_classes: int = 1000,
) -> ResNet18Reference:
    """Build a deterministic ResNet-18 without weights or network access."""

    torch.manual_seed(seed)
    return ResNet18Reference(num_classes=num_classes).eval()


def build_model(model_identifier: str, seed: int = 1234) -> nn.Module:
    """Resolve a supported local model identifier."""

    if model_identifier == ReferenceCNN.identifier:
        return build_reference_model(seed=seed)
    if model_identifier == ResNet18Reference.identifier:
        return build_resnet18_reference(seed=seed)
    raise ValueError(f"unsupported model identifier: {model_identifier}")

"""Frozen VGG-19 feature extractor for Neural Style Transfer."""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_HOME", ".torch_cache")

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

CONTENT_LAYERS: dict[int, str] = {
    21: "conv4_2",
}

STYLE_LAYERS: dict[int, str] = {
    0: "conv1_1",
    5: "conv2_1",
    10: "conv3_1",
    19: "conv4_1",
    28: "conv5_1",
}

DEFAULT_CONTENT_LAYER_NAMES: tuple[str, ...] = ("conv4_2",)
DEFAULT_STYLE_LAYER_NAMES: tuple[str, ...] = (
    "conv1_1",
    "conv2_1",
    "conv3_1",
    "conv4_1",
    "conv5_1",
)


class VGG19FeatureExtractor(nn.Module):
    """Extract content and style feature maps from a frozen pretrained VGG-19 backbone."""

    def __init__(self, use_avg_pool: bool = False, freeze: bool = True) -> None:
        """Initialize the VGG-19 feature extractor.

        Args:
            use_avg_pool: If True, replace all `nn.MaxPool2d` layers with `nn.AvgPool2d`.
            freeze: If True, disable gradients on all VGG-19 parameters and keep eval mode.
        """
        super().__init__()
        self.use_avg_pool = use_avg_pool
        self._frozen = freeze

        max_layer_idx = max(*CONTENT_LAYERS.keys(), *STYLE_LAYERS.keys())
        backbone = vgg19(weights=VGG19_Weights.DEFAULT).features[: max_layer_idx + 1]

        layers: list[nn.Module] = []
        for layer in backbone:
            if use_avg_pool and isinstance(layer, nn.MaxPool2d):
                layers.append(
                    nn.AvgPool2d(
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
                )
            elif isinstance(layer, nn.ReLU):
                # Avoid in-place ReLU overwriting pre-ReLU activations during backprop.
                layers.append(nn.ReLU(inplace=False))
            else:
                layers.append(layer)

        self.features = nn.Sequential(*layers)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.eval()

    def train(self, mode: bool = True) -> VGG19FeatureExtractor:
        """Keep the frozen feature extractor in evaluation mode."""
        super().train(mode)
        if self._frozen:
            self.features.eval()
        return self

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Extract content and style activations from a `(B, 3, H, W)` normalized tensor.

        Args:
            x: ImageNet-normalized input tensor of shape `(B, 3, H, W)`.

        Returns:
            Tuple `(content_features, style_features)` where:
            - `content_features` maps `"conv4_2"` to its activation tensor.
            - `style_features` maps `"conv1_1"`, `"conv2_1"`, `"conv3_1"`, `"conv4_1"`,
              and `"conv5_1"` to their activation tensors.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected 4D input tensor with shape (B, 3, H, W), got {tuple(x.shape)}."
            )

        content_features: dict[str, torch.Tensor] = {}
        style_features: dict[str, torch.Tensor] = {}

        current = x
        for idx, layer in enumerate(self.features):
            current = layer(current)
            if idx in CONTENT_LAYERS:
                content_features[CONTENT_LAYERS[idx]] = current
            if idx in STYLE_LAYERS:
                style_features[STYLE_LAYERS[idx]] = current

        return content_features, style_features

"""Manual loss functions and Gram Matrix computation for Neural Style Transfer."""

from __future__ import annotations

from collections.abc import Iterable

import torch

try:
    from src.vgg_extractor import (
        DEFAULT_CONTENT_LAYER_NAMES,
        DEFAULT_STYLE_LAYER_NAMES,
    )
except ImportError:
    from vgg_extractor import (
        DEFAULT_CONTENT_LAYER_NAMES,
        DEFAULT_STYLE_LAYER_NAMES,
    )

DEFAULT_STYLE_WEIGHTS: dict[str, float] = {
    layer_name: 1.0 / len(DEFAULT_STYLE_LAYER_NAMES)
    for layer_name in DEFAULT_STYLE_LAYER_NAMES
}


def _validate_feature_layers(
    generated_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
    layer_names: Iterable[str],
    feature_kind: str,
) -> tuple[str, ...]:
    """Validate that all requested feature layers exist in both feature dicts."""
    layers = tuple(layer_names)
    if not layers:
        raise ValueError(f"{feature_kind} layer list cannot be empty.")

    missing_generated = [name for name in layers if name not in generated_features]
    if missing_generated:
        raise KeyError(
            f"Missing {feature_kind} layers in generated_features: {missing_generated}."
        )

    missing_target = [name for name in layers if name not in target_features]
    if missing_target:
        raise KeyError(
            f"Missing {feature_kind} layers in target_features: {missing_target}."
        )

    return layers


def compute_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Compute the normalized Gram Matrix of a 4D feature map `(B, C, H, W)`.

    Reshapes `(1, C, H, W)` to `(C, H * W)`, multiplies by its transpose to
    obtain `(C, C)`, and normalizes by dividing by `C * H * W`. For `B > 1`,
    computes the normalized Gram Matrix independently per sample `(B, C, C)`.
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected 4D feature map with shape (B, C, H, W), got {tuple(feature_map.shape)}."
        )

    batch_size, channels, height, width = feature_map.shape
    if batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"All feature map dimensions must be positive, got {tuple(feature_map.shape)}."
        )

    norm_factor = float(channels * height * width)

    if batch_size == 1:
        features = feature_map.reshape(channels, height * width)
        return torch.mm(features, features.t()) / norm_factor

    features = feature_map.reshape(batch_size, channels, height * width)
    return torch.bmm(features, features.transpose(1, 2)) / norm_factor


def compute_content_loss(
    generated_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYER_NAMES,
) -> torch.Tensor:
    """Compute manual MSE content loss over the selected VGG content layers."""
    layers = _validate_feature_layers(
        generated_features, target_features, content_layers, "content"
    )

    layer_losses: list[torch.Tensor] = []
    for layer_name in layers:
        gen_feat = generated_features[layer_name]
        target_feat = target_features[layer_name].detach()

        if gen_feat.shape != target_feat.shape:
            raise ValueError(
                f"Content feature shape mismatch at '{layer_name}': "
                f"{tuple(gen_feat.shape)} vs {tuple(target_feat.shape)}."
            )

        diff = gen_feat - target_feat
        layer_losses.append(torch.mean(diff * diff))

    return torch.stack(layer_losses).mean()


def compute_style_loss(
    generated_features: dict[str, torch.Tensor],
    target_features: dict[str, torch.Tensor],
    style_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute weighted manual MSE style loss over Gram Matrices of 5 style layers."""
    weights = DEFAULT_STYLE_WEIGHTS if style_weights is None else style_weights
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("All style layer weights must be non-negative.")

    layers = _validate_feature_layers(
        generated_features, target_features, weights.keys(), "style"
    )

    ref_tensor = generated_features[layers[0]]
    total_style_loss = torch.zeros((), dtype=ref_tensor.dtype, device=ref_tensor.device)

    for layer_name in layers:
        weight = weights[layer_name]
        gen_feat = generated_features[layer_name]
        target_feat = target_features[layer_name].detach()

        if gen_feat.ndim != 4 or target_feat.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors at '{layer_name}', got "
                f"{tuple(gen_feat.shape)} and {tuple(target_feat.shape)}."
            )
        if gen_feat.shape[:2] != target_feat.shape[:2]:
            raise ValueError(
                f"Style batch/channel mismatch at '{layer_name}': "
                f"{tuple(gen_feat.shape[:2])} vs {tuple(target_feat.shape[:2])}."
            )

        gen_gram = compute_gram_matrix(gen_feat)
        target_gram = compute_gram_matrix(target_feat)
        diff = gen_gram - target_gram
        layer_loss = torch.mean(diff * diff)
        total_style_loss = total_style_loss + weight * layer_loss

    return total_style_loss


def compute_total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """Compute anisotropic L1 Total Variation loss on a `(B, C, H, W)` image tensor."""
    if image.ndim != 4:
        raise ValueError(
            f"Expected 4D image tensor with shape (B, C, H, W), got {tuple(image.shape)}."
        )

    batch_size, channels, height, width = image.shape
    if batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"All image dimensions must be positive, got {tuple(image.shape)}."
        )

    tv_loss = torch.zeros((), dtype=image.dtype, device=image.device)

    if height > 1:
        diff_h = image[:, :, 1:, :] - image[:, :, :-1, :]
        tv_loss = tv_loss + torch.mean(torch.abs(diff_h))

    if width > 1:
        diff_w = image[:, :, :, 1:] - image[:, :, :, :-1]
        tv_loss = tv_loss + torch.mean(torch.abs(diff_w))

    return tv_loss


def compute_total_loss(
    content_loss: torch.Tensor,
    style_loss: torch.Tensor,
    tv_loss: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1e6,
    tv_weight: float = 0.0,
) -> torch.Tensor:
    """Combine content, style, and optional total variation losses into a scalar objective."""
    if alpha < 0.0 or beta < 0.0 or tv_weight < 0.0:
        raise ValueError("Loss weights alpha, beta, and tv_weight must be non-negative.")

    total_loss = alpha * content_loss + beta * style_loss
    if tv_loss is not None and tv_weight > 0.0:
        total_loss = total_loss + tv_weight * tv_loss

    return total_loss

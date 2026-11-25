"""Image preprocessing and postprocessing utilities for Neural Style Transfer."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE: int = 512


def build_transform(
    target_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    keep_aspect_ratio: bool = False,
) -> transforms.Compose:
    """Build a torchvision transform pipeline for resizing and ImageNet normalization.

    Args:
        target_size: Target spatial resolution. An integer produces `(target_size, target_size)`
            when `keep_aspect_ratio` is False, or scales the shorter edge when True.
            A `(height, width)` tuple always resizes to exact dimensions.
        keep_aspect_ratio: Whether to preserve aspect ratio when `target_size` is an int.

    Returns:
        Composed transform converting an RGB PIL image to a `(3, H, W)` float32 tensor.
    """
    if isinstance(target_size, int):
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}.")
        resize_op = (
            transforms.Resize(target_size)
            if keep_aspect_ratio
            else transforms.Resize((target_size, target_size))
        )
    elif isinstance(target_size, tuple) and len(target_size) == 2:
        height, width = target_size
        if height <= 0 or width <= 0:
            raise ValueError(f"target_size dimensions must be positive, got {target_size}.")
        resize_op = transforms.Resize((height, width))
    else:
        raise TypeError(
            "target_size must be a positive int or a (height, width) tuple."
        )

    return transforms.Compose(
        [
            resize_op,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(
    image_path: str | Path,
    target_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    keep_aspect_ratio: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load an image file as a normalized `(1, 3, H, W)` RGB tensor.

    Args:
        image_path: Path to the input image file (PNG, JPEG, etc.).
        target_size: Target resolution `(H, W)` or integer size (e.g., 256 or 512).
        keep_aspect_ratio: Whether to preserve aspect ratio when `target_size` is an int.
        device: Optional PyTorch device to move the output tensor to.

    Returns:
        ImageNet-normalized tensor with shape `(1, 3, H, W)` and dtype `torch.float32`.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    transform = build_transform(
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
    )

    with Image.open(path) as img:
        rgb_image = img.convert("RGB")
        tensor = transform(rgb_image).unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def load_content_and_style(
    content_path: str | Path,
    style_path: str | Path,
    target_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    match_style_to_content: bool = True,
    keep_aspect_ratio: bool = False,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load content and style images with consistent `(1, 3, H, W)` tensor shapes.

    Args:
        content_path: Path to the content image.
        style_path: Path to the style image.
        target_size: Target resolution for the content image (and style image if not matched).
        match_style_to_content: If True, resize style image to match content `(H, W)`.
        keep_aspect_ratio: Whether to preserve aspect ratio for the content image when
            `target_size` is an int.
        device: Optional PyTorch device for both tensors.

    Returns:
        Tuple of `(content_tensor, style_tensor)`, each shaped `(1, 3, H, W)`.
    """
    content_tensor = load_image(
        image_path=content_path,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
        device=device,
    )

    style_target_size: int | tuple[int, int] = (
        (content_tensor.shape[2], content_tensor.shape[3])
        if match_style_to_content
        else target_size
    )
    style_tensor = load_image(
        image_path=style_path,
        target_size=style_target_size,
        keep_aspect_ratio=False if match_style_to_content else keep_aspect_ratio,
        device=device,
    )

    return content_tensor, style_tensor


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization and clamp pixel values to `[0.0, 1.0]`.

    Args:
        tensor: Tensor of shape `(1, 3, H, W)` or `(3, H, W)`.

    Returns:
        CPU float32 tensor with the same shape as input and values clamped to `[0.0, 1.0]`.
    """
    if tensor.ndim == 4:
        if tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 4D tensor with shape (1, 3, H, W), got {tuple(tensor.shape)}."
            )
        view_shape = (1, 3, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape[0] != 3:
            raise ValueError(
                f"Expected 3D tensor with shape (3, H, W), got {tuple(tensor.shape)}."
            )
        view_shape = (3, 1, 1)
    else:
        raise ValueError(
            f"Expected 3D or 4D tensor, got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}."
        )

    denorm = tensor.detach().clone().cpu().to(torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=denorm.dtype).view(*view_shape)
    std = torch.tensor(IMAGENET_STD, dtype=denorm.dtype).view(*view_shape)

    denorm = denorm * std + mean
    return torch.clamp(denorm, 0.0, 1.0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert an ImageNet-normalized tensor into an RGB PIL Image.

    Args:
        tensor: Tensor of shape `(1, 3, H, W)` or `(3, H, W)`.

    Returns:
        RGB `PIL.Image.Image` reconstructed from denormalized `[0, 1]` values.
    """
    clamped = denormalize_tensor(tensor)
    image_tensor = clamped.squeeze(0) if clamped.ndim == 4 else clamped
    return transforms.ToPILImage()(image_tensor)

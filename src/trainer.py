"""Optimization loop and artifact persistence for Neural Style Transfer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

try:
    from src.losses import (
        compute_content_loss,
        compute_style_loss,
        compute_total_loss,
        compute_total_variation_loss,
    )
    from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD, tensor_to_image
    from src.vgg_extractor import VGG19FeatureExtractor
except ImportError:
    from losses import (
        compute_content_loss,
        compute_style_loss,
        compute_total_loss,
        compute_total_variation_loss,
    )
    from preprocessing import IMAGENET_MEAN, IMAGENET_STD, tensor_to_image
    from vgg_extractor import VGG19FeatureExtractor

DEFAULT_ADAM_LR: float = 0.05
DEFAULT_LBFGS_LR: float = 1.0


def _normalized_bounds(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel valid ImageNet-normalized bounds corresponding to [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    min_bounds = (0.0 - mean) / std
    max_bounds = (1.0 - mean) / std
    return min_bounds, max_bounds


def _clamp_to_valid_normalized_range_(
    target_img: torch.Tensor, min_bounds: torch.Tensor, max_bounds: torch.Tensor
) -> None:
    """Clamp the optimized image tensor in-place to valid normalized RGB bounds."""
    with torch.no_grad():
        target_img.copy_(torch.clamp(target_img, min=min_bounds, max=max_bounds))


def initialize_target_image(
    content_img: torch.Tensor, init_mode: str = "content"
) -> torch.Tensor:
    """Initialize the optimizable target image tensor `(1, 3, H, W)`."""
    mode = init_mode.lower()
    if mode == "content":
        target_img = content_img.clone().detach()
    elif mode == "noise":
        mean = torch.tensor(
            IMAGENET_MEAN, device=content_img.device, dtype=content_img.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            IMAGENET_STD, device=content_img.device, dtype=content_img.dtype
        ).view(1, 3, 1, 1)
        raw_noise = torch.rand_like(content_img)
        target_img = (raw_noise - mean) / std
    else:
        raise ValueError(
            f"Unsupported init_mode '{init_mode}'. Expected 'content' or 'noise'."
        )

    return target_img.requires_grad_(True)


def create_optimizer(
    target_img: torch.Tensor, optimizer_name: str = "adam", lr: float | None = None
) -> torch.optim.Optimizer:
    """Create an Adam or LBFGS optimizer targeting only the generated image tensor."""
    opt_key = optimizer_name.lower()
    if lr is not None and lr <= 0.0:
        raise ValueError(f"Learning rate must be positive, got {lr}.")

    if opt_key == "adam":
        effective_lr = DEFAULT_ADAM_LR if lr is None else lr
        return torch.optim.Adam([target_img], lr=effective_lr)

    if opt_key == "lbfgs":
        effective_lr = DEFAULT_LBFGS_LR if lr is None else lr
        return torch.optim.LBFGS([target_img], lr=effective_lr, max_iter=1)

    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Expected 'adam' or 'lbfgs'."
    )


def save_loss_history(
    history: list[dict[str, float | int]], csv_path: Path
) -> Path:
    """Write per-step content, style, TV, and total loss records to a CSV file."""
    fieldnames = ["step", "content_loss", "style_loss", "tv_loss", "total_loss"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return csv_path


def save_loss_curve(
    history: list[dict[str, float | int]], plot_path: Path
) -> Path:
    """Plot total, content, and style loss curves and save to a PNG file."""
    steps = [int(row["step"]) for row in history]
    total_losses = [float(row["total_loss"]) for row in history]
    content_losses = [float(row["content_loss"]) for row in history]
    style_losses = [float(row["style_loss"]) for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    series = [
        (axes[0], total_losses, "Total Loss", "tab:blue"),
        (axes[1], content_losses, "Content Loss", "tab:green"),
        (axes[2], style_losses, "Style Loss", "tab:orange"),
    ]

    for ax, values, label, color in series:
        ax.plot(steps, values, color=color, linewidth=1.8, label=label)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

    axes[0].set_title("Neural Style Transfer Loss History")
    axes[-1].set_xlabel("Optimization Step")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def run_style_transfer(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    extractor: VGG19FeatureExtractor,
    output_dir: str | Path = "outputs",
    steps: int = 300,
    alpha: float = 1.0,
    beta: float = 1e6,
    tv_weight: float = 0.0,
    optimizer_name: str = "adam",
    lr: float | None = None,
    init_mode: str = "content",
    save_every: int | None = None,
) -> dict[str, Any]:
    """Run Neural Style Transfer pixel optimization and save all output artifacts."""
    if content_img.ndim != 4 or content_img.shape[:2] != (1, 3):
        raise ValueError(
            f"Expected content_img shape (1, 3, H, W), got {tuple(content_img.shape)}."
        )
    if style_img.ndim != 4 or style_img.shape[:2] != (1, 3):
        raise ValueError(
            f"Expected style_img shape (1, 3, H, W), got {tuple(style_img.shape)}."
        )
    if steps <= 0:
        raise ValueError(f"steps must be a positive integer, got {steps}.")
    if save_every is not None and save_every <= 0:
        raise ValueError(f"save_every must be positive when set, got {save_every}.")

    out_path = Path(output_dir)
    progress_dir = out_path / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    extractor.eval()
    with torch.no_grad():
        target_content, _ = extractor(content_img)
        _, target_style = extractor(style_img)

    target_img = initialize_target_image(content_img, init_mode=init_mode)
    optimizer = create_optimizer(target_img, optimizer_name=optimizer_name, lr=lr)
    min_bounds, max_bounds = _normalized_bounds(target_img.device, target_img.dtype)

    progress_interval = max(1, steps // 5) if save_every is None else save_every
    opt_key = optimizer_name.lower()
    history: list[dict[str, float | int]] = []
    latest_metrics: dict[str, float] = {}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        gen_content, gen_style = extractor(target_img)
        c_loss = compute_content_loss(gen_content, target_content)
        s_loss = compute_style_loss(gen_style, target_style)

        if tv_weight > 0.0:
            tv_loss_tensor: torch.Tensor | None = compute_total_variation_loss(
                target_img
            )
            tv_value = float(tv_loss_tensor.detach().item())
        else:
            tv_loss_tensor = None
            tv_value = 0.0

        total_loss = compute_total_loss(
            content_loss=c_loss,
            style_loss=s_loss,
            tv_loss=tv_loss_tensor,
            alpha=alpha,
            beta=beta,
            tv_weight=tv_weight,
        )
        total_loss.backward()

        latest_metrics["content_loss"] = float(c_loss.detach().item())
        latest_metrics["style_loss"] = float(s_loss.detach().item())
        latest_metrics["tv_loss"] = tv_value
        latest_metrics["total_loss"] = float(total_loss.detach().item())
        return total_loss

    for step in range(1, steps + 1):
        if opt_key == "lbfgs":
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

        _clamp_to_valid_normalized_range_(target_img, min_bounds, max_bounds)

        history.append(
            {
                "step": step,
                "content_loss": latest_metrics["content_loss"],
                "style_loss": latest_metrics["style_loss"],
                "tv_loss": latest_metrics["tv_loss"],
                "total_loss": latest_metrics["total_loss"],
            }
        )

        if step % progress_interval == 0 or step == steps:
            progress_path = progress_dir / f"step_{step:04d}.png"
            tensor_to_image(target_img).save(progress_path)

    result_path = out_path / "result.png"
    tensor_to_image(target_img).save(result_path)

    history_path = save_loss_history(history, out_path / "loss_history.csv")
    curve_path = save_loss_curve(history, out_path / "loss_curve.png")

    return {
        "result_path": result_path,
        "history_path": history_path,
        "curve_path": curve_path,
        "progress_dir": progress_dir,
        "final_losses": history[-1],
        "target_img": target_img.detach(),
    }

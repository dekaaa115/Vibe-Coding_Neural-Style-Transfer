"""CLI entry point for Neural Style Transfer."""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_HOME", ".torch_cache")

import matplotlib

matplotlib.use("Agg")

import argparse
from collections.abc import Sequence

import torch

from src.preprocessing import DEFAULT_IMAGE_SIZE, load_content_and_style
from src.trainer import run_style_transfer
from src.vgg_extractor import VGG19FeatureExtractor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Neural Style Transfer."""
    parser = argparse.ArgumentParser(
        description="Run Neural Style Transfer from scratch using frozen VGG-19."
    )
    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Path to the content image.",
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Path to the style image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Target spatial size for resizing images (default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of optimization steps (default: 300).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for content loss (default: 1.0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e6,
        help="Weight for style loss (default: 1e6).",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Weight for total variation denoising loss (default: 0.0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 0.05 for adam, 1.0 for lbfgs).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("adam", "lbfgs"),
        default="adam",
        help="Optimizer used to update target image pixels (default: adam).",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=("content", "noise"),
        default="content",
        help="Target image initialization mode (default: content).",
    )
    parser.add_argument(
        "--avg-pool",
        action="store_true",
        help="Replace VGG-19 MaxPool2d layers with AvgPool2d.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where result, progress images, CSV, and loss curve are saved.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Optional step interval for saving periodic progress images.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the Neural Style Transfer pipeline from CLI arguments."""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img, style_img = load_content_and_style(
        content_path=args.content,
        style_path=args.style,
        target_size=args.size,
        device=device,
    )
    extractor = VGG19FeatureExtractor(use_avg_pool=args.avg_pool).to(device)

    result = run_style_transfer(
        content_img=content_img,
        style_img=style_img,
        extractor=extractor,
        output_dir=args.output_dir,
        steps=args.steps,
        alpha=args.alpha,
        beta=args.beta,
        tv_weight=args.tv_weight,
        optimizer_name=args.optimizer,
        lr=args.lr,
        init_mode=args.init,
        save_every=args.save_every,
    )

    final_losses = result["final_losses"]
    print(f"Device: {device}")
    print(
        f"Finished {args.steps} steps ({args.optimizer}, init={args.init}) -> "
        f"total={final_losses['total_loss']:.6f}, "
        f"content={final_losses['content_loss']:.6f}, "
        f"style={final_losses['style_loss']:.6f}, "
        f"tv={final_losses['tv_loss']:.6f}"
    )
    print(f"Result image : {result['result_path']}")
    print(f"Loss history : {result['history_path']}")
    print(f"Loss curve   : {result['curve_path']}")
    print(f"Progress dir : {result['progress_dir']}")


if __name__ == "__main__":
    main()

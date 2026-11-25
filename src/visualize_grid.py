"""Generate a 2x3 comparison grid of content, style, and experiment outputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

GRID_ROWS: int = 2
GRID_COLS: int = 3
DEFAULT_DPI: int = 200
DEFAULT_FIGSIZE: tuple[float, float] = (15.0, 10.0)
DEFAULT_OUTPUT_PATH: Path = Path("outputs/perbandingan_eksperimen.png")
DEFAULT_SUPTITLE: str = "Perbandingan Eksperimen Neural Style Transfer"

DEFAULT_PANELS: tuple[tuple[str, Path], ...] = (
    ("Content Image", Path("data/content/batik-content1.jpg")),
    ("Style Image", Path("data/style/batik-style1.jpg")),
    ("Exp 1: Baseline (Adam)", Path("outputs/exp1_baseline/result.png")),
    ("Exp 2: High Style", Path("outputs/exp2_high_style/result.png")),
    ("Exp 3: L-BFGS", Path("outputs/exp3_lbfgs/result.png")),
    ("Exp 4: AvgPool", Path("outputs/exp4_avgpool/result.png")),
)


def load_rgb_image(
    image_path: Path,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Load an image as RGB, optionally resizing to `(width, height)`."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")

    if target_size is not None:
        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError(f"target_size dimensions must be positive, got {target_size}.")
        rgb_img = rgb_img.resize((width, height), Image.Resampling.LANCZOS)

    return rgb_img


def create_comparison_grid(
    panels: Sequence[tuple[str, Path]] = DEFAULT_PANELS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    suptitle: str | None = DEFAULT_SUPTITLE,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Render up to six labeled images into a 2x3 grid and save to `output_path`."""
    max_panels = GRID_ROWS * GRID_COLS
    if not panels:
        raise ValueError("panels must contain at least one (title, path) entry.")
    if len(panels) > max_panels:
        raise ValueError(f"panels cannot exceed {max_panels} items for a 2x3 grid, got {len(panels)}.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}.")

    loaded_panels = [
        (title, load_rgb_image(Path(path), target_size=target_size))
        for title, path in panels
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=figsize, facecolor="white")
    try:
        for ax, panel in zip(axes.flat, loaded_panels):
            title, image = panel
            ax.imshow(image)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            ax.axis("off")

        for ax in list(axes.flat)[len(loaded_panels) :]:
            ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        else:
            fig.tight_layout()

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)

    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for generating the 2x3 experiment comparison grid."""
    parser = argparse.ArgumentParser(
        description="Compose a 2x3 comparison grid of content, style, and experiment outputs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output image DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_SUPTITLE,
        help="Optional figure suptitle; pass an empty string to omit.",
    )
    parser.add_argument(
        "--uniform-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional uniform (width, height) resize for every panel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point to build and save the comparison grid."""
    args = parse_args(argv)
    target_size = tuple(args.uniform_size) if args.uniform_size is not None else None
    suptitle = args.title.strip() or None

    saved_path = create_comparison_grid(
        panels=DEFAULT_PANELS,
        output_path=args.output,
        suptitle=suptitle,
        dpi=args.dpi,
        target_size=target_size,
    )
    print(f"Saved comparison grid to {saved_path}")


if __name__ == "__main__":
    main()

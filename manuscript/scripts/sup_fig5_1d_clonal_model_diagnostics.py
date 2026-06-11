#!/usr/bin/env python
"""
Assemble Trim28-Setdb1 clonal-model diagnostic panels for the manuscript.

Each perturbation contributes a cropped ECDF/isotonic pair with a title
reporting the perturbation, best loss, and clone count. Layout can be stacked
by rows or arranged in a single row.
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATIONS = ["NT", "Trim28", "Setdb1"]
SOURCE_DIR = REPO_ROOT / "clonal_analysis" / "intermediate_files" / "trim28_setdb1"
REAL_DATA_DIR = REPO_ROOT / "scripts" / "clonal_modeling" / "real_data" / "trim28_setdb1" / "input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_12"
DEFAULT_OUTPUT_STEM = "d"
WHITE_THRESHOLD = 0.995
TRIM_PADDING = 1
TITLE_PAD = 2


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble Trim28-Setdb1 clonal-model diagnostic panels."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the assembled PNG and SVG outputs.",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem.",
    )
    parser.add_argument(
        "--layout",
        choices=["column", "row"],
        default="column",
        help="Arrange perturbation summaries in a column or a single row.",
    )
    return parser.parse_args(args)


def diagnostic_paths(perturbation: str) -> tuple[Path, Path]:
    base = SOURCE_DIR / perturbation
    ecdf = base / f"lowest_loss_points_best_{perturbation}_ecdf.png"
    isotonic = base / f"lowest_loss_points_best_{perturbation}_isotonic.png"
    return ecdf, isotonic


def parameter_path(perturbation: str) -> Path:
    return SOURCE_DIR / perturbation / f"lowest_loss_points_best_{perturbation}_parameters.txt"


def real_data_path(perturbation: str) -> Path:
    return REAL_DATA_DIR / f"{perturbation}.csv"


def validate_inputs() -> None:
    missing: list[str] = []
    for perturbation in PERTURBATIONS:
        for path in (*diagnostic_paths(perturbation), parameter_path(perturbation), real_data_path(perturbation)):
            if not path.exists():
                missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing source diagnostic panels for Figure S12a:\n"
            + "\n".join(missing)
        )


def crop_whitespace(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        mask = image < WHITE_THRESHOLD
    else:
        rgb = image[..., :3]
        mask = np.any(rgb < WHITE_THRESHOLD, axis=-1)
        if image.shape[-1] == 4:
            mask |= image[..., 3] > 0

    if not np.any(mask):
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    row_start = max(int(rows[0]) - TRIM_PADDING, 0)
    row_stop = min(int(rows[-1]) + TRIM_PADDING + 1, image.shape[0])
    col_start = max(int(cols[0]) - TRIM_PADDING, 0)
    col_stop = min(int(cols[-1]) + TRIM_PADDING + 1, image.shape[1])
    return image[row_start:row_stop, col_start:col_stop]


def load_image(path: Path) -> np.ndarray:
    return crop_whitespace(plt.imread(path))


def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image

    pad_total = target_height - image.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_width = [(pad_top, pad_bottom), (0, 0)]
    if image.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode="constant", constant_values=1.0)


def combine_row_images(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_height = max(left.shape[0], right.shape[0])
    left_padded = pad_to_height(left, target_height)
    right_padded = pad_to_height(right, target_height)
    return np.concatenate([left_padded, right_padded], axis=1)


def load_loss(perturbation: str) -> float:
    path = parameter_path(perturbation)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("loss ="):
                return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find loss in {path}")


def load_num_clones(perturbation: str) -> int:
    return int(pd.read_csv(real_data_path(perturbation)).shape[0])


def build_figure(layout: str) -> plt.Figure:
    if layout == "row":
        fig, axes = plt.subplots(1, len(PERTURBATIONS), figsize=(10.3, 2.65), squeeze=False)
        axes_iter = axes[0, :]
        fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.02, wspace=0.06)
    else:
        n_rows = len(PERTURBATIONS)
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, 1.9 * n_rows + 0.35), squeeze=False)
        axes_iter = axes[:, 0]
        fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.005, hspace=0.12)

    for ax, perturbation in zip(axes_iter, PERTURBATIONS):
        ecdf_path, isotonic_path = diagnostic_paths(perturbation)
        combined_image = combine_row_images(load_image(ecdf_path), load_image(isotonic_path))
        ax.imshow(combined_image)
        ax.axis("off")
        ax.set_title(
            f"{perturbation}, loss={load_loss(perturbation):.4g}, n={load_num_clones(perturbation)}",
            fontsize=10.5,
            pad=TITLE_PAD,
        )

    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    args = parse_args()
    validate_inputs()
    fig = build_figure(args.layout)
    save_figure(fig, Path(args.output_dir), args.stem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate a standalone legend for the lowest-loss multi-view projections figure.
The legend mirrors the styling used in `plot_lowest_loss_points_multi_view.py`
and is emitted as both PNG and SVG outputs for downstream figure assembly.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Required font settings for manuscript-quality figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import plot_lowest_loss_points_multi_view as mv  # noqa: E402

STAR_COLOR = getattr(mv, "STAR_COLOR", "#d62728")
get_color = mv.get_color
load_and_extract_lowest_points = mv.load_and_extract_lowest_points
parse_dataset_spec = mv.parse_dataset_spec


def _load_datasets(
    dataset_args: Iterable[str],
    n_points: int,
) -> List[Dict[str, object]]:
    """Load the requested datasets and return plotting metadata."""
    datasets: List[Dict[str, object]] = []
    for idx, spec in enumerate(dataset_args):
        label, path = parse_dataset_spec(spec)
        points = load_and_extract_lowest_points(path, n_points)
        if points.shape[0] < n_points:
            raise ValueError(f"Dataset '{label}' does not contain {n_points} valid lowest-loss points.")
        datasets.append(
            {
                "label": label,
                "color": get_color(label, idx),
                "points": points[:n_points],
            }
        )
    return datasets


def create_legend_figure(
    datasets: List[Dict[str, object]],
    highlight_label: str,
    output_prefix: str,
    legend_columns: int | None = None,
) -> Tuple[str, str]:
    """Render and save the standalone legend, returning the output paths."""
    legend_handles: List[Line2D] = []
    for dataset in datasets:
        label = cast(str, dataset["label"])
        color = dataset["color"]
        points = cast(np.ndarray, dataset["points"])
        count = int(points.shape[0])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=6,
                alpha=0.5,
                label=f"{label} {count} lowest-loss points",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.5,
                label=f"{label} linear fit",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=STAR_COLOR,
            markeredgecolor="black",
            markersize=10,
            label=f"{highlight_label} best-loss point",
        )
    )

    fig, ax = plt.subplots(figsize=(3.0, 4.0))
    ax.axis("off")
    ncol = legend_columns or 1
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=12,
        borderaxespad=0.0,
    )
    fig.tight_layout()

    png_path = f"{output_prefix}.png"
    svg_path = f"{output_prefix}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, svg_path


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a standalone legend for the multi-view lowest-loss plot.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset specification in the form LABEL=PATH. Provide at least two entries.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=(
            "./clonal_analysis/intermediate_files/"
            "lowest_loss_points_multi_view_projections_legend"
        ),
        help="Output path prefix (without extension) for the legend figure.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Number of lowest loss points used in the legend labels.",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Dataset label whose best-loss point is highlighted.",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=None,
        help="Optional override for the number of legend columns.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if len(args.dataset) < 2:
        raise ValueError("At least two --dataset arguments are required.")

    datasets = _load_datasets(args.dataset, args.n_points)
    default_highlight = cast(str, datasets[0]["label"])
    highlight_label = args.highlight or default_highlight
    if highlight_label not in {cast(str, ds["label"]) for ds in datasets}:
        raise ValueError(f"Highlight label '{highlight_label}' not found among provided datasets.")

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    png_path, svg_path = create_legend_figure(datasets, highlight_label, args.output_prefix, args.legend_columns)
    print(f"Saved legend PNG to {png_path}")
    print(f"Saved legend SVG to {svg_path}")


if __name__ == "__main__":
    main()

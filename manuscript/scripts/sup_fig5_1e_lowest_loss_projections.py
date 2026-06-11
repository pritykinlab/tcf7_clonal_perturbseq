#!/usr/bin/env python
"""Generate lowest-loss parameter projections for the Trim28-Setdb1 dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.grid"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
CLONAL_SCRIPTS_PATH = REPO_ROOT / "clonal_analysis" / "scripts"
UTILS_PATH = REPO_ROOT / "scripts" / "utils"

for path in (CLONAL_SCRIPTS_PATH, UTILS_PATH):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from plot_lowest_loss_points_multi_view import (  # noqa: E402
    calculate_shared_direction,
    create_line_points,
    determine_axis_limits,
    load_and_extract_lowest_points,
    set_axis_style,
)
from figure_format import perturbation_colors  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_12"
N_POINTS = 100
FOREGROUND_DATASETS = {"NT"}
DATASET_PATHS = {
    "NT": REPO_ROOT
    / "clonal_analysis"
    / "intermediate_files"
    / "trim28_setdb1"
    / "NT"
    / "grid_search_loss.npz",
    "Trim28": REPO_ROOT
    / "clonal_analysis"
    / "intermediate_files"
    / "trim28_setdb1"
    / "Trim28"
    / "grid_search_loss.npz",
}
PROJECTIONS = [
    (0, 1, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{asym}}$"),
    (0, 2, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{dif}}$"),
    (1, 2, r"$p_{\mathrm{asym}}$", r"$p_{\mathrm{dif}}$"),
]


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lowest-loss parameter projections."
    )
    parser.add_argument(
        "--mode",
        choices=["projection", "views", "all"],
        default="all",
        help="Choose the 3D projection, the orthogonal 2D views, or both.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved PNG and SVG outputs.",
    )
    parser.add_argument(
        "--projection-stem",
        default="e",
        help="Filename stem for the 3D projection output.",
    )
    parser.add_argument(
        "--views-stem",
        default="f",
        help="Filename stem for the orthogonal-view output.",
    )
    return parser.parse_args(args)


def load_datasets() -> list[dict[str, object]]:
    datasets: list[dict[str, object]] = []
    for idx, (label, path) in enumerate(DATASET_PATHS.items()):
        if not path.exists():
            raise FileNotFoundError(f"Missing clonal modeling results for {label}: {path}")
        datasets.append(
            {
                "label": label,
                "color": perturbation_colors.get(label, f"C{idx}"),
                "points": load_and_extract_lowest_points(str(path), N_POINTS),
            }
        )
    return datasets


def compute_line_coords(
    datasets: list[dict[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    centroids, direction = calculate_shared_direction(datasets)
    return {
        ds["label"]: create_line_points(
            {"centroid": centroids[ds["label"]], "direction": direction}
        )
        for ds in datasets
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK for {stem}: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def plot_3d_projection(
    datasets: list[dict[str, object]],
    axis_limits: tuple[float, float],
    line_coords: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    stem: str,
) -> None:
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    set_axis_style(ax, axis_limits)
    ax.tick_params(labelsize=12, pad=4)
    base_label_size = ax.xaxis.label.get_size()
    scaled = base_label_size * 2.0
    ax.set_xlabel(r"$p_{\mathrm{sym}}$", labelpad=10, fontsize=scaled)
    ax.set_ylabel(r"$p_{\mathrm{asym}}$", labelpad=10, fontsize=scaled)
    ax.set_zlabel(r"$p_{\mathrm{dif}}$", labelpad=10, fontsize=scaled)
    ax.view_init(elev=20, azim=60)

    for ds in datasets:
        coords = ds["points"][:, :3]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            color=ds["color"],
            s=20.25,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
        )

        line = line_coords.get(ds["label"])
        if line and line[0].size > 0:
            ax.plot(line[0], line[1], line[2], color=ds["color"], linewidth=4.5, alpha=0.8)

    save_figure(fig, output_dir, stem)


def plot_projection_views(
    datasets: list[dict[str, object]],
    axis_limits: tuple[float, float],
    line_coords: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    lower, upper = axis_limits

    nt_line = line_coords.get("NT")
    missing_labels = {
        0: r"$p_{\mathrm{sym}}$",
        1: r"$p_{\mathrm{asym}}$",
        2: r"$p_{\mathrm{dif}}$",
    }
    missing_index_map = {
        (0, 1): 2,
        (0, 2): 1,
        (1, 2): 0,
    }

    for ax, (x_idx, y_idx, x_label, y_label) in zip(axes, PROJECTIONS):
        for ds in datasets:
            coords = ds["points"][:, :3]
            scatter_zorder = 1 if ds["label"] in FOREGROUND_DATASETS else 0
            ax.scatter(
                coords[:, x_idx],
                coords[:, y_idx],
                color=ds["color"],
                s=22.5,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
                zorder=scatter_zorder,
            )

            line = line_coords.get(ds["label"])
            if line and line[0].size > 0:
                ax.plot(
                    line[x_idx],
                    line[y_idx],
                    color=ds["color"],
                    linewidth=3.0,
                    alpha=0.85,
                    zorder=4,
                )

        if nt_line and nt_line[0].size > 1:
            missing_idx = missing_index_map[(x_idx, y_idx)]
            x_vals = nt_line[x_idx]
            y_vals = nt_line[y_idx]
            endpoints = [
                (x_vals[0], y_vals[0], nt_line[missing_idx][0]),
                (x_vals[-1], y_vals[-1], nt_line[missing_idx][-1]),
            ]
            for ex, ey, mval in endpoints:
                ax.scatter(ex, ey, color="black", s=40, zorder=6)
                label_value = f"{mval:.2f}".rstrip("0").rstrip(".")
                x_mid = (lower + upper) / 2.0
                y_mid = (lower + upper) / 2.0
                x_offset = 15 if ex <= x_mid else -55
                y_offset = -18 if ey >= y_mid else 15
                ax.annotate(
                    f"{missing_labels[missing_idx]}={label_value}",
                    xy=(ex, ey),
                    xycoords="data",
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=12,
                    color="black",
                    zorder=7,
                    arrowprops=dict(arrowstyle="-", color="black", lw=1),
                )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ticks = np.arange(
            np.floor(lower * 10) / 10.0,
            np.ceil(upper * 10) / 10.0 + 0.001,
            0.1,
        )
        ticks = np.round(ticks, 2)
        ticks = [t for t in ticks if not np.isclose(t % 0.5, 0.0) or np.isclose(t, 0.0)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        base = ax.xaxis.label.get_size()
        scaled = base * 1.5
        ax.set_xlabel(x_label, fontsize=scaled)
        ax.set_ylabel(y_label, fontsize=scaled)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.tick_params(labelsize=10)

    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def main(args: Iterable[str] | None = None) -> None:
    parsed_args = parse_args(args)
    output_dir = Path(parsed_args.output_dir)
    datasets = load_datasets()
    axis_limits = determine_axis_limits(datasets)
    line_coords = compute_line_coords(datasets)

    if parsed_args.mode in {"projection", "all"}:
        plot_3d_projection(
            datasets,
            axis_limits,
            line_coords,
            output_dir,
            parsed_args.projection_stem,
        )
    if parsed_args.mode in {"views", "all"}:
        plot_projection_views(
            datasets,
            axis_limits,
            line_coords,
            output_dir,
            parsed_args.views_stem,
        )


if __name__ == "__main__":
    main()

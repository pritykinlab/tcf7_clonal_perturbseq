#!/usr/bin/env python
"""
Create a multi-view static visualization of the lowest-loss points for one or more perturbations.
Generates a 3x3 grid of perspectives with linear fits derived from a shared PCA direction and
highlights the best-loss point from a designated reference perturbation (default: first provided).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projection
from sklearn.decomposition import PCA

# Required font settings for all manuscript figures / static outputs
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from figure_format import perturbation_colors  # noqa: E402

def parse_dataset_spec(spec: str) -> Tuple[str, str]:
    """Parse a dataset specification of the form LABEL=PATH."""
    if "=" not in spec:
        raise ValueError(f"Dataset specification '{spec}' must be of the form LABEL=PATH")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Dataset specification '{spec}' is invalid")
    return label, path


def get_color(label: str, idx: int) -> str:
    """Return the project-defined color for the label, falling back to matplotlib defaults."""
    return perturbation_colors.get(label, f"C{idx}")


def load_and_extract_lowest_points(data_file: str, n_points: int = 100) -> np.ndarray:
    """
    Load loss landscape data and extract the n lowest loss points.

    Returns an array of shape (n_points, 4) with columns [p_self, p_split_one, p_split_two, loss].
    """
    data = np.load(data_file)
    p_self_values = data["p_self_values"]
    p_split_one_values = data["p_split_one_values"]
    p_split_two_values = data["p_split_two_values"]
    loss_grid = data["loss_grid"]
    data.close()

    all_points = []
    for i, p_self in enumerate(p_self_values):
        for j, p_split_one in enumerate(p_split_one_values):
            for k, p_split_two in enumerate(p_split_two_values):
                loss = loss_grid[i, j, k]
                if not np.isnan(loss) and (p_self + p_split_one + p_split_two) <= 1.0:
                    all_points.append([p_self, p_split_one, p_split_two, loss])

    if not all_points:
        raise ValueError(f"No valid points found in {data_file}")

    all_points = np.array(all_points)
    sorted_indices = np.argsort(all_points[:, 3])
    return all_points[sorted_indices[: min(n_points, len(all_points))]]


def create_line_points(
    line_params: Dict[str, np.ndarray], t_range: Tuple[float, float] = (-0.3, 0.3), n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points along a fitted line that respect the simplex constraints."""
    centroid = line_params["centroid"]
    direction = line_params["direction"]
    t_values = np.linspace(t_range[0], t_range[1], n_points)

    valid_points = []
    for t in t_values:
        point = centroid + t * direction
        if np.all(point >= 0) and np.all(point <= 1) and point.sum() <= 1.0:
            valid_points.append(point)

    if not valid_points:
        return np.array([]), np.array([]), np.array([])

    points = np.array(valid_points)
    return points[:, 0], points[:, 1], points[:, 2]


def calculate_shared_direction(datasets: List[Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute a shared PCA direction using centered points from all provided datasets.

    Returns a mapping of label -> centroid and the shared direction vector.
    """
    centroids: Dict[str, np.ndarray] = {}
    centered_blocks = []

    for ds in datasets:
        coords = ds["points"][:, :3]
        centroid = np.mean(coords, axis=0)
        centroids[ds["label"]] = centroid
        centered_blocks.append(coords - centroid)

    combined_centered = np.vstack(centered_blocks)
    pca = PCA(n_components=1)
    pca.fit(combined_centered)
    direction = pca.components_[0]
    if direction[0] < 0:
        direction = -direction

    return centroids, direction


def determine_axis_limits(datasets: List[Dict[str, np.ndarray]], padding: float = 0.02) -> Tuple[float, float]:
    """Determine shared axis limits with optional padding."""
    combined_points = np.vstack([ds["points"][:, :3] for ds in datasets])
    axis_min = max(0.0, combined_points.min() - padding)
    axis_max = min(1.0, combined_points.max() + padding)
    return axis_min, axis_max


def set_axis_style(ax, axis_limits: Tuple[float, float]):
    """Apply consistent styling to each 3D axis."""
    lower, upper = axis_limits
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    ax.set_xlabel(r"$p_{\mathrm{sym}}$")
    ax.set_ylabel(r"$p_{\mathrm{asym}}$")
    ax.set_zlabel(r"$p_{\mathrm{dif}}$")
    ax.tick_params(labelsize=8)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False  # type: ignore[attr-defined]
        axis._axinfo["grid"]["linewidth"] = 0.2  # type: ignore[attr-defined]


def add_points_and_lines(
    ax,
    datasets: List[Dict[str, np.ndarray]],
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    highlight_point: np.ndarray | None,
    highlight_color: str | None,
) -> None:
    """Add scatter points, fitted lines, and the highlighted point to an axis."""
    for ds in datasets:
        coords = ds["points"][:, :3]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            color=ds["color"],
            s=10,
            alpha=0.5,
            edgecolors="none",
        )

        line = line_coords.get(ds["label"])
        if line and line[0].size > 0:
            ax.plot(line[0], line[1], line[2], color=ds["color"], linewidth=1.5)

    if highlight_point is not None and highlight_color is not None:
        ax.scatter(
            highlight_point[0],
            highlight_point[1],
            highlight_point[2],
            color=highlight_color,
            marker="*",
            s=140,
            edgecolors="black",
            linewidths=0.5,
            alpha=1.0,
            zorder=10,
        )


def add_projection_plot(
    ax,
    datasets: List[Dict[str, np.ndarray]],
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    highlight_point: np.ndarray | None,
    highlight_color: str | None,
    axis_limits: Tuple[float, float],
    x_idx: int,
    y_idx: int,
    x_label: str,
    y_label: str,
) -> None:
    """Render 2D projection scatter with shared-direction lines."""
    for ds in datasets:
        coords = ds["points"][:, :3]
        ax.scatter(
            coords[:, x_idx],
            coords[:, y_idx],
            color=ds["color"],
            s=10,
            alpha=0.5,
            edgecolors="none",
        )

        line = line_coords.get(ds["label"])
        if line and line[0].size > 0:
            ax.plot(line[x_idx], line[y_idx], color=ds["color"], linewidth=1.5)

    if highlight_point is not None and highlight_color is not None:
        ax.scatter(
            highlight_point[x_idx],
            highlight_point[y_idx],
            color=highlight_color,
            marker="*",
            s=140,
            edgecolors="black",
            linewidths=0.5,
            alpha=1.0,
            zorder=10,
        )

    lower, upper = axis_limits
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)
    ax.grid(False)


def create_multi_view_plot(
    datasets: List[Dict[str, np.ndarray]],
    axis_limits: Tuple[float, float],
    output_prefix: str,
    highlight_label: str | None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Render the 3x3 grid of views, save PNG outputs, and return plot metadata."""
    centroids, direction = calculate_shared_direction(datasets)

    line_params: Dict[str, Dict[str, np.ndarray]] = {}
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ds in datasets:
        params = {"centroid": centroids[ds["label"]], "direction": direction}
        line_params[ds["label"]] = params
        line_coords[ds["label"]] = create_line_points(params)

    highlight_point: np.ndarray | None = None
    highlight_color: str | None = None
    highlight_loss: float | None = None
    if highlight_label is not None:
        highlight_dataset = next(ds for ds in datasets if ds["label"] == highlight_label)
        highlight_idx = int(np.argmin(highlight_dataset["points"][:, 3]))
        highlight_point = highlight_dataset["points"][highlight_idx, :3]
        highlight_loss = float(highlight_dataset["points"][highlight_idx, 3])
        highlight_color = highlight_dataset["color"]

    fig = plt.figure(figsize=(12, 12), constrained_layout=False)
    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            index = row * 3 + col + 1
            axes[row, col] = fig.add_subplot(3, 3, index, projection="3d")

    elevations = np.linspace(60, 20, 3)
    azimuths = np.linspace(-60, 60, 3)

    for row, elev in enumerate(elevations):
        for col, azim in enumerate(azimuths):
            ax = axes[row, col]
            add_points_and_lines(ax, datasets, line_coords, highlight_point, highlight_color)
            set_axis_style(ax, axis_limits)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("")

    legend_handles: List[Line2D] = []
    for ds in datasets:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ds["color"],
                markersize=6,
                alpha=0.5,
                label=f"{ds['label']} {len(ds['points'])} lowest-loss points",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=ds["color"],
                linewidth=1.5,
                label=f"{ds['label']} linear fit",
            )
        )

    if highlight_label is not None and highlight_color is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=highlight_color,
                markeredgecolor="black",
                markersize=10,
                label=f"{highlight_label} best-loss point",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=12,
    )

    fig.subplots_adjust(top=0.9, wspace=0.05, hspace=0.1)

    png_path = f"{output_prefix}.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved multi-view plot to {png_path}")

    return {
        "line_params": line_params,
        "highlight_label": highlight_label,
        "highlight_point": highlight_point,
        "highlight_loss": highlight_loss,
        "line_coords": line_coords,
        "axis_limits": axis_limits,
        "highlight_color": highlight_color,
    }


def create_projection_figure(
    datasets: List[Dict[str, np.ndarray]],
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    axis_limits: Tuple[float, float],
    highlight_point: np.ndarray | None,
    highlight_color: str | None,
    output_prefix: str,
) -> None:
    """Create stacked 2D projections (self-one, self-two, one-two)."""
    projections = [
        (0, 1, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{asym}}$"),
        (0, 2, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{dif}}$"),
        (1, 2, r"$p_{\mathrm{asym}}$", r"$p_{\mathrm{dif}}$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(4.5, 12), constrained_layout=False)

    for ax, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        add_projection_plot(
            ax,
            datasets,
            line_coords,
            highlight_point,
            highlight_color,
            axis_limits,
            x_idx,
            y_idx,
            x_label,
            y_label,
        )

    fig.subplots_adjust(hspace=0.25)
    output_path = f"{output_prefix}_projections.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved projection plot to {output_path}")


def _format_vector(vec: np.ndarray) -> str:
    return ", ".join(f"{v:.6f}" for v in vec)


def _format_line_equation(centroid: np.ndarray, direction: np.ndarray) -> str:
    return (
        f"p_sym = {centroid[0]:.6f} + t*{direction[0]:.6f}\n"
        f"p_asym = {centroid[1]:.6f} + t*{direction[1]:.6f}\n"
        f"p_dif = {centroid[2]:.6f} + t*{direction[2]:.6f}"
    )


def write_equation_summary(
    output_prefix: str,
    datasets: List[Dict[str, np.ndarray]],
    line_params: Dict[str, Dict[str, np.ndarray]],
    highlight_label: str | None,
    highlight_point: np.ndarray | None,
    highlight_loss: float | None,
    n_points: int,
) -> None:
    """Persist the fitted line equations and highlighted point coordinates."""
    lines: List[str] = []
    for ds in datasets:
        label = ds["label"]
        lines.append(f"{label} linear fit using {n_points} lowest-loss points:")
        params = line_params[label]
        lines.append(_format_line_equation(params["centroid"], params["direction"]))
        lines.append("")

    if highlight_label is not None and highlight_point is not None and highlight_loss is not None:
        lines.append(f"Highlighted {highlight_label} lowest-loss point (star marker):")
        lines.append(f"coordinates = [{_format_vector(highlight_point)}]")
        lines.append(f"loss = {highlight_loss:.6f}")

    output_path = f"{output_prefix}_equations.txt"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Saved line equations to {output_path}")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create static multi-view plots of lowest loss points.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset specification in the form LABEL=PATH. Provide at least two entries.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./clonal_analysis/intermediate_files/lowest_loss_points_multi_view",
        help="Output path prefix (without extension) for the generated figure.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Number of lowest loss points to plot from each dataset.",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Label of the dataset to highlight with a star (defaults to the first provided dataset).",
    )
    parser.add_argument(
        "--skip-highlight",
        action="store_true",
        help="Disable highlighting and star marker for the dataset with lowest loss.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    if len(args.dataset) < 2:
        raise ValueError("At least two --dataset arguments are required.")

    dataset_specs = [parse_dataset_spec(spec) for spec in args.dataset]
    highlight_label = None if args.skip_highlight else (args.highlight or dataset_specs[0][0])

    datasets: List[Dict[str, np.ndarray]] = []
    for idx, (label, path) in enumerate(dataset_specs):
        print(f"Loading dataset '{label}' from {path}")
        points_all = load_and_extract_lowest_points(path, args.n_points)
        if points_all.shape[0] < args.n_points:
            raise ValueError(f"Dataset '{label}' does not contain {args.n_points} valid lowest-loss points.")
        datasets.append(
            {
                "label": label,
                "path": path,
                "color": get_color(label, idx),
                "points": points_all[: args.n_points],
            }
        )

    if highlight_label is not None and highlight_label not in {ds["label"] for ds in datasets}:
        raise ValueError(f"Highlight label '{highlight_label}' not found among provided datasets.")

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    axis_limits = determine_axis_limits(datasets)
    summary = create_multi_view_plot(datasets, axis_limits, args.output_prefix, highlight_label)
    create_projection_figure(
        datasets,
        summary["line_coords"],
        summary["axis_limits"],
        summary["highlight_point"],
        summary["highlight_color"],
        args.output_prefix,
    )
    write_equation_summary(
        args.output_prefix,
        datasets,
        summary["line_params"],
        summary["highlight_label"],
        summary["highlight_point"],
        summary["highlight_loss"],
        args.n_points,
    )


if __name__ == "__main__":
    main()

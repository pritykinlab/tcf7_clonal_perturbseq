#!/usr/bin/env python
"""Generate Figure 5f: parameter range bar plot matching the Figure 4i format."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = (6.5, 4.5)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLONAL_SCRIPTS_PATH = os.path.join(REPO_ROOT, "clonal_analysis", "scripts")
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")

for path in (CLONAL_SCRIPTS_PATH, UTILS_PATH):
    if path not in sys.path:
        sys.path.append(path)

from plot_lowest_loss_points_multi_view import calculate_shared_direction  # noqa: E402
from figure_format import perturbation_colors  # noqa: E402

DATASET_SPECS = [
    (
        "NT",
        os.path.join(
            REPO_ROOT,
            "clonal_analysis",
            "intermediate_files",
            "trim28_setdb1",
            "NT",
            "grid_search_loss.npz",
        ),
    ),
    (
        "Trim28",
        os.path.join(
            REPO_ROOT,
            "clonal_analysis",
            "intermediate_files",
            "trim28_setdb1",
            "Trim28",
            "grid_search_loss.npz",
        ),
    ),
]

OUTPUT_DIR = os.path.join(REPO_ROOT, "manuscript", "plots", "Figure5")
FIGURE_5M_NAME = "f"
FIGURE_5M_LEGEND_NAME = "f_legend"
N_POINTS = 100

PARAM_ORDER: List[Tuple[str, str]] = [
    ("p_self", r"$p_{\mathrm{sym}}$"),
    ("p_one", r"$p_{\mathrm{asym}}$"),
    ("p_two", r"$p_{\mathrm{dif}}$"),
    ("p_sum", r"$p_{\mathrm{sym}} + p_{\mathrm{asym}} + p_{\mathrm{dif}}$"),
]


@dataclass
class DatasetPoints:
    label: str
    color: str
    points: np.ndarray


def load_and_extract_lowest_points(data_file: str, n_points: int = 100) -> np.ndarray:
    """
    Load loss landscape data and extract the lowest-loss points.

    Returns an array of shape (n_subset, 4) with [p_self, p_split_one, p_split_two, loss].
    """
    data = np.load(data_file)
    p_self_values = data["p_self_values"]
    p_split_one_values = data["p_split_one_values"]
    p_split_two_values = data["p_split_two_values"]
    loss_grid = data["loss_grid"]
    data.close()

    records = []
    for i, p_self in enumerate(p_self_values):
        for j, p_split_one in enumerate(p_split_one_values):
            for k, p_split_two in enumerate(p_split_two_values):
                loss = loss_grid[i, j, k]
                if np.isnan(loss):
                    continue
                if (p_self + p_split_one + p_split_two) > 1.0:
                    continue
                records.append([p_self, p_split_one, p_split_two, loss])

    if not records:
        raise ValueError(f"No valid points found in {data_file}")

    points = np.array(records)
    sorted_indices = np.argsort(points[:, 3])
    return points[sorted_indices[: min(n_points, len(points))]]


def load_datasets(n_points: int = N_POINTS) -> List[DatasetPoints]:
    datasets: List[DatasetPoints] = []
    for idx, (label, path) in enumerate(DATASET_SPECS):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing clonal modeling results for {label}: {path}")

        points = load_and_extract_lowest_points(path, n_points)
        datasets.append(
            DatasetPoints(
                label=label,
                color=perturbation_colors.get(label, f"C{idx}"),
                points=points,
            )
        )
    return datasets


def find_permissible_ranges(centroid: np.ndarray, direction: np.ndarray) -> Dict[str, Tuple[float, float]]:
    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        if direction[i] > 0:
            t_max = min(t_max, (1 - centroid[i]) / direction[i])
            t_min = max(t_min, -centroid[i] / direction[i])
        elif direction[i] < 0:
            t_max = min(t_max, -centroid[i] / direction[i])
            t_min = max(t_min, (1 - centroid[i]) / direction[i])

    sum_centroid = np.sum(centroid)
    sum_direction = np.sum(direction)

    if sum_direction > 0:
        t_max = min(t_max, (1 - sum_centroid) / sum_direction)
    elif sum_direction < 0:
        t_min = max(t_min, (1 - sum_centroid) / sum_direction)

    p_min = centroid + t_min * direction
    p_max = centroid + t_max * direction

    return {
        "p_self": (
            max(0.0, min(p_min[0], p_max[0])),
            min(1.0, max(p_min[0], p_max[0])),
        ),
        "p_one": (
            max(0.0, min(p_min[1], p_max[1])),
            min(1.0, max(p_min[1], p_max[1])),
        ),
        "p_two": (
            max(0.0, min(p_min[2], p_max[2])),
            min(1.0, max(p_min[2], p_max[2])),
        ),
        "p_sum": (
            max(0.0, min(p_min.sum(), p_max.sum())),
            min(1.0, max(p_min.sum(), p_max.sum())),
        ),
    }


def create_bar_plot(
    datasets: List[DatasetPoints],
    range_map: Dict[str, Dict[str, Tuple[float, float]]],
) -> Tuple[List[Line2D], List[str]]:
    params = [key for key, _ in PARAM_ORDER]
    labels = [label for _, label in PARAM_ORDER]

    n_datasets = len(datasets)
    width = 0.6 / n_datasets if n_datasets else 0.6
    x = np.arange(len(params))

    fig, ax = plt.subplots()

    offsets_map: Dict[str, np.ndarray] = {}
    handles: List[Line2D] = []
    legend_labels: List[str] = []

    cap_half = min(0.18, width * 0.45 if n_datasets else 0.18)

    for idx, ds in enumerate(datasets):
        ranges = range_map[ds.label]
        mins = [ranges[p][0] for p in params]
        maxs = [ranges[p][1] for p in params]
        offsets = x + (idx - (n_datasets - 1) / 2) * width
        offsets_map[ds.label] = offsets

        for param_idx, offset in enumerate(offsets):
            y_min = mins[param_idx]
            y_max = maxs[param_idx]

            if np.isclose(y_min, y_max):
                ax.hlines(
                    y_min,
                    offset - cap_half,
                    offset + cap_half,
                    color=ds.color,
                    linewidth=2.2,
                    zorder=3,
                )
            else:
                ax.vlines(
                    offset,
                    y_min,
                    y_max,
                    color=ds.color,
                    linewidth=2.4,
                    zorder=2,
                )
                ax.hlines(
                    [y_min, y_max],
                    offset - cap_half,
                    offset + cap_half,
                    color=ds.color,
                    linewidth=2.2,
                    zorder=2,
                )

        handles.append(
            Line2D([0, 1], [0, 0], color=ds.color, linewidth=3.0, solid_capstyle="projecting")
        )
        legend_labels.append(ds.label)

    max_upper = max(r[p][1] for r in range_map.values() for p in params)
    ax.set_ylim(0, max(max_upper * 1.2, 1.2))
    ax.set_ylabel("Division probability range")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.4, alpha=0.9, zorder=3)
    ax.text(
        x[-1] + 0.35,
        1.01,
        "probability of differentiated division",
        color="red",
        fontsize=12,
        va="bottom",
        ha="right",
    )

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, f"{FIGURE_5M_NAME}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{FIGURE_5M_NAME}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return handles, legend_labels


def create_legend(handles: List[Line2D], labels: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.axis("off")
    fig.legend(handles, labels, loc="center", frameon=False, ncol=1)
    legend_png = os.path.join(OUTPUT_DIR, f"{FIGURE_5M_LEGEND_NAME}.png")
    legend_svg = os.path.join(OUTPUT_DIR, f"{FIGURE_5M_LEGEND_NAME}.svg")
    fig.savefig(legend_png, dpi=300, bbox_inches="tight")
    fig.savefig(legend_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets = load_datasets()

    centroids, shared_direction = calculate_shared_direction(
        [{"label": ds.label, "points": ds.points} for ds in datasets]
    )
    range_map = {
        ds.label: find_permissible_ranges(centroids[ds.label], shared_direction) for ds in datasets
    }

    handles, labels = create_bar_plot(datasets, range_map)
    create_legend(handles, labels)
    print(f"Figure 5f saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

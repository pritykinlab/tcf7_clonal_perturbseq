#!/usr/bin/env python
"""Generate Figure 4i: parameter range bar plot with highlighted random points."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Required fonts for manuscript figures
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

from plot_lowest_loss_points_multi_view import (  # noqa: E402
    calculate_shared_direction,
    load_and_extract_lowest_points,
)
from figure_format import perturbation_colors  # noqa: E402

DATASET_SPECS = [
    ("NT", os.path.join(REPO_ROOT, "clonal_analysis", "intermediate_files", "gene10", "NT", "grid_search_loss.npz")),
    ("Bach2", os.path.join(REPO_ROOT, "clonal_analysis", "intermediate_files", "gene10", "Bach2", "grid_search_loss.npz")),
]

OUTPUT_DIR = os.path.join(REPO_ROOT, "manuscript", "plots", "Figure4")
FIGURE4I_NAME = "i"
FIGURE4I_LEGEND_NAME = "i_legend"
HIGHLIGHT_METADATA = os.path.join(OUTPUT_DIR, "f_highlight_points.json")
N_POINTS = 100
HIGHLIGHT_COLORS = [
    (0.87, 0.17, 0.17),
    (0.16, 0.32, 0.82),
]
HIGHLIGHT_SEED = 98765
HIGHLIGHT_COUNT = 2


@dataclass
class DatasetPoints:
    label: str
    color: str
    points: np.ndarray


@dataclass
class HighlightPoint:
    label: str
    index: int
    p_self: float
    p_split_one: float
    p_split_two: float
    color: Tuple[float, float, float]

    @property
    def p_sum(self) -> float:
        return self.p_self + self.p_split_one + self.p_split_two


def load_datasets(n_points: int = N_POINTS) -> List[DatasetPoints]:
    datasets: List[DatasetPoints] = []
    for idx, (label, path) in enumerate(DATASET_SPECS):
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


def select_highlight_points(dataset: DatasetPoints) -> List[HighlightPoint]:
    if dataset.points.shape[0] < HIGHLIGHT_COUNT + 1:
        raise ValueError("Not enough points to select random points.")

    candidate_indices = np.arange(1, dataset.points.shape[0])
    rng = np.random.default_rng(HIGHLIGHT_SEED)
    chosen = rng.choice(candidate_indices, size=HIGHLIGHT_COUNT, replace=False)
    chosen.sort()

    highlights: List[HighlightPoint] = []
    for color_idx, idx in enumerate(chosen):
        params = dataset.points[idx]
        highlights.append(
            HighlightPoint(
                label=dataset.label,
                index=int(idx),
                p_self=float(params[0]),
                p_split_one=float(params[1]),
                p_split_two=float(params[2]),
                color=HIGHLIGHT_COLORS[color_idx % len(HIGHLIGHT_COLORS)],
            )
        )
    return highlights


def load_highlights(dataset: DatasetPoints) -> List[HighlightPoint]:
    if os.path.exists(HIGHLIGHT_METADATA):
        with open(HIGHLIGHT_METADATA, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        highlights: List[HighlightPoint] = []
        for idx, entry in enumerate(data):
            highlights.append(
                HighlightPoint(
                    label=entry["label"],
                    index=entry["index"],
                    p_self=entry["p_self"],
                    p_split_one=entry["p_split_one"],
                    p_split_two=entry["p_split_two"],
                    color=HIGHLIGHT_COLORS[idx % len(HIGHLIGHT_COLORS)],
                )
            )
        return highlights
    return select_highlight_points(dataset)


def create_bar_plot(
    datasets: List[DatasetPoints],
    range_map: Dict[str, Dict[str, Tuple[float, float]]],
    highlights: List[HighlightPoint],
) -> Tuple[List[Line2D], List[str]]:
    params = ["p_self", "p_one", "p_two", "p_sum"]
    labels = [
        r"$p_{\mathrm{sym}}$",
        r"$p_{\mathrm{asym}}$",
        r"$p_{\mathrm{dif}}$",
        r"$p_{\mathrm{sym}} + p_{\mathrm{asym}} + p_{\mathrm{dif}}$",
    ]

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
        legend_labels.append(f"{ds.label} range")

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

    highlight_counts: Dict[str, int] = {}
    for highlight in highlights:
        highlight_counts[highlight.label] = highlight_counts.get(highlight.label, 0) + 1

    jitter_scale_default = min(cap_half * 0.9, width * 0.35) if cap_half > 0 else width * 0.2
    jitter_values_map: Dict[str, np.ndarray] = {}
    for label, count in highlight_counts.items():
        if count <= 1:
            jitter_values_map[label] = np.array([0.0])
        else:
            base = np.linspace(-1.0, 1.0, count)
            jitter_values_map[label] = base * jitter_scale_default

    jitter_index_map = {label: 0 for label in highlight_counts}

    for idx, highlight in enumerate(highlights):
        label = highlight.label
        if label not in offsets_map:
            continue
        offsets = offsets_map[label]
        values = [highlight.p_self, highlight.p_split_one, highlight.p_split_two, highlight.p_sum]
        jitter_array = jitter_values_map.get(label, np.array([0.0]))
        array_idx = min(jitter_index_map.get(label, 0), len(jitter_array) - 1)
        jitter = jitter_array[array_idx]
        jitter_index_map[label] = array_idx + 1
        ax.scatter(
            offsets + jitter,
            values,
            marker="*",
            color=highlight.color,
            s=260,
            edgecolors="black",
            linewidths=0.8,
            zorder=1.5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=highlight.color,
                markeredgecolor="black",
                markersize=12,
            )
        )
        legend_labels.append(f"random point {idx + 1}")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, f"{FIGURE4I_NAME}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{FIGURE4I_NAME}.svg")
    pdf_path = os.path.join(OUTPUT_DIR, f"{FIGURE4I_NAME}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return handles, legend_labels


def create_legend(handles: List[Line2D], labels: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.axis("off")
    fig.legend(handles, labels, loc="center", frameon=False, ncol=1)
    legend_png = os.path.join(OUTPUT_DIR, f"{FIGURE4I_LEGEND_NAME}.png")
    legend_svg = os.path.join(OUTPUT_DIR, f"{FIGURE4I_LEGEND_NAME}.svg")
    fig.savefig(legend_png, dpi=300, bbox_inches="tight")
    fig.savefig(legend_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets = load_datasets()
    nt_dataset = next(ds for ds in datasets if ds.label == "NT")

    highlights = load_highlights(nt_dataset)

    centroids, shared_direction = calculate_shared_direction(
        [{"label": ds.label, "points": ds.points} for ds in datasets]
    )
    range_map = {
        ds.label: find_permissible_ranges(centroids[ds.label], shared_direction) for ds in datasets
    }

    handles, labels = create_bar_plot(datasets, range_map, highlights)
    create_legend(handles, labels)
    print(f"Figure 4i saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

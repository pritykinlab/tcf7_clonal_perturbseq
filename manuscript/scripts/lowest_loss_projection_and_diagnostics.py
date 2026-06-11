#!/usr/bin/env python
"""
Generate Figure 4f, 4g, and 4h panels from the lowest-loss parameter grid search.

Outputs
-------
Figure 4f: Bottom-right 3D projection of the lowest-loss points with dataset-colored
           markers and two random NT points.
Figure 4g: Diagnostic plots for the same random NT points showing
           (i) ECDF comparison and (ii) progenitor proportion vs log2 clone size.
           The diagnostics are arranged as a 2x2 grid (rows = random points).
Figure 4h: Three orthogonal 2D projections of the lowest-loss points with shared
           PCA-derived lines and highlighted random NT points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)
from sklearn.isotonic import IsotonicRegression

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLONAL_SCRIPTS_PATH = os.path.join(REPO_ROOT, "clonal_analysis", "scripts")
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")
CLONAL_MODELING_PATH = os.path.join(REPO_ROOT, "scripts", "clonal_modeling", "scripts")

for path in (CLONAL_SCRIPTS_PATH, UTILS_PATH, CLONAL_MODELING_PATH):
    if path not in sys.path:
        sys.path.append(path)

from plot_lowest_loss_points_multi_view import (  # noqa: E402
    calculate_shared_direction,
    create_line_points,
    determine_axis_limits,
    load_and_extract_lowest_points,
    set_axis_style,
)
from figure_format import perturbation_colors  # noqa: E402
from isotonic_regression import simulate_clone_sizes_advanced  # noqa: E402

# =============================================================================
# Constants
# =============================================================================
DATASET_PATHS = {
    "NT": os.path.join(
        REPO_ROOT,
        "clonal_analysis",
        "intermediate_files",
        "gene10",
        "NT",
        "grid_search_loss.npz",
    ),
    "Bach2": os.path.join(
        REPO_ROOT,
        "clonal_analysis",
        "intermediate_files",
        "gene10",
        "Bach2",
        "grid_search_loss.npz",
    ),
}

REAL_DATA_PATH = os.path.join(
    REPO_ROOT,
    "scripts",
    "clonal_modeling",
    "real_data",
    "gene10",
    "input",
    "NT.csv",
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "manuscript", "plots", "Figure4")
FIGURE4F_NAME = "f"
FIGURE4G_NAME = "g"
FIGURE4H_NAME = "h"
FIGURE4F_EQUATION_NAME = "f_equation"
HIGHLIGHT_METADATA = os.path.join(OUTPUT_DIR, "f_highlight_points.json")
PARAM_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "f_highlight_parameters.txt")

# Selection / simulation settings
N_LOWEST_POINTS = 100
HIGHLIGHT_COUNT = 2
HIGHLIGHT_SEED = 98765
NUM_PROGENITORS = 300_000
MEAN_DIVISIONS = 5.0
SUBSAMPLE_FRACTION = 0.15
HIGHLIGHT_COLORS = [
    (0.87, 0.17, 0.17),  # red
    (0.16, 0.32, 0.82),  # blue
]

# Datasets that should be drawn on top in 2D projections (e.g., NT grey points)
FOREGROUND_DATASETS = {"NT"}


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class DatasetPoints:
    label: str
    color: str
    points: np.ndarray  # shape (n, 4) columns: [p_self, p_split_one, p_split_two, loss]


@dataclass
class HighlightPoint:
    label: str
    index: int
    p_self: float
    p_split_one: float
    p_split_two: float
    loss: float
    color: Tuple[float, float, float]


# =============================================================================
# Utility functions
# =============================================================================
def load_datasets(n_points: int) -> List[DatasetPoints]:
    """Load the lowest-loss points for each dataset defined in DATASET_PATHS."""
    datasets: List[DatasetPoints] = []
    for idx, (label, path) in enumerate(DATASET_PATHS.items()):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected dataset file not found: {path}")
        points = load_and_extract_lowest_points(path, n_points)
        if points.shape[0] < n_points:
            raise ValueError(
                f"Dataset '{label}' provides only {points.shape[0]} valid points; "
                f"{n_points} required."
            )
        datasets.append(
            DatasetPoints(
                label=label,
                color=perturbation_colors.get(label, f"C{idx}"),
                points=points,
            )
        )
    return datasets


def select_highlight_points(dataset: DatasetPoints) -> List[HighlightPoint]:
    """
    Select two reproducible random NT points (excluding the absolute lowest-loss point)
    and return their metadata.
    """
    if dataset.points.shape[0] < HIGHLIGHT_COUNT + 1:
        raise ValueError("Not enough points to select random points.")

    candidate_indices = np.arange(1, dataset.points.shape[0])  # skip the very best (index 0)
    rng = np.random.default_rng(HIGHLIGHT_SEED)
    chosen = rng.choice(candidate_indices, size=HIGHLIGHT_COUNT, replace=False)
    chosen.sort()

    highlights: List[HighlightPoint] = []
    for color_idx, idx in enumerate(chosen):
        highlight_color = HIGHLIGHT_COLORS[color_idx % len(HIGHLIGHT_COLORS)]
        params = dataset.points[idx]
        highlights.append(
            HighlightPoint(
                label="NT",
                index=int(idx),
                p_self=float(params[0]),
                p_split_one=float(params[1]),
                p_split_two=float(params[2]),
                loss=float(params[3]),
                color=highlight_color,
            )
        )
    return highlights


def save_highlight_metadata(highlights: List[HighlightPoint]) -> None:
    """Persist random point metadata for reproducibility."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = [
        {
            "label": h.label,
            "index": h.index,
            "p_self": h.p_self,
            "p_split_one": h.p_split_one,
            "p_split_two": h.p_split_two,
            "loss": h.loss,
        }
        for h in highlights
    ]
    with open(HIGHLIGHT_METADATA, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_parameter_summary(highlights: List[HighlightPoint]) -> None:
    """Write a human-readable summary of random point parameters."""
    lines: List[str] = []
    for idx, h in enumerate(highlights, start=1):
        lines.append(f"random point {idx} ({h.label})")
        lines.append(f"  index = {h.index}")
        lines.append(f"  p_sym = {h.p_self:.6f}")
        lines.append(f"  p_asym = {h.p_split_one:.6f}")
        lines.append(f"  p_dif = {h.p_split_two:.6f}")
        lines.append(f"  loss = {h.loss:.6f}")
        lines.append("")
    with open(PARAM_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def load_real_nt_data(csv_path: str) -> pd.DataFrame:
    """Load NT real clone data and derive useful columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real NT data CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"Num Progenitor", "Num Effector"}
    if not required.issubset(df.columns):
        raise ValueError(f"Real NT data must contain columns: {required}")

    total = df["Num Progenitor"] + df["Num Effector"]
    df["Clone Size"] = total
    df["Proportion of Progenitors"] = df["Num Progenitor"] / total.clip(lower=1)
    df["Log2 Clone Size"] = np.log2(total.clip(lower=1))
    return df


def compute_observed_curves(df_real: pd.DataFrame, num_points: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ECDF and isotonic regression curves for the observed data."""
    max_log_clone_size = df_real["Log2 Clone Size"].max()
    grid = np.linspace(0, max_log_clone_size * 0.95, num_points)

    ecdf = sm.distributions.ECDF(df_real["Log2 Clone Size"].dropna())
    ecdf_values = ecdf(grid)

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(df_real["Log2 Clone Size"].fillna(0), df_real["Proportion of Progenitors"].fillna(0))
    iso_values = iso.predict(grid)

    return grid, ecdf_values, iso_values


def simulate_for_highlight(highlight: HighlightPoint, seed_offset: int = 0) -> pd.DataFrame:
    """Simulate clone sizes for a selected random point parameter set."""
    df_sim = simulate_clone_sizes_advanced(
        num_progenitors=NUM_PROGENITORS,
        mean_divisions=MEAN_DIVISIONS,
        subsample_fraction=SUBSAMPLE_FRACTION,
        p_self=highlight.p_self,
        p_split_one=highlight.p_split_one,
        p_split_two=highlight.p_split_two,
        division_type="poisson",
        random_seed=HIGHLIGHT_SEED + seed_offset,
    )
    if df_sim.empty:
        raise RuntimeError("Simulation returned no clones; check parameter validity.")
    return df_sim


def compute_simulated_curves(df_sim: pd.DataFrame, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ECDF and isotonic regression curves for simulated clones."""
    ecdf = sm.distributions.ECDF(df_sim["Log2 Clone Size"].dropna())
    ecdf_values = ecdf(grid)

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(df_sim["Log2 Clone Size"].fillna(0), df_sim["Proportion of Progenitors"].fillna(0))
    iso_values = iso.predict(grid)
    return ecdf_values, iso_values


# =============================================================================
# Plotting functions
# =============================================================================
def plot_projection(
    datasets: List[DatasetPoints],
    highlights: List[HighlightPoint],
    axis_limits: Tuple[float, float],
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Create the 3D projection figure (Figure 4f)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        coords = ds.points[:, :3]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            color=ds.color,
            s=20.25,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
        )

        line = line_coords.get(ds.label)
        if line and line[0].size > 0:
            ax.plot(line[0], line[1], line[2], color=ds.color, linewidth=4.5, alpha=0.8)

    for highlight in highlights:
        ax.scatter(
            highlight.p_self,
            highlight.p_split_one,
            highlight.p_split_two,
            color=highlight.color,
            marker="*",
            s=360,
            edgecolors="black",
            linewidths=0.8,
            zorder=20,
        )

    figure_png = os.path.join(OUTPUT_DIR, f"{FIGURE4F_NAME}.png")
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_line_coordinates(datasets: List[DatasetPoints]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    line_params: Dict[str, Dict[str, np.ndarray]] = {}
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    centroids, direction = calculate_shared_direction(
        [{"label": ds.label, "points": ds.points} for ds in datasets]
    )
    for ds in datasets:
        params = {"centroid": centroids[ds.label], "direction": direction}
        line_params[ds.label] = params
        line_coords[ds.label] = create_line_points(params)
    return line_params, line_coords


def _format_equation_lines(centroid: np.ndarray, direction: np.ndarray) -> List[str]:
    def format_component(base: float, coef: float) -> str:
        base_str = f"{base:.2f}"
        if np.isclose(coef, 0.0):
            return base_str
        sign = "+" if coef >= 0 else "-"
        return f"{base_str} {sign} {abs(coef):.2f}t"

    return [
        rf"$p_{{\mathrm{{sym}}}} = {format_component(centroid[0], direction[0])}$",
        rf"$p_{{\mathrm{{asym}}}} = {format_component(centroid[1], direction[1])}$",
        rf"$p_{{\mathrm{{dif}}}} = {format_component(centroid[2], direction[2])}$",
    ]


def create_line_equation_image(
    datasets: List[DatasetPoints],
    line_params: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Render standalone PNG/PDF summaries of the shared-direction line equations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_height = 1.0 + 1.2 * len(datasets)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))
    ax.axis("off")

    heading_kwargs = {
        "transform": ax.transAxes,
        "ha": "left",
        "va": "top",
        "fontsize": 18,
    }
    equation_kwargs = {
        "transform": ax.transAxes,
        "ha": "left",
        "va": "top",
        "fontsize": 16,
        "color": "black",
    }

    y_cursor = 0.95
    heading_spacing = 0.08
    line_spacing = 0.08
    block_spacing = 0.04

    for ds in datasets:
        params = line_params.get(ds.label)
        if not params:
            continue

        ax.text(0.02, y_cursor, f"{ds.label} shared-direction line", color=ds.color, **heading_kwargs)
        y_cursor -= heading_spacing

        for equation_line in _format_equation_lines(params["centroid"], params["direction"]):
            ax.text(0.05, y_cursor, equation_line, **equation_kwargs)
            y_cursor -= line_spacing

        y_cursor -= block_spacing

    equation_base = os.path.join(OUTPUT_DIR, FIGURE4F_EQUATION_NAME)
    fig.savefig(f"{equation_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{equation_base}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_projection_views(
    datasets: List[DatasetPoints],
    highlights: List[HighlightPoint],
    axis_limits: Tuple[float, float],
    line_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Create three orthogonal 2D projections (Figure 4h)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    projections = [
        (0, 1, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{asym}}$"),
        (0, 2, r"$p_{\mathrm{sym}}$", r"$p_{\mathrm{dif}}$"),
        (1, 2, r"$p_{\mathrm{asym}}$", r"$p_{\mathrm{dif}}$"),
    ]

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

    for ax, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        for ds in datasets:
            coords = ds.points[:, :3]
            scatter_zorder = 1 if ds.label in FOREGROUND_DATASETS else 0
            ax.scatter(
                coords[:, x_idx],
                coords[:, y_idx],
                color=ds.color,
                s=22.5,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
                zorder=scatter_zorder,
            )

            line = line_coords.get(ds.label)
            if line and line[0].size > 0:
                ax.plot(
                    line[x_idx],
                    line[y_idx],
                    color=ds.color,
                    linewidth=3.0,
                    alpha=0.85,
                    zorder=4,
                )

        if nt_line and nt_line[0].size > 1:
            missing_idx = missing_index_map[(x_idx, y_idx)]
            x_vals = nt_line[x_idx]
            y_vals = nt_line[y_idx]
            endpoints = [(x_vals[0], y_vals[0], nt_line[missing_idx][0]),
                         (x_vals[-1], y_vals[-1], nt_line[missing_idx][-1])]
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

        for highlight in highlights:
            values = np.array([highlight.p_self, highlight.p_split_one, highlight.p_split_two])
            ax.scatter(
                values[x_idx],
                values[y_idx],
                color=highlight.color,
                marker="*",
                s=240,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ticks = np.arange(np.floor(lower * 10) / 10.0, np.ceil(upper * 10) / 10.0 + 0.001, 0.1)
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
    figure_png = os.path.join(OUTPUT_DIR, f"{FIGURE4H_NAME}.png")
    figure_pdf = os.path.join(OUTPUT_DIR, f"{FIGURE4H_NAME}.pdf")
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def create_projection_legend(datasets: List[DatasetPoints], highlights: List[HighlightPoint]) -> None:
    """Generate standalone legend files for the projection plot."""
    handles: List[Line2D] = []
    labels: List[str] = []

    for ds in datasets:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ds.color,
                markeredgecolor="none",
                markersize=8,
                label=f"{ds.label} 100 lowest-loss points",
            )
        )
        labels.append(handles[-1].get_label())
        handles.append(
            Line2D(
                [0],
                [0],
                color=ds.color,
                linewidth=2.0,
                label=f"{ds.label} shared-direction line",
            )
        )
        labels.append(handles[-1].get_label())

    for idx, highlight in enumerate(highlights):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=highlight.color,
                markeredgecolor="black",
                markersize=12,
                label=f"random point {idx + 1}",
            )
        )
        labels.append(handles[-1].get_label())

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.axis("off")
    fig.legend(handles, labels, loc="center", frameon=False, ncol=1)
    legend_png = os.path.join(OUTPUT_DIR, f"{FIGURE4F_NAME}_legend.png")
    legend_svg = os.path.join(OUTPUT_DIR, f"{FIGURE4F_NAME}_legend.svg")
    fig.savefig(legend_png, dpi=300, bbox_inches="tight")
    fig.savefig(legend_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(
    highlights: List[HighlightPoint],
    df_real: pd.DataFrame,
    grid: np.ndarray,
    real_ecdf: np.ndarray,
    real_iso: np.ndarray,
) -> None:
    """Create the diagnostic figure (Figure 4g) with combined panels."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=False)
    ax_ecdf, ax_iso = axes

    nt_color = perturbation_colors.get("NT", "grey")

    # ECDF plot
    ax_ecdf.plot(
        grid,
        real_ecdf,
        color=nt_color,
        linewidth=2.0,
        label="NT real data",
    )

    # Progenitor proportion plot
    ax_iso.plot(
        grid,
        real_iso,
        color=nt_color,
        linewidth=2.0,
        label="NT real data",
    )

    title_lines = []

    for idx, highlight in enumerate(highlights):
        df_sim = simulate_for_highlight(highlight, seed_offset=idx + 1)
        sim_ecdf, sim_iso = compute_simulated_curves(df_sim, grid)
        sim_label = f"random point {idx + 1} simulated"

        ax_ecdf.plot(
            grid,
            sim_ecdf,
            color=highlight.color,
            linewidth=2.2,
            linestyle="--",
            label=sim_label,
        )

        ax_iso.plot(
            grid,
            sim_iso,
            color=highlight.color,
            linewidth=2.2,
            linestyle="--",
            label=sim_label,
        )

        title_lines.append(f"random point {idx + 1} (loss={highlight.loss:.4f})")

    base_x = ax_ecdf.xaxis.label.get_size()
    ax_ecdf.set_xlabel("log2(clone size)", fontsize=base_x * 1.5)
    base_y = ax_ecdf.yaxis.label.get_size()
    ax_ecdf.set_ylabel("ECDF", fontsize=base_y * 1.5)
    ax_ecdf.set_xlim(grid.min(), grid.max())
    ax_ecdf.set_ylim(0, 1)
    ax_ecdf.tick_params(labelsize=15)
    ax_ecdf.legend(frameon=False, loc="lower right")
    ax_ecdf.set_title(
        "\n".join(title_lines),
        fontsize=ax_ecdf.xaxis.label.get_size(),
        pad=8,
    )

    base_iso_x = ax_iso.xaxis.label.get_size()
    ax_iso.set_xlabel("log2(clone size)", fontsize=base_iso_x * 1.5)
    ax_iso.set_ylabel("Progenitor\ncell type proportion", fontsize=1.5 * ax_iso.yaxis.label.get_size())
    ax_iso.set_xlim(grid.min(), grid.max())
    ax_iso.set_ylim(0, 1)
    ax_iso.tick_params(labelsize=15)
    ax_iso.legend(frameon=False, loc="upper right")

    fig.subplots_adjust(wspace=0.25, hspace=0.3)

    figure_png = os.path.join(OUTPUT_DIR, f"{FIGURE4G_NAME}.png")
    figure_svg = os.path.join(OUTPUT_DIR, f"{FIGURE4G_NAME}.svg")
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 4f (projection), Figure 4g (diagnostics), and Figure 4h (projection views).")
    parser.add_argument(
        "--mode",
        choices=["projection", "diagnostics", "all"],
        default="all",
        help="Select which outputs to generate.",
    )
    return parser.parse_args(args)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = load_datasets(N_LOWEST_POINTS)
    nt_dataset = next(ds for ds in datasets if ds.label == "NT")

    highlights = select_highlight_points(nt_dataset)
    save_highlight_metadata(highlights)
    write_parameter_summary(highlights)

    axis_limits = determine_axis_limits([{"points": ds.points} for ds in datasets])
    line_params, line_coords = _compute_line_coordinates(datasets)

    if args.mode in {"projection", "all"}:
        plot_projection(datasets, highlights, axis_limits, line_coords)
        plot_projection_views(datasets, highlights, axis_limits, line_coords)
        create_projection_legend(datasets, highlights)
        create_line_equation_image(datasets, line_params)

    if args.mode in {"diagnostics", "all"}:
        df_real = load_real_nt_data(REAL_DATA_PATH)
        grid, real_ecdf, real_iso = compute_observed_curves(df_real)
        plot_diagnostics(highlights, df_real, grid, real_ecdf, real_iso)


if __name__ == "__main__":
    main()

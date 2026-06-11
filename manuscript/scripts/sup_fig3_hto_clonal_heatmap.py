#!/usr/bin/env python3
"""
Generate Figure S4c_1-c_2.

The exact legacy notebook cell that produced the original HTO-versus-clonal
heatmap could not be recovered from saved source cells. This script recreates
the bottom-right 2x2 block of the recovered legacy contingency table as two
panels: raw cell counts and column-normalized proportions.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_4"

COUNTS_OUTPUT = OUTPUT_DIR / "c_counts.tsv"
PROPORTIONS_OUTPUT = OUTPUT_DIR / "c_proportions.tsv"
COUNT_PNG_OUTPUT = OUTPUT_DIR / "c_1.png"
COUNT_SVG_OUTPUT = OUTPUT_DIR / "c_1.svg"
PROPORTION_PNG_OUTPUT = OUTPUT_DIR / "c_2.png"
PROPORTION_SVG_OUTPUT = OUTPUT_DIR / "c_2.svg"


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.grid"] = False
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"


def build_legacy_counts() -> pd.DataFrame:
    full_counts = pd.DataFrame(
        {
            "Negative": [1158, 124, 24],
            "Singlet": [16026, 53601, 4395],
            "Doublet": [1507, 6318, 6866],
        },
        index=["no_guide", "singlet", "doublet"],
    )
    return full_counts.loc[["singlet", "doublet"], ["Singlet", "Doublet"]].copy()


def build_proportions(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.divide(counts.sum(axis=0), axis=1)


def format_tick_labels(counts: pd.DataFrame) -> tuple[list[str], list[str]]:
    col_name_map = {
        "Singlet": "Singlet",
        "Doublet": "Doublet",
    }
    row_name_map = {
        "singlet": "Singlet",
        "doublet": "Doublet",
    }
    x_tick_labels = [
        f"{col_name_map[column]}\nn={counts[column].sum():,}"
        for column in counts.columns
    ]
    y_tick_labels = [
        f"{row_name_map[row]}\nn={counts.loc[row].sum():,}"
        for row in counts.index
    ]
    return x_tick_labels, y_tick_labels


def format_p_value(p_value: float) -> str:
    if p_value == 0 or p_value < 1e-300:
        return "Fisher's exact P < 1e-300"
    if p_value < 1e-3:
        return f"Fisher's exact P = {p_value:.2e}"
    return f"Fisher's exact P = {p_value:.3f}"


def style_heatmap_axes(ax: plt.Axes, counts: pd.DataFrame, title: str) -> None:
    x_tick_labels, y_tick_labels = format_tick_labels(counts)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Mouse hashtag oligo\ndoublet calling")
    ax.set_ylabel("Clonal doublet calling")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    for spine in ax.spines.values():
        spine.set_visible(False)


def blank_cmap():
    return sns.color_palette(["#FFFFFF"], as_cmap=True)


def plot_count_heatmap(counts: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.05, 3.6))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap=blank_cmap(),
        cbar=False,
        linewidths=1.0,
        linecolor="#D0D7DE",
        square=True,
        annot_kws={"fontsize": 12, "color": "black"},
        vmin=0,
        vmax=1,
        ax=ax,
    )
    style_heatmap_axes(ax, counts, title="number of cells")
    fig.tight_layout()
    return fig


def plot_proportion_heatmap(counts: pd.DataFrame, proportions: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.05, 3.825))
    annotations = proportions.apply(lambda column: column.map(lambda value: f"{100 * value:.1f}%"))
    sns.heatmap(
        proportions,
        annot=annotations,
        fmt="",
        cmap=blank_cmap(),
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=1.0,
        linecolor="#D0D7DE",
        square=True,
        annot_kws={"fontsize": 12, "color": "black"},
        ax=ax,
    )
    _, p_value = fisher_exact(counts.to_numpy(), alternative="greater")
    style_heatmap_axes(
        ax,
        counts,
        title=f"Column-normalized proportion of cells\n{format_p_value(p_value)}",
    )
    fig.tight_layout()
    return fig


def report_file_sizes(path_png: Path, path_svg: Path) -> None:
    png_size_mb = os.path.getsize(path_png) / 1024 / 1024
    svg_size_kb = os.path.getsize(path_svg) / 1024
    if png_size_mb > 5:
        print(f"ERROR: PNG {png_size_mb:.1f}MB exceeds 5MB limit.")
    if svg_size_kb > 500:
        print(f"ERROR: SVG {svg_size_kb:.1f}KB exceeds 500KB limit.")
    else:
        print(f"File sizes OK for {path_png.name}: PNG {png_size_mb:.2f}MB, SVG {svg_size_kb:.1f}KB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = build_legacy_counts()
    proportions = build_proportions(counts)
    counts.to_csv(COUNTS_OUTPUT, sep="\t", index=True)
    proportions.to_csv(PROPORTIONS_OUTPUT, sep="\t", index=True)

    count_fig = plot_count_heatmap(counts)
    count_fig.savefig(COUNT_PNG_OUTPUT, dpi=300, bbox_inches="tight")
    count_fig.savefig(COUNT_SVG_OUTPUT, bbox_inches="tight")
    plt.close(count_fig)

    proportion_fig = plot_proportion_heatmap(counts, proportions)
    proportion_fig.savefig(PROPORTION_PNG_OUTPUT, dpi=300, bbox_inches="tight")
    proportion_fig.savefig(PROPORTION_SVG_OUTPUT, bbox_inches="tight")
    plt.close(proportion_fig)

    print(f"Saved Figure S4c_1 PNG: {COUNT_PNG_OUTPUT}")
    print(f"Saved Figure S4c_1 SVG: {COUNT_SVG_OUTPUT}")
    print(f"Saved Figure S4c_2 PNG: {PROPORTION_PNG_OUTPUT}")
    print(f"Saved Figure S4c_2 SVG: {PROPORTION_SVG_OUTPUT}")
    print(f"Saved Figure S4 counts: {COUNTS_OUTPUT}")
    print(f"Saved Figure S4 proportions: {PROPORTIONS_OUTPUT}")

    report_file_sizes(COUNT_PNG_OUTPUT, COUNT_SVG_OUTPUT)
    report_file_sizes(PROPORTION_PNG_OUTPUT, PROPORTION_SVG_OUTPUT)


if __name__ == "__main__":
    main()

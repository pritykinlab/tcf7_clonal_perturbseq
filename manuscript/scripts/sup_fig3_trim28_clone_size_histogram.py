#!/usr/bin/env python
"""
Generate Figure S6e as a pooled Trim28-versus-NT clone-size histogram.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import get_adata, perturbation_colors  # noqa: E402


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams["axes.grid"] = False
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["legend.edgecolor"] = "dimgray"
plt.rcParams["figure.figsize"] = (7.2, 5.4)


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_6"
OUTPUT_STEM = "e"
TESTS_PATH = OUTPUT_DIR / "d_clone_size_tests.tsv"
PERTURBATION_ORDER = ["NT", "Trim28"]
LOG10_BIN_WIDTH = 0.5


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def resolve_mode(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def get_clone_df() -> pd.DataFrame:
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
    )

    obs = adata.obs.copy()
    for column in ["clone", "perturbation", "guide", "condition"]:
        obs[column] = obs[column].astype(str)

    grouped = obs.groupby("clone", observed=False)

    clone_df = pd.DataFrame(index=grouped.size().index)
    clone_df["num_cells"] = grouped.size()
    clone_df["perturbation"] = grouped["perturbation"].agg(resolve_mode)
    clone_df["guide"] = grouped["guide"].agg(resolve_mode)
    clone_df["condition"] = grouped["condition"].agg(resolve_mode)
    clone_df = clone_df.reset_index()
    clone_df["log10_clone_size"] = np.log10(clone_df["num_cells"])
    return clone_df


def load_trim28_stats() -> pd.Series:
    if not TESTS_PATH.exists():
        raise FileNotFoundError(
            f"Expected clone-size test table at {TESTS_PATH}. Run Figure S6d first."
        )

    tests_df = pd.read_csv(TESTS_PATH, sep="\t")
    match = tests_df.loc[
        (tests_df["Condition"] == "All") & (tests_df["Perturbation"] == "Trim28")
    ]
    if match.empty:
        raise ValueError("Could not find the pooled Trim28 row in the Figure S6d test table.")
    return match.iloc[0]


def plot_trim28_histogram(clone_df: pd.DataFrame, stats_row: pd.Series) -> plt.Figure:
    plot_df = clone_df.loc[clone_df["perturbation"].isin(PERTURBATION_ORDER)].copy()
    plot_df["perturbation"] = pd.Categorical(
        plot_df["perturbation"],
        categories=PERTURBATION_ORDER,
        ordered=True,
    )
    plot_df = plot_df.sort_values("perturbation")

    max_log10 = plot_df["log10_clone_size"].max()
    upper_edge = LOG10_BIN_WIDTH * np.ceil(max_log10 / LOG10_BIN_WIDTH)
    bins = np.arange(0, upper_edge + LOG10_BIN_WIDTH, LOG10_BIN_WIDTH)

    fig, ax = plt.subplots()

    for perturbation in PERTURBATION_ORDER:
        values = plot_df.loc[plot_df["perturbation"] == perturbation, "log10_clone_size"]
        ax.hist(
            values,
            bins=bins,
            color=perturbation_colors[perturbation],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.7,
            density=True,
            label=f"{perturbation} (n={len(values)})",
        )

    p_adj = float(stats_row["Corrected P-Value"])
    ax.set_title(f"Wilcoxon rank-sum test Trim28 vs NT\n(padj={p_adj:.2g})")
    ax.set_xlabel("log10(clone size)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, bins[-1])
    ax.legend(frameon=True, fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    validate_file_sizes(png_path, svg_path)


def main() -> None:
    print("Loading clone-level gene_10 data...")
    clone_df = get_clone_df()
    print(
        clone_df.loc[clone_df["perturbation"].isin(PERTURBATION_ORDER)]
        .groupby("perturbation", observed=False)["num_cells"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )

    print("Loading pooled Trim28 statistics from Figure S6d...")
    stats_row = load_trim28_stats()
    print(stats_row.to_string())

    print("Creating pooled Trim28-versus-NT histogram...")
    fig = plot_trim28_histogram(clone_df, stats_row)
    save_figure(fig)
    plt.close(fig)
    print("Figure S6e completed successfully.")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()

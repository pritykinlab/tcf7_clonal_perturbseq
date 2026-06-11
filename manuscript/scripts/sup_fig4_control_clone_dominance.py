#!/usr/bin/env python
"""
Generate Figure S7a: largest-clone share across control guides.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


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
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.edgecolor"] = "dimgray"
plt.rcParams["figure.figsize"] = (7.2, 5.6)


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_7"
PNG_PATH = OUTPUT_DIR / "a.png"
SVG_PATH = OUTPUT_DIR / "a.svg"
SUMMARY_PATH = OUTPUT_DIR / "a_summary.tsv"

CONTROL_GUIDE_ORDER = [
    "NO_SITE-1",
    "NO_SITE-2",
    "NO_SITE-3",
    "NO_SITE-4",
    "NO_SITE-5",
    "INTERGENIC-1",
    "INTERGENIC-2",
    "INTERGENIC-3",
    "INTERGENIC-4",
    "INTERGENIC-5",
]
def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def load_control_clone_summary() -> pd.DataFrame:
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=False,
        get_raw_scaled=False,
    )

    obs = adata.obs[["guide", "clone"]].copy()
    obs["guide"] = obs["guide"].astype(str)
    obs["clone"] = obs["clone"].astype(str)
    obs = obs.loc[
        obs["guide"].str.startswith("NO_SITE")
        | obs["guide"].str.startswith("INTERGENIC")
    ].copy()

    clone_sizes = (
        obs.groupby(["guide", "clone"], observed=False)
        .size()
        .rename("clone_size")
        .reset_index()
    )

    summary = (
        clone_sizes.groupby("guide", observed=False)
        .agg(
            clones_per_guide=("clone", "size"),
            total_cells=("clone_size", "sum"),
            largest_clone_cells=("clone_size", "max"),
        )
        .reset_index()
    )
    summary["largest_clone_fraction"] = (
        summary["largest_clone_cells"] / summary["total_cells"]
    )
    summary["guide_class"] = np.where(
        summary["guide"].str.startswith("NO_SITE"),
        "NO-SITE",
        "Intergenic",
    )
    summary["guide"] = pd.Categorical(
        summary["guide"],
        categories=CONTROL_GUIDE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("guide").reset_index(drop=True)
    summary["guide"] = summary["guide"].astype(str)
    return summary


def create_plot(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    x_positions = np.arange(len(summary))

    ax.bar(
        x_positions,
        summary["largest_clone_fraction"] * 100,
        color=perturbation_colors["NT"],
        edgecolor="black",
        linewidth=0.8,
        width=0.82,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Largest clone\nshare (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [guide.replace("_", "-") for guide in summary["guide"]],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, 85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_control_clone_summary()
    summary.to_csv(SUMMARY_PATH, sep="\t", index=False)

    fig = create_plot(summary)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(PNG_PATH, SVG_PATH)

    mean_clones = summary["clones_per_guide"].mean()
    mean_fraction = summary["largest_clone_fraction"].mean() * 100
    print(f"Mean largest clone share: {mean_fraction:.1f}%")
    print(f"Mean recovered clones per guide: {mean_clones:.1f}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

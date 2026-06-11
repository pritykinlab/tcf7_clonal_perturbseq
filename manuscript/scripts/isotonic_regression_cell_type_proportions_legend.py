#!/usr/bin/env python
"""
Generate standalone legend for Figure 3d isotonic regression cell type proportions.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import cell_type_colors as base_cell_type_colors, cell_type_orders_gene10

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 18

COMBINED_LABEL = "Progenitor + Progenitor Cycling"
LEGEND_CELL_TYPES = [
    COMBINED_LABEL,
    "Progenitor",
    "Progenitor Cycling",
    "Cycling",
    "Ccl5 High",
    "Granzyme High",
    "TRM-like",
]

CELL_TYPE_COLORS = base_cell_type_colors.copy()
CELL_TYPE_COLORS.setdefault(COMBINED_LABEL, "black")


def create_legend():
    fig, ax = plt.subplots(figsize=(3.2, 5.5))
    ax.axis("off")

    handles = [
        mpatches.Patch(color=CELL_TYPE_COLORS[cell_type], label=cell_type)
        for cell_type in LEGEND_CELL_TYPES
        if cell_type in CELL_TYPE_COLORS
    ]

    ax.legend(
        handles=handles,
        loc="center",
        title="Cell type",
        frameon=True,
        fontsize=18,
        title_fontsize=20,
        edgecolor="black",
        fancybox=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.6,
        labelspacing=0.8,
    )

    output_dir = "../manuscript/plots/Figure3"
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, "d_legend.png")
    svg_path = os.path.join(output_dir, "d_legend.svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ Legend file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)
    print("Figure 3d legend generated successfully!")


def main():
    create_legend()


if __name__ == "__main__":
    main()

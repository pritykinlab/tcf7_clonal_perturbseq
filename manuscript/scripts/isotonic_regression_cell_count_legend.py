#!/usr/bin/env python
"""
Generate standalone legend for Figure 3e isotonic regression cell count plot.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import cell_type_colors as base_cell_type_colors

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Match Figure 2 legend styling
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20

COMBINED_LABEL = "Progenitor + Progenitor Cycling"
LEGEND_CELL_TYPES = [COMBINED_LABEL, "Progenitor", "Progenitor Cycling"]
CELL_TYPE_COLORS = base_cell_type_colors.copy()
CELL_TYPE_COLORS.setdefault(COMBINED_LABEL, "black")


def create_legend():
    """Create and save standalone legend for Figure 3e, matching Figure 2 formatting."""

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=CELL_TYPE_COLORS[cell_type], label=cell_type)
        for cell_type in LEGEND_CELL_TYPES
        if cell_type in CELL_TYPE_COLORS
    ]

    legend = ax.legend(
        handles=legend_patches,
        loc="center",
        title="Cell type",
        frameon=True,
        fontsize=20,
        title_fontsize=22,
        edgecolor="black",
        fancybox=False,
    )

    output_dir = "../manuscript/plots/Figure3"
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, "e_legend.png")
    svg_path = os.path.join(output_dir, "e_legend.svg")

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
    print("Figure 3e legend generated successfully!")


def main():
    create_legend()


if __name__ == "__main__":
    main()

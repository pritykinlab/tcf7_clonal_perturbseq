#!/usr/bin/env python
"""
Generate standalone legend for Figure S6b.

The legend shows Isotype and anti-PD-1 line styles for each plotted cell type and
provides a standalone key for the Figure S6b panel.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sup_fig3_nt_condition_isotonic import (
    CELL_TYPE_COLORS,
    COMBINED_LABEL,
    CONDITION_DISPLAY,
    CONDITION_LINESTYLES,
    CONDITION_ORDER,
    OUTPUT_DIR,
    get_adata,
    plotted_cell_types,
    prepare_clone_proportions_df,
    _add_combined_cell_type,
    cell_type_orders_gene10,
)

# Required manuscript font settings
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

plt.rcParams["legend.fontsize"] = 16
plt.rcParams["legend.title_fontsize"] = 18


def create_legend():
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )
    clone_df = prepare_clone_proportions_df(adata, cell_type_orders_gene10)
    clone_df = _add_combined_cell_type(clone_df)

    cell_types = plotted_cell_types(clone_df)
    handles = []
    for cell_type in cell_types:
        for condition in CONDITION_ORDER:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=CELL_TYPE_COLORS[cell_type],
                    linestyle=CONDITION_LINESTYLES[condition],
                    linewidth=3 if cell_type == COMBINED_LABEL else 2,
                    label=f"{cell_type} ({CONDITION_DISPLAY[condition]})",
                )
            )

    fig_height = max(4.8, 0.33 * len(handles) + 1.2)
    fig, ax = plt.subplots(figsize=(5.8, fig_height))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        ncol=1,
        handlelength=2.6,
        handletextpad=0.8,
        borderpad=0.8,
        labelspacing=0.7,
    )

    png_path = os.path.join(OUTPUT_DIR, "b_legend.png")
    svg_path = os.path.join(OUTPUT_DIR, "b_legend.svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Legend file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    create_legend()


if __name__ == "__main__":
    main()

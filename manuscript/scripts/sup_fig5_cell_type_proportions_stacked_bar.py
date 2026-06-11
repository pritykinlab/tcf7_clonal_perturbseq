#!/usr/bin/env python
"""
legacy internal Sup_Figure_5_6: cell type proportions stacked bar chart for the trim28_setdb1 dataset.
Layout mirrors Figure 2f with hierarchical clustering of guides and stacked bars of cell-type composition.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import (  # noqa: E402
    cell_type_colors,
    cell_type_orders_trim28_setdb1,
    get_adata,
)

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['figure.figsize'] = (20, 8)

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_5_6_old"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trim28_setdb1 data...")
    adata = get_adata(
        'trim28_setdb1',
        use_cells_only_one_construct=True,
        remove_NG=True,
        remove_lymph_node=False,
    )

    print("Computing cell type proportions by guide...")
    df = pd.DataFrame({
        'cell_type': adata.obs['cell_type'],
        'guide': adata.obs['guide']
    })

    counts = df.groupby(['guide', 'cell_type']).size().unstack(fill_value=0)
    proportions = counts.divide(counts.sum(axis=1), axis=0)

    guide_cell_counts = df.groupby('guide').size()

    # Hierarchical clustering on proportions
    linkage_matrix = linkage(pdist(proportions), method='ward')
    dendrogram_order = leaves_list(linkage_matrix)
    ordered_proportions = proportions.iloc[dendrogram_order]

    # Use reverse cell type order for stacking
    cell_types = [ct for ct in cell_type_orders_trim28_setdb1 if ct in ordered_proportions.columns]
    ordered_proportions = ordered_proportions[cell_types[::-1]]

    print("Creating stacked bar chart with dendrogram...")
    fig = plt.figure(figsize=(20, 9))

    ax_dend = fig.add_subplot(2, 1, 1)
    ax_dend.set_position([0.08, 0.86, 0.84, 0.12])
    dendrogram(linkage_matrix,
               labels=proportions.index.tolist(),
               orientation='top',
               color_threshold=0,
               ax=ax_dend,
               no_labels=True)
    ax_dend.set_axis_off()

    ax_main = fig.add_subplot(2, 1, 2)
    ax_main.set_position([0.08, 0.12, 0.84, 0.70])

    ordered_proportions.plot(
        kind='bar',
        stacked=True,
        ax=ax_main,
        legend=False,
        color=[cell_type_colors[ct] for ct in ordered_proportions.columns],
        width=0.8
    )

    ax_main.set_ylabel('Cell type proportion')
    ax_main.set_xlabel('')
    ax_main.set_ylim(0, 1)

    # x-axis labels with cell counts
    new_labels = []
    for guide in ordered_proportions.index:
        cell_count = guide_cell_counts.get(guide, 0)
        new_labels.append(f"{guide} (n={cell_count})")
    ax_main.set_xticks(range(len(new_labels)))
    plt.draw()
    labels = [item for item in ax_main.get_xticklabels()]
    for i, label in enumerate(labels):
        if i < len(new_labels):
            label.set_text(new_labels[i])
            label.set_rotation(90)
    ax_main.set_xticklabels(labels, fontsize=13)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "6.png")
    svg_path = os.path.join(OUTPUT_DIR, "6.svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Consider simplifying or rasterizing.")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    print("legacy internal Sup_Figure_5_6 cell type proportions stacked bar completed successfully!")
    print("\nSummary:")
    print(f"Guides: {len(ordered_proportions)}")
    print(f"Cell types: {list(ordered_proportions.columns[::-1])}")


if __name__ == "__main__":
    main()

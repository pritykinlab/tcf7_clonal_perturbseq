#!/usr/bin/env python
"""
Generate cell type proportions stacked bar chart (Figure 2f)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['figure.figsize'] = (20, 7)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=True,
                      remove_NG=True)

    print("Computing cell type proportions...")

    # Create DataFrame with cell types and guides
    df = pd.DataFrame({
        'cell_type': adata.obs['cell_type'],
        'guide': adata.obs['guide']
    })

    # Get counts and proportions
    counts = df.groupby(['guide', 'cell_type']).size().unstack(fill_value=0)
    proportions = counts.divide(counts.sum(axis=1), axis=0)

    # Get total cell counts per guide for labels
    guide_cell_counts = df.groupby('guide').size()

    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    linkage_matrix = linkage(pdist(proportions), method='ward')
    dendrogram_order = leaves_list(linkage_matrix)
    ordered_proportions = proportions.iloc[dendrogram_order]

    # Reverse cell type order for stacking
    cell_type_orders_gene10_rev = cell_type_orders_gene10[::-1]
    ordered_proportions = ordered_proportions[cell_type_orders_gene10_rev]

    print("Creating stacked bar chart...")

    # Create figure with subplots for dendrogram and bar chart
    fig = plt.figure(figsize=(20, 8))

    # Create dendrogram subplot (small at top)
    ax_dend = fig.add_subplot(2, 1, 1)
    ax_dend.set_position([0.1, 0.85, 0.8, 0.1])

    # Create dendrogram
    dendrogram(linkage_matrix,
               labels=proportions.index.tolist(),
               orientation='top',
               color_threshold=0,
               ax=ax_dend,
               no_labels=True)
    ax_dend.set_axis_off()

    # Create main bar chart subplot
    ax_main = fig.add_subplot(2, 1, 2)
    ax_main.set_position([0.1, 0.1, 0.8, 0.7])

    # Create stacked bar chart
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

    # Create new x-axis labels with cell counts
    new_labels = []
    for guide in ordered_proportions.index:
        cell_count = guide_cell_counts[guide]
        new_labels.append(f"{guide} (n={cell_count})")

    # Set the tick positions explicitly
    ax_main.set_xticks(range(len(new_labels)))

    # Draw to ensure plot is rendered
    plt.draw()

    # Get the current labels and modify them individually
    labels = [item for item in ax_main.get_xticklabels()]
    for i, label in enumerate(labels):
        if i < len(new_labels):
            label.set_text(new_labels[i])
            label.set_rotation(90)
            label.set_ha('center')  # Center horizontally on tick

    # Re-apply the modified labels
    ax_main.set_xticklabels(labels)

    # Legend removed - now created as separate figure

    # Save the figure
    print("Saving figures...")
    fig.savefig(os.path.join(output_dir, "f.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "f.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Figure 2f completed successfully!")

    print(f"\nAnalyzed {len(ordered_proportions)} guides across {len(ordered_proportions.columns)} cell types")

if __name__ == "__main__":
    main()

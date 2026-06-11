#!/usr/bin/env python
"""
Generate differential expression genes heatmap (Figure 2g)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from matplotlib.colors import LinearSegmentedColormap

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata
import DE_testing

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['figure.figsize'] = (12, 30)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def custom_cmap():
    """Create custom colormap with extended smooth middle"""
    original_cmap = plt.cm.RdBu_r
    colors = original_cmap(np.linspace(0.1, 0.9, 256))

    mid_start = 0.4 * 256
    mid_end = 0.6 * 256
    mid_colors = colors[int(mid_start):int(mid_end)]

    enhanced_mid = np.repeat(mid_colors, 4, axis=0)

    new_colors = np.concatenate(
        (colors[:int(mid_start)], enhanced_mid, colors[int(mid_end):]),
        axis=0
    )

    return LinearSegmentedColormap.from_list('extended_smooth_mid', new_colors)

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=True,
                      remove_NG=True,
                      get_raw_scaled=False)

    # Get perturbations (exclude NT)
    perturbations = adata.obs['perturbation'].unique().tolist()
    perturbations = [p for p in perturbations if p != "NT"]
    perturbations.sort()

    print(f"Performing DE analysis for {len(perturbations)} perturbations...")

    # Initialize dictionary to store DE results
    de_results = {}

    # Loop over each perturbation for DE testing
    for perturbation in perturbations:
        print(f"  Processing {perturbation}...")

        # Define cells for perturbation vs control
        perturb_cells = adata.obs['perturbation'] == perturbation
        control_cells = adata.obs['perturbation'] == 'NT'

        perturb_cells = adata.obs_names[perturb_cells].tolist()
        control_cells = adata.obs_names[control_cells].tolist()

        # Perform DE test
        results_df = DE_testing.wilcoxon_DE_test(
            adata,
            perturb_cells,
            control_cells,
            group1_threshold_pct=0.05,  # At least 5% of cells in group1 must express
            group2_threshold_pct=0.0,   # No minimum expression threshold for controls
            norm_layer='norm_log'
        )

        de_results[perturbation] = results_df

    print("Processing DE results for heatmap...")

    # Combine all DE results
    all_de_results = pd.concat([df.assign(perturbation=pert) for pert, df in de_results.items()])

    # Function to get top genes for each perturbation
    def get_top_genes(df):
        # Filter genes with logFC >= 0.5
        df_filtered = df[df['Log2FC'] >= 0.5]
        df_sorted = df_filtered.sort_values(by='Log2FC', ascending=False)
        top_genes = df_sorted['Gene'].head(10).tolist()
        return top_genes

    # Get top genes for each perturbation
    grouped = all_de_results.groupby('perturbation')
    top_genes = grouped.apply(get_top_genes).reset_index(level=0)

    # Flatten and get unique genes
    selected_genes = list(set(top_genes[0].sum()))

    # Create pivot table for heatmap
    subset_de_results = all_de_results[all_de_results['Gene'].isin(selected_genes)]
    heatmap_data = subset_de_results.pivot_table(
        index='Gene',
        columns='perturbation',
        values='Log2FC',
        fill_value=0
    )

    print("Creating clustered heatmap...")

    # Create clustered heatmap
    g = sns.clustermap(
        heatmap_data.fillna(0),
        cmap=custom_cmap(),
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(12, 30),
        linewidths=0.5,
        method='average',
        metric='correlation',
        dendrogram_ratio=(0.2, 0.01),
        cbar_pos=(1.05, 0.5, 0.02, 0.3)
    )

    # Get sorted labels from clustering
    sorted_labels = heatmap_data.index[g.dendrogram_row.reordered_ind]

    # Set y-tick labels
    midpoints = np.arange(0.5, len(heatmap_data.index))
    g.ax_heatmap.set_yticks(midpoints)
    g.ax_heatmap.set_yticklabels(sorted_labels, rotation=0)

    # Format labels
    plt.setp(g.ax_heatmap.get_yticklabels(), va="center", fontsize=20)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=20)

    g.ax_heatmap.set_xlabel("perturbations")
    g.ax_heatmap.set_ylabel("top DE genes")

    # Save the figure
    print("Saving figures...")
    g.savefig(os.path.join(output_dir, "g.png"),
              dpi=300,
              bbox_inches='tight')
    g.savefig(os.path.join(output_dir, "g.svg"),
              format='svg',
              bbox_inches='tight')

    plt.close()
    print("Figure 2g completed successfully!")

    print(f"\nAnalyzed {len(selected_genes)} genes across {len(perturbations)} perturbations")

if __name__ == "__main__":
    main()

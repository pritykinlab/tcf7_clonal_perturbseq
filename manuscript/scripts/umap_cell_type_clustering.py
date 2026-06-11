#!/usr/bin/env python
"""
Generate UMAP clustering visualization colored by cell type (Figure 2b)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.edgecolor'] = 'dimgray'
plt.rcParams['figure.figsize'] = (7, 7)
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
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=True)

    # Set colors for cell types
    ordered_colors = [cell_type_colors[cat] for cat in adata.obs['cell_type'].cat.categories]
    adata.uns['cell_type_colors'] = ordered_colors

    # Generate UMAP plot
    print("Generating UMAP plot...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Use scanpy to generate the UMAP with black border
    sc.pl.umap(adata,
               color='cell_type',
               title='',
               size=3,
               frameon=True,
               ax=ax,
               show=False,
               legend_loc=None)

    # Add axis labels
    ax.set_xlabel('UMAP1', fontsize=25)
    ax.set_ylabel('UMAP2', fontsize=25)

    # Save the figure
    print("Saving figures...")
    fig.savefig(os.path.join(output_dir, "b.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "b.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Figure 2b completed successfully!")

    # Print cell type counts for reference
    print("\nCell type distribution:")
    print(adata.obs['cell_type'].value_counts())

if __name__ == "__main__":
    main()
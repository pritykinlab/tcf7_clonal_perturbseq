#!/usr/bin/env python
"""
Generate ATAC UMAP with cell type clustering and Trim28 overlay (Figure 6a)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_colors, cell_type_orders_multiome, perturbation_colors

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def draw_overlap_UMAP_perturbation(adata):
    """Draw UMAP with Trim28 overlay"""
    if 'X_umap' not in adata.obsm.keys():
        raise ValueError("UMAP data not found in adata. Make sure UMAP embeddings are computed.")

    # Create masks first to avoid AnnData slicing issues
    trim28_mask = adata.obs['condition'] == 'Trim28'
    other_mask = ~adata.obs['condition'].isin(['Trim28'])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']

    # Plot 'other' group first in light grey
    ax.scatter(umap_coords[other_mask, 0], umap_coords[other_mask, 1],
               color='lightgray', s=10, alpha=0.7)

    # Plot 'Trim28' group in red with reduced opacity
    ax.scatter(umap_coords[trim28_mask, 0], umap_coords[trim28_mask, 1],
               color=perturbation_colors['Trim28'], s=10, alpha=0.5)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Trim28')
    plt.tight_layout()

    return fig

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Load multiome RNA data
    print("Loading multiome RNA data...")
    adata = get_adata('multiome_rna')

    # Set up cell type colors
    ordered_colors = [cell_type_colors[cat] for cat in adata.obs['cell_type'].cat.categories]
    adata.uns['cell_type_colors'] = ordered_colors

    print("Creating cell type UMAP...")
    # Create cell type UMAP
    fig_celltype = sc.pl.umap(adata, color=['cell_type'], show=False, return_fig=True, frameon=True, legend_loc=None)
    fig_celltype.savefig(os.path.join(output_dir, "a_celltype.png"),
                        dpi=300,
                        bbox_inches='tight')
    fig_celltype.savefig(os.path.join(output_dir, "a_celltype.svg"),
                        format='svg',
                        bbox_inches='tight')
    plt.close(fig_celltype)

    print("Creating Trim28 overlay UMAP...")
    # Create Trim28 overlay
    fig_trim28 = draw_overlap_UMAP_perturbation(adata)
    fig_trim28.savefig(os.path.join(output_dir, "a_trim28.png"),
                      dpi=300,
                      bbox_inches='tight')
    fig_trim28.savefig(os.path.join(output_dir, "a_trim28.svg"),
                      format='svg',
                      bbox_inches='tight')
    plt.close(fig_trim28)

    print("Figure 6a completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Trim28 cells: {(adata.obs['condition'] == 'Trim28').sum():,}")
    print(f"Control cells: {(adata.obs['condition'] == 'Control').sum():,}")
    print(f"Cell types: {', '.join(adata.obs['cell_type'].cat.categories)}")

if __name__ == "__main__":
    main()
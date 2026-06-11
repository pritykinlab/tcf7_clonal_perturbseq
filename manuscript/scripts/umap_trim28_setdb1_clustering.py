#!/usr/bin/env python
"""
Generate UMAP clustering visualization colored by cell type for trim28_setdb1 dataset (Figure 5b)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_colors, cell_type_orders_trim28_setdb1

# Set figure parameters - exact same as Figure 2b
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

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load the trim28_setdb1 data (as shown in the notebook)
    print("Loading trim28_setdb1 data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False)

    # Set colors for cell types
    ordered_colors = [cell_type_colors[cat] for cat in adata.obs['cell_type'].cat.categories]
    adata.uns['cell_type_colors'] = ordered_colors

    # Generate UMAP plot with legend
    print("Generating UMAP plot...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Create scatter plot manually with rasterization for smaller SVG
    for cell_type in adata.obs['cell_type'].cat.categories:
        mask = adata.obs['cell_type'] == cell_type
        ax.scatter(adata.obsm['X_umap'][mask, 0],
                  adata.obsm['X_umap'][mask, 1],
                  c=[cell_type_colors[cell_type]],
                  label=cell_type,
                  s=3,
                  rasterized=True)  # Rasterize for smaller SVG

    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, edgecolor='dimgray',
             fontsize=12, title='Cell type')

    # Add axis labels
    ax.set_xlabel('UMAP1', fontsize=25)
    ax.set_ylabel('UMAP2', fontsize=25)

    # Save the figure
    print("Saving figures...")
    png_path = os.path.join(output_dir, "b.png")
    svg_path = os.path.join(output_dir, "b.svg")

    fig.savefig(png_path,
                dpi=300,
                bbox_inches='tight')
    fig.savefig(svg_path,
                format='svg',
                bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()
    print("Figure 5b completed successfully!")

    # Print cell type counts for reference
    print("\nCell type distribution:")
    print(adata.obs['cell_type'].value_counts())

if __name__ == "__main__":
    main()
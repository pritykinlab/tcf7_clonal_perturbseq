#!/usr/bin/env python
"""
Generate UMAP plots showing cell types and Trim28/Setdb1 perturbations (Figure 5c)
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

from figure_format import get_adata, cell_type_colors, cell_type_orders_trim28_setdb1, perturbation_colors

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
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False)

    # Set up colors for cell types
    ordered_colors = [cell_type_colors[cat] for cat in adata.obs['cell_type'].cat.categories]
    adata.uns['cell_type_colors'] = ordered_colors

    # =================================================================
    # LEFT PANEL: Cell type UMAP (exactly like Figure 2b)
    # =================================================================
    print("Creating cell type UMAP (left panel)...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Use scanpy to generate the UMAP with black border
    # Note: Using size=5 for better visibility (trim28_setdb1 has fewer cells than gene_10)
    sc.pl.umap(adata,
               color='cell_type',
               title='',
               size=5,
               frameon=True,
               ax=ax,
               show=False,
               legend_loc=None)

    # Add axis labels
    ax.set_xlabel('UMAP1', fontsize=25)
    ax.set_ylabel('UMAP2', fontsize=25)

    # Save left panel
    png_path = os.path.join(output_dir, "c_left.png")
    svg_path = os.path.join(output_dir, "c_left.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: Left panel PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: Left panel SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Left panel file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    # =================================================================
    # RIGHT PANEL: Perturbation UMAP (Trim28 and Setdb1) WITHOUT legend
    # =================================================================
    print("Creating perturbation UMAP (right panel - UMAP only)...")

    # Create perturbation colors
    perturbation_colors_subset = {
        'Trim28': perturbation_colors.get('Trim28', (1.0, 0.4, 0.4)),
        'Setdb1': perturbation_colors.get('Setdb1', (0.4, 0.4, 1.0)),
        'Other': (0.8, 0.8, 0.8)  # Gray for non-highlighted cells
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    # First plot all cells in gray (background)
    ax.scatter(adata.obsm['X_umap'][:, 0],
              adata.obsm['X_umap'][:, 1],
              c=[perturbation_colors_subset['Other']],
              s=5,
              alpha=0.3,
              rasterized=True)

    # Highlight Trim28 cells
    trim28_mask = adata.obs['perturbation'].str.contains('Trim28', na=False)
    if trim28_mask.sum() > 0:
        ax.scatter(adata.obsm['X_umap'][trim28_mask, 0],
                  adata.obsm['X_umap'][trim28_mask, 1],
                  c=[perturbation_colors_subset['Trim28']],
                  s=5,
                  rasterized=True)

    # Highlight Setdb1 cells (only cells with Setdb1 but not Trim28)
    setdb1_mask = (adata.obs['perturbation'].str.contains('Setdb1', na=False) &
                   ~adata.obs['perturbation'].str.contains('Trim28', na=False))
    if setdb1_mask.sum() > 0:
        ax.scatter(adata.obsm['X_umap'][setdb1_mask, 0],
                  adata.obsm['X_umap'][setdb1_mask, 1],
                  c=[perturbation_colors_subset['Setdb1']],
                  s=5,
                  rasterized=True)

    # Remove axis labels and frame for right panel
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save right panel UMAP (without legend)
    png_path = os.path.join(output_dir, "c_right_umap.png")
    svg_path = os.path.join(output_dir, "c_right_umap.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: Right UMAP PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: Right UMAP SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Right UMAP file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    # =================================================================
    # RIGHT PANEL LEGEND: Separate legend for Trim28 and Setdb1
    # =================================================================
    print("Creating perturbation legend (right panel - legend only)...")

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.axis('off')

    # Create legend entries for Trim28 and Setdb1
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=perturbation_colors_subset['Trim28'], label='Trim28'),
        Patch(facecolor=perturbation_colors_subset['Setdb1'], label='Setdb1')
    ]

    # Add legend
    legend = ax.legend(handles=handles,
                       loc='center',
                       frameon=True,
                       edgecolor='dimgray',
                       fontsize=15,
                       title='Perturbation',
                       title_fontsize=15)

    # Save legend separately
    png_path = os.path.join(output_dir, "c_right_legend.png")
    svg_path = os.path.join(output_dir, "c_right_legend.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: Legend PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: Legend SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Legend file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    print("Figure 5c completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {adata.n_obs}")
    print(f"Cells with Trim28: {trim28_mask.sum()}")
    print(f"Cells with Setdb1 (not Trim28): {setdb1_mask.sum()}")
    print(f"Cell types: {list(adata.obs['cell_type'].cat.categories)}")

if __name__ == "__main__":
    main()

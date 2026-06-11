#!/usr/bin/env python
"""
Generate UMAP plots showing cell types and Trim28 perturbation for multiome RNA data (Figure 6a)
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

from figure_format import get_adata, cell_type_colors, perturbation_colors

# Set figure parameters - matching Figure 5b format
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 10  # Increased from 5 to 10 for larger dots
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
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Load multiome RNA data
    print("Loading multiome RNA data...")
    adata = get_adata('multiome_rna')

    # Set up colors for cell types using the standard cell_type_colors
    ordered_colors = [cell_type_colors[cat] for cat in adata.obs['cell_type'].cat.categories]
    adata.uns['cell_type_colors'] = ordered_colors

    # =================================================================
    # LEFT PANEL: Cell type UMAP (no legend, no title)
    # =================================================================
    print("Creating cell type UMAP (left panel)...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot each cell type separately with consistent size
    for cell_type in adata.obs['cell_type'].cat.categories:
        mask = adata.obs['cell_type'] == cell_type
        color = cell_type_colors[cell_type]
        ax.scatter(adata.obsm['X_umap'][mask, 0],
                  adata.obsm['X_umap'][mask, 1],
                  c=[color],
                  s=10,  # Larger dots as requested
                  rasterized=False)  # No rasterization for PNG

    # Add axis labels
    ax.set_xlabel('UMAP1', fontsize=25)
    ax.set_ylabel('UMAP2', fontsize=25)

    # Remove ticks and tick labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Save PNG version first
    png_path = os.path.join(output_dir, "a_left.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    # Now create SVG version with rasterization
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    # Plot each cell type separately with rasterization for SVG
    for cell_type in adata.obs['cell_type'].cat.categories:
        mask = adata.obs['cell_type'] == cell_type
        color = cell_type_colors[cell_type]
        ax2.scatter(adata.obsm['X_umap'][mask, 0],
                   adata.obsm['X_umap'][mask, 1],
                   c=[color],
                   s=10,  # Same size as PNG
                   rasterized=True)  # Rasterize for SVG to reduce size

    # Add axis labels
    ax2.set_xlabel('UMAP1', fontsize=25)
    ax2.set_ylabel('UMAP2', fontsize=25)

    # Remove ticks and tick labels for cleaner look
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Save SVG version
    svg_path = os.path.join(output_dir, "a_left.svg")
    fig2.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: Left panel PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: Left panel SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Left panel file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)
    plt.close(fig2)

    # =================================================================
    # RIGHT PANEL: Trim28 perturbation UMAP (highlighting Trim28 cells)
    # =================================================================
    print("Creating Trim28 perturbation UMAP (right panel - UMAP only)...")

    # Define Trim28 color (orange as in perturbation_colors)
    trim28_color = perturbation_colors.get('Trim28', (1.0, 0.4, 0.4))  # Should be orangish
    other_color = (0.8, 0.8, 0.8)  # Gray for non-Trim28 cells

    fig, ax = plt.subplots(figsize=(7, 7))

    # First plot all cells in gray (background)
    ax.scatter(adata.obsm['X_umap'][:, 0],
              adata.obsm['X_umap'][:, 1],
              c=[other_color],
              s=10,  # Larger dots to match left panel
              alpha=0.3,
              rasterized=True)

    # Highlight Trim28 cells - use 'condition' column for multiome data
    if 'condition' in adata.obs.columns:
        trim28_mask = adata.obs['condition'] == 'Trim28'
    elif 'perturbation' in adata.obs.columns:
        trim28_mask = adata.obs['perturbation'].str.contains('Trim28', na=False)
    else:
        print("Warning: Could not find perturbation/condition column")
        trim28_mask = pd.Series([False] * adata.n_obs, index=adata.obs.index)

    if trim28_mask.sum() > 0:
        ax.scatter(adata.obsm['X_umap'][trim28_mask, 0],
                  adata.obsm['X_umap'][trim28_mask, 1],
                  c=[trim28_color],
                  s=10,  # Larger dots to match left panel
                  rasterized=True)
        print(f"Found {trim28_mask.sum()} Trim28 cells")
    else:
        print("No Trim28 cells found in the data")

    # Remove axis labels and frame for right panel (matching Figure 5b style)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save right panel UMAP (without legend)
    png_path = os.path.join(output_dir, "a_right_umap.png")
    svg_path = os.path.join(output_dir, "a_right_umap.svg")

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
    # RIGHT PANEL LEGEND: Separate legend for Trim28 (matching Figure 5b style)
    # =================================================================
    print("Creating perturbation legend (right panel - legend only)...")

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.axis('off')

    # Create legend entry for Trim28 only (since we only have Trim28 in multiome)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=trim28_color, label='Trim28')
    ]

    # Add legend
    legend = ax.legend(handles=handles,
                       loc='center',
                       frameon=True,
                       edgecolor='dimgray',
                       fontsize=15)

    # Save legend separately
    png_path = os.path.join(output_dir, "a_right_legend.png")
    svg_path = os.path.join(output_dir, "a_right_legend.svg")

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

    # =================================================================
    # LEFT PANEL LEGEND: Separate legend for cell types (matching Figure 5 style)
    # =================================================================
    print("Creating cell type legend (left panel - legend only)...")

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.axis('off')

    # Create legend entries for all cell types
    from matplotlib.patches import Patch
    handles = []
    for cell_type in adata.obs['cell_type'].cat.categories:
        color = cell_type_colors[cell_type]
        patch = Patch(facecolor=color, label=cell_type)
        handles.append(patch)

    # Add legend with box
    legend = ax.legend(handles=handles,
                       loc='center',
                       title="Cell type",
                       frameon=True,
                       fontsize=20,
                       title_fontsize=22,
                       edgecolor='black',
                       fancybox=False)

    # Save legend separately
    png_path = os.path.join(output_dir, "a_left_legend.png")
    svg_path = os.path.join(output_dir, "a_left_legend.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: Cell type legend PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: Cell type legend SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Cell type legend file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    print("Figure 6a completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {adata.n_obs}")
    print(f"Cells with Trim28: {trim28_mask.sum() if 'trim28_mask' in locals() else 'N/A'}")
    print(f"Cell types: {list(adata.obs['cell_type'].cat.categories)}")
    print(f"Number of cells per type:")
    print(adata.obs['cell_type'].value_counts().sort_index())

if __name__ == "__main__":
    main()

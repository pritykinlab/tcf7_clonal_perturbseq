#!/usr/bin/env python
"""
Generate UMAP colored by clone size (Figure 3e)
Shows UMAP plot with cells colored by log2(clone size)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata

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

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Sup_Figure_5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=True,
                      remove_NG=True,
                      get_raw_scaled=True)

    # Calculate log2 clone size
    print("Processing clone size data...")
    adata.obs['log2_clone_size'] = np.log2(adata.obs['clone_size'] + 1)

    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']
    clone_size_log2 = adata.obs['log2_clone_size']

    print("Creating UMAP visualization...")
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Create scatter plot with clone size coloring (rasterized to reduce file size)
    sc = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=clone_size_log2,
        s=5,
        cmap='viridis',
        alpha=0.7,
        rasterized=True
    )

    # Add colorbar
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('log2(clone size)', fontsize=20)
    cb.ax.tick_params(labelsize=16)

    # Remove title, axis labels, ticks, and bounding box
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "b.png")
    svg_path = os.path.join(output_dir, "b.svg")

    fig.savefig(png_path,
                dpi=300,
                bbox_inches='tight')
    fig.savefig(svg_path,
                format='svg',
                bbox_inches='tight')

    # Validate file sizes (critical requirement)
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()
    print("Figure 3e completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {len(adata)}")
    print(f"Log2 clone size range: {clone_size_log2.min():.2f} - {clone_size_log2.max():.2f}")
    print(f"Clone size range: {adata.obs['clone_size'].min()} - {adata.obs['clone_size'].max()}")

if __name__ == "__main__":
    main()

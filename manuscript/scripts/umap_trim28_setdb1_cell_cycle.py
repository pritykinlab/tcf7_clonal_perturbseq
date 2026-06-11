#!/usr/bin/env python
"""
Generate UMAP plot showing cell cycle phases for trim28_setdb1 dataset (Figure 5e)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata

# Set figure parameters - exact same as Figure 2d
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
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data (exact same as Figure 2d but for trim28_setdb1 dataset)
    print("Loading data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False)

    print("Creating cell cycle phase UMAP...")
    # Create cell cycle phase UMAP - EXACTLY like Figure 2d
    fig, ax = plt.subplots(figsize=(7, 7))

    # Use sc.pl.umap directly with NO legend (exactly like Figure 2d)
    # Note: Using size=5 for consistency with Figure 5c
    sc.pl.umap(adata,
               color='phase',
               title='',
               size=5,
               frameon=True,
               ax=ax,
               show=False,
               legend_loc=None)

    # Add axis labels (exactly like Figure 2d)
    ax.set_xlabel('UMAP1', fontsize=25)
    ax.set_ylabel('UMAP2', fontsize=25)

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "e.png")
    svg_path = os.path.join(output_dir, "e.svg")

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
        print(f"WARNING: PNG {png_size:.1f}MB exceeds 5MB limit")
    if svg_size > 500:
        print(f"WARNING: SVG {svg_size:.1f}KB exceeds 500KB limit (expected with sc.pl.umap)")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()
    print("Figure 5e completed successfully!")

    # Print summary statistics (exactly like Figure 2d)
    if 'phase' in adata.obs.columns:
        phase_counts = adata.obs['phase'].value_counts()
        print(f"\nSummary:")
        print(f"Cell cycle phase distribution:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count}")
    else:
        print("\nNo cell cycle phase information available")

if __name__ == "__main__":
    main()

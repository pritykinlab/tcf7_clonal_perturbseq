#!/usr/bin/env python
"""
Generate UMAP plot showing cell cycle phases across CD8+ TIL populations (Figure 5d)
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
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False)

    print("Creating cell cycle phase UMAP...")
    # Create cell cycle phase UMAP
    fig, ax = plt.subplots(figsize=(7, 7))
    sc.pl.umap(adata, color='phase', title='', ax=ax, show=False, legend_loc=None)

    # Save outputs
    print("Saving figures...")
    fig.savefig(os.path.join(output_dir, "d.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "d.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Figure 2d completed successfully!")

    # Print summary statistics
    phase_counts = adata.obs['phase'].value_counts()
    print(f"\nSummary:")
    print(f"Cell cycle phase distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count}")

if __name__ == "__main__":
    main()
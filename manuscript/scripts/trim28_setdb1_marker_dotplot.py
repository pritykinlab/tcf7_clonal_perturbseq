#!/usr/bin/env python
"""
Generate cell type marker dotplot for trim28_setdb1 dataset (Figure 5d)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_colors, cell_type_orders_trim28_setdb1, dotplot_genes_LN
from dotplot_utils import create_proper_dotplot

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

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load the trim28_setdb1 data with control-based z-score normalization
    print("Loading trim28_setdb1 data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_norm_log_zscore_control=True)

    # First, create the full dotplot (all cells)
    print("Creating dotplot for all cells...")

    # Create dotplot with proper fractions and z-score colors
    dp = create_proper_dotplot(adata,
                              groupby="cell_type",
                              var_names=dotplot_genes_LN,
                              categories_order=cell_type_orders_trim28_setdb1,
                              figsize=(8, 2),
                              cmap="Reds",
                              vmin=-1,
                              vmax=1)

    # Add totals (cell count barplot) and show to create the figure
    dp.add_totals(color="maroon")

    # We need to call show() to actually render the plot, then get the figure
    dp.show()  # This renders the plot

    # Now get the current figure from matplotlib
    fig = plt.gcf()

    # Save outputs
    print("Saving figures for all cells...")
    png_path = os.path.join(output_dir, "d.png")
    svg_path = os.path.join(output_dir, "d.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    # Now create the control-only version
    print("\nCreating dotplot for control cells only...")
    adata_control = adata[adata.obs['perturbation'] == 'NT'].copy()
    print(f"Using {adata_control.n_obs} control cells")

    # Filter categories_order to only include cell types present in control data,
    # and omit the TRM-like row for the supplementary control-only panel.
    control_cell_types = adata_control.obs['cell_type'].unique()
    control_categories_order = [
        ct for ct in cell_type_orders_trim28_setdb1
        if ct != 'TRM-like' and ct in control_cell_types
    ]
    if not control_categories_order:
        raise ValueError("No non-TRM-like control cell types found for Figure 5d control panel.")
    print(f"Cell types in control data: {control_categories_order}")

    # Create dotplot for control cells only with proper fractions
    dp_control = create_proper_dotplot(adata_control,
                                      groupby="cell_type",
                                      var_names=dotplot_genes_LN,
                                      categories_order=control_categories_order,
                                      figsize=(8, 2),
                                      cmap="Reds",
                                      vmin=-1,
                                      vmax=1)

    # Add totals (cell count barplot) and show to create the figure
    dp_control.add_totals(color="maroon")

    # We need to call show() to actually render the plot, then get the figure
    dp_control.show()  # This renders the plot

    # Now get the current figure from matplotlib
    fig_control = plt.gcf()

    # Save control-only outputs
    print("Saving figures for control cells only...")
    png_path_control = os.path.join(output_dir, "d_control.png")
    svg_path_control = os.path.join(output_dir, "d_control.svg")

    fig_control.savefig(png_path_control, dpi=300, bbox_inches='tight')
    fig_control.savefig(svg_path_control, format='svg', bbox_inches='tight')

    # Validate file sizes for control plots
    png_size_control = os.path.getsize(png_path_control) / 1024 / 1024  # MB
    svg_size_control = os.path.getsize(svg_path_control) / 1024  # KB

    if png_size_control > 5:
        print(f"ERROR: Control PNG {png_size_control:.1f}MB exceeds 5MB limit!")
    if svg_size_control > 500:
        print(f"ERROR: Control SVG {svg_size_control:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Control file sizes OK: PNG {png_size_control:.2f}MB, SVG {svg_size_control:.1f}KB")

    plt.close()
    print("Figure 5d completed successfully (both versions)!")

    # Print summary
    print(f"\nSummary:")
    print(f"Total cells: {adata.n_obs}")
    print(f"Control cells: {adata_control.n_obs}")
    print(f"Genes plotted: {dotplot_genes_LN}")

if __name__ == "__main__":
    main()

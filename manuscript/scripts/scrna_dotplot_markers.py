#!/usr/bin/env python
"""
Generate scRNA dot plot with cell markers (Figure 6b)
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

from figure_format import get_adata, cell_type_colors, cell_type_orders_multiome, dotplot_genes
from dotplot_utils import create_proper_dotplot

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (8, 2)

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Load multiome RNA data
    print("Loading multiome RNA data...")
    # Note: multiome_rna doesn't need get_norm_log_zscore_control since it's already processed
    adata = get_adata('multiome_rna')

    # First, create the full dotplot (all cells)
    print("Creating dot plot for all cells...")
    # Create dot plot with proper fractions
    # Check if norm_log_zscore_control layer exists, otherwise use norm_log_zscore
    if 'norm_log_zscore_control' in adata.layers:
        dp = create_proper_dotplot(adata,
                                  groupby="cell_type",
                                  var_names=dotplot_genes,
                                  categories_order=cell_type_orders_multiome,
                                  figsize=(8, 2),
                                  cmap="Reds",
                                  vmin=-1,
                                  vmax=1)
    else:
        # Fallback to original method if the layer doesn't exist
        dp = sc.pl.DotPlot(adata,
                          groupby="cell_type",
                          cmap="Reds",
                          norm=mpl.colors.Normalize(vmin=-1, vmax=1),
                          var_names=dotplot_genes,
                          layer='norm_log_zscore',
                          figsize=(8, 2),
                          categories_order=cell_type_orders_multiome)
    dp.legend(
        size_title="fraction of cells\nin group (%)",
        colorbar_title="mean expression\nin group",
    )

    # Add totals and show
    dp.add_totals(color="maroon")

    # Save the dot plot
    dp.savefig(os.path.join(output_dir, "b.png"),
               dpi=300,
               bbox_inches='tight')
    dp.savefig(os.path.join(output_dir, "b.svg"),
               format='svg',
               bbox_inches='tight')

    print("Dot plot for all cells saved!")

    # Print summary statistics for all cells
    print(f"\nSummary (all cells):")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Genes in dot plot: {len(dotplot_genes)}")
    cell_type_counts = adata.obs['cell_type'].value_counts()
    for ct in cell_type_orders_multiome:
        if ct in cell_type_counts:
            print(f"{ct}: {cell_type_counts[ct]:,} cells")

    # Now create the control-only version
    print("\nCreating dot plot for control cells only...")
    adata_control = adata[adata.obs['condition'] == 'Control'].copy()
    print(f"Using {adata_control.n_obs:,} control cells")

    # Filter categories_order to only include cell types present in control data
    control_cell_types = adata_control.obs['cell_type'].unique()
    control_categories_order = [ct for ct in cell_type_orders_multiome if ct in control_cell_types]
    print(f"Cell types in control data: {control_categories_order}")

    # Create dot plot for control cells only with proper fractions
    if 'norm_log_zscore_control' in adata_control.layers:
        dp_control = create_proper_dotplot(adata_control,
                                          groupby="cell_type",
                                          var_names=dotplot_genes,
                                          categories_order=control_categories_order,
                                          figsize=(8, 2),
                                          cmap="Reds",
                                          vmin=-1,
                                          vmax=1)
    else:
        # Fallback to original method if the layer doesn't exist
        dp_control = sc.pl.DotPlot(adata_control,
                                   groupby="cell_type",
                                   cmap="Reds",
                                   norm=mpl.colors.Normalize(vmin=-1, vmax=1),
                                   var_names=dotplot_genes,
                                   layer='norm_log_zscore',
                                   figsize=(8, 2),
                                   categories_order=control_categories_order)
    dp_control.legend(
        size_title="fraction of cells\nin group (%)",
        colorbar_title="mean expression\nin group",
    )

    # Add totals and show
    dp_control.add_totals(color="maroon")

    # Save the control-only dot plot
    dp_control.savefig(os.path.join(output_dir, "b_control.png"),
                       dpi=300,
                       bbox_inches='tight')
    dp_control.savefig(os.path.join(output_dir, "b_control.svg"),
                       format='svg',
                       bbox_inches='tight')

    print("Dot plot for control cells saved!")

    # Print summary statistics for control cells
    print(f"\nSummary (control cells):")
    print(f"Total control cells: {adata_control.n_obs:,}")
    control_cell_type_counts = adata_control.obs['cell_type'].value_counts()
    for ct in cell_type_orders_multiome:
        if ct in control_cell_type_counts:
            print(f"{ct}: {control_cell_type_counts[ct]:,} cells")

    # Validate file sizes
    png_path = os.path.join(output_dir, "b.png")
    svg_path = os.path.join(output_dir, "b.svg")
    png_control_path = os.path.join(output_dir, "b_control.png")
    svg_control_path = os.path.join(output_dir, "b_control.svg")

    for png_file, svg_file in [(png_path, svg_path), (png_control_path, svg_control_path)]:
        png_size = os.path.getsize(png_file) / 1024 / 1024  # MB
        svg_size = os.path.getsize(svg_file) / 1024  # KB

        if png_size > 5:
            print(f"ERROR: PNG {os.path.basename(png_file)} {png_size:.1f}MB exceeds 5MB limit!")
        if svg_size > 500:
            print(f"ERROR: SVG {os.path.basename(svg_file)} {svg_size:.1f}KB exceeds 500KB limit!")
        else:
            print(f"✓ File sizes OK for {os.path.basename(png_file)}: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    print("\nFigure 6b completed successfully (both versions)!")

if __name__ == "__main__":
    main()

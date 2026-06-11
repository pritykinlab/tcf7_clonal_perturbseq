#!/usr/bin/env python
"""
Generate cell type marker dotplot (Figure 2c)
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

from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10, dotplot_genes
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
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data with control-based z-score normalization
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_norm_log_zscore_control=True)

    # First, create the full dotplot (all cells)
    print("Creating dotplot for all cells...")

    # Create dotplot with proper fractions and z-score colors
    dp = create_proper_dotplot(adata,
                              groupby="cell_type",
                              var_names=dotplot_genes,
                              categories_order=cell_type_orders_gene10,
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
    fig.savefig(os.path.join(output_dir, "c.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "c.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()

    # Now create the control-only version
    print("\nCreating dotplot for control cells only...")
    adata_control = adata[adata.obs['perturbation'] == 'NT'].copy()
    print(f"Using {adata_control.n_obs} control cells")

    # Filter categories_order to only include cell types present in control data
    control_cell_types = adata_control.obs['cell_type'].unique()
    control_categories_order = [ct for ct in cell_type_orders_gene10 if ct in control_cell_types]
    print(f"Cell types in control data: {control_categories_order}")

    # Create dotplot for control cells only with proper fractions
    dp_control = create_proper_dotplot(adata_control,
                                      groupby="cell_type",
                                      var_names=dotplot_genes,
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
    fig_control.savefig(os.path.join(output_dir, "c_control.png"),
                        dpi=300,
                        bbox_inches='tight')
    fig_control.savefig(os.path.join(output_dir, "c_control.svg"),
                        format='svg',
                        bbox_inches='tight')

    plt.close()
    print("Figure 2c completed successfully (both versions)!")

if __name__ == "__main__":
    main()
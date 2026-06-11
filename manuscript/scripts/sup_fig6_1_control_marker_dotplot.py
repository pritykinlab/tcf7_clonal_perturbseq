#!/usr/bin/env python
"""
Generate a control-only marker dotplot for Figure S13b.

This panel shows only control cells and omits the TRM-like cell-type row.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse

sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

from figure_format import get_adata, cell_type_orders_multiome, dotplot_genes
from dotplot_utils import create_proper_dotplot


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

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


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_13"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "b.png")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "b.svg")


def add_control_zscore_layer(adata):
    if "norm_log" not in adata.layers:
        raise ValueError("Expected 'norm_log' layer in multiome RNA AnnData.")

    control_mask = (adata.obs["condition"] == "Control").to_numpy()
    if control_mask.sum() == 0:
        raise ValueError("No cells with condition == 'Control' found.")

    x_all = adata.layers["norm_log"]
    if issparse(x_all):
        x_all = x_all.toarray()
    else:
        x_all = np.asarray(x_all)

    x_control = x_all[control_mask]
    control_mean = np.mean(x_control, axis=0)
    control_std = np.std(x_control, axis=0, ddof=1)
    control_std[control_std == 0] = 1e-9

    adata.layers["norm_log_zscore_control"] = np.clip(
        (x_all - control_mean) / control_std,
        a_min=-10,
        a_max=10,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading multiome RNA data...")
    adata = get_adata("multiome_rna")
    add_control_zscore_layer(adata)

    print("Restricting to control cells...")
    adata_control = adata[adata.obs["condition"] == "Control"].copy()
    print(f"Using {adata_control.n_obs} control cells")

    categories_order = [
        cell_type
        for cell_type in cell_type_orders_multiome
        if cell_type != "TRM-like" and cell_type in set(adata_control.obs["cell_type"])
    ]
    if not categories_order:
        raise ValueError("No non-TRM-like control cell types found for Figure S13b.")
    print(f"Plotting cell types: {categories_order}")

    dp = create_proper_dotplot(
        adata_control,
        groupby="cell_type",
        var_names=dotplot_genes,
        categories_order=categories_order,
        figsize=(8, 1.8),
        cmap="Reds",
        vmin=-1,
        vmax=1,
    )
    dp.add_totals(color="maroon")
    dp.show()

    fig = plt.gcf()

    print("Saving Figure S13b...")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_SVG, format='svg', bbox_inches='tight')
    plt.close(fig)

    png_size = os.path.getsize(OUTPUT_PNG) / 1024 / 1024
    svg_size = os.path.getsize(OUTPUT_SVG) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate TRM-like distance analysis for the bottom panel of Figure 5m.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_colors, cell_type_orders_trim28_setdb1

# Set figure parameters
FONT_SCALE = 1.25  # midway between original and previous 1.5x scaling
TITLE_SIZE = 25 * FONT_SCALE
LABEL_SIZE = 25 * FONT_SCALE
TICK_SIZE = 20 * FONT_SCALE
MARKER_SIZE = 5 * FONT_SCALE

plt.rcParams["axes.titlesize"] = TITLE_SIZE
plt.rcParams["axes.labelsize"] = LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_SIZE
plt.rcParams["ytick.labelsize"] = TICK_SIZE
plt.rcParams["lines.markersize"] = MARKER_SIZE
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (6, 6)

def compute_average_distance(adata, reference_cell_type='TRM-like', obsm_key='X_pca', n_pcs=50):
    """
    Compute average distance from reference cell type to all other cell types in PCA space.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    reference_cell_type : str
        Cell type to use as reference
    obsm_key : str
        Key for dimensionality reduction in obsm
    n_pcs : int
        Number of principal components to use

    Returns:
    --------
    pd.DataFrame
        DataFrame with cell types and their average distances to reference
    """
    pca_coords = adata.obsm[obsm_key][:, :n_pcs]

    cell_types = adata.obs['cell_type'].unique()
    ref_coords = pca_coords[adata.obs['cell_type'] == reference_cell_type]

    distance_list = []
    for ct in cell_types:
        if ct == reference_cell_type:
            continue
        ct_coords = pca_coords[adata.obs['cell_type'] == ct]
        pairwise_distances = cdist(ref_coords, ct_coords, metric='euclidean')
        avg_distance = pairwise_distances.mean()
        distance_list.append({'cell_type': ct, 'distance': avg_distance})

    return pd.DataFrame(distance_list).sort_values('distance')

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data - filter for Trim28-perturbed cells only
    print("Loading Trim28 Setdb1 perturb-seq dataset...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      remove_lymph_node=True,  # Usually want to remove lymph node for this dataset
                      get_raw_scaled=True)

    # Filter for Trim28-perturbed cells only (this panel uses only Trim28 cells)
    print("Filtering for Trim28-perturbed cells...")
    adata_trim28_only = adata[adata.obs['perturbation'] == 'Trim28'].copy()
    print(f"Number of Trim28-perturbed cells: {adata_trim28_only.n_obs}")

    # Compute distances
    print("Computing distances from TRM-like cells...")
    distances_df = compute_average_distance(
        adata_trim28_only,
        reference_cell_type='TRM-like',
        obsm_key='X_pca',
        n_pcs=50
    )

    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create bar plot
    bars = ax.bar(distances_df['cell_type'], distances_df['distance'], color='gray', width=0.8)

    # Customize plot
    ax.set_title('Trim28/Setdb1 experiment', fontsize=TITLE_SIZE)
    ax.set_ylim(0, max(distances_df['distance']) * 1.1)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(distances_df['cell_type'], rotation=45, ha='right')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid for easier reading (optional - can be removed if not desired)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "m_bottom.png")
    svg_path = os.path.join(output_dir, "m_bottom.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

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
    print("Figure 5m_bottom completed successfully!")

    # Print summary
    print(f"\nSummary:")
    print(f"- Cell types analyzed: {len(distances_df)}")
    print(f"- Closest to TRM-like: {distances_df.iloc[0]['cell_type']} (distance: {distances_df.iloc[0]['distance']:.2f})")
    print(f"- Farthest from TRM-like: {distances_df.iloc[-1]['cell_type']} (distance: {distances_df.iloc[-1]['distance']:.2f})")

if __name__ == "__main__":
    main()

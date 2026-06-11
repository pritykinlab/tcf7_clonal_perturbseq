#!/usr/bin/env python
"""
Generate clone size distribution with stacked bar chart (Figure 3b)
and clone-specific UMAP visualizations (Figure 3c)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

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
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.edgecolor'] = 'dimgray'

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure3"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=True,
                      remove_NG=True,
                      get_raw_scaled=True)

    # Convert categorical columns to strings
    print("Processing clone data...")
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    adata.obs['clone'] = adata.obs['clone'].astype(str)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

    # =========================================================================
    # Select clones for visualization (NO_SITE-4 clones)
    # =========================================================================

    # Select clones containing "NO_SITE-4"
    no_site4_subset = adata.obs.query("clone.str.contains('NO_SITE-4')").copy()

    # Get unique clones sorted by size descending
    unique_clones_sorted = (
        no_site4_subset[['clone', 'clone_size']]
        .drop_duplicates('clone')
        .sort_values('clone_size', ascending=False)
    )

    print(f"Found {len(unique_clones_sorted)} NO_SITE-4 clones")

    # Select clones evenly spaced by cumulative weight (deterministic)
    n_clones_to_show = 9

    # Calculate weights proportional to log2 clone size
    clone_sizes = unique_clones_sorted['clone_size'].values
    log2_clone_sizes = np.log2(clone_sizes + 0.5)  # Add 0.5 pseudo count
    weights = log2_clone_sizes / log2_clone_sizes.sum()

    # Calculate cumulative weights
    cumulative_weights = np.cumsum(weights)

    # Select clones at evenly spaced cumulative weight positions
    if len(unique_clones_sorted) >= n_clones_to_show:
        # Create evenly spaced bins and use their midpoints
        # This creates n_clones_to_show bins from 0 to 1
        bin_edges = np.linspace(0, 1, n_clones_to_show + 1)
        # Calculate midpoint of each bin
        target_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

        selected_indices = []
        for target in target_positions:
            # Find the clone whose cumulative weight is closest to the target
            idx = np.searchsorted(cumulative_weights, target)
            # Ensure we don't go out of bounds
            idx = min(idx, len(cumulative_weights) - 1)
            selected_indices.append(idx)

        # Remove duplicates while preserving order
        selected_indices = list(dict.fromkeys(selected_indices))

        selected_clones = unique_clones_sorted.iloc[selected_indices]
    else:
        selected_clones = unique_clones_sorted

    # Keep clones in selection order (not sorted)
    selected_clone_ids = selected_clones['clone'].tolist()

    print(f"Selected {n_clones_to_show} clones for visualization")

    # =========================================================================
    # Prepare data for Figure 3b
    # =========================================================================

    # Calculate the count of each cell_type within each (perturbation, clone) pair
    count_data = adata.obs.groupby(['perturbation', 'clone', 'cell_type']).size().unstack(fill_value=0)

    # Convert counts to proportions
    proportion_data = count_data.div(count_data.sum(axis=1), axis=0)

    # Calculate clone sizes for ALL clones
    clone_sizes = adata.obs.groupby(['perturbation', 'clone']).size()
    proportion_data['clone_size'] = clone_sizes
    proportion_data['log2_clone_size'] = np.log2(proportion_data['clone_size'] + 1)

    # Reset index for sorting
    proportion_data = proportion_data.reset_index()

    # Sort by log2_clone_size descending, then by cell type proportions
    sort_cols = ['log2_clone_size'] + cell_type_orders_gene10[::-1]
    proportion_data.sort_values(by=sort_cols, ascending=False, inplace=True)

    # Create a mapping of clone to its x-position in the sorted data
    clone_to_x_position = {}
    for idx, row in enumerate(proportion_data.itertuples()):
        clone_id = f"{row.perturbation}, {row.clone}"
        clone_to_x_position[clone_id] = idx

    # Find x-positions of selected clones
    selected_x_positions = []
    selected_clone_sizes = []
    for clone_id in selected_clone_ids:
        # Need to find the perturbation for this clone
        pert = no_site4_subset[no_site4_subset['clone'] == clone_id]['perturbation'].iloc[0]
        full_clone_id = f"{pert}, {clone_id}"
        if full_clone_id in clone_to_x_position:
            x_pos = clone_to_x_position[full_clone_id]
            selected_x_positions.append(x_pos)
            # Get the actual clone size
            clone_size = clone_sizes.loc[(pert, clone_id)]
            selected_clone_sizes.append(clone_size)

    print(f"Found {len(selected_x_positions)} clone positions in sorted data")

    # Drop size columns after sorting for the bar chart
    proportion_data.drop(columns=['clone_size', 'log2_clone_size'], inplace=True)

    # Create perturbation_clone identifier and set as index
    proportion_data['perturbation_clone'] = (
        proportion_data['perturbation'] + ', ' + proportion_data['clone']
    )
    proportion_data.set_index('perturbation_clone', inplace=True)
    proportion_data.drop(columns=['perturbation', 'clone'], inplace=True)

    # Reorder columns based on cell type order
    proportion_data = proportion_data[cell_type_orders_gene10]

    # Calculate clone sizes for the line plot (all clones)
    clone_sizes_all = adata.obs.groupby(['perturbation', 'clone']).size()
    sorted_clone_sizes = clone_sizes_all.sort_values(ascending=False).reset_index(drop=True)
    log2_sorted_clone_sizes = np.log2(sorted_clone_sizes)

    # =========================================================================
    # Create Figure 3b
    # =========================================================================

    print("Creating Figure 3b...")
    fig_3b, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(20*1.1, 4*1.1),
        gridspec_kw={'height_ratios': [0.5, 1], 'hspace': 0.01}
    )

    # Top subplot: log2(clone size) line plot
    ax1.plot(log2_sorted_clone_sizes, linestyle='-', color='gray', rasterized=True)
    ax1.set_ylabel('log2(clone size)', fontsize=20, rotation=45, labelpad=50)
    ax1.set_xlim(0, len(sorted_clone_sizes))
    ax1.set_yticks([])
    ax1.set_xticks([])

    # Highlight selected clones with red dots and annotate with actual sizes
    for x_pos, clone_size in zip(selected_x_positions, selected_clone_sizes):
        y_value = log2_sorted_clone_sizes[x_pos]
        ax1.plot(x_pos, y_value, 'ro', markersize=8, rasterized=True)
        ax1.text(x_pos, y_value, f'{clone_size}', fontsize=16,
                 verticalalignment='bottom', horizontalalignment='center',
                 color='red')

    # Bottom subplot: Stacked bar chart of proportions
    colors = [cell_type_colors.get(x) for x in cell_type_orders_gene10]

    # Create the stacked bar plot
    proportion_data.plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        width=1,
        color=colors
    )

    # Rasterize all bar artists to reduce file size
    for patch in ax2.patches:
        patch.set_rasterized(True)

    ax2.set_xlabel('')
    ax2.set_xticks([])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel('Cell type\nproportion')
    ax2.set_ylim(0, 1)

    # Add NT clone label
    ax2.text(0.5, -0.15, f'NT clone (n={len(proportion_data)})',
             transform=ax2.transAxes, ha='center', fontsize=20)

    # Remove legend from bar plot
    ax2.legend().remove()

    plt.tight_layout()

    # Save Figure 3b
    print("Saving Figure 3b...")
    png_path_3b = os.path.join(output_dir, "b.png")
    svg_path_3b = os.path.join(output_dir, "b.svg")

    fig_3b.savefig(png_path_3b, dpi=300, bbox_inches='tight')
    fig_3b.savefig(svg_path_3b, format='svg', bbox_inches='tight')

    # Validate file sizes for Figure 3b
    png_size_3b = os.path.getsize(png_path_3b) / 1024 / 1024  # MB
    svg_size_3b = os.path.getsize(svg_path_3b) / 1024  # KB

    if png_size_3b > 5:
        print(f"ERROR: Figure 3b PNG {png_size_3b:.1f}MB exceeds 5MB limit!")
    if svg_size_3b > 500:
        print(f"ERROR: Figure 3b SVG {svg_size_3b:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Figure 3b file sizes OK: PNG {png_size_3b:.2f}MB, SVG {svg_size_3b:.1f}KB")

    plt.close(fig_3b)

    # =========================================================================
    # Create Figure 3c - Single row of UMAPs
    # =========================================================================

    print("Creating Figure 3c...")

    # Create figure with single row
    fig_3c, axes = plt.subplots(1, n_clones_to_show,
                                figsize=(3*n_clones_to_show, 3))

    # Ensure axes is always a list
    if n_clones_to_show == 1:
        axes = [axes]

    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']

    # Create UMAP for each selected clone (in size order)
    for idx, (_, row) in enumerate(selected_clones.iterrows()):
        clone_id = row['clone']
        clone_size = row['clone_size']

        ax = axes[idx]

        print(f"Creating UMAP for clone: {clone_id} (size: {clone_size})")

        # Create mask for this clone
        clone_mask = adata.obs['clone'] == clone_id

        # Plot all cells in gray (background)
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                  c='lightgray', s=0.3, alpha=0.15, rasterized=True)

        # Highlight cells from this clone in red
        if clone_mask.sum() > 0:
            ax.scatter(umap_coords[clone_mask, 0], umap_coords[clone_mask, 1],
                      c='red', s=12, alpha=0.8, rasterized=True)

        # Extract just the 6bp barcode (last part after __)
        barcode_6bp = clone_id.split('__')[-1] if '__' in clone_id else clone_id
        # Set title with just the 6bp barcode
        ax.set_title(barcode_6bp, fontsize=27)

        # Remove axes and all borders
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_frame_on(False)

    plt.tight_layout()

    # Save Figure 3c
    print("Saving Figure 3c...")
    png_path_3c = os.path.join(output_dir, "c.png")
    svg_path_3c = os.path.join(output_dir, "c.svg")

    fig_3c.savefig(png_path_3c, dpi=150, bbox_inches='tight')
    # Save SVG with lower dpi for rasterized elements to reduce file size
    fig_3c.savefig(svg_path_3c, format='svg', bbox_inches='tight', dpi=72)

    # Validate file sizes for Figure 3c
    png_size_3c = os.path.getsize(png_path_3c) / 1024 / 1024  # MB
    svg_size_3c = os.path.getsize(svg_path_3c) / 1024  # KB

    if png_size_3c > 5:
        print(f"ERROR: Figure 3c PNG {png_size_3c:.1f}MB exceeds 5MB limit!")
    if svg_size_3c > 500:
        print(f"ERROR: Figure 3c SVG {svg_size_3c:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Figure 3c file sizes OK: PNG {png_size_3c:.2f}MB, SVG {svg_size_3c:.1f}KB")

    plt.close(fig_3c)

    print("\nFigure 3b and 3c completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total NT clones: {len(proportion_data)}")
    print(f"Total NO_SITE-4 clones: {len(unique_clones_sorted)}")
    print(f"Clones visualized in UMAPs: {n_clones_to_show}")
    print(f"Clone size range (all): {sorted_clone_sizes.min()} - {sorted_clone_sizes.max()}")
    print(f"Selected clone sizes: {selected_clone_sizes}")

if __name__ == "__main__":
    main()

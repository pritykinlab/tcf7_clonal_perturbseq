#!/usr/bin/env python
"""
Generate stacked bar chart showing TRM-like cells by clone for trim28_setdb1 dataset (Figure 5j)
Colored by mouse with different shades for clones within each mouse
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading trim28_setdb1 data...")
    adata = get_adata('trim28_setdb1',
                      # Keep multi-construct / multi-perturbation clones
                      use_cells_only_one_construct=False,
                      remove_NG=True,
                      get_raw_scaled=True)

    # Filter to TRM-like cells and include mouse information
    print("Filtering to TRM-like cells...")
    df = pd.DataFrame({
        'cell_type': adata.obs['cell_type'],
        'clone': adata.obs['clone'],
        'mouse': adata.obs['mouse']
    })
    df_trm_like = df[df['cell_type'] == 'TRM-like'].copy()

    # Count cells per clone and mouse - only include actual occurrences
    clone_mouse_counts = df_trm_like.groupby(['mouse', 'clone'], observed=True).size().reset_index(name='count')

    # Remove any zero counts (shouldn't happen but just in case)
    clone_mouse_counts = clone_mouse_counts[clone_mouse_counts['count'] > 0]

    # Calculate total cells per mouse for ordering
    mouse_totals = clone_mouse_counts.groupby('mouse', observed=True)['count'].sum().sort_values(ascending=False)

    # Create ordered dataframe: first by mouse (ordered by total cells), then by clone size within each mouse
    ordered_data = []
    for mouse in mouse_totals.index:
        mouse_data = clone_mouse_counts[clone_mouse_counts['mouse'] == mouse].sort_values('count', ascending=False)
        ordered_data.append(mouse_data)

    clone_mouse_counts = pd.concat(ordered_data, ignore_index=True)

    print(f"Found {len(clone_mouse_counts)} clone-mouse combinations")
    print(f"Total TRM-like cells: {clone_mouse_counts['count'].sum()}")
    print(f"Number of unique mice: {clone_mouse_counts['mouse'].nunique()}")

    # Get unique mice in order of total cells
    unique_mice = mouse_totals.index.tolist()

    # Use tab10 colormap cycling for more distinct colors between mice
    # tab10 has 10 distinct colors, so we'll cycle through them
    tab10_colors = plt.cm.tab10(np.arange(10))
    mouse_base_colors = [tab10_colors[i % 10] for i in range(len(unique_mice))]

    # Create color mapping for each clone
    clone_colors = []
    for i, mouse in enumerate(unique_mice):
        mouse_data = clone_mouse_counts[clone_mouse_counts['mouse'] == mouse]
        n_clones = len(mouse_data)

        # Get base color for this mouse
        base_color = mouse_base_colors[i]

        if n_clones == 1:
            # If only one clone for this mouse, use the base color
            clone_colors.append(base_color)
        else:
            # Create stronger gradient from very light to dark for multiple clones
            # Start at 0.2 (much lighter) and go to 1.0 (full color) for stronger contrast
            shades = np.linspace(0.2, 1.0, n_clones)
            for shade in shades:
                # Adjust the color by mixing with white
                adjusted_color = tuple([
                    1 - (1 - base_color[j]) * shade for j in range(3)
                ] + [base_color[3]])  # Keep alpha unchanged
                clone_colors.append(adjusted_color)

    # Prepare data for stacked bar plot
    plot_data = pd.DataFrame([clone_mouse_counts['count'].values],
                              columns=[f"{row['mouse']}_{row['clone']}" for _, row in clone_mouse_counts.iterrows()])

    # Create the plot
    print("Creating stacked bar visualization...")
    fig, ax = plt.subplots(figsize=(2, 5))

    # Plot a single stacked bar with rasterization for smaller SVG
    plot_data.plot(
        kind='bar',
        stacked=True,
        color=clone_colors,
        width=0.5,
        ax=ax,
        legend=False,  # No legend with clone names
        rasterized=True  # Rasterize to reduce SVG size
    )

    # Style the axes
    ax.set_ylabel("number of TRM-like cells", fontsize=20.25)  # Reduced from 22.5 to 20.25 (0.9x again, 0.81x original)
    ax.set_xlabel("")
    ax.set_xticks([])  # Remove the x-axis tick label (only one bar)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust y-axis tick label size
    ax.tick_params(axis='y', labelsize=16.2)  # Reduced from 18 to 16.2 (0.9x again, 0.81x original)

    # Add clone count labels for the first 5 mice
    cumulative_height = 0
    for i, mouse in enumerate(unique_mice[:5]):  # Only first 5 mice
        mouse_data = clone_mouse_counts[clone_mouse_counts['mouse'] == mouse]
        mouse_height = mouse_data['count'].sum()
        n_clones = len(mouse_data)

        # Position label at the middle of this mouse's section
        label_y = cumulative_height + mouse_height / 2

        # Add text to the right of the bar (x position is in axis coordinates, y in data)
        # Get the x position slightly to the right of the bar
        bar_right = 0.27  # Slightly past the bar's right edge
        ax.text(bar_right, label_y, f'n={n_clones}',
                fontsize=12, va='center', ha='left')  # Reduced from 16 to 12 (0.75x)

        cumulative_height += mouse_height

    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "j.png")
    svg_path = os.path.join(output_dir, "j.svg")

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
    print("Figure 5j completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total TRM-like cells: {clone_mouse_counts['count'].sum()}")
    print(f"- Number of unique mice: {clone_mouse_counts['mouse'].nunique()}")
    print(f"- Number of clone-mouse combinations: {len(clone_mouse_counts)}")

    # Print per-mouse summary (only mice with cells)
    print("\nPer-mouse breakdown (ordered by total TRM-like cells):")
    for mouse in unique_mice:
        mouse_data = clone_mouse_counts[clone_mouse_counts['mouse'] == mouse]
        total_cells = mouse_data['count'].sum()
        n_clones = len(mouse_data)
        if total_cells > 0:  # Only print mice that have TRM-like cells
            print(f"  {mouse}: {total_cells} cells across {n_clones} clone(s)")
            if n_clones <= 3:
                for _, row in mouse_data.iterrows():
                    print(f"    - Clone {row['clone']}: {row['count']} cells")

if __name__ == "__main__":
    main()

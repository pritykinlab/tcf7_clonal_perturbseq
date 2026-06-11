#!/usr/bin/env python
"""
Generate standalone cell type legend with reversed order (Figure 2 legend)
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import cell_type_colors, cell_type_orders_gene10

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating standalone legend...")

    # Use original order (not reversed)
    cell_type_order = cell_type_orders_gene10

    # Create figure for legend only
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.axis('off')

    # Create legend patches with original order
    legend_patches = []
    for cell_type in cell_type_order:
        if cell_type in cell_type_colors:
            patch = mpatches.Patch(color=cell_type_colors[cell_type], label=cell_type)
            legend_patches.append(patch)

    # Add legend with box (frameon=True)
    legend = ax.legend(handles=legend_patches,
                       loc='center',
                       title="Cell type",
                       frameon=True,
                       fontsize=20,
                       title_fontsize=22,
                       edgecolor='black',
                       fancybox=False)

    # Save the figure
    print("Saving legend figures...")
    fig.savefig(os.path.join(output_dir, "legend.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "legend.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Figure 2 legend completed successfully!")

    print(f"\nCreated legend with {len(legend_patches)} cell types")

if __name__ == "__main__":
    main()

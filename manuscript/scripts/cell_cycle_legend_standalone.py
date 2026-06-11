#!/usr/bin/env python
"""
Generate standalone cell cycle phase legends for multiple figures
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add utils to path
sys.path.append('../scripts/utils')

# Cell cycle phase colors (matching scanpy defaults)
PHASE_COLORS = {
    'G1': '#ff7f0e',  # Orange
    'S': '#2ca02c',   # Green
    'G2M': '#1f77b4'  # Blue
}

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def create_cell_cycle_legend(output_path):
    """
    Create a standalone cell cycle legend figure

    Parameters
    ----------
    output_path : str
        Full path for the output file (including filename)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.axis('off')

    # Create legend patches
    legend_patches = []
    for phase in ['G1', 'S', 'G2M']:
        patch = mpatches.Patch(color=PHASE_COLORS[phase], label=phase)
        legend_patches.append(patch)

    # Add legend with box (similar to cell type legend)
    legend = ax.legend(handles=legend_patches,
                       loc='center',
                       title="Cell cycle",
                       frameon=True,
                       fontsize=20,
                       title_fontsize=22,
                       edgecolor='black',
                       fancybox=False)

    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cell cycle legend saved to {output_path}")

def main():
    # Only create legend for Figure 2
    figure_dir = "../manuscript/plots/Figure2"

    # Create output directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)

    # Generate PNG version
    output_path = os.path.join(figure_dir, "cell_cycle_legend.png")
    create_cell_cycle_legend(output_path)

    # Generate SVG version
    output_path_svg = os.path.join(figure_dir, "cell_cycle_legend.svg")
    create_cell_cycle_legend(output_path_svg)

    print(f"Created cell cycle legends in {figure_dir}")
    print("\nFigure 2 cell cycle legend created successfully!")

if __name__ == "__main__":
    main()

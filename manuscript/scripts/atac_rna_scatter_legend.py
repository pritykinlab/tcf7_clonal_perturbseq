#!/usr/bin/env python
"""
Generate standalone legend for ATAC vs RNA scatter plot (Figure 6d)
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Create figure for legend
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_subplot(111)

    # Hide the axes
    ax.axis('off')

    # Create custom legend patches with the same colors as the scatter plots
    legend_elements = [
        mpatches.Patch(color='lightgrey', label='Non-significant', alpha=0.5),
        mpatches.Patch(color='tab:cyan', label='Significant in RNA only', alpha=0.7),
        mpatches.Patch(color='tab:purple', label='Significant in ATAC only', alpha=0.7),
        mpatches.Patch(color='tab:red', label='Significant in both', alpha=0.7)
    ]

    # Create the legend
    legend = ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=True,
        fancybox=False,
        fontsize=14,
        ncol=2,  # Display in 2 columns
        columnspacing=2,
        handlelength=2,
        handleheight=1,
        title='significance categories',
        title_fontsize=16
    )

    # Style the legend
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('dimgray')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)

    # Adjust layout
    plt.tight_layout()

    # Save outputs
    print("Saving legend figures...")
    png_path = os.path.join(output_dir, "d_legend.png")
    svg_path = os.path.join(output_dir, "d_legend.svg")

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
    print("Figure 6d legend completed successfully!")

if __name__ == "__main__":
    main()

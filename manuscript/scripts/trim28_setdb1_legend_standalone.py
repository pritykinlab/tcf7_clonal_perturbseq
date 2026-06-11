#!/usr/bin/env python
"""
Generate standalone cell type legend for trim28_setdb1 dataset (Figure 5 legend)
"""

import argparse
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

from figure_format import cell_type_colors, cell_type_orders_trim28_setdb1

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

DEFAULT_OUTPUT_DIR = "../manuscript/plots/Figure5"
DEFAULT_STEM = "legend"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a standalone cell type legend for the trim28_setdb1 dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where legend files should be written.",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Creating standalone legend for trim28_setdb1 cell types...")

    # Use trim28_setdb1 cell type order
    cell_type_order = cell_type_orders_trim28_setdb1

    # Create figure for legend only
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    # Create legend patches
    legend_patches = []
    for cell_type in cell_type_order:
        if cell_type in cell_type_colors:
            patch = mpatches.Patch(color=cell_type_colors[cell_type], label=cell_type)
            legend_patches.append(patch)

    # Add legend with box (frameon=True)
    legend = ax.legend(handles=legend_patches,
                       loc='center',
                       frameon=True,
                       ncol=2,
                       fontsize=20,
                       edgecolor='black',
                       fancybox=False,
                       columnspacing=1.5,
                       handletextpad=0.6)

    # Save the figure
    print("Saving legend figures...")
    png_path = os.path.join(output_dir, f"{args.stem}.png")
    svg_path = os.path.join(output_dir, f"{args.stem}.svg")

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
    print("Figure 5 cell type legend completed successfully!")

    print(f"\nCreated legend with {len(legend_patches)} cell types:")
    for ct in cell_type_order:
        print(f"  - {ct}")

if __name__ == "__main__":
    main()

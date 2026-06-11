#!/usr/bin/env python
"""
Generate standalone perturbation legend for Figure 5b.
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('../scripts/utils')

from figure_format import perturbation_colors  # noqa: E402


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 20


def validate_file_sizes(png_path, svg_path):
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.axis('off')

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=perturbation_colors['NT'],
            markeredgecolor=perturbation_colors['NT'],
            markersize=10,
            label='NT',
        ),
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=perturbation_colors['Trim28'],
            markeredgecolor=perturbation_colors['Trim28'],
            markersize=10,
            label='Trim28',
        ),
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=perturbation_colors['Setdb1'],
            markeredgecolor=perturbation_colors['Setdb1'],
            markersize=10,
            label='Setdb1',
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc='center',
        frameon=True,
        edgecolor='dimgray',
        fancybox=False,
    )

    png_path = os.path.join(output_dir, "b_legend.png")
    svg_path = os.path.join(output_dir, "b_legend.svg")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    validate_file_sizes(png_path, svg_path)

    plt.close(fig)
    print("Figure 5b legend completed successfully!")


if __name__ == "__main__":
    main()

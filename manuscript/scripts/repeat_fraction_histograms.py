#!/usr/bin/env python
"""
Generate histograms showing fraction of reads mapping to repeat elements for ATAC and RNA (Figure 6f)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import mannwhitneyu

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import perturbation_colors

# Set figure parameters
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["axes.labelsize"] = 27
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.fontsize'] = 21
plt.rcParams['legend.edgecolor'] = 'dimgray'
plt.rcParams['figure.figsize'] = (8, 6)

def plot_lib_size_ratio_histogram(repeat_adata, condition_col, group1, group2, modality=''):
    """
    Plots overlapping histograms for the ratio (repeat library / total library)
    for two conditions, keeps only the middle 98% of data, and performs
    a one-sided (group1 > group2) Mann-Whitney U test.

    Parameters
    ----------
    repeat_adata : AnnData
        AnnData object with repeat information
    condition_col : str
        Column name for condition
    group1 : str
        First group (e.g., "Trim28")
    group2 : str
        Second group (e.g., "Control")
    modality : str
        "ATAC" or "RNA" to label in the plot

    Returns
    -------
    fig : matplotlib.Figure
        The generated figure
    """

    # 1) Compute the ratio of repeat library to total library
    repeat_adata.obs["frac_mapping_to_repeat"] = (
        repeat_adata.obs["repeat_lib_size"] /
        (repeat_adata.obs["lib_size_reference"] + repeat_adata.obs["repeat_lib_size"])
    )

    # 2) Subset the required columns
    df_plot = repeat_adata.obs[[condition_col, "frac_mapping_to_repeat"]].copy()

    # 3) Keep only the middle 98% (remove below 1st percentile and above 99th)
    lower_bound = df_plot["frac_mapping_to_repeat"].quantile(0.01)
    upper_bound = df_plot["frac_mapping_to_repeat"].quantile(0.99)
    df_plot = df_plot[
        (df_plot["frac_mapping_to_repeat"] >= lower_bound) &
        (df_plot["frac_mapping_to_repeat"] <= upper_bound)
    ]

    # 4) Split data for each group
    group1_vals = df_plot.loc[df_plot[condition_col] == group1, "frac_mapping_to_repeat"]
    group2_vals = df_plot.loc[df_plot[condition_col] == group2, "frac_mapping_to_repeat"]

    # Use shared bin edges so both histograms line up
    bins = np.histogram_bin_edges(
        np.concatenate([group1_vals.values, group2_vals.values]),
        bins=30  # Fixed number of bins for consistency
    )

    # 5) Create figure and plot overlapping histograms
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use NT instead of Control in legend
    label2 = "NT" if group2 == "Control" else group2

    # Plot histograms with appropriate colors
    ax.hist(
        group2_vals,
        bins=bins,
        color=perturbation_colors.get(group2, '#888888'),
        alpha=0.5,
        density=True,
        label=label2,
        edgecolor='black',
        linewidth=0.5
    )
    ax.hist(
        group1_vals,
        bins=bins,
        color=perturbation_colors.get(group1, '#FF6B6B'),
        alpha=0.5,
        density=True,
        label=group1,
        edgecolor='black',
        linewidth=0.5
    )

    # 6) Perform one-sided Mann-Whitney U test (group1 > group2)
    stat, pval = mannwhitneyu(group1_vals, group2_vals, alternative='greater')

    # 7) Add title with modality
    ax.set_title(f"{modality}", fontsize=33, pad=10)

    # Add statistical test result as text inside the plot (no box)
    # Position it in the upper right area
    ax.text(0.95, 0.95, f"Mann-Whitney U\n({group1} > {label2})\np = {pval:.2e}",
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment='top',
            horizontalalignment='right')

    # 8) Set labels
    ax.set_xlabel("fraction of reads mapping to repeat elements", fontsize=24)
    ax.set_ylabel("Density", fontsize=24)

    # 9) Add legend with proper frame styling
    legend = ax.legend(loc='upper left', frameon=True)
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('dimgray')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)

    # Add subtle grid for readability
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    print(f"[{modality}] Mann-Whitney U ({group1} > {label2}): stat={stat:.2f}, p={pval:.2e}")
    print(f"[{modality}] {group1} median: {group1_vals.median():.4f}, {label2} median: {group2_vals.median():.4f}")

    return fig

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Define data directory
    REPEAT_DATA_BASE_DIR = "../data/repeat_enrichment/multiome"

    # Load normalized aggregated repeat data
    print("Loading normalized aggregated repeat data...")
    multiome_rna_repeat_agg_adata = sc.read(
        os.path.join(REPEAT_DATA_BASE_DIR, "multiome_rna_repeat_agg_norm_adata.h5ad"))
    multiome_atac_repeat_agg_adata = sc.read(
        os.path.join(REPEAT_DATA_BASE_DIR, "multiome_atac_repeat_agg_norm_adata.h5ad"))

    print(f"RNA data shape: {multiome_rna_repeat_agg_adata.shape}")
    print(f"ATAC data shape: {multiome_atac_repeat_agg_adata.shape}")

    # Create ATAC histogram
    print("\nCreating ATAC histogram...")
    fig_atac = plot_lib_size_ratio_histogram(
        repeat_adata=multiome_atac_repeat_agg_adata,
        condition_col="condition",
        group1="Trim28",
        group2="Control",
        modality="ATAC"
    )

    # Save ATAC histogram
    png_path_atac = os.path.join(output_dir, "f_atac.png")
    svg_path_atac = os.path.join(output_dir, "f_atac.svg")

    fig_atac.savefig(png_path_atac, dpi=300, bbox_inches='tight')
    fig_atac.savefig(svg_path_atac, format='svg', bbox_inches='tight')

    # Validate file sizes for ATAC
    png_size = os.path.getsize(png_path_atac) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path_atac) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: ATAC PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: ATAC SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ ATAC file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig_atac)

    # Create RNA histogram
    print("\nCreating RNA histogram...")
    fig_rna = plot_lib_size_ratio_histogram(
        repeat_adata=multiome_rna_repeat_agg_adata,
        condition_col="condition",
        group1="Trim28",
        group2="Control",
        modality="RNA"
    )

    # Save RNA histogram
    png_path_rna = os.path.join(output_dir, "f_rna.png")
    svg_path_rna = os.path.join(output_dir, "f_rna.svg")

    fig_rna.savefig(png_path_rna, dpi=300, bbox_inches='tight')
    fig_rna.savefig(svg_path_rna, format='svg', bbox_inches='tight')

    # Validate file sizes for RNA
    png_size = os.path.getsize(png_path_rna) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path_rna) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: RNA PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: RNA SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ RNA file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig_rna)

    print("\nFigure 6f completed successfully!")
    print("Generated files:")
    print(f"  - {png_path_atac}")
    print(f"  - {svg_path_atac}")
    print(f"  - {png_path_rna}")
    print(f"  - {svg_path_rna}")

if __name__ == "__main__":
    main()

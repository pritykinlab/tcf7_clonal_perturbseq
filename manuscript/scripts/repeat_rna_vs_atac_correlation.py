#!/usr/bin/env python
"""
Generate Figure 6g: RNA vs ATAC log2FC correlation for repeat regions
Shows scatter plot of differential repeat element expression between TRM-like and other cells
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import spearmanr, pearsonr

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Import utils
import DE_testing

# Set figure parameters
plt.rcParams["axes.titlesize"] = 27
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 21
plt.rcParams["ytick.labelsize"] = 21
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)

def main():
    print("Loading repeat element data...")

    # Load repeat data
    REPEAT_DATA_BASE_DIR = "../data/repeat_enrichment/multiome"

    # Load aggregated and normalized repeat data
    multiome_rna_repeat_agg_adata = sc.read_h5ad(
        os.path.join(REPEAT_DATA_BASE_DIR, "multiome_rna_repeat_agg_norm_adata.h5ad")
    )
    multiome_atac_repeat_agg_adata = sc.read_h5ad(
        os.path.join(REPEAT_DATA_BASE_DIR, "multiome_atac_repeat_agg_norm_adata.h5ad")
    )

    print(f"RNA repeat data shape: {multiome_rna_repeat_agg_adata.shape}")
    print(f"ATAC repeat data shape: {multiome_atac_repeat_agg_adata.shape}")

    # Perform DE test (TRM-like vs. other cells)
    print("\nPerforming differential expression analysis...")

    # Create masks for Trim28 TRM-like cells and non-TRM control cells
    mask_trm_like_rna = (
        (multiome_rna_repeat_agg_adata.obs["cell_type"] == "TRM-like")
        & (multiome_rna_repeat_agg_adata.obs["condition"] == "Trim28")
    )
    mask_trm_like_atac = (
        (multiome_atac_repeat_agg_adata.obs["cell_type"] == "TRM-like")
        & (multiome_atac_repeat_agg_adata.obs["condition"] == "Trim28")
    )
    mask_other_rna = (
        (multiome_rna_repeat_agg_adata.obs["cell_type"] != "TRM-like")
        & (multiome_rna_repeat_agg_adata.obs["condition"] == "Control")
    )
    mask_other_atac = (
        (multiome_atac_repeat_agg_adata.obs["cell_type"] != "TRM-like")
        & (multiome_atac_repeat_agg_adata.obs["condition"] == "Control")
    )

    # Get cell names
    trm_like_cells_rna = multiome_rna_repeat_agg_adata.obs_names[mask_trm_like_rna].tolist()
    other_cells_rna = multiome_rna_repeat_agg_adata.obs_names[mask_other_rna].tolist()
    n_trm_like = len(trm_like_cells_rna)
    n_other = len(other_cells_rna)

    trm_like_cells_atac = multiome_atac_repeat_agg_adata.obs_names[mask_trm_like_atac].tolist()
    other_cells_atac = multiome_atac_repeat_agg_adata.obs_names[mask_other_atac].tolist()

    print(f"RNA: {n_trm_like} Trim28 TRM-like cells vs {n_other} non-TRM control cells")
    print(f"ATAC: {len(trm_like_cells_atac)} Trim28 TRM-like cells vs {len(other_cells_atac)} non-TRM control cells")

    # Run DE tests
    results_rna_df = DE_testing.wilcoxon_DE_test(
        multiome_rna_repeat_agg_adata,
        trm_like_cells_rna,
        other_cells_rna,
        norm_layer="norm"
    )

    results_atac_df = DE_testing.wilcoxon_DE_test(
        multiome_atac_repeat_agg_adata,
        trm_like_cells_atac,
        other_cells_atac,
        norm_layer="norm"
    )

    # Create the plot
    print("\nCreating RNA vs ATAC correlation plot...")

    # Merge on 'Gene' (repeat element name)
    merged_df = pd.merge(
        results_rna_df,
        results_atac_df,
        on="Gene",
        suffixes=("_rna", "_atac")
    )

    # Define significance thresholds
    alpha = 0.05
    fc_threshold = 1.0

    # Define significance masks
    merged_df['significant_rna'] = merged_df["Adjusted P-value_rna"] < alpha
    merged_df['significant_atac'] = merged_df["Adjusted P-value_atac"] < alpha

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot different categories
    # Non-significant
    mask_non_sig = ~(merged_df['significant_atac'] | merged_df['significant_rna'])
    ax.scatter(
        merged_df.loc[mask_non_sig, 'Log2FC_rna'],
        merged_df.loc[mask_non_sig, 'Log2FC_atac'],
        color='lightgrey', label='Non-significant', s=30, alpha=0.5, rasterized=True
    )

    # Significant in RNA only
    mask_rna_only = ~merged_df['significant_atac'] & merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_rna_only, 'Log2FC_rna'],
        merged_df.loc[mask_rna_only, 'Log2FC_atac'],
        color='tab:cyan', label='Significant in RNA only', s=30, alpha=0.7, rasterized=True
    )

    # Significant in ATAC only
    mask_atac_only = merged_df['significant_atac'] & ~merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_atac_only, 'Log2FC_rna'],
        merged_df.loc[mask_atac_only, 'Log2FC_atac'],
        color='tab:purple', label='Significant in ATAC only', s=30, alpha=0.7, rasterized=True
    )

    # Significant in both
    mask_both = merged_df['significant_atac'] & merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_both, 'Log2FC_rna'],
        merged_df.loc[mask_both, 'Log2FC_atac'],
        color='tab:red', label='Significant in both', s=30, alpha=0.7, rasterized=True
    )

    # Annotate significant genes (selective labeling)
    print("Adding repeat element labels...")
    for _, row in merged_df.iterrows():
        # Label repeats significant in both
        if row['significant_atac'] and row['significant_rna']:
            if abs(row['Log2FC_rna']) > 1 or abs(row['Log2FC_atac']) > 1:
                ax.text(row['Log2FC_rna'], row['Log2FC_atac'], row['Gene'],
                        fontsize=10.5, alpha=0.7, ha='center')
        # Label extreme changes
        elif (row['significant_atac'] or row['significant_rna']) and \
             (abs(row['Log2FC_rna']) > 2 or abs(row['Log2FC_atac']) > 2):
            ax.text(row['Log2FC_rna'], row['Log2FC_atac'], row['Gene'],
                    fontsize=10.5, alpha=0.7, ha='center')

    # Calculate correlations for significant points
    sig_subset = merged_df[(merged_df['significant_atac']) | (merged_df['significant_rna'])]
    if len(sig_subset) >= 2:
        spearman_val, _ = spearmanr(sig_subset['Log2FC_rna'], sig_subset['Log2FC_atac'])
        pearson_val, _ = pearsonr(sig_subset['Log2FC_rna'], sig_subset['Log2FC_atac'])
        # Add correlation text inside plot (upper left corner) without box
        # Use non-breaking spaces (\u00A0) to ensure spacing is preserved in SVG
        correlation_text = f'Correlation across significant points (n={len(sig_subset)}):\n\u00A0\u00A0\u00A0Spearman ρ: {spearman_val:.2f}\n\u00A0\u00A0\u00A0Pearson r: {pearson_val:.2f}'
        ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes,
                fontsize=18, verticalalignment='top')
    else:
        ax.text(0.05, 0.95, 'Insufficient data for correlation', transform=ax.transAxes,
                fontsize=18, verticalalignment='top')

    # Set labels and title
    ax.set_xlabel('log2FC RNA', fontsize=24)
    ax.set_ylabel('log2FC ATAC', fontsize=24)
    ax.set_title(
        f"Trim28 TRM-like (n={n_trm_like}) vs NT other cell types (n={n_other})",
        fontsize=27,
        pad=20
    )

    # Add grid
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Remove legend from this plot (will be created separately)
    # ax.legend(loc='upper left', fontsize=10)

    # Save outputs
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    print("Saving figures...")
    png_path = os.path.join(output_dir, "g.png")
    svg_path = os.path.join(output_dir, "g.svg")

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

    # Print summary
    print(f"\nSummary:")
    print(f"  Total repeat elements analyzed: {len(merged_df)}")
    print(f"  Significant in ATAC only: {mask_atac_only.sum()}")
    print(f"  Significant in RNA only: {mask_rna_only.sum()}")
    print(f"  Significant in both: {mask_both.sum()}")
    print(f"  Non-significant: {mask_non_sig.sum()}")

    print("Figure 6g completed successfully!")

if __name__ == "__main__":
    main()

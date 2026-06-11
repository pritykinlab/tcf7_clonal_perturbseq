#!/usr/bin/env python
"""
Generate scatter plot of ATAC vs RNA log2FC for Control vs Trim28 in Progenitor cells (Figure 6e)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata
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
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading multiome RNA data...")
    multiome_rna_adata = get_adata('multiome_rna')

    print("Loading multiome ATAC data...")
    multiome_atac_adata = get_adata('multiome_atac')

    # Create nearest gene mapping
    nearest_gene_dict = multiome_atac_adata.var['gene'].to_dict()

    # ------------------------------------------------------------
    # DE test for ATAC - Trim28 vs NT in Progenitor cells
    # ------------------------------------------------------------
    print("Performing ATAC DE testing: Trim28 vs NT in Progenitor cells...")
    mask_Trim28 = (multiome_atac_adata.obs['condition'] == 'Trim28') & (multiome_atac_adata.obs['cell_type'] == 'Progenitor')
    mask_NT = (multiome_atac_adata.obs['condition'] == 'Control') & (multiome_atac_adata.obs['cell_type'] == 'Progenitor')

    CBC_group_Trim28_atac = multiome_atac_adata.obs_names[mask_Trim28].tolist()
    CBC_group_NT_atac = multiome_atac_adata.obs_names[mask_NT].tolist()
    n_trim28_atac = len(CBC_group_Trim28_atac)
    n_nt_atac = len(CBC_group_NT_atac)

    print(f"  Trim28 Progenitor cells (ATAC): {n_trim28_atac:,}")
    print(f"  NT Progenitor cells (ATAC): {n_nt_atac:,}")

    results_atac_df = DE_testing.wilcoxon_DE_test(
        multiome_atac_adata,
        CBC_group_Trim28_atac,
        CBC_group_NT_atac,
        norm_layer='norm'
    )
    results_atac_df['nearest_gene'] = results_atac_df['Gene'].apply(lambda e: nearest_gene_dict.get(e))

    # ------------------------------------------------------------
    # DE test for RNA - Trim28 vs NT in Progenitor cells
    # ------------------------------------------------------------
    print("Performing RNA DE testing: Trim28 vs NT in Progenitor cells...")
    mask_Trim28 = (multiome_rna_adata.obs['condition'] == 'Trim28') & (multiome_rna_adata.obs['cell_type'] == 'Progenitor')
    mask_NT = (multiome_rna_adata.obs['condition'] == 'Control') & (multiome_rna_adata.obs['cell_type'] == 'Progenitor')

    CBC_group_Trim28_rna = multiome_rna_adata.obs_names[mask_Trim28].tolist()
    CBC_group_NT_rna = multiome_rna_adata.obs_names[mask_NT].tolist()

    print(f"  Trim28 Progenitor cells (RNA): {len(CBC_group_Trim28_rna):,}")
    print(f"  NT Progenitor cells (RNA): {len(CBC_group_NT_rna):,}")

    results_rna_df = DE_testing.wilcoxon_DE_test(
        multiome_rna_adata,
        CBC_group_Trim28_rna,
        CBC_group_NT_rna,
        norm_layer='norm'
    )

    # ------------------------------------------------------------
    # Create scatter plot with mean aggregation
    # ------------------------------------------------------------
    print("Creating scatter plot with mean aggregation...")
    alpha = 0.05
    aggregate_func = 'mean'

    # Aggregate ATAC by gene (mean)
    atac_grouped = results_atac_df.groupby('nearest_gene').agg(
        mean_log2FC_atac=('Log2FC', 'mean'),
        significant_atac=('Adjusted P-value', lambda x: (x < alpha).any())
    ).reset_index()

    # Mark significant RNA genes
    results_rna_df['significant_rna'] = results_rna_df['Adjusted P-value'] < alpha

    # Merge datasets
    merged_df = pd.merge(
        atac_grouped,
        results_rna_df,
        left_on='nearest_gene',
        right_on='Gene'
    )
    merged_df.rename(
        columns={
            'Log2FC': 'log2FC_rna',
            'Adjusted P-value': 'Adjusted P-value RNA'
        },
        inplace=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot different categories
    # Non-significant
    mask_non_sig = ~(merged_df['significant_atac'] | merged_df['significant_rna'])
    ax.scatter(
        merged_df.loc[mask_non_sig, 'log2FC_rna'],
        merged_df.loc[mask_non_sig, 'mean_log2FC_atac'],
        color='lightgrey', label='Non-significant', s=30, alpha=0.5, rasterized=True
    )

    # Significant in RNA only
    mask_rna_only = ~merged_df['significant_atac'] & merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_rna_only, 'log2FC_rna'],
        merged_df.loc[mask_rna_only, 'mean_log2FC_atac'],
        color='tab:cyan', label='Significant in RNA only', s=30, alpha=0.7, rasterized=True
    )

    # Significant in ATAC only
    mask_atac_only = merged_df['significant_atac'] & ~merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_atac_only, 'log2FC_rna'],
        merged_df.loc[mask_atac_only, 'mean_log2FC_atac'],
        color='tab:purple', label='Significant in ATAC only', s=30, alpha=0.7, rasterized=True
    )

    # Significant in both
    mask_both = merged_df['significant_atac'] & merged_df['significant_rna']
    ax.scatter(
        merged_df.loc[mask_both, 'log2FC_rna'],
        merged_df.loc[mask_both, 'mean_log2FC_atac'],
        color='tab:red', label='Significant in both', s=30, alpha=0.7, rasterized=True
    )

    # Annotate significant genes (optional - selective labeling)
    print("Adding gene labels...")
    for _, row in merged_df.iterrows():
        # Label genes significant in both
        if row['significant_atac'] and row['significant_rna']:
            if abs(row['log2FC_rna']) > 1 or abs(row['mean_log2FC_atac']) > 0.3:
                ax.text(row['log2FC_rna'], row['mean_log2FC_atac'], row['nearest_gene'],
                       fontsize=10.5, alpha=0.7, ha='center')
        # Label genes with strong ATAC changes
        elif row['significant_atac'] and abs(row['mean_log2FC_atac']) > 0.5:
            ax.text(row['log2FC_rna'], row['mean_log2FC_atac'], row['nearest_gene'],
                   fontsize=10.5, alpha=0.7, ha='center')
        # Label extreme changes
        elif (row['significant_atac'] or row['significant_rna']) and \
             (abs(row['log2FC_rna']) > 2 or abs(row['mean_log2FC_atac']) > 1):
            ax.text(row['log2FC_rna'], row['mean_log2FC_atac'], row['nearest_gene'],
                   fontsize=10.5, alpha=0.7, ha='center')

    # Calculate correlations for significant points
    sig_subset = merged_df[(merged_df['significant_atac']) | (merged_df['significant_rna'])]
    if len(sig_subset) >= 2:
        spearman_val, _ = spearmanr(sig_subset['log2FC_rna'], sig_subset['mean_log2FC_atac'])
        pearson_val, _ = pearsonr(sig_subset['log2FC_rna'], sig_subset['mean_log2FC_atac'])
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
    ax.set_ylabel('Mean log2FC ATAC', fontsize=24)
    ax.set_title(
        f"NT progenitor (n={n_nt_atac}) vs Trim28 progenitor (n={n_trim28_atac})",
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
    print("Saving figures...")
    png_path = os.path.join(output_dir, "e.png")
    svg_path = os.path.join(output_dir, "e.svg")

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

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total genes analyzed: {len(merged_df)}")
    print(f"  Significant in ATAC only: {mask_atac_only.sum()}")
    print(f"  Significant in RNA only: {mask_rna_only.sum()}")
    print(f"  Significant in both: {mask_both.sum()}")
    print(f"  Non-significant: {mask_non_sig.sum()}")

    print("Figure 6e completed successfully!")

if __name__ == "__main__":
    main()

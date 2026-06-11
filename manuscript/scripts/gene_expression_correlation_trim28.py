#!/usr/bin/env python
"""
Generate gene expression correlation plot between Trim28 vs NT in both experiments (legacy internal Sup_Figure_5_1_old)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_orders_gene10, cell_type_orders_trim28_setdb1
import DE_testing
from importlib import reload

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 10)

def run_de_tests(adata, condition='Both', comparison='Trim28 vs NT'):
    """Run differential expression test for Trim28 vs NT"""
    adata = adata[~adata.obs['cell_type'].str.contains('Lymph')].copy()

    # Trim28 vs NT in all cells
    cbc_group1 = adata[adata.obs['perturbation'] == 'Trim28'].obs_names
    cbc_group2 = adata[adata.obs['perturbation'] == 'NT'].obs_names

    results_df = DE_testing.wilcoxon_DE_test(adata, cbc_group1, cbc_group2)
    return results_df.set_index('Gene')

def plot_gene_expression_correlation(gene_10_results, trim28_setdb1_results,
                                    xlim=(-3, 5.5), ylim=(-3, 5.5), sig_thresh=0.1):
    """Plot correlation between gene expression in both experiments"""

    # Find shared genes
    shared_genes = trim28_setdb1_results.index.intersection(gene_10_results.index)

    # Identify significant genes
    significant_genes = [
        gene for gene in shared_genes
        if (trim28_setdb1_results.loc[gene, 'Adjusted P-value'] <= sig_thresh and
            gene_10_results.loc[gene, 'Adjusted P-value'] <= sig_thresh)
    ]

    # Prepare data for plotting
    plot_data = []
    de_genes_list = []

    # Add significant genes
    for gene in significant_genes:
        logFC_trim = trim28_setdb1_results.loc[gene, 'Log2FC']
        logFC_gene_10 = gene_10_results.loc[gene, 'Log2FC']

        # Label genes with logFC > 1 in both experiments
        if logFC_trim > 1 and logFC_gene_10 > 1:
            de_genes_list.append(gene)

        plot_data.append({
            'Gene': gene,
            'logFC_Trim28_Setdb1': logFC_trim,
            'logFC_10_gene': logFC_gene_10,
            'Significant': True
        })

    # Add non-significant genes
    non_significant_genes = [gene for gene in shared_genes if gene not in significant_genes]
    for gene in non_significant_genes:
        logFC_trim = trim28_setdb1_results.loc[gene, 'Log2FC']
        logFC_gene_10 = gene_10_results.loc[gene, 'Log2FC']
        plot_data.append({
            'Gene': gene,
            'logFC_Trim28_Setdb1': logFC_trim,
            'logFC_10_gene': logFC_gene_10,
            'Significant': False
        })

    # Create DataFrame
    df_plot = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot non-significant points in light grey
    ax.scatter(df_plot[~df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[~df_plot['Significant']]['logFC_10_gene'],
               color='lightgrey', label='Non-significant', s=50, alpha=0.7, rasterized=True)

    # Plot significant points in blue
    ax.scatter(df_plot[df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[df_plot['Significant']]['logFC_10_gene'],
               label='Significant', s=50, alpha=0.7, rasterized=True)

    # Add axis labels
    ax.set_xlabel('log2FC - Trim28 and Setdb1 experiment', fontsize=25)
    ax.set_ylabel('log2FC - 10 gene experiment', fontsize=25)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add reference lines at logFC = 1
    ax.plot([1, xlim[1]], [1, 1], color='red', linestyle='--', alpha=0.5)
    ax.plot([1, 1], [1, ylim[1]], color='red', linestyle='--', alpha=0.5)

    # Calculate correlations
    pearson_corr, _ = pearsonr(df_plot['logFC_Trim28_Setdb1'], df_plot['logFC_10_gene'])
    spearman_corr, _ = spearmanr(df_plot['logFC_Trim28_Setdb1'], df_plot['logFC_10_gene'])

    # Add title with comparison description
    ax.set_title('Trim28 perturbed cells vs NT cells', fontsize=25, pad=20)

    # Add correlation text inside plot (no box)
    correlation_text = f'Spearman ρ: {spearman_corr:.2f}\nPearson r: {pearson_corr:.2f}'
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes,
            fontsize=20, verticalalignment='top')

    # Label genes with logFC > 1 in both experiments
    for _, row in df_plot[df_plot['Gene'].isin(de_genes_list)].iterrows():
        ax.text(row['logFC_Trim28_Setdb1'], row['logFC_10_gene'], row['Gene'],
                horizontalalignment='left', size=12, color='black')

    # Add legend
    ax.legend(loc='lower right', fontsize=15)

    # Keep all spines visible for box frame
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    return fig, de_genes_list

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Sup_Figure_5_1_old"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading gene_10 dataset...")
    gene10_adata = get_adata('gene_10', use_cells_only_one_construct=True, remove_NG=True)

    print("Loading trim28_setdb1 dataset...")
    trim28_setdb1_adata = get_adata('trim28_setdb1', use_cells_only_one_construct=False,
                                    remove_NG=True, remove_lymph_node=True)

    print("Running DE analysis for gene_10 dataset...")
    results_gene_10 = run_de_tests(gene10_adata)

    print("Running DE analysis for trim28_setdb1 dataset...")
    results_trim28_setdb1 = run_de_tests(trim28_setdb1_adata)

    print("Creating correlation plot...")
    fig, de_genes = plot_gene_expression_correlation(
        results_gene_10,
        results_trim28_setdb1,
        xlim=(-3, 5.5),
        ylim=(-3, 5.5),
        sig_thresh=0.1  # Using higher threshold as in notebook
    )

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "1.png")
    svg_path = os.path.join(output_dir, "1.svg")

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
    print("legacy internal Sup_Figure_5_1_old completed successfully!")

    # Print summary
    print(f"\nSummary:")
    print(f"- Shared genes analyzed: {len(results_gene_10.index.intersection(results_trim28_setdb1.index))}")
    print(f"- DE genes labeled (logFC > 1 in both): {len(de_genes)}")
    if de_genes:
        print(f"- Top DE genes: {', '.join(de_genes[:5])}")

if __name__ == "__main__":
    reload(DE_testing)
    main()

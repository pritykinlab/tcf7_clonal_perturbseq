#!/usr/bin/env python
"""
Generate gene expression correlation plot between Setdb1 vs NT in both experiments (legacy internal Setdb1 correlation panel)
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

from figure_format import get_adata
import DE_testing
from importlib import reload

# Set figure parameters (matched to Figure 5g styling)
plt.rcParams["axes.titlesize"] = 32.5
plt.rcParams["axes.labelsize"] = 32.5
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26
plt.rcParams["lines.markersize"] = 6.5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 10)


def run_de_tests(adata, exclude_trm_like_setdb1=False):
    """Run differential expression test for Setdb1 vs NT (cells outside lymph nodes).

    If exclude_trm_like_setdb1 is True, removes Setdb1 cells labeled as TRM-like
    before performing DE (legacy internal Setdb1 correlation panel v2).
    """
    adata = adata[~adata.obs['cell_type'].str.contains('Lymph')].copy()

    if exclude_trm_like_setdb1:
        remove_mask = (
            (adata.obs['perturbation'] == 'Setdb1')
            & (adata.obs['cell_type'] == 'TRM-like')
        )
        adata = adata[~remove_mask].copy()

    setdb1_cells = adata[adata.obs['perturbation'] == 'Setdb1'].obs_names
    nt_cells = adata[adata.obs['perturbation'] == 'NT'].obs_names

    results_df = DE_testing.wilcoxon_DE_test(adata, setdb1_cells, nt_cells)
    return results_df.set_index('Gene')


def plot_gene_expression_correlation(gene_10_results, trim28_setdb1_results,
                                     sig_thresh=0.05, title=None):
    """Plot correlation of Setdb1 vs NT log2FC values between experiments.

    Returns:
        fig, de_genes_list, df_plot, corr_dict
    where corr_dict contains Pearson/Spearman for all, significant, and non-significant genes.
    """

    # Find shared genes
    shared_genes = trim28_setdb1_results.index.intersection(gene_10_results.index)

    # Identify significant genes
    significant_genes = [
        gene for gene in shared_genes
        if (trim28_setdb1_results.loc[gene, 'Adjusted P-value'] <= sig_thresh and
            gene_10_results.loc[gene, 'Adjusted P-value'] <= sig_thresh)
    ]

    plot_data = []
    de_genes_list = []

    for gene in significant_genes:
        logfc_trim_setdb1 = trim28_setdb1_results.loc[gene, 'Log2FC']
        logfc_gene10 = gene_10_results.loc[gene, 'Log2FC']

        if logfc_trim_setdb1 > 1 and logfc_gene10 > 1:
            de_genes_list.append(gene)

        plot_data.append({
            'Gene': gene,
            'logFC_Trim28_Setdb1': logfc_trim_setdb1,
            'logFC_10_gene': logfc_gene10,
            'Significant': True
        })

    non_significant_genes = [gene for gene in shared_genes if gene not in significant_genes]
    for gene in non_significant_genes:
        plot_data.append({
            'Gene': gene,
            'logFC_Trim28_Setdb1': trim28_setdb1_results.loc[gene, 'Log2FC'],
            'logFC_10_gene': gene_10_results.loc[gene, 'Log2FC'],
            'Significant': False
        })

    df_plot = pd.DataFrame(plot_data)

    # Dynamically set axes to capture the full range with padding
    max_range = np.max(np.abs(df_plot[['logFC_Trim28_Setdb1', 'logFC_10_gene']].values)) + 0.5
    xlim = (-max_range, max_range)
    ylim = (-max_range, max_range)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(df_plot[~df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[~df_plot['Significant']]['logFC_10_gene'],
               color='lightgrey', label='Non-significant', s=50, alpha=0.7, rasterized=True)

    ax.scatter(df_plot[df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[df_plot['Significant']]['logFC_10_gene'],
               label='Significant', s=50, alpha=0.7, rasterized=True)

    ax.set_xlabel('log2FC (Setdb1 vs NT) - Trim28/Setdb1 experiment', fontsize=32.5)
    ax.set_ylabel('log2FC (Setdb1 vs NT) - 10 gene experiment', fontsize=32.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Reference lines at logFC = 1
    ax.plot([1, xlim[1]], [1, 1], color='red', linestyle='--', alpha=0.5)
    ax.plot([1, 1], [1, ylim[1]], color='red', linestyle='--', alpha=0.5)

    def _corr(x, y):
        if len(x) < 2:
            return (np.nan, np.nan)
        return pearsonr(x, y)[0], spearmanr(x, y)[0]

    all_pearson, all_spearman = _corr(df_plot['logFC_Trim28_Setdb1'], df_plot['logFC_10_gene'])
    sig_df = df_plot[df_plot['Significant']]
    nonsig_df = df_plot[~df_plot['Significant']]
    sig_pearson, sig_spearman = _corr(sig_df['logFC_Trim28_Setdb1'], sig_df['logFC_10_gene'])
    nonsig_pearson, nonsig_spearman = _corr(nonsig_df['logFC_Trim28_Setdb1'], nonsig_df['logFC_10_gene'])

    if title is None:
        title = 'Setdb1 perturbed cells vs NT cells'
    ax.set_title(title, fontsize=32.5, pad=20)

    correlation_text = (
        f'All: Spearman ρ {all_spearman:.2f}, Pearson r {all_pearson:.2f}\n'
        f'Sig: Spearman ρ {sig_spearman:.2f}, Pearson r {sig_pearson:.2f}\n'
        f'Non-sig: Spearman ρ {nonsig_spearman:.2f}, Pearson r {nonsig_pearson:.2f}'
    )
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes,
            fontsize=22, verticalalignment='top')

    for _, row in df_plot[df_plot['Gene'].isin(de_genes_list)].iterrows():
        ax.text(row['logFC_Trim28_Setdb1'], row['logFC_10_gene'], row['Gene'],
                horizontalalignment='left', size=16, color='black')

    ax.legend(loc='lower right', fontsize=19.5)

    # Show full frame
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

    corr_dict = {
        "all": {"pearson": all_pearson, "spearman": all_spearman, "n": len(df_plot)},
        "significant": {"pearson": sig_pearson, "spearman": sig_spearman, "n": len(sig_df)},
        "non_significant": {"pearson": nonsig_pearson, "spearman": nonsig_spearman, "n": len(nonsig_df)},
    }

    return fig, de_genes_list, df_plot, corr_dict


def main():
    output_dir = "../manuscript/plots/Sup_Figure_5_2_old"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading gene_10 dataset...")
    gene10_adata = get_adata('gene_10', use_cells_only_one_construct=True, remove_NG=True)

    print("Loading trim28_setdb1 dataset...")
    trim28_setdb1_adata = get_adata('trim28_setdb1', use_cells_only_one_construct=False,
                                    remove_NG=True, remove_lymph_node=True)

    print("Running DE analysis for gene_10 dataset (Setdb1 vs NT)...")
    results_gene_10 = run_de_tests(gene10_adata)

    print("Running DE analysis for trim28_setdb1 dataset (Setdb1 vs NT)...")
    results_trim28_setdb1 = run_de_tests(trim28_setdb1_adata)

    print("Creating correlation plot...")
    fig, de_genes, df_plot, corr_v1 = plot_gene_expression_correlation(
        results_gene_10,
        results_trim28_setdb1,
        sig_thresh=0.05,
        title='Setdb1 perturbed cells vs NT cells'
    )

    png_path = os.path.join(output_dir, "2.png")
    svg_path = os.path.join(output_dir, "2.svg")

    print("Saving figures...")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)
    print("legacy internal Setdb1 correlation panel completed successfully!")

    # v2: remove TRM-like Setdb1 cells before DE in both datasets
    print("\nRunning DE analysis excluding TRM-like Setdb1 cells (v2)...")
    results_gene_10_v2 = run_de_tests(gene10_adata, exclude_trm_like_setdb1=True)
    results_trim28_setdb1_v2 = run_de_tests(trim28_setdb1_adata, exclude_trm_like_setdb1=True)

    print("Creating correlation plot (v2)...")
    fig_v2, de_genes_v2, df_plot_v2, corr_v2 = plot_gene_expression_correlation(
        results_gene_10_v2,
        results_trim28_setdb1_v2,
        sig_thresh=0.05,
        title='Setdb1 vs NT (TRM-like Setdb1 cells removed)'
    )

    png_path_v2 = os.path.join(output_dir, "2_v2.png")
    svg_path_v2 = os.path.join(output_dir, "2_v2.svg")

    print("Saving figures (v2)...")
    fig_v2.savefig(png_path_v2, dpi=300, bbox_inches='tight')
    fig_v2.savefig(svg_path_v2, format='svg', bbox_inches='tight')

    png_size_v2 = os.path.getsize(png_path_v2) / 1024 / 1024  # MB
    svg_size_v2 = os.path.getsize(svg_path_v2) / 1024  # KB

    if png_size_v2 > 5:
        print(f"ERROR: PNG {png_size_v2:.1f}MB exceeds 5MB limit!")
    if svg_size_v2 > 500:
        print(f"ERROR: SVG {svg_size_v2:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v2:.2f}MB, SVG {svg_size_v2:.1f}KB")

    plt.close(fig_v2)
    print("legacy internal Setdb1 correlation panel v2 completed successfully!")

    # v3: progenitor-only (Progenitor + Progenitor Cycling)
    print("\nPreparing progenitor-only comparison (v3)...")
    progenitor_types = {"Progenitor", "Progenitor Cycling"}

    gene10_prog = gene10_adata[gene10_adata.obs["cell_type"].isin(progenitor_types)].copy()
    trimset_prog = trim28_setdb1_adata[trim28_setdb1_adata.obs["cell_type"].isin(progenitor_types)].copy()

    print(f"gene_10 progenitor cells: {gene10_prog.n_obs}")
    print(f"trim28_setdb1 progenitor cells: {trimset_prog.n_obs}")

    print("Running DE in progenitors (Setdb1 vs NT)...")
    results_gene_10_prog = run_de_tests(gene10_prog)
    results_trimset_prog = run_de_tests(trimset_prog)

    print("Creating correlation plot (v3)...")
    fig_v3, de_genes_v3, df_plot_v3, corr_v3 = plot_gene_expression_correlation(
        results_gene_10_prog,
        results_trimset_prog,
        sig_thresh=0.05,
        title="Setdb1 vs NT in progenitors"
    )

    png_path_v3 = os.path.join(output_dir, "2_v3.png")
    svg_path_v3 = os.path.join(output_dir, "2_v3.svg")

    print("Saving figures (v3)...")
    fig_v3.savefig(png_path_v3, dpi=300, bbox_inches='tight')
    fig_v3.savefig(svg_path_v3, format='svg', bbox_inches='tight')

    png_size_v3 = os.path.getsize(png_path_v3) / 1024 / 1024  # MB
    svg_size_v3 = os.path.getsize(svg_path_v3) / 1024  # KB

    if png_size_v3 > 5:
        print(f"ERROR: PNG {png_size_v3:.1f}MB exceeds 5MB limit!")
    if svg_size_v3 > 500:
        print(f"ERROR: SVG {svg_size_v3:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v3:.2f}MB, SVG {svg_size_v3:.1f}KB")

    plt.close(fig_v3)
    print("legacy internal Setdb1 correlation panel v3 completed successfully!")

    print("\nSummary:")
    print(f"- Shared genes analyzed (v1): {len(df_plot)}")
    print(f"- DE genes labeled (log2FC > 1 in both, v1): {len(de_genes)}")
    print(f"- Shared genes analyzed (v2): {len(df_plot_v2)}")
    print(f"- DE genes labeled (log2FC > 1 in both, v2): {len(de_genes_v2)}")
    print(f"- Shared genes analyzed (v3 progenitors): {len(df_plot_v3)}")
    print(f"- DE genes labeled (log2FC > 1 in both, v3 progenitors): {len(de_genes_v3)}")
    print("\nCorrelations:")
    for label, corr in [("v1 all cells", corr_v1), ("v2 TRM-like removed", corr_v2), ("v3 progenitors", corr_v3)]:
        print(f"{label}:")
        print(f"  all (n={corr['all']['n']}): pearson {corr['all']['pearson']:.3f}, spearman {corr['all']['spearman']:.3f}")
        print(f"  significant (n={corr['significant']['n']}): pearson {corr['significant']['pearson']:.3f}, spearman {corr['significant']['spearman']:.3f}")
        print(f"  non-significant (n={corr['non_significant']['n']}): pearson {corr['non_significant']['pearson']:.3f}, spearman {corr['non_significant']['spearman']:.3f}")
    if de_genes:
        print(f"- Top DE genes (v1): {', '.join(de_genes[:5])}")
    if de_genes_v2:
        print(f"- Top DE genes (v2): {', '.join(de_genes_v2[:5])}")
    if de_genes_v3:
        print(f"- Top DE genes (v3): {', '.join(de_genes_v3[:5])}")


if __name__ == "__main__":
    reload(DE_testing)
    main()

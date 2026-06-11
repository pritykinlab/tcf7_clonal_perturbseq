#!/usr/bin/env python
"""
Per-guide log2FC correlation plots: Trim28 vs NT (x) vs Setdb1 guide vs NT (y) for the trim28_setdb1 experiment.
Outputs one plot per Setdb1 guide under Sup_Figure_5_8_old/8_trim28_setdb1.
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

# Styling matched to Sup Figure 5.8 tweaks
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26
plt.rcParams["lines.markersize"] = 6.5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 10)

GUIDES = [
    "Setdb1-1", "Setdb1-2", "Setdb1-3", "Setdb1-4", "Setdb1-5",
    "Setdb1-6", "Setdb1-7", "Setdb1-8", "Setdb1-9", "Setdb1-10"
]
SIG_THRESH = 0.1


def run_de_vs_nt(adata, mask):
    """Run DE (mask vs NT) using Wilcoxon test."""
    group1_cells = adata.obs_names[mask]
    nt_cells = adata.obs_names[adata.obs['perturbation'] == 'NT']
    if len(group1_cells) == 0 or len(nt_cells) == 0:
        raise ValueError("Empty group for DE testing.")

    results_df = DE_testing.wilcoxon_DE_test(adata, group1_cells, nt_cells)
    return results_df.set_index('Gene')


def plot_correlation(trim28_results, guide_results, guide_name):
    """Scatter and correlations for one guide."""
    shared_genes = guide_results.index.intersection(trim28_results.index)

    plot_data = []
    labeled_genes = []
    for gene in shared_genes:
        logfc_trim28 = trim28_results.loc[gene, 'Log2FC']
        logfc_guide = guide_results.loc[gene, 'Log2FC']
        significant = (
            trim28_results.loc[gene, 'Adjusted P-value'] <= SIG_THRESH and
            guide_results.loc[gene, 'Adjusted P-value'] <= SIG_THRESH
        )
        if significant and (abs(logfc_trim28) > 1 or abs(logfc_guide) > 1):
            labeled_genes.append(gene)

        plot_data.append({
            'Gene': gene,
            'logFC_Trim28': logfc_trim28,
            'logFC_Guide': logfc_guide,
            'Significant': significant
        })

    df_plot = pd.DataFrame(plot_data)

    # Determine axis limits with padding
    max_range = np.max(np.abs(df_plot[['logFC_Trim28', 'logFC_Guide']].values)) + 0.5
    xlim = (-max_range, max_range)
    ylim = (-max_range, max_range)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(df_plot[~df_plot['Significant']]['logFC_Trim28'],
               df_plot[~df_plot['Significant']]['logFC_Guide'],
               color='lightgrey', label='Non-significant', s=50, alpha=0.7, rasterized=True)

    ax.scatter(df_plot[df_plot['Significant']]['logFC_Trim28'],
               df_plot[df_plot['Significant']]['logFC_Guide'],
               label='Significant', s=50, alpha=0.7, rasterized=True)

    ax.set_xlabel('log2FC (Trim28 vs NT) - Trim28/Setdb1 experiment', fontsize=28)
    ax.set_ylabel(f'log2FC ({guide_name} vs NT) - Trim28/Setdb1 experiment', fontsize=28)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Correlations on all genes
    pearson_corr, _ = pearsonr(df_plot['logFC_Trim28'], df_plot['logFC_Guide'])
    spearman_corr, _ = spearmanr(df_plot['logFC_Trim28'], df_plot['logFC_Guide'])
    corr_label = f"Spearman ρ: {spearman_corr:.2f} | Pearson r: {pearson_corr:.2f} (all genes)"

    ax.text(0.05, 0.95, corr_label.replace('|', '\n'), transform=ax.transAxes,
            fontsize=26, verticalalignment='top')
    ax.set_title(f'{guide_name} (correlation on all genes)', fontsize=28, pad=14)

    for _, row in df_plot[df_plot['Gene'].isin(labeled_genes)].iterrows():
        ax.text(row['logFC_Trim28'], row['logFC_Guide'], row['Gene'],
                horizontalalignment='left', size=16, color='black')

    ax.legend(loc='lower right', fontsize=19.5)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

    return fig, df_plot, labeled_genes


def save_figure(fig, png_path, svg_path):
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


def main():
    output_dir = "../manuscript/plots/Sup_Figure_5_8_old/8_trim28_setdb1"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading trim28_setdb1 dataset...")
    adata = get_adata('trim28_setdb1', use_cells_only_one_construct=True,
                      remove_NG=True, remove_lymph_node=True)

    print("Running DE for Trim28 vs NT...")
    trim28_results = run_de_vs_nt(adata, adata.obs['perturbation'] == 'Trim28')

    summaries = []

    for guide in GUIDES:
        print(f"\nProcessing {guide}...")
        guide_mask = (adata.obs['guide'] == guide) & (adata.obs['perturbation'] == 'Setdb1')
        if guide_mask.sum() == 0:
            print(f"  No cells for {guide}; skipping.")
            continue

        guide_results = run_de_vs_nt(adata, guide_mask)

        fig, df_plot, labeled_genes = plot_correlation(trim28_results, guide_results, guide)

        png_path = os.path.join(output_dir, f"{guide}.png")
        svg_path = os.path.join(output_dir, f"{guide}.svg")
        save_figure(fig, png_path, svg_path)
        plt.close(fig)

        summaries.append((guide, len(df_plot[df_plot['Significant']]),
                          len(labeled_genes), labeled_genes[:5]))

    print("\nlegacy internal Sup_Figure_5_8 (per-guide) completed successfully!")
    print("Summary (guide, significant genes, labeled genes):")
    for guide, sig_n, label_n, top_labels in summaries:
        print(f"- {guide}: sig {sig_n}, labeled {label_n}, top labels: {', '.join(top_labels) if top_labels else 'none'}")


if __name__ == "__main__":
    reload(DE_testing)
    main()

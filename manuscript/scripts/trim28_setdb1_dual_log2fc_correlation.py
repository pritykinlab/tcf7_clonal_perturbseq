#!/usr/bin/env python
"""
Generate log2FC correlation plots comparing Trim28 vs NT (x-axis) and Setdb1 vs NT (y-axis).
The combined two-panel output is used as the Figure S11g source panel; individual panels are retained as legacy internal outputs.
"""

import argparse
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
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26
plt.rcParams["lines.markersize"] = 6.5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 10)

OUT_DIR = "../manuscript/plots/Sup_Figure_5_8_old"
DEFAULT_GENE10_PNG = os.path.join(OUT_DIR, "8_gene10.png")
DEFAULT_GENE10_SVG = os.path.join(OUT_DIR, "8_gene10.svg")
DEFAULT_TRIM28_SETDB1_PNG = os.path.join(OUT_DIR, "8_trim28_setdb1.png")
DEFAULT_TRIM28_SETDB1_SVG = os.path.join(OUT_DIR, "8_trim28_setdb1.svg")

def run_de_vs_nt(adata, perturbation):
    """Run DE (perturbation vs NT) using Wilcoxon test, excluding lymph node cells."""
    adata = adata[~adata.obs['cell_type'].str.contains('Lymph')].copy()
    perturb_cells = adata[adata.obs['perturbation'] == perturbation].obs_names
    nt_cells = adata[adata.obs['perturbation'] == 'NT'].obs_names

    results_df = DE_testing.wilcoxon_DE_test(adata, perturb_cells, nt_cells)
    return results_df.set_index('Gene')


def plot_log2fc_correlation(
    trim28_results,
    setdb1_results,
    dataset_label,
    sig_thresh=0.1,
    ax=None,
    show_legend=True,
):
    """Scatter of Trim28 vs NT log2FC (x) against Setdb1 vs NT log2FC (y) for one dataset."""
    shared_genes = trim28_results.index.intersection(setdb1_results.index)

    plot_data = []
    labeled_genes = []
    for gene in shared_genes:
        logfc_trim28 = trim28_results.loc[gene, 'Log2FC']
        logfc_setdb1 = setdb1_results.loc[gene, 'Log2FC']
        significant = (
            trim28_results.loc[gene, 'Adjusted P-value'] <= sig_thresh and
            setdb1_results.loc[gene, 'Adjusted P-value'] <= sig_thresh
        )
        if significant and (abs(logfc_trim28) > 1 or abs(logfc_setdb1) > 1):
            labeled_genes.append(gene)

        plot_data.append({
            'Gene': gene,
            'logFC_Trim28': logfc_trim28,
            'logFC_Setdb1': logfc_setdb1,
            'Significant': significant
        })

    df_plot = pd.DataFrame(plot_data)

    # Determine axis limits with padding
    max_range = np.max(np.abs(df_plot[['logFC_Trim28', 'logFC_Setdb1']].values)) + 0.5
    xlim = (-max_range, max_range)
    ylim = (-max_range, max_range)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Non-significant genes
    ax.scatter(df_plot[~df_plot['Significant']]['logFC_Trim28'],
               df_plot[~df_plot['Significant']]['logFC_Setdb1'],
               color='lightgrey', label='Non-significant', s=50, alpha=0.7, rasterized=True)

    # Significant genes
    ax.scatter(df_plot[df_plot['Significant']]['logFC_Trim28'],
               df_plot[df_plot['Significant']]['logFC_Setdb1'],
               label='Significant', s=50, alpha=0.7, rasterized=True)

    ax.set_xlabel(f'log2FC (Trim28 vs NT)\n{dataset_label}', fontsize=28)
    ax.set_ylabel(f'log2FC (Setdb1 vs NT)\n{dataset_label}', fontsize=28)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Correlations (all genes)
    pearson_corr, _ = pearsonr(df_plot['logFC_Trim28'], df_plot['logFC_Setdb1'])
    spearman_corr, _ = spearmanr(df_plot['logFC_Trim28'], df_plot['logFC_Setdb1'])
    correlation_text = (
        "Correlation (across all genes):\n"
        f"Spearman ρ: {spearman_corr:.2f}\n"
        f"Pearson r: {pearson_corr:.2f}"
    )
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes,
            fontsize=26, verticalalignment='top')

    # Label genes strongly upregulated in both
    for _, row in df_plot[df_plot['Gene'].isin(labeled_genes)].iterrows():
        ax.text(row['logFC_Trim28'], row['logFC_Setdb1'], row['Gene'],
                horizontalalignment='left', size=16, color='black')

    if show_legend:
        ax.legend(loc='lower right', fontsize=19.5)

    # Show full frame
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)

    return fig, labeled_genes, df_plot


def save_figure(fig, png_path, svg_path):
    """Save PNG/SVG and report sizes."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Trim28-vs-Setdb1 log2FC correlation plots within each experiment, "
            "optionally including a combined two-panel figure."
        )
    )
    parser.add_argument("--output-dir", default=OUT_DIR, help="Output directory for standalone legacy plots.")
    parser.add_argument("--gene10-png", default=DEFAULT_GENE10_PNG, help="Output PNG for standalone gene_10 panel.")
    parser.add_argument("--gene10-svg", default=DEFAULT_GENE10_SVG, help="Output SVG for standalone gene_10 panel.")
    parser.add_argument(
        "--trim28-setdb1-png",
        default=DEFAULT_TRIM28_SETDB1_PNG,
        help="Output PNG for standalone Trim28/Setdb1 panel.",
    )
    parser.add_argument(
        "--trim28-setdb1-svg",
        default=DEFAULT_TRIM28_SETDB1_SVG,
        help="Output SVG for standalone Trim28/Setdb1 panel.",
    )
    parser.add_argument("--combined-png", default=None, help="Optional output PNG path for a combined two-panel figure.")
    parser.add_argument("--combined-svg", default=None, help="Optional output SVG path for a combined two-panel figure.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading gene_10 dataset...")
    gene10_adata = get_adata('gene_10', use_cells_only_one_construct=True, remove_NG=True)

    print("Running DE for gene_10 (Trim28 vs NT)...")
    gene10_trim28 = run_de_vs_nt(gene10_adata, 'Trim28')
    print("Running DE for gene_10 (Setdb1 vs NT)...")
    gene10_setdb1 = run_de_vs_nt(gene10_adata, 'Setdb1')

    print("Creating correlation plot for gene_10 dataset...")
    gene10_fig, gene10_labeled, gene10_df = plot_log2fc_correlation(
        gene10_trim28, gene10_setdb1, dataset_label="10 gene experiment", sig_thresh=0.1
    )
    save_figure(gene10_fig, args.gene10_png, args.gene10_svg)
    plt.close(gene10_fig)

    print("Loading trim28_setdb1 dataset...")
    trim28_setdb1_adata = get_adata('trim28_setdb1', use_cells_only_one_construct=False,
                                    remove_NG=True, remove_lymph_node=True)

    print("Running DE for trim28_setdb1 (Trim28 vs NT)...")
    trim28_setdb1_trim28 = run_de_vs_nt(trim28_setdb1_adata, 'Trim28')
    print("Running DE for trim28_setdb1 (Setdb1 vs NT)...")
    trim28_setdb1_setdb1 = run_de_vs_nt(trim28_setdb1_adata, 'Setdb1')

    print("Creating correlation plot for trim28_setdb1 dataset...")
    trim_fig, trim_labeled, trim_df = plot_log2fc_correlation(
        trim28_setdb1_trim28, trim28_setdb1_setdb1,
        dataset_label="Trim28/Setdb1 experiment", sig_thresh=0.1
    )
    save_figure(trim_fig, args.trim28_setdb1_png, args.trim28_setdb1_svg)
    plt.close(trim_fig)

    if args.combined_png and args.combined_svg:
        print("Creating combined two-panel figure...")
        combined_fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
        plot_log2fc_correlation(
            gene10_trim28,
            gene10_setdb1,
            dataset_label="10 gene experiment",
            sig_thresh=0.1,
            ax=axes[0],
            show_legend=False,
        )
        plot_log2fc_correlation(
            trim28_setdb1_trim28,
            trim28_setdb1_setdb1,
            dataset_label="Trim28/Setdb1 experiment",
            sig_thresh=0.1,
            ax=axes[1],
            show_legend=True,
        )
        combined_fig.tight_layout()
        save_figure(combined_fig, args.combined_png, args.combined_svg)
        plt.close(combined_fig)

    print("Figure S11g source correlation panel completed successfully!")

    print("\nSummary:")
    print(f"- gene_10 shared genes: {len(gene10_df)}; labeled (|log2FC| > 1 in either & sig): {len(gene10_labeled)}")
    if gene10_labeled:
        print(f"- gene_10 top labeled genes: {', '.join(gene10_labeled[:5])}")
    print(f"- trim28_setdb1 shared genes: {len(trim_df)}; labeled (|log2FC| > 1 in either & sig): {len(trim_labeled)}")
    if trim_labeled:
        print(f"- trim28_setdb1 top labeled genes: {', '.join(trim_labeled[:5])}")


if __name__ == "__main__":
    reload(DE_testing)
    main()

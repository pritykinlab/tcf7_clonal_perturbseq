#!/usr/bin/env python
"""
Generate gene expression correlation plot between Trim28 TRM-like vs NT other cell types in both experiments (Figure 5l)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import ConnectionPatch, Ellipse
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
plt.rcParams["axes.titlesize"] = 32.5  # 1.3x
plt.rcParams["axes.labelsize"] = 32.5  # 1.3x
plt.rcParams["xtick.labelsize"] = 26  # 1.3x
plt.rcParams["ytick.labelsize"] = 26  # 1.3x
plt.rcParams["lines.markersize"] = 6.5  # 1.3x
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 10)


def run_de_tests(adata, condition='Both', comparison='Trim28 TRM-like vs NT others'):
    """Run differential expression test for Trim28 TRM-like vs NT other cell types"""
    adata = adata[~adata.obs['cell_type'].str.contains('Lymph')].copy()

    # TRM-like cells from Trim28 KO vs all other NT cells
    cbc_group1 = adata[(adata.obs['cell_type'] == 'TRM-like') &
                       (adata.obs['perturbation'] == 'Trim28')].obs_names
    cbc_group2 = adata[(adata.obs['cell_type'] != 'TRM-like') &
                       (adata.obs['perturbation'] == 'NT')].obs_names

    results_df = DE_testing.wilcoxon_DE_test(adata, cbc_group1, cbc_group2)
    return results_df.set_index('Gene')


def _build_boxed_gene_inset(ax, boxed_genes):
    """Add a boxed, two-column inset listing the selected genes."""
    sorted_genes = sorted(boxed_genes)
    midpoint = int(np.ceil(len(sorted_genes) / 2))
    left_genes = sorted_genes[:midpoint]
    right_genes = sorted_genes[midpoint:]

    text_props = dict(color='black', fontsize=16, family='Arial')
    left_column = VPacker(
        children=[TextArea(gene, textprops=text_props) for gene in left_genes],
        align="left",
        pad=0,
        sep=2,
    )
    right_column = VPacker(
        children=[TextArea(gene, textprops=text_props) for gene in right_genes],
        align="left",
        pad=0,
        sep=2,
    )
    inset_columns = HPacker(
        children=[left_column, right_column],
        align="top",
        pad=0,
        sep=20,
    )

    anchored_box = AnchoredOffsetbox(
        loc='upper left',
        child=inset_columns,
        pad=0.25,
        borderpad=0.6,
        frameon=True,
        bbox_to_anchor=(0.12, 0.82),
        bbox_transform=ax.transAxes,
    )
    anchored_box.patch.set_facecolor('white')
    anchored_box.patch.set_edgecolor('black')
    anchored_box.patch.set_linewidth(1)
    ax.add_artist(anchored_box)
    return anchored_box


def _ellipse_boundary_point(center_x, center_y, width, height, target_x, target_y):
    """Return the point where the line to target intersects the ellipse boundary."""
    radius_x = width / 2.0
    radius_y = height / 2.0
    delta_x = target_x - center_x
    delta_y = target_y - center_y
    scale = 1.0 / np.sqrt((delta_x ** 2) / (radius_x ** 2) + (delta_y ** 2) / (radius_y ** 2))
    return center_x + delta_x * scale, center_y + delta_y * scale


def plot_gene_expression_correlation(gene_10_results, trim28_setdb1_results,
                                    sig_thresh=0.05):
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
    boxed_gene_candidates = df_plot[
        df_plot['Significant']
        & (df_plot['logFC_Trim28_Setdb1'] > 1)
        & (df_plot['logFC_10_gene'] > 1)
    ].copy()
    boxed_gene_candidates['distance_to_target'] = np.sqrt(
        (boxed_gene_candidates['logFC_Trim28_Setdb1'] - 1.5) ** 2
        + (boxed_gene_candidates['logFC_10_gene'] - 1.5) ** 2
    )
    boxed_gene_df = (
        boxed_gene_candidates
        .sort_values(
            ['distance_to_target', 'logFC_Trim28_Setdb1', 'logFC_10_gene', 'Gene'],
            ascending=[True, True, True, True],
        )
        .head(20)
        .copy()
    )
    boxed_gene_set = set(boxed_gene_df['Gene'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use a shared data-driven range on both axes without forcing symmetry around 0.
    min_logfc = df_plot[['logFC_Trim28_Setdb1', 'logFC_10_gene']].min().min()
    max_logfc = df_plot[['logFC_Trim28_Setdb1', 'logFC_10_gene']].max().max()
    axis_padding = 0.5
    shared_limits = (min_logfc - axis_padding, max_logfc + axis_padding)
    xlim = shared_limits
    ylim = shared_limits

    # Plot non-significant points in light grey
    ax.scatter(df_plot[~df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[~df_plot['Significant']]['logFC_10_gene'],
               color='lightgrey', label='Non-significant', s=50, alpha=0.7, rasterized=True)

    # Plot significant points in blue
    ax.scatter(df_plot[df_plot['Significant']]['logFC_Trim28_Setdb1'],
               df_plot[df_plot['Significant']]['logFC_10_gene'],
               label='Significant', s=50, alpha=0.7, rasterized=True)

    # Add axis labels
    ax.set_xlabel('log2FC - Trim28/Setdb1 experiment', fontsize=32.5)
    ax.set_ylabel('log2FC - 10 gene experiment', fontsize=32.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add reference lines at logFC = 1
    ax.plot([1, xlim[1]], [1, 1], color='red', linestyle='--', alpha=0.5)
    ax.plot([1, 1], [1, ylim[1]], color='red', linestyle='--', alpha=0.5)

    # Calculate correlations
    pearson_corr, _ = pearsonr(df_plot['logFC_Trim28_Setdb1'], df_plot['logFC_10_gene'])
    spearman_corr, _ = spearmanr(df_plot['logFC_Trim28_Setdb1'], df_plot['logFC_10_gene'])

    # Add title with comparison description
    ax.set_title('Trim28 TRM-like vs NT other cell types', fontsize=32.5, pad=20)

    # Add correlation text inside plot (no box)
    correlation_text = f'Spearman ρ: {spearman_corr:.2f}\nPearson r: {pearson_corr:.2f}'
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes,
            fontsize=26, verticalalignment='top')

    # Label genes with logFC > 1 in both experiments
    for _, row in df_plot[df_plot['Gene'].isin(de_genes_list)].iterrows():
        if row['Gene'] in boxed_gene_set:
            continue
        ax.text(
            row['logFC_Trim28_Setdb1'],
            row['logFC_10_gene'],
            row['Gene'],
            horizontalalignment='left',
            size=16,
            color='black',
            clip_on=False,
        )

    inset_box = _build_boxed_gene_inset(ax, boxed_gene_df['Gene'].tolist())

    cluster_center_x = boxed_gene_df['logFC_Trim28_Setdb1'].mean()
    cluster_center_y = boxed_gene_df['logFC_10_gene'].mean()
    ellipse_width = max(
        boxed_gene_df['logFC_Trim28_Setdb1'].max() - boxed_gene_df['logFC_Trim28_Setdb1'].min() + 0.45,
        0.9,
    )
    ellipse_height = max(
        boxed_gene_df['logFC_10_gene'].max() - boxed_gene_df['logFC_10_gene'].min() + 0.45,
        0.9,
    )
    highlight_ellipse = Ellipse(
        xy=(cluster_center_x, cluster_center_y),
        width=ellipse_width,
        height=ellipse_height,
        facecolor='none',
        edgecolor='black',
        linewidth=1,
        zorder=4,
    )
    ax.add_patch(highlight_ellipse)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box_bbox = inset_box.get_window_extent(renderer=renderer)
    box_anchor_display = (
        box_bbox.x1,
        box_bbox.y0 + 0.3 * box_bbox.height,
    )
    box_anchor_axes = ax.transAxes.inverted().transform(box_anchor_display)
    box_anchor_data = ax.transData.inverted().transform(box_anchor_display)
    ellipse_anchor_x, ellipse_anchor_y = _ellipse_boundary_point(
        cluster_center_x,
        cluster_center_y,
        ellipse_width,
        ellipse_height,
        box_anchor_data[0],
        box_anchor_data[1],
    )
    leader_line = ConnectionPatch(
        xyA=box_anchor_axes,
        coordsA=ax.transAxes,
        xyB=(ellipse_anchor_x, ellipse_anchor_y),
        coordsB=ax.transData,
        axesA=ax,
        axesB=ax,
        arrowstyle='-',
        color='black',
        linewidth=1,
        zorder=5,
    )
    ax.add_artist(leader_line)

    # Add legend
    ax.legend(loc='lower right', fontsize=19.5)

    # Keep all spines visible for box frame
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    return fig, de_genes_list, boxed_gene_df['Gene'].tolist()

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
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
    fig, de_genes, boxed_genes = plot_gene_expression_correlation(
        results_gene_10,
        results_trim28_setdb1,
        sig_thresh=0.05
    )

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "l.png")
    svg_path = os.path.join(output_dir, "l.svg")

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
    print("Figure 5l completed successfully!")

    # Print summary
    print(f"\nSummary:")
    print(f"- Shared genes analyzed: {len(results_gene_10.index.intersection(results_trim28_setdb1.index))}")
    print(f"- DE genes labeled (logFC > 1 in both): {len(de_genes)}")
    print(f"- Genes in boxed inset: {len(boxed_genes)}")
    if de_genes:
        print(f"- Top DE genes: {', '.join(de_genes[:5])}")

if __name__ == "__main__":
    reload(DE_testing)
    main()

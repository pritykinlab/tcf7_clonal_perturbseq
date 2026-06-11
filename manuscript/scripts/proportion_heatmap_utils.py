#!/usr/bin/env python
"""
Shared utilities for proportion heatmap generation (Figure 4c and 4d)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from statsmodels.stats.multitest import multipletests

SIG_THRESHOLD = 0.10


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Create a truncated version of a colormap"""
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def load_and_prepare_data(get_adata, remove_guides=None):
    """Load and prepare data for proportion analysis

    Returns:
        adata: Prepared AnnData object
    """
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=True,
                      remove_NG=False,
                      get_raw_scaled=True)

    if remove_guides:
        remove_guides = set(remove_guides)
        mask = ~adata.obs['guide'].isin(remove_guides)
        adata = adata[mask].copy()

    # Convert to strings for safety
    adata.obs['guide'] = adata.obs['guide'].astype(str)
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

    print(f"Analyzing {adata.n_obs} cells from all conditions...")
    return adata


def run_statistical_tests(adata, test_class, cell_type_orders):
    """Run statistical tests for all cell types

    Args:
        adata: AnnData object
        test_class: Statistical test class to use
        cell_type_orders: List of cell types to analyze

    Returns:
        combined: DataFrame with all test results
    """
    all_results = []

    for cell_type in cell_type_orders:
        print(f"Processing {cell_type}...")

        # Run the test
        test = test_class(adata, cell_type_col_name='cell_type', cell_type=cell_type)
        results = test.get_guide_pvalues()
        results['cell_type'] = cell_type
        all_results.append(results)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Apply multiple testing correction within each cell type
    print("Applying multiple testing correction...")
    combined['p_adjusted'] = 1.0
    for cell_type in cell_type_orders:
        mask = combined['cell_type'] == cell_type
        pvals = combined.loc[mask, 'p_value'].values
        corr_pvals = multipletests(pvals, method='fdr_bh')[1]
        combined.loc[mask, 'p_adjusted'] = corr_pvals

    # Invert test statistic (as done in notebook)
    combined['test_statistic'] = -combined['test_statistic']

    return combined


def create_proportion_heatmap(
    combined_results,
    cell_type_orders,
    title,
    colorbar_label,
    output_dir,
    panel_letter,
    highlight_guides=None,
    show_test_fractions=False,
):
    """Create and save proportion heatmap

    Args:
        combined_results: DataFrame with statistical test results
        cell_type_orders: List of cell types for ordering
        title: Title for the heatmap
        colorbar_label: Label for the colorbar
        output_dir: Directory to save outputs
        panel_letter: Panel letter (e.g., 'c' or 'd')
    """
    print("Creating heatmap...")

    # Pivot data for heatmap
    pivoted = combined_results.pivot_table(index='cell_type', columns='guide',
                                          values='test_statistic', aggfunc='mean')
    pvals = combined_results.pivot_table(index='cell_type', columns='guide',
                                        values='p_adjusted', aggfunc='min')

    # Reorder columns: NO_SITE first, then INTERGENIC, then others
    col_order_nosite = sorted([c for c in pivoted.columns if c.startswith('NO_SITE')])
    col_order_inter = sorted([c for c in pivoted.columns if 'INTERGENIC' in c])
    col_order_other = sorted(list(set(pivoted.columns) - set(col_order_nosite) - set(col_order_inter)))
    new_col_order = col_order_nosite + col_order_inter + col_order_other

    pivoted = pivoted.reindex(columns=new_col_order)
    pvals = pvals.reindex(columns=new_col_order)

    # Reindex rows with cell type order
    pivoted = pivoted.reindex(cell_type_orders).dropna(axis=0, how='all')
    pvals = pvals.reindex(cell_type_orders).dropna(axis=0, how='all')

    # Create figure with better size for x-axis labels
    fig = plt.figure(figsize=(16, 5))

    # Create truncated colormap
    my_cmap = truncate_colormap(plt.cm.PuOr, 0.3, 0.7)

    # Create mask for non-significant values
    mask = (pvals >= SIG_THRESHOLD)

    # Create heatmap
    ax = sns.heatmap(
        pivoted, mask=mask, cmap=my_cmap,
        center=0, vmin=-2, vmax=2,
        annot=True, fmt=".1f", annot_kws={"size": 10.8},
        cbar_kws={'label': colorbar_label, 'shrink': 0.8, 'aspect': 10},
        linewidths=0.5, linecolor='lightgray',
        rasterized=True  # Rasterize heatmap for smaller file size
    )

    # Get colorbar and adjust tick label size to match axis labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(colorbar_label, fontsize=20)

    # Customize appearance - remove axis labels
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('', fontsize=14)
    ax.set_title(title, fontsize=28)  # 22 * 1.2 = 26.4

    # Rotate labels for better alignment
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)

    # Count significant results for controls and guides
    control_guides = [g for g in pivoted.columns if 'NO_SITE' in g or 'INTERGENIC' in g]
    ko_guides = [g for g in pivoted.columns if g not in control_guides]

    control_sig_count = 0
    ko_sig_count = 0

    for ct in pivoted.index:
        for guide in control_guides:
            if pvals.loc[ct, guide] < SIG_THRESHOLD:
                control_sig_count += 1
        for guide in ko_guides:
            if pvals.loc[ct, guide] < SIG_THRESHOLD:
                ko_sig_count += 1

    total_control_tests = len(pivoted.index) * len(control_guides)
    total_ko_tests = len(pivoted.index) * len(ko_guides)

    if show_test_fractions:
        control_pct = round(100 * control_sig_count / total_control_tests)
        ko_pct = round(100 * ko_sig_count / total_ko_tests)
        control_label = (
            f"{control_sig_count}/{total_control_tests}={control_pct}% control tests\n"
            "are significant"
        )
        ko_label = (
            f"{ko_sig_count}/{total_ko_tests}={ko_pct}% gene tests\n"
            "are significant"
        )
    else:
        control_label = f"{control_sig_count} control tests\nare significant"
        ko_label = f"{ko_sig_count} guide tests\nare significant"

    # Add text annotations for significance counts
    ax.text(0.02, 0.98, control_label,
            transform=ax.transAxes, fontsize=18, verticalalignment='top',
            horizontalalignment='left', multialignment='center')

    ax.text(0.98, 0.98, ko_label,
            transform=ax.transAxes, fontsize=18, verticalalignment='top',
            horizontalalignment='right', multialignment='center')

    highlight_guides = set(highlight_guides or [])
    for guide in highlight_guides:
        if guide in pivoted.columns:
            col_idx = pivoted.columns.get_loc(guide)
            ax.add_patch(
                Rectangle(
                    (col_idx, 0),
                    1,
                    pivoted.shape[0],
                    fill=False,
                    edgecolor='red',
                    linewidth=2.5,
                    clip_on=False,
                    zorder=10,
                )
            )

    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, f"{panel_letter}.png")
    svg_path = os.path.join(output_dir, f"{panel_letter}.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    validate_file_sizes(png_path, svg_path)

    plt.close()

    # Return significance counts for summary
    return control_sig_count, ko_sig_count


def validate_file_sizes(png_path, svg_path):
    """Validate that file sizes are within limits"""
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def print_summary(combined_results):
    """Print summary statistics"""
    sig_results = combined_results[combined_results['p_adjusted'] < SIG_THRESHOLD]
    ctrl_sig = sig_results[sig_results['guide'].str.contains('NO_SITE|INTERGENIC', na=False)].shape[0]
    ko_sig = sig_results.shape[0] - ctrl_sig

    print(f"\nSummary:")
    print(f"Total significant results (p_adj < {SIG_THRESHOLD:.2f}): {sig_results.shape[0]}")
    print(f"Control guides: {ctrl_sig}")
    print(f"KO guides: {ko_sig}")

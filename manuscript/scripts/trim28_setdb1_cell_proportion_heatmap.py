#!/usr/bin/env python
"""
Generate cell-based cell type proportion heatmap for trim28_setdb1 dataset using Fisher exact tests (Figure 5h)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# Add utils and scripts to path
sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_orders_trim28_setdb1
from proportion_heatmap_utils import (
    truncate_colormap,
    validate_file_sizes
)
from statsmodels.stats.multitest import multipletests

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 3)


class FisherTest:
    """
    Implements a 2x2 fisher_exact test comparing each perturbation vs. NT,
    in terms of the proportion of cells that belong to the chosen cell_type vs. not.
    """
    def __init__(self, adata, cell_type_col_name='cell_type', cell_type='Progenitor'):
        self.adata = adata
        self.cell_type = cell_type
        self.cell_type_col_name = cell_type_col_name

    def _calculate_pvalue(self, control_counts, perturb_counts):
        """
        control_counts: (count_celltype, count_not_celltype)
        perturb_counts: (count_celltype, count_not_celltype)
        """
        obs = np.array([control_counts, perturb_counts])
        if obs.sum() == 0:
            return 1.0, 0.0

        # Fisher's exact test on raw counts (p-value uses unadjusted table)
        odds_ratio, pval = fisher_exact(obs)
        # Use log2 odds ratio as test statistic with pseudocount
        # (second call computes effect size for plotting after adding +1)
        obs_pc = obs + 1
        odds_ratio_pc, _ = fisher_exact(obs_pc)
        test_stat = np.log2(odds_ratio_pc)
        return pval, test_stat

    def _cell_type_counts(self, subset_mask):
        """
        Return (count_in_celltype, count_not_celltype) for a mask of adata.
        """
        sub = self.adata[subset_mask]
        ctype_count = (sub.obs[self.cell_type_col_name] == self.cell_type).sum()
        total = sub.n_obs
        return (ctype_count, total - ctype_count)

    def get_perturbation_pvalues(self):
        """
        Compare each perturbation's distribution vs. NT reference.
        """
        # Prepare the control (NT) counts once
        nt_mask = (self.adata.obs['perturbation'] == 'NT')
        control_counts = self._cell_type_counts(nt_mask)

        # Get all perturbations excluding NT
        all_perts = set(self.adata.obs['perturbation'].unique())
        all_perts.discard('NT')
        all_perts.discard('Setdb1+Trim28')
        all_perts = [p for p in all_perts if isinstance(p, str)]

        res = []
        for pert in all_perts:
            p_mask = (self.adata.obs['perturbation'] == pert)
            p_counts = self._cell_type_counts(p_mask)
            pval, statistic = self._calculate_pvalue(control_counts, p_counts)
            res.append({
                'perturbation': pert,
                'p_value': pval,
                'test_statistic': statistic
            })

        df = pd.DataFrame(res)
        df['-log10(p_value)'] = -np.log10(df['p_value'])
        return df


def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # Load data for trim28_setdb1 dataset
    print("Loading trim28_setdb1 data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=False,  # Allow multi-construct cells
                      remove_NG=False,  # Filter on curated perturbation labels below
                      remove_lymph_node=False,  # Keep lymph node cells to include Progenitor dLN
                      get_raw_scaled=True)

    # Use curated perturbation labels (set by get_adata) and drop unlabeled/no-site cells.
    # This keeps: NT, Trim28, Setdb1, Setdb1+Trim28.
    adata = adata[adata.obs['perturbation'].astype(str) != ''].copy()

    # Convert to strings for safety
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)
    adata.obs['condition'] = adata.obs['condition'].astype(str)

    print(f"Analyzing {adata.n_obs} cells...")

    # Run statistical tests for all cell types
    all_results = []

    # Use full cell type order (including Progenitor dLN)
    cell_types = cell_type_orders_trim28_setdb1

    for cell_type in cell_types:
        print(f"Processing {cell_type}...")

        # Run the test
        test = FisherTest(adata, cell_type_col_name='cell_type', cell_type=cell_type)
        results = test.get_perturbation_pvalues()
        results['cell_type'] = cell_type
        results['condition'] = 'All'  # For compatibility with heatmap display
        all_results.append(results)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Apply multiple testing correction within each cell type
    print("Applying multiple testing correction...")
    combined['p_adjusted'] = 1.0
    for cell_type in cell_types:
        mask = combined['cell_type'] == cell_type
        pvals = combined.loc[mask, 'p_value'].values
        if len(pvals) > 0:
            corr_pvals = multipletests(pvals, method='fdr_bh')[1]
            combined.loc[mask, 'p_adjusted'] = corr_pvals

    # Invert test statistic (as done in notebook)
    combined['test_statistic'] = -combined['test_statistic']

    # Create heatmap
    print("Creating heatmap...")

    # Pivot data for heatmap
    pivoted = combined.pivot_table(index='cell_type', columns='perturbation',
                                  values='test_statistic', aggfunc='mean')
    pvals = combined.pivot_table(index='cell_type', columns='perturbation',
                                values='p_adjusted', aggfunc='min')

    # Sort columns alphabetically
    ordered_columns = sorted(pivoted.columns)
    pivoted = pivoted[ordered_columns]
    pvals = pvals[ordered_columns]

    # Reindex rows with cell type order
    pivoted = pivoted.reindex(cell_types).dropna(axis=0, how='all')
    pvals = pvals.reindex(cell_types).dropna(axis=0, how='all')

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 10))

    # Create truncated colormap
    my_cmap = truncate_colormap(plt.cm.PuOr, 0.3, 0.7)

    # Create mask for non-significant values (p >= 0.2)
    mask = (pvals >= 0.2)

    # Create heatmap
    import seaborn as sns
    sns.heatmap(
        pivoted, mask=mask, cmap=my_cmap,
        center=0, vmin=-4, vmax=4,
        annot=True, fmt=".1f", annot_kws={"size": 18},
        cbar_kws={'label': 'log2 guide/NT odds'},
        linewidths=0.5, linecolor='lightgray',
        rasterized=True  # Rasterize heatmap for smaller file size
    )

    # Get colorbar and adjust
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('log2 guide/NT odds', fontsize=20)

    # Customize appearance
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Change in cell type\nproportion', fontsize=28, pad=14)

    # Rotate labels
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "h.png")
    svg_path = os.path.join(output_dir, "h.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    validate_file_sizes(png_path, svg_path)

    plt.close()

    # Print summary
    sig_results = combined[combined['p_adjusted'] < 0.1]
    print(f"\nSummary:")
    print(f"Total significant results (p_adj < 0.1): {sig_results.shape[0]}")
    print("Figure 5h completed successfully!")


if __name__ == "__main__":
    main()

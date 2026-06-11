#!/usr/bin/env python
"""
legacy internal Sup_Figure_5_5: cell-based cell type proportion heatmap (guide-level)
for the trim28_setdb1 dataset using Fisher exact tests vs NT.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import seaborn as sns

# Add utils and scripts to path
sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_orders_trim28_setdb1  # noqa: E402
from proportion_heatmap_utils import truncate_colormap, validate_file_sizes  # noqa: E402

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (26, 3)

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_5_5_old"
TABLE_DIR = os.path.join(OUTPUT_DIR, "5_individual_tables")
SUMMARY_TABLE = os.path.join(TABLE_DIR, "all_results.tsv")


class FisherTestGuides:
    """
    Implements a 2x2 fisher_exact test comparing each guide vs. NT,
    in terms of the proportion of cells that belong to the chosen cell_type vs. not.
    """
    def __init__(self, adata, cell_type_col_name='cell_type', cell_type='Progenitor'):
        self.adata = adata
        self.cell_type = cell_type
        self.cell_type_col_name = cell_type_col_name

    def _calculate_pvalue(self, control_counts, guide_counts):
        """
        control_counts: (count_celltype, count_not_celltype)
        guide_counts: (count_celltype, count_not_celltype)
        """
        obs = np.array([control_counts, guide_counts])
        if obs.sum() == 0:
            return 1.0, 0.0

        odds_ratio, pval = fisher_exact(obs)
        obs_pc = obs + 1
        odds_ratio_pc, _ = fisher_exact(obs_pc)
        test_stat = np.log2(odds_ratio_pc)
        return pval, test_stat

    def _cell_type_counts(self, subset_mask):
        """Return (count_in_celltype, count_not_celltype) for a mask of adata."""
        sub = self.adata[subset_mask]
        ctype_count = (sub.obs[self.cell_type_col_name] == self.cell_type).sum()
        total = sub.n_obs
        return (ctype_count, total - ctype_count)

    def get_guide_pvalues(self):
        """Compare each guide's distribution vs. NT reference (perturbation == 'NT')."""
        nt_mask = (self.adata.obs['perturbation'] == 'NT')
        control_counts = self._cell_type_counts(nt_mask)

        all_guides = [g for g in self.adata.obs['guide'].unique() if isinstance(g, str)]

        res = []
        for guide in all_guides:
            g_mask = (self.adata.obs['guide'] == guide)
            g_counts = self._cell_type_counts(g_mask)
            pval, statistic = self._calculate_pvalue(control_counts, g_counts)
            res.append({
                'guide': guide,
                'p_value': pval,
                'test_statistic': statistic,
                'nt_in_cell_type': control_counts[0],
                'nt_not_cell_type': control_counts[1],
                'guide_in_cell_type': g_counts[0],
                'guide_not_cell_type': g_counts[1]
            })

        df = pd.DataFrame(res)
        df['-log10(p_value)'] = -np.log10(df['p_value'])
        return df


def main():
    def order_by_numeric_suffix(guides, prefix):
        """Order guides by numeric suffix to avoid lexicographic issues (e.g., -10 after -1)."""
        ordered = []
        others = []
        prefix_lower = prefix.lower()
        for g in guides:
            gl = g.lower()
            if gl.startswith(prefix_lower):
                parts = gl.replace(prefix_lower, '', 1).lstrip('-_')
                try:
                    num = int(parts)
                    ordered.append((num, g))
                except ValueError:
                    others.append(g)
            else:
                others.append(g)
        ordered_sorted = [g for _, g in sorted(ordered, key=lambda x: x[0])]
        return ordered_sorted + sorted(others)

    def sanitize_path_component(value):
        """Make strings safe for file and directory names."""
        safe = str(value).replace('/', '-').replace('\\', '-')
        safe = safe.replace(' ', '_').replace(':', '-')
        return safe

    def write_individual_tables(combined_df, table_dir, cell_type_order, guide_order):
        """
        Save per-cell-type/guide contingency tables and a consolidated summary.
        """
        if os.path.exists(table_dir):
            shutil.rmtree(table_dir)
        os.makedirs(table_dir, exist_ok=True)

        results = combined_df.copy()
        results['nt_total'] = results['nt_in_cell_type'] + results['nt_not_cell_type']
        results['guide_total'] = results['guide_in_cell_type'] + results['guide_not_cell_type']
        results['nt_fraction_in_cell_type'] = np.where(
            results['nt_total'] > 0,
            results['nt_in_cell_type'] / results['nt_total'],
            np.nan
        )
        results['guide_fraction_in_cell_type'] = np.where(
            results['guide_total'] > 0,
            results['guide_in_cell_type'] / results['guide_total'],
            np.nan
        )
        results['log2_guide_over_nt_odds'] = results['test_statistic']

        results['cell_type'] = pd.Categorical(results['cell_type'], categories=cell_type_order, ordered=True)
        results['guide'] = pd.Categorical(results['guide'], categories=guide_order, ordered=True)
        results = results.sort_values(['cell_type', 'guide'])

        summary_records = []
        for _, row in results.iterrows():
            ct_dir = os.path.join(table_dir, sanitize_path_component(row['cell_type']))
            os.makedirs(ct_dir, exist_ok=True)
            table_path = os.path.join(ct_dir, f"{sanitize_path_component(row['guide'])}.tsv")
            table_records = [
                {
                    'cell_type': row['cell_type'],
                    'group': 'NT',
                    'cells_in_cell_type': int(row['nt_in_cell_type']),
                    'cells_not_in_cell_type': int(row['nt_not_cell_type'])
                },
                {
                    'cell_type': row['cell_type'],
                    'group': row['guide'],
                    'cells_in_cell_type': int(row['guide_in_cell_type']),
                    'cells_not_in_cell_type': int(row['guide_not_cell_type'])
                }
            ]
            table_df = pd.DataFrame(table_records)
            table_df.to_csv(table_path, sep="\t", index=False)
            summary_records.extend(table_records)

        summary_df = pd.DataFrame(summary_records)
        summary_df['cell_type'] = pd.Categorical(summary_df['cell_type'], categories=cell_type_order, ordered=True)
        group_order = ['NT'] + [g for g in guide_order if g != 'NT']
        summary_df['group'] = pd.Categorical(summary_df['group'], categories=group_order, ordered=True)
        summary_df = summary_df.sort_values(['cell_type', 'group'])

        summary_path = SUMMARY_TABLE
        summary_df.to_csv(summary_path, sep="\t", index=False)
        return summary_path

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Load data for trim28_setdb1 dataset
    print("Loading trim28_setdb1 data...")
    adata = get_adata('trim28_setdb1',
                      use_cells_only_one_construct=True,  # Only single-construct cells
                      remove_NG=True,  # Remove non-guide controls
                      remove_lymph_node=False,  # Keep lymph node cells to include Progenitor dLN
                      get_raw_scaled=True)

    # Convert to strings for safety
    adata.obs['guide'] = adata.obs['guide'].astype(str)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str)

    print(f"Analyzing {adata.n_obs} cells across {adata.obs['guide'].nunique()} guides...")

    # Run statistical tests for all cell types (including Progenitor dLN)
    cell_types = cell_type_orders_trim28_setdb1
    all_results = []
    for cell_type in cell_types:
        print(f"Processing {cell_type}...")
        test = FisherTestGuides(adata, cell_type_col_name='cell_type', cell_type=cell_type)
        results = test.get_guide_pvalues()
        results['cell_type'] = cell_type
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)

    # Multiple testing correction within each cell type
    print("Applying multiple testing correction...")
    combined['p_adjusted'] = 1.0
    for cell_type in cell_types:
        mask = combined['cell_type'] == cell_type
        pvals = combined.loc[mask, 'p_value'].values
        if len(pvals) > 0:
            corr_pvals = multipletests(pvals, method='fdr_bh')[1]
            combined.loc[mask, 'p_adjusted'] = corr_pvals

    # Invert test statistic for display
    combined['test_statistic'] = -combined['test_statistic']

    # Pivot data for heatmap
    pivoted = combined.pivot_table(index='cell_type', columns='guide',
                                   values='test_statistic', aggfunc='mean')
    pvals = combined.pivot_table(index='cell_type', columns='guide',
                                 values='p_adjusted', aggfunc='min')

    guide_to_perturbation = adata.obs.groupby('guide')['perturbation'].agg(lambda s: s.value_counts().idxmax())

    # Order: NO_SITE controls (numeric order), INTERGENIC if present, Setdb1 guides (numeric), Trim28 guides (numeric), then others (NT column excluded as reference)
    no_sites = order_by_numeric_suffix([g for g in pivoted.columns if g.upper().startswith('NO_SITE')], 'no_site')
    intergenic = sorted([g for g in pivoted.columns if 'INTERGENIC' in g.upper()])
    setdb_guides = order_by_numeric_suffix([g for g in pivoted.columns if g.lower().startswith('setdb1')], 'setdb1')
    trim_guides = order_by_numeric_suffix([g for g in pivoted.columns if g.lower().startswith('trim28')], 'trim28')
    other_guides = sorted(list(set(pivoted.columns) - set(no_sites) - set(intergenic) - set(trim_guides) - set(setdb_guides)))
    ordered_guides = no_sites + intergenic + setdb_guides + trim_guides + other_guides

    pivoted = pivoted.reindex(columns=ordered_guides)
    pvals = pvals.reindex(columns=ordered_guides)

    pivoted = pivoted.reindex(cell_types).dropna(axis=0, how='all')
    pvals = pvals.reindex(cell_types).dropna(axis=0, how='all')

    summary_path = write_individual_tables(
        combined_df=combined,
        table_dir=TABLE_DIR,
        cell_type_order=list(pivoted.index),
        guide_order=list(pivoted.columns)
    )
    print(f"Saved per-grid contingency tables to {TABLE_DIR}")
    print(f"Summary table written to {summary_path}")

    # Create figure
    fig, ax = plt.subplots(figsize=(26, 10))

    my_cmap = truncate_colormap(plt.cm.PuOr, 0.3, 0.7)
    sig_thresh = 0.1

    sns.heatmap(
        pivoted, cmap=my_cmap,
        center=0, vmin=-2, vmax=2,
        annot=True, fmt=".1f", annot_kws={"size": 10.8},
        cbar_kws={'label': 'log2 guide/NT odds'},
        linewidths=0.5, linecolor='lightgray',
        rasterized=True
    )

    # Overlay stars for significant results
    for i, ct in enumerate(pivoted.index):
        for j, guide in enumerate(pivoted.columns):
            if pvals.loc[ct, guide] < sig_thresh:
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                        color='black', fontsize=16, fontweight='normal')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('log2 guide/NT odds', fontsize=20)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Change in cell type\nproportion (by guide)', fontsize=28, pad=14)

    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Significance counts (control vs KO style)
    control_guides = [g for g in pivoted.columns if guide_to_perturbation.get(g) == 'NT' or ('NO_SITE' in g.upper()) or ('INTERGENIC' in g.upper())]
    ko_guides = [g for g in pivoted.columns if g not in control_guides]

    ctrl_sig = 0
    ko_sig = 0
    for ct in pivoted.index:
        for guide in control_guides:
            if guide in pvals.columns and pvals.loc[ct, guide] < sig_thresh:
                ctrl_sig += 1
        for guide in ko_guides:
            if guide in pvals.columns and pvals.loc[ct, guide] < sig_thresh:
                ko_sig += 1

    ax.text(0.02, 0.98, f"{ctrl_sig} control tests\nare significant",
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='left', multialignment='center')

    ax.text(0.98, 0.98, f"{ko_sig} guide tests\nare significant",
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', multialignment='center')

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(OUTPUT_DIR, "5.png")
    svg_path = os.path.join(OUTPUT_DIR, "5.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    validate_file_sizes(png_path, svg_path)
    plt.close()

    sig_results = combined[combined['p_adjusted'] < sig_thresh]
    print(f"\nSummary:")
    print(f"Total significant results (p_adj < 0.1): {sig_results.shape[0]}")
    print("legacy internal Sup_Figure_5_5 completed successfully!")


if __name__ == "__main__":
    main()

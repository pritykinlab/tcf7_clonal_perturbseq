#!/usr/bin/env python
"""
Legacy clone-based cell type proportion heatmap (guide-level).
for the trim28_setdb1 dataset using Wilcoxon rank-sum tests vs NT clones.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums
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

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_5_4_old"


class CloneMannWhitneyGuides:
    """
    Tests whether the proportion of each clone's cells in cell_type
    differs between NT clones vs. a given guide's clones using the Wilcoxon rank-sum test.
    """
    def __init__(self, adata, cell_type_col_name='cell_type', cell_type='Progenitor'):
        self.adata = adata
        self.cell_type = cell_type
        self.cell_type_col_name = cell_type_col_name
        self.clone_by_cell_type_count = self._compute_clone_by_cell_type_count()
        self.clone_by_cell_type_proportion = self._compute_proportions()

    def _compute_clone_by_cell_type_count(self):
        """Returns a DataFrame: rows=clone, columns=cell_types, values=count."""
        clones = self.adata.obs['clone'].astype(str).unique()
        ctypes = self.adata.obs[self.cell_type_col_name].astype(str).unique()
        df = pd.DataFrame(0, index=clones, columns=ctypes)
        for cl in clones:
            sub = self.adata[self.adata.obs['clone'] == cl]
            counts = sub.obs[self.cell_type_col_name].value_counts()
            for ctype, cnt in counts.items():
                df.loc[cl, ctype] = cnt
        return df

    def _compute_proportions(self):
        """Convert the above counts to proportions for each clone."""
        df = self.clone_by_cell_type_count.astype(float)
        sums = df.sum(axis=1)
        for cl in df.index:
            if sums[cl] > 0:
                df.loc[cl] = df.loc[cl] / sums[cl]
        return df.fillna(0.0)

    def _calculate_pvalue(self, control_prop, test_prop):
        if len(control_prop) == 0 or len(test_prop) == 0:
            return 1.0, 0.0
        test_stat, pval = ranksums(control_prop, test_prop, alternative='two-sided')
        return pval, test_stat

    def get_guide_pvalues(self):
        """Compare each guide's clones to NT clones (perturbation == 'NT')."""
        nt_clones = self.adata.obs.loc[self.adata.obs['perturbation'] == 'NT', 'clone'].unique()
        nt_prop = self.clone_by_cell_type_proportion.loc[nt_clones, self.cell_type]

        all_guides = [g for g in self.adata.obs['guide'].unique() if isinstance(g, str)]

        res = []
        for guide in all_guides:
            guide_clones = self.adata.obs.loc[self.adata.obs['guide'] == guide, 'clone'].unique()
            guide_prop = self.clone_by_cell_type_proportion.loc[guide_clones, self.cell_type]
            pval, stat = self._calculate_pvalue(nt_prop, guide_prop)
            res.append({'guide': guide, 'p_value': pval, 'test_statistic': stat})

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
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        test = CloneMannWhitneyGuides(adata, cell_type_col_name='cell_type', cell_type=cell_type)
        results = test.get_guide_pvalues()
        results['cell_type'] = cell_type
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    guide_to_perturbation = adata.obs.groupby('guide')['perturbation'].agg(lambda s: s.value_counts().idxmax())

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

    # Create figure
    fig, ax = plt.subplots(figsize=(26, 10))

    my_cmap = truncate_colormap(plt.cm.PuOr, 0.3, 0.7)
    sig_thresh = 0.1

    sns.heatmap(
        pivoted, cmap=my_cmap,
        center=0, vmin=-2, vmax=2,
        annot=True, fmt=".1f", annot_kws={"size": 10.8},
        cbar_kws={'label': 'Wilcoxon rank-sum\nz-statistic'},
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
    cbar.set_label('Wilcoxon rank-sum\nz-statistic', fontsize=20)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Change in cell type\nproportion in clone (by guide)', fontsize=28, pad=14)

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
    png_path = os.path.join(OUTPUT_DIR, "4.png")
    svg_path = os.path.join(OUTPUT_DIR, "4.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    validate_file_sizes(png_path, svg_path)
    plt.close()

    sig_results = combined[combined['p_adjusted'] < sig_thresh]
    print(f"\nSummary:")
    print(f"Total significant results (p_adj < 0.1): {sig_results.shape[0]}")
    print("Legacy clone-based guide heatmap completed successfully!")


if __name__ == "__main__":
    main()

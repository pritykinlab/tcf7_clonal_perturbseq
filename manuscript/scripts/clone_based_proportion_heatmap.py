#!/usr/bin/env python
"""
Generate clone-based cell type proportion heatmap using Wilcoxon rank-sum tests (Figure 4d)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# Add utils and scripts to path
sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_orders_gene10
from proportion_heatmap_utils import (
    load_and_prepare_data,
    run_statistical_tests,
    create_proportion_heatmap,
    print_summary
)

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (10, 3)


class CloneMannWhitney:
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
                df.loc[cl] = df.loc[cl]/sums[cl]
        return df.fillna(0)

    def _calculate_pvalue(self, control_prop, test_prop):
        if len(control_prop) == 0 or len(test_prop) == 0:
            return 1.0, 0.0
        test_stat, pval = ranksums(control_prop, test_prop, alternative='two-sided')
        return pval, test_stat

    def get_guide_pvalues(self):
        """Compare each guide's clones to NT clones"""
        nt_mask = self.adata.obs['perturbation'] == 'NT'

        # Get all guides excluding NT
        all_guides = set(self.adata.obs['guide'].unique())
        all_guides.discard('NT')
        all_guides = [g for g in all_guides if isinstance(g, str)]

        res = []
        for guide in all_guides:
            guide_mask = self.adata.obs['guide'] == guide
            guide_clones = self.adata.obs.loc[guide_mask, 'clone'].unique()
            guide_prop = self.clone_by_cell_type_proportion.loc[guide_clones, self.cell_type]

            is_control_guide = bool(
                (self.adata.obs.loc[guide_mask, 'perturbation'] == 'NT').all()
            )
            if is_control_guide:
                reference_mask = nt_mask & (~guide_mask)
            else:
                reference_mask = nt_mask

            nt_clones = self.adata.obs.loc[reference_mask, 'clone'].unique()
            nt_prop = self.clone_by_cell_type_proportion.loc[nt_clones, self.cell_type]
            pval, stat = self._calculate_pvalue(nt_prop, guide_prop)
            res.append({'guide': guide, 'p_value': pval, 'test_statistic': stat})

        df = pd.DataFrame(res)
        df['-log10(p_value)'] = -np.log10(df['p_value'])
        return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../manuscript/plots/Figure4",
    )
    parser.add_argument("--panel-letter", default="d")
    parser.add_argument(
        "--include-all-guides",
        action="store_true",
        help="Retain INTERGENIC-3 instead of applying the Figure 4 exclusion.",
    )
    parser.add_argument(
        "--highlight-guide",
        action="append",
        default=[],
        help="Guide label to outline with a red rectangle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    remove_guides = [] if args.include_all_guides else ['INTERGENIC-3']
    adata = load_and_prepare_data(get_adata, remove_guides=remove_guides)

    # Run statistical tests
    combined = run_statistical_tests(adata, CloneMannWhitney, cell_type_orders_gene10)

    # Create and save heatmap
    create_proportion_heatmap(
        combined_results=combined,
        cell_type_orders=cell_type_orders_gene10,
        title='Change in cell type proportion in clone',
        colorbar_label='Wilcoxon rank-sum\nz-statistic',
        output_dir=output_dir,
        panel_letter=args.panel_letter,
        highlight_guides=args.highlight_guide,
    )

    # Print summary
    print_summary(combined)
    print("Figure 4d completed successfully!")


if __name__ == "__main__":
    main()

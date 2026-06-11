#!/usr/bin/env python
"""
Generate cell-based cell type proportion heatmap using Fisher exact tests (Figure 4c)
"""

import os
import sys
import argparse
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


class ChiTest:
    """
    Implements a 2x2 fisher_exact test comparing each guide vs. NT,
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

    def get_guide_pvalues(self):
        """
        Compare each guide's distribution vs. NT reference.
        """
        nt_mask = (self.adata.obs['perturbation'] == 'NT')

        # Get all guides excluding NT
        all_guides = set(self.adata.obs['guide'].unique())
        all_guides.discard('NT')
        all_guides = [g for g in all_guides if isinstance(g, str)]

        res = []
        for guide in all_guides:
            g_mask = (self.adata.obs['guide'] == guide)
            g_counts = self._cell_type_counts(g_mask)

            is_control_guide = bool(
                (self.adata.obs.loc[g_mask, 'perturbation'] == 'NT').all()
            )
            if is_control_guide:
                reference_mask = nt_mask & (~g_mask)
            else:
                reference_mask = nt_mask

            control_counts = self._cell_type_counts(reference_mask)
            pval, statistic = self._calculate_pvalue(control_counts, g_counts)
            res.append({
                'guide': guide,
                'p_value': pval,
                'test_statistic': statistic
            })

        df = pd.DataFrame(res)
        df['-log10(p_value)'] = -np.log10(df['p_value'])
        return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../manuscript/plots/Figure4",
    )
    parser.add_argument("--panel-letter", default="c")
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
    combined = run_statistical_tests(adata, ChiTest, cell_type_orders_gene10)

    # Create and save heatmap
    create_proportion_heatmap(
        combined_results=combined,
        cell_type_orders=cell_type_orders_gene10,
        title='Change in cell type proportion',
        colorbar_label='log2 guide/NT odds',
        output_dir=output_dir,
        panel_letter=args.panel_letter,
        highlight_guides=args.highlight_guide,
        show_test_fractions=True,
    )

    # Print summary
    print_summary(combined)
    print("Figure 4c completed successfully!")


if __name__ == "__main__":
    main()

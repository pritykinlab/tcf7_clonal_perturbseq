#!/usr/bin/env python
"""
Generate supplementary table of TRM-like signature genes
Genes are defined as significant and log2FC > 1 in both experiments.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

LOG2FC_THRESHOLD = 1
ADJ_P_THRESHOLD = 0.05


def run_de_tests(adata):
    """Run DE test for Trim28 TRM-like vs NT other cell types."""
    adata = adata[~adata.obs['cell_type'].str.contains('Lymph')].copy()

    group1 = adata[(adata.obs['cell_type'] == 'TRM-like') &
                   (adata.obs['perturbation'] == 'Trim28')].obs_names
    group2 = adata[(adata.obs['cell_type'] != 'TRM-like') &
                   (adata.obs['perturbation'] == 'NT')].obs_names

    results_df = DE_testing.wilcoxon_DE_test(adata, group1, group2)
    return results_df.set_index('Gene')


def main():
    output_dir = "../manuscript/supplementary_tables"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading gene_10 dataset...")
    gene10_adata = get_adata('gene_10', use_cells_only_one_construct=True, remove_NG=True)

    print("Loading trim28_setdb1 dataset...")
    trim28_setdb1_adata = get_adata('trim28_setdb1', use_cells_only_one_construct=False,
                                    remove_NG=True, remove_lymph_node=True)

    print("Running DE analysis for gene_10 dataset...")
    results_gene10 = run_de_tests(gene10_adata)

    print("Running DE analysis for trim28_setdb1 dataset...")
    results_trim28_setdb1 = run_de_tests(trim28_setdb1_adata)

    shared_genes = results_gene10.index.intersection(results_trim28_setdb1.index)

    combined = pd.DataFrame({
        'Gene': shared_genes,
        'Log2FC_gene10': results_gene10.loc[shared_genes, 'Log2FC'].values,
        'P-value_gene10': results_gene10.loc[shared_genes, 'P-value'].values,
        'Adjusted P-value_gene10': results_gene10.loc[shared_genes, 'Adjusted P-value'].values,
        'rank_sum_statistic_gene10': results_gene10.loc[shared_genes, 'rank_sum_statistic'].values,
        'Log2FC_trim28_setdb1': results_trim28_setdb1.loc[shared_genes, 'Log2FC'].values,
        'P-value_trim28_setdb1': results_trim28_setdb1.loc[shared_genes, 'P-value'].values,
        'Adjusted P-value_trim28_setdb1': results_trim28_setdb1.loc[shared_genes, 'Adjusted P-value'].values,
        'rank_sum_statistic_trim28_setdb1': results_trim28_setdb1.loc[shared_genes, 'rank_sum_statistic'].values
    })

    signature_genes = combined[
        (combined['Adjusted P-value_gene10'] <= ADJ_P_THRESHOLD) &
        (combined['Adjusted P-value_trim28_setdb1'] <= ADJ_P_THRESHOLD) &
        (combined['Log2FC_gene10'] > LOG2FC_THRESHOLD) &
        (combined['Log2FC_trim28_setdb1'] > LOG2FC_THRESHOLD)
    ].copy()

    signature_genes['min_log2fc'] = signature_genes[
        ['Log2FC_gene10', 'Log2FC_trim28_setdb1']
    ].min(axis=1)
    signature_genes = signature_genes.sort_values(
        ['min_log2fc', 'Gene'],
        ascending=[False, True]
    ).drop(columns=['min_log2fc'])

    output_path = os.path.join(output_dir, "5_2.tsv")
    signature_genes.to_csv(output_path, sep='\t', index=False)

    print(f"\nSupplementary table saved to: {output_path}")
    print(f"Total shared genes tested: {len(shared_genes)}")
    print(
        "TRM-like signature genes "
        f"(adjusted P <= {ADJ_P_THRESHOLD} and log2FC > {LOG2FC_THRESHOLD} in both): "
        f"{len(signature_genes)}"
    )


if __name__ == "__main__":
    reload(DE_testing)
    main()

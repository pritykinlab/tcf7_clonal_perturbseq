#!/usr/bin/env python
"""
Generate differentially expressed genes per cell type supplementary table
Compares each cell type vs all other cell types using all cells in the dataset
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

from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10
import DE_testing

def main():
    # Create output directory
    output_dir = "../manuscript/supplementary_tables"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False)

    # Check cell type composition
    print("\nCell type composition in dataset:")
    cell_type_counts = adata.obs['cell_type'].value_counts()
    print(cell_type_counts)
    print(f"Total number of cells: {adata.n_obs}")

    # Get unique cell types (using the defined order)
    cell_types = cell_type_orders_gene10

    # Store all DE results
    all_de_results = []

    # Perform DE test for each cell type vs rest
    print("\nPerforming DE tests...")
    for cell_type in cell_types:
        if cell_type not in adata.obs['cell_type'].values:
            print(f"Warning: Cell type '{cell_type}' not found in dataset, skipping...")
            continue

        print(f"\nProcessing cell type: {cell_type}")

        # Get cells for this cell type vs all others
        cell_type_mask = adata.obs['cell_type'] == cell_type
        other_mask = ~cell_type_mask

        # Get cell barcodes
        cell_type_cells = adata.obs_names[cell_type_mask].tolist()
        other_cells = adata.obs_names[other_mask].tolist()

        print(f"  {cell_type}: {len(cell_type_cells)} cells")
        print(f"  Other cell types: {len(other_cells)} cells")

        # Perform DE test using wilcoxon method
        # Use 'norm_log' layer which contains normalized log-transformed data
        # The DE_testing module expects either raw counts in adata.raw or a normalized layer
        de_results = DE_testing.wilcoxon_DE_test(
            adata,
            CBC_group1=cell_type_cells,  # Cell type of interest
            CBC_group2=other_cells,       # All other cell types
            group1_threshold_pct=0.05,    # Default from module
            group2_threshold_pct=0.0,     # Default from module
            pseudo_count=0.01,            # Default from module
            norm_layer='norm_log'         # Use the normalized layer
        )

        # Add cell type column
        de_results['Cell_Type'] = cell_type

        # Append to all results
        all_de_results.append(de_results)

        # Print summary statistics
        significant_genes = de_results[de_results['Adjusted P-value'] < 0.05]
        upregulated = significant_genes[significant_genes['Log2FC'] > 0]
        downregulated = significant_genes[significant_genes['Log2FC'] < 0]

        print(f"  Total genes tested: {len(de_results)}")
        print(f"  Significant genes (adj p < 0.05): {len(significant_genes)}")
        print(f"    Upregulated: {len(upregulated)}")
        print(f"    Downregulated: {len(downregulated)}")

        if len(upregulated) > 0:
            top_genes = upregulated.nlargest(5, 'Log2FC')['Gene'].tolist()
            print(f"  Top 5 most positive log2FC genes: {', '.join(top_genes)}")

    # Combine all results
    print("\nCombining all DE results...")
    combined_results = pd.concat(all_de_results, ignore_index=True)

    # Filter to only significant genes
    print(f"Total genes before filtering: {len(combined_results)}")
    combined_results = combined_results[combined_results['Adjusted P-value'] < 0.05]
    print(f"Total significant genes (adj p < 0.05): {len(combined_results)}")

    combined_results = combined_results.sort_values(
        ['Cell_Type', 'Log2FC', 'Gene'],
        ascending=[True, False, True]
    )

    # Reorder columns for better readability
    column_order = ['Cell_Type', 'Gene', 'Log2FC', 'P-value', 'Adjusted P-value', 'rank_sum_statistic']
    combined_results = combined_results[column_order]

    # Save to TSV file (following naming convention: {figure}_{number}.tsv)
    output_path = os.path.join(output_dir, "2_1.tsv")
    combined_results.to_csv(output_path, sep='\t', index=False)

    print(f"\nSupplementary table saved to: {output_path}")
    print(f"Total rows in table: {len(combined_results)}")
    print(f"Cell types included: {', '.join(combined_results['Cell_Type'].unique())}")

    # Print final summary
    print("\nFinal summary:")
    for cell_type in combined_results['Cell_Type'].unique():
        ct_data = combined_results[combined_results['Cell_Type'] == cell_type]
        sig_data = ct_data[ct_data['Adjusted P-value'] < 0.05]
        print(f"  {cell_type}: {len(ct_data)} genes tested, {len(sig_data)} significant")

if __name__ == "__main__":
    main()

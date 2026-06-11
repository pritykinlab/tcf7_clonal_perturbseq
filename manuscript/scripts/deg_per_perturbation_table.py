#!/usr/bin/env python
"""
Generate differentially expressed genes per perturbation supplementary table
Compares each perturbation vs NT (non-targeting control) cells
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
                      use_cells_only_one_construct=True,  # Use single-construct cells for cleaner signal
                      remove_NG=True,  # Remove cells without guide assignment (keeps NT cells)
                      get_raw_scaled=False)

    # Get list of perturbations (excluding NT)
    print("\nGetting perturbation list...")
    all_perturbations = adata.obs['perturbation'].unique().tolist()
    perturbations = [p for p in all_perturbations if p != 'NT']
    perturbations.sort()  # Sort alphabetically for consistency

    print(f"Found {len(perturbations)} perturbations to analyze: {', '.join(perturbations)}")

    # Get NT control cells
    nt_cells = adata.obs[adata.obs['perturbation'] == 'NT'].index.tolist()
    print(f"Number of NT control cells: {len(nt_cells)}")

    # Store all DE results
    all_de_results = []

    # Perform DE test for each perturbation vs NT controls
    print("\nPerforming DE tests...")
    for perturbation in perturbations:
        print(f"\nProcessing perturbation: {perturbation}")

        # Get cells for this perturbation
        perturb_mask = adata.obs['perturbation'] == perturbation
        perturb_cells = adata.obs_names[perturb_mask].tolist()

        print(f"  {perturbation}: {len(perturb_cells)} cells")
        print(f"  NT controls: {len(nt_cells)} cells")

        # Perform DE test using wilcoxon method
        de_results = DE_testing.wilcoxon_DE_test(
            adata,
            CBC_group1=perturb_cells,      # Perturbation cells
            CBC_group2=nt_cells,           # NT control cells
            group1_threshold_pct=0.05,     # At least 5% of cells in group1 must express
            group2_threshold_pct=0.0,      # No minimum expression threshold for controls
            pseudo_count=0.01,             # Default from module
            norm_layer='norm_log'          # Use the normalized layer
        )

        # Add perturbation column
        de_results['Perturbation'] = perturbation

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

        if len(downregulated) > 0:
            bottom_genes = downregulated.nsmallest(5, 'Log2FC')['Gene'].tolist()
            print(f"  Top 5 most negative log2FC genes: {', '.join(bottom_genes)}")

    # Combine all results
    print("\nCombining all DE results...")
    combined_results = pd.concat(all_de_results, ignore_index=True)

    # Filter to only significant genes
    print(f"Total genes before filtering: {len(combined_results)}")
    combined_results = combined_results[combined_results['Adjusted P-value'] < 0.05]
    print(f"Total significant genes (adj p < 0.05): {len(combined_results)}")

    combined_results = combined_results.sort_values(
        ['Perturbation', 'Log2FC', 'Gene'],
        ascending=[True, False, True]
    )

    # Reorder columns for better readability
    column_order = ['Perturbation', 'Gene', 'Log2FC', 'P-value', 'Adjusted P-value', 'rank_sum_statistic']
    combined_results = combined_results[column_order]

    # Save to TSV file (following naming convention: {figure}_{number}.tsv)
    output_path = os.path.join(output_dir, "2_3.tsv")
    combined_results.to_csv(output_path, sep='\t', index=False)

    print(f"\nSupplementary table saved to: {output_path}")
    print(f"Total rows in table: {len(combined_results)}")
    print(f"Perturbations included: {', '.join(combined_results['Perturbation'].unique())}")

    # Print final summary
    print("\nFinal summary:")
    for perturbation in combined_results['Perturbation'].unique():
        pert_data = combined_results[combined_results['Perturbation'] == perturbation]
        sig_data = pert_data[pert_data['Adjusted P-value'] < 0.05]
        print(f"  {perturbation}: {len(pert_data)} genes tested, {len(sig_data)} significant")

    # Also print the most significant genes across all perturbations
    print("\nMost significant DE genes across all perturbations (top 10 by adj p-value):")
    top_overall = combined_results.nsmallest(10, 'Adjusted P-value')
    for _, row in top_overall.iterrows():
        print(f"  {row['Perturbation']}: {row['Gene']} (Log2FC={row['Log2FC']:.2f}, adj p={row['Adjusted P-value']:.2e})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate supplementary table of GO term enrichment scores per cell type (Table S2.2)
Using control-normalized z-scores for GO term scoring
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata, cell_type_orders_gene10

def main():
    print("Loading GO scores from pre-computed file...")
    # Load the control-normalized GO scores (computed by gene10_GO_term.py)
    go_scores_path = "../intermediate_final_plots/gene10_CBC_GO_score_control.tsv"
    go_scores_df = pd.read_csv(go_scores_path, sep="\t", index_col=0)
    print(f"Loaded GO scores with shape: {go_scores_df.shape}")

    # Load adata to get cell type annotations
    print("Loading adata to get cell type annotations...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=True)

    # Ensure cell barcodes match
    common_cells = go_scores_df.index.intersection(adata.obs.index)
    print(f"Found {len(common_cells)} cells in common")

    # Subset to common cells
    go_scores_df = go_scores_df.loc[common_cells]
    adata = adata[common_cells].copy()

    # Add cell type annotations to GO scores
    go_scores_df['cell_type'] = adata.obs['cell_type'].values
    go_scores_df['perturbation'] = adata.obs['perturbation'].values
    go_scores_df['guide'] = adata.obs['guide'].values

    # Identify control cells (non-targeting guides: NT, NO_SITE, or INTERGENIC)
    control_cells = (go_scores_df['perturbation'] == 'NT') | \
                    go_scores_df['guide'].str.contains('NO_SITE', na=False) | \
                    go_scores_df['guide'].str.contains('INTERGENIC', na=False)
    print(f"Found {control_cells.sum()} control cells")

    # Get GO term columns (exclude metadata columns)
    go_term_columns = [col for col in go_scores_df.columns
                       if col not in ['cell_type', 'perturbation', 'guide']]
    print(f"Processing {len(go_term_columns)} GO terms")

    # Calculate statistics for each GO term and cell type
    results = []

    for cell_type in cell_type_orders_gene10:
        print(f"Processing cell type: {cell_type}")

        # Get cells for this cell type
        ct_mask = go_scores_df['cell_type'] == cell_type

        # Get perturbed cells for this cell type (non-control)
        perturbed_mask = ct_mask & ~control_cells

        # Get control cells for this cell type
        control_mask = ct_mask & control_cells

        if perturbed_mask.sum() == 0 or control_mask.sum() == 0:
            print(f"  Skipping {cell_type}: insufficient cells")
            continue

        for go_term in go_term_columns:
            # Get scores for perturbed and control cells
            perturbed_scores = go_scores_df.loc[perturbed_mask, go_term].values
            control_scores = go_scores_df.loc[control_mask, go_term].values

            # Remove NaN values
            perturbed_scores = perturbed_scores[~np.isnan(perturbed_scores)]
            control_scores = control_scores[~np.isnan(control_scores)]

            if len(perturbed_scores) == 0 or len(control_scores) == 0:
                continue

            # Calculate mean scores
            mean_perturbed = np.mean(perturbed_scores)
            mean_control = np.mean(control_scores)
            mean_difference = mean_perturbed - mean_control

            # Perform Mann-Whitney U test
            try:
                statistic, p_value = mannwhitneyu(perturbed_scores, control_scores,
                                                   alternative='two-sided')
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(perturbed_scores), len(control_scores)
                effect_size = 1 - (2 * statistic) / (n1 * n2)
            except:
                p_value = 1.0
                effect_size = 0.0

            results.append({
                'GO_Term': go_term,
                'Cell_Type': cell_type,
                'Mean_Score_Perturbed': mean_perturbed,
                'Mean_Score_Control': mean_control,
                'Mean_Difference': mean_difference,
                'P_value': p_value,
                'Effect_Size': effect_size,
                'N_Perturbed': len(perturbed_scores),
                'N_Control': len(control_scores)
            })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Apply multiple testing correction within each cell type
    print("Applying multiple testing correction...")
    for cell_type in results_df['Cell_Type'].unique():
        ct_mask = results_df['Cell_Type'] == cell_type
        p_values = results_df.loc[ct_mask, 'P_value'].values

        # Benjamini-Hochberg correction
        _, adjusted_p, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
        results_df.loc[ct_mask, 'Adjusted_P_value'] = adjusted_p

    # Sort by cell type, then by adjusted p-value
    results_df = results_df.sort_values(['Cell_Type', 'Adjusted_P_value', 'P_value'])

    # Filter to significant results only (adjusted p < 0.05)
    significant_df = results_df[results_df['Adjusted_P_value'] < 0.05].copy()

    print(f"\nFound {len(significant_df)} significant GO term-cell type combinations")
    print(f"Out of {len(results_df)} total combinations tested")

    # Save the table
    output_path = "../manuscript/supplementary_tables/2_2.tsv"
    significant_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved supplementary table to {output_path}")

    # Print summary statistics
    print("\nSummary by cell type:")
    for ct in cell_type_orders_gene10:
        ct_data = significant_df[significant_df['Cell_Type'] == ct]
        if len(ct_data) > 0:
            print(f"  {ct}: {len(ct_data)} significant GO terms")

    print("\nTop 5 most significant GO terms overall:")
    # Sort to get top 5 instead of nsmallest to avoid dtype issues
    top_go = significant_df.sort_values('Adjusted_P_value').head(5)[['GO_Term', 'Cell_Type', 'Mean_Difference', 'Adjusted_P_value']]
    # Format for display
    display_df = top_go.copy()
    display_df['Mean_Difference'] = display_df['Mean_Difference'].round(4)
    display_df['Adjusted_P_value'] = display_df['Adjusted_P_value'].map(lambda x: f"{x:.2e}")
    print(display_df.to_string(index=False))

if __name__ == "__main__":
    main()
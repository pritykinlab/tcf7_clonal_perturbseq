#!/usr/bin/env python
"""
Generate scatter plot of initial guide abundance vs number of clones for Trim28/Setdb1 dataset (Figure 5b)
"""

import sys
import pandas as pd

# Add current directory to path to import the modular script
sys.path.append('../manuscript/scripts')
from guide_abundance_correlation_analysis import perform_analysis, format_numeric_suffix_label

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata

def convert_long_to_short(long_label):
    """Convert long guide label to short form"""
    guide_label_mapping = {
        'NO_SITE_26': 'NO_SITE-1',
        'NO_SITE_80': 'NO_SITE-2',
        'NO_SITE_223': 'NO_SITE-3',
        'NO_SITE_913': 'NO_SITE-4',
        'NO_SITE_738': 'NO_SITE-5',
        'ONE_INTERGENIC_SITE_1': 'INTERGENIC-1',
        'ONE_INTERGENIC_SITE_21': 'INTERGENIC-2',
        'ONE_INTERGENIC_SITE_5': 'INTERGENIC-3',
        'ONE_INTERGENIC_SITE_24': 'INTERGENIC-4',
        'ONE_INTERGENIC_SITE_18': 'INTERGENIC-5',
        'NO_SITE_5': 'NO_SITE-6',
        'NO_SITE_3': 'NO_SITE-7',
        'NO_SITE_1': 'NO_SITE-8',
        'NO_SITE_2': 'NO_SITE-9',
        'NO_SITE_4': 'NO_SITE-10',
        'NO_SITE_6': 'NO_SITE-11',
    }
    return guide_label_mapping.get(long_label, long_label)

def main():
    # Load data
    print("Loading trim28_setdb1 data...")
    adata = get_adata(
        dataset='trim28_setdb1',
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True
    )

    # Load guide concentration data
    print("Loading guide concentration data for trim28_setdb1...")
    guide_conc_df = pd.read_csv(
        "../data/guide_concentration/trim28_setdb1.csv",
        sep='\t'  # Tab-separated file
    )

    # Check column names and use appropriate one
    if "CP14_input" in guide_conc_df.columns:
        conc_column = "CP14_input"
    elif "cp14 raw counts" in guide_conc_df.columns:
        conc_column = "cp14 raw counts"
    else:
        # Use first numeric column that looks like counts
        numeric_cols = guide_conc_df.select_dtypes(include=['int64', 'float64']).columns
        conc_column = numeric_cols[0] if len(numeric_cols) > 0 else "cp14 raw counts"
        print(f"Warning: Using column '{conc_column}' for concentration data")

    guide_conc_df["short_guide"] = guide_conc_df["Row Barcode IDs"].apply(convert_long_to_short)
    guide_conc_dict = guide_conc_df.set_index("short_guide")[conc_column].to_dict()

    # Perform analysis
    perform_analysis(
        adata=adata,
        guide_conc_dict=guide_conc_dict,
        dataset_name='trim28_setdb1',
        analysis_type='clones',
        figure_num='5',
        panel_letter='b',
        table_suffix='5_1',
        remove_guides=['INTERGENIC-3'],  # Remove INTERGENIC-3 as specified in notebook
        x_scale_factor=1e4,
        x_axis_label="initial guide abundance (reads x 10^4)",
        xtick_values=[0, 0.5, 1.0, 1.5, 2.0],
        ko_color_mode='by_perturbation',
        control_label_formatter=format_numeric_suffix_label,
        ko_label_formatter=format_numeric_suffix_label,
    )

    print("\nFigure 5b completed successfully!")

if __name__ == "__main__":
    main()

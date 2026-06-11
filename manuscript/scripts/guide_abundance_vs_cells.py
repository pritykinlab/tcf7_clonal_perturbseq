#!/usr/bin/env python
"""
Generate scatter plot of initial guide abundance vs number of cells (Figure 4a)
"""

import sys
import argparse
import pandas as pd

# Add current directory to path to import the modular script
sys.path.append('../manuscript/scripts')
from guide_abundance_correlation_analysis import perform_analysis

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../manuscript/plots/Figure4",
    )
    parser.add_argument("--panel-letter", default="a")
    parser.add_argument(
        "--include-all-guides",
        action="store_true",
        help="Retain INTERGENIC-3 instead of applying the Figure 4 exclusion.",
    )
    parser.add_argument(
        "--skip-table",
        action="store_true",
        help="Do not write a supplementary table for this plot.",
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

    # Load data
    print("Loading gene10 data...")
    adata = get_adata(
        dataset='gene_10',
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True
    )

    # Load guide concentration data
    print("Loading guide concentration data...")
    guide_conc_df = pd.read_csv(
        "../data/guide_concentration/gene10.csv"
    )
    guide_conc_df["short_guide"] = guide_conc_df["Row Barcode IDs"].apply(convert_long_to_short)
    guide_conc_dict = guide_conc_df.set_index("short_guide")["cp8 raw counts"].to_dict()

    # Perform analysis
    perform_analysis(
        adata=adata,
        guide_conc_dict=guide_conc_dict,
        dataset_name='gene10',
        analysis_type='cells',
        figure_num='4',
        panel_letter=args.panel_letter,
        table_suffix=None if args.skip_table else '4_1',
        remove_guides=[] if args.include_all_guides else ['INTERGENIC-3'],
        x_scale_factor=1e4,
        x_axis_label="initial guide abundance (reads x 10^4)",
        xtick_values=[1, 2, 3, 4, 5],
        output_dir=args.output_dir,
        highlight_guides=args.highlight_guide,
    )

    print("\nFigure 4a completed successfully!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Figure S10a: initial guide abundance versus recovered cell counts.
"""

import sys

import pandas as pd

sys.path.append("../manuscript/scripts")
from guide_abundance_correlation_analysis import perform_analysis, format_numeric_nt_label  # noqa: E402

sys.path.append("../scripts/utils")
from figure_format import get_adata  # noqa: E402


def convert_long_to_short(long_label: str) -> str:
    guide_label_mapping = {
        "NO_SITE_26": "NO_SITE-1",
        "NO_SITE_80": "NO_SITE-2",
        "NO_SITE_223": "NO_SITE-3",
        "NO_SITE_913": "NO_SITE-4",
        "NO_SITE_738": "NO_SITE-5",
        "ONE_INTERGENIC_SITE_1": "INTERGENIC-1",
        "ONE_INTERGENIC_SITE_21": "INTERGENIC-2",
        "ONE_INTERGENIC_SITE_5": "INTERGENIC-3",
        "ONE_INTERGENIC_SITE_24": "INTERGENIC-4",
        "ONE_INTERGENIC_SITE_18": "INTERGENIC-5",
        "NO_SITE_5": "NO_SITE-6",
        "NO_SITE_3": "NO_SITE-7",
        "NO_SITE_1": "NO_SITE-8",
        "NO_SITE_2": "NO_SITE-9",
        "NO_SITE_4": "NO_SITE-10",
        "NO_SITE_6": "NO_SITE-11",
    }
    return guide_label_mapping.get(long_label, long_label)


def main() -> None:
    print("Loading trim28_setdb1 data...")
    adata = get_adata(
        dataset="trim28_setdb1",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    print("Loading guide concentration data for trim28_setdb1...")
    guide_conc_df = pd.read_csv(
        "../data/guide_concentration/trim28_setdb1.csv",
        sep="\t",
    )

    if "CP14_input" in guide_conc_df.columns:
        conc_column = "CP14_input"
    elif "cp14 raw counts" in guide_conc_df.columns:
        conc_column = "cp14 raw counts"
    else:
        numeric_cols = guide_conc_df.select_dtypes(include=["int64", "float64"]).columns
        conc_column = numeric_cols[0] if len(numeric_cols) > 0 else "cp14 raw counts"
        print(f"Warning: Using column '{conc_column}' for concentration data")

    guide_conc_df["short_guide"] = guide_conc_df["Row Barcode IDs"].apply(convert_long_to_short)
    guide_conc_dict = guide_conc_df.set_index("short_guide")[conc_column].to_dict()

    perform_analysis(
        adata=adata,
        guide_conc_dict=guide_conc_dict,
        dataset_name="trim28_setdb1",
        analysis_type="cells",
        figure_num="5",
        panel_letter="a",
        table_suffix=None,
        remove_guides=["INTERGENIC-3"],
        x_scale_factor=1e4,
        x_axis_label="initial guide abundance (reads x 10^4)",
        xtick_values=[0, 0.5, 1.0, 1.5, 2.0],
        output_dir="../manuscript/plots/Sup_Figure_10",
        label_ko_guides=False,
        ko_color_mode="by_perturbation",
        control_label_formatter=format_numeric_nt_label,
    )

    print("\nFigure S10a completed successfully!")


if __name__ == "__main__":
    main()

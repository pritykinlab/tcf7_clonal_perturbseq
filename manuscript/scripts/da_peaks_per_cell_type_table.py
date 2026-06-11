#!/usr/bin/env python
"""
Generate supplementary table of differentially accessible peaks per cell type
for the multiome ATAC dataset (Table S6.1).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../scripts/utils')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

from figure_format import get_adata, cell_type_orders_multiome
import DE_testing


OUTPUT_PATH = "../manuscript/supplementary_tables/6_1.tsv"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading multiome ATAC data...")
    adata = get_adata("multiome_atac")

    peak_metadata = adata.var[["chr", "start", "end", "gene"]].rename(
        columns={"gene": "Nearest_Gene"}
    )

    all_results = []
    for cell_type in cell_type_orders_multiome:
        if cell_type not in adata.obs["cell_type"].values:
            continue

        cell_mask = adata.obs["cell_type"] == cell_type
        other_mask = ~cell_mask
        cell_cbcs = adata.obs_names[cell_mask].tolist()
        other_cbcs = adata.obs_names[other_mask].tolist()

        print(
            f"Running DA test for {cell_type}: "
            f"{len(cell_cbcs)} cells vs {len(other_cbcs)} other cells"
        )

        results_df = DE_testing.wilcoxon_DE_test(
            adata,
            cell_cbcs,
            other_cbcs,
            norm_layer="norm",
        )
        results_df["Cell_Type"] = cell_type
        all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[combined["Adjusted P-value"] < 0.05].copy()
    combined = combined.merge(
        peak_metadata,
        left_on="Gene",
        right_index=True,
        how="left",
    )
    combined = combined.rename(columns={"Gene": "Peak"})
    combined = combined.sort_values(
        ["Cell_Type", "Log2FC", "Peak"],
        ascending=[True, False, True],
    )
    combined = combined[
        [
            "Cell_Type",
            "Peak",
            "chr",
            "start",
            "end",
            "Nearest_Gene",
            "Log2FC",
            "P-value",
            "Adjusted P-value",
            "rank_sum_statistic",
        ]
    ].rename(columns={"chr": "Chr", "start": "Start", "end": "End"})

    combined.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Saved {len(combined)} significant cell-type peak rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

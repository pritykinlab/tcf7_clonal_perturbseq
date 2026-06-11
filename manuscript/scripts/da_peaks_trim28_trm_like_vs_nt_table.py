#!/usr/bin/env python
"""
Generate supplementary table of differentially accessible peaks for the
Figure 6d comparison (Table S6.2).
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

from figure_format import get_adata
import DE_testing


OUTPUT_PATH = "../manuscript/supplementary_tables/6_2.tsv"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading multiome ATAC data...")
    adata = get_adata("multiome_atac")
    peak_metadata = adata.var[["chr", "start", "end", "gene"]].rename(
        columns={"gene": "Nearest_Gene"}
    )

    mask_trm_like = (
        (adata.obs["cell_type"] == "TRM-like")
        & (adata.obs["condition"] == "Trim28")
    )
    mask_control_non_trm = (
        (adata.obs["cell_type"] != "TRM-like")
        & (adata.obs["condition"] == "Control")
    )

    trm_like_cbcs = adata.obs_names[mask_trm_like].tolist()
    control_cbcs = adata.obs_names[mask_control_non_trm].tolist()
    print(
        "Running Figure 6d DA test: "
        f"{len(trm_like_cbcs)} Trim28 TRM-like cells vs "
        f"{len(control_cbcs)} NT non-TRM cells"
    )

    results_df = DE_testing.wilcoxon_DE_test(
        adata,
        trm_like_cbcs,
        control_cbcs,
        norm_layer="norm",
    )
    results_df = results_df[results_df["Adjusted P-value"] < 0.05].copy()
    results_df = results_df.merge(
        peak_metadata,
        left_on="Gene",
        right_index=True,
        how="left",
    )
    results_df = results_df.rename(columns={"Gene": "Peak"})
    results_df = results_df.sort_values(
        ["Log2FC", "Peak"],
        ascending=[False, True],
    )
    results_df = results_df[
        [
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

    results_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Saved {len(results_df)} significant peaks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

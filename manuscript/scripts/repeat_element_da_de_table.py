#!/usr/bin/env python
"""
Generate supplementary table of repeat-element differential accessibility and
differential expression for the Figure 6g comparison (Table S6.3).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

sys.path.append('../scripts/utils')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

import DE_testing


OUTPUT_PATH = "../manuscript/supplementary_tables/6_3.tsv"
RNA_REPEAT_PATH = (
    "../data/repeat_enrichment/multiome/"
    "multiome_rna_repeat_agg_norm_adata.h5ad"
)
ATAC_REPEAT_PATH = (
    "../data/repeat_enrichment/multiome/"
    "multiome_atac_repeat_agg_norm_adata.h5ad"
)


def _get_groups(adata):
    mask_trm_like = (
        (adata.obs["cell_type"] == "TRM-like")
        & (adata.obs["condition"] == "Trim28")
    )
    mask_control_non_trm = (
        (adata.obs["cell_type"] != "TRM-like")
        & (adata.obs["condition"] == "Control")
    )
    return (
        adata.obs_names[mask_trm_like].tolist(),
        adata.obs_names[mask_control_non_trm].tolist(),
    )


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading repeat-element RNA and ATAC data...")
    rna_adata = sc.read_h5ad(RNA_REPEAT_PATH)
    atac_adata = sc.read_h5ad(ATAC_REPEAT_PATH)

    rna_group1, rna_group2 = _get_groups(rna_adata)
    atac_group1, atac_group2 = _get_groups(atac_adata)
    print(
        f"RNA comparison: {len(rna_group1)} Trim28 TRM-like vs "
        f"{len(rna_group2)} NT non-TRM"
    )
    print(
        f"ATAC comparison: {len(atac_group1)} Trim28 TRM-like vs "
        f"{len(atac_group2)} NT non-TRM"
    )

    rna_results = DE_testing.wilcoxon_DE_test(
        rna_adata,
        rna_group1,
        rna_group2,
        norm_layer="norm",
    ).rename(
        columns={
            "Gene": "Repeat_Element",
            "Log2FC": "Log2FC_RNA",
            "rank_sum_statistic": "rank_sum_statistic_RNA",
            "P-value": "P-value_RNA",
            "Adjusted P-value": "Adjusted P-value_RNA",
        }
    )
    atac_results = DE_testing.wilcoxon_DE_test(
        atac_adata,
        atac_group1,
        atac_group2,
        norm_layer="norm",
    ).rename(
        columns={
            "Gene": "Repeat_Element",
            "Log2FC": "Log2FC_ATAC",
            "rank_sum_statistic": "rank_sum_statistic_ATAC",
            "P-value": "P-value_ATAC",
            "Adjusted P-value": "Adjusted P-value_ATAC",
        }
    )

    merged = rna_results.merge(atac_results, on="Repeat_Element", how="outer")
    merged["Significant_RNA"] = merged["Adjusted P-value_RNA"].lt(0.05)
    merged["Significant_ATAC"] = merged["Adjusted P-value_ATAC"].lt(0.05)
    merged = merged[merged["Significant_RNA"] | merged["Significant_ATAC"]].copy()
    merged["Significant_Both"] = merged["Significant_RNA"] & merged["Significant_ATAC"]
    merged["Max_Abs_Log2FC"] = merged[["Log2FC_RNA", "Log2FC_ATAC"]].abs().max(axis=1)
    merged = merged.sort_values(
        ["Significant_Both", "Significant_ATAC", "Significant_RNA", "Max_Abs_Log2FC", "Repeat_Element"],
        ascending=[False, False, False, False, True],
    )
    merged = merged[
        [
            "Repeat_Element",
            "Log2FC_RNA",
            "P-value_RNA",
            "Adjusted P-value_RNA",
            "rank_sum_statistic_RNA",
            "Log2FC_ATAC",
            "P-value_ATAC",
            "Adjusted P-value_ATAC",
            "rank_sum_statistic_ATAC",
        ]
    ]

    merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Saved {len(merged)} significant repeat-element rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

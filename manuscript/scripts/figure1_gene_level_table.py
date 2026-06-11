#!/usr/bin/env python
"""
Generate gene-level Figure 1 supplementary tables from guide-level tables.

Each row corresponds to one targeted gene. Non-targeting and intergenic
controls are excluded. Exact guide-level negative-binomial p-values are
converted to signed normal-equivalent z-scores, then combined with a signed
Stouffer method to produce a gene-level z-score and p-value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


CONTROL_PREFIXES = ("NO_SITE", "ONE_INTERGENIC")
REQUIRED_COLUMNS = {
    "guide",
    "tcf7_negative_count",
    "tcf7_positive_count",
    "expected_mean_tcf7_positive",
    "z_score",
    "pvalue",
    "bh_adj_pvalue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def extract_gene(guide_name: str) -> str:
    return guide_name.rsplit("-", 1)[0]


def is_control_guide(guide_name: str) -> bool:
    return guide_name.startswith(CONTROL_PREFIXES)


def signed_normal_equivalent_zscores(df: pd.DataFrame) -> pd.Series:
    """Convert two-sided p-values to signed normal-equivalent z-scores."""
    pvalues = pd.to_numeric(df["pvalue"], errors="raise").astype(float)
    signs = np.sign(pd.to_numeric(df["z_score"], errors="raise").astype(float))
    clipped = np.clip(pvalues, np.finfo(float).tiny, 1.0)
    return pd.Series(signs * norm.isf(clipped / 2), index=df.index)


def load_guide_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {path}: {missing_str}")
    return df


def collapse_to_gene_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["guide"].astype(str).apply(is_control_guide)].copy()
    df["gene"] = df["guide"].astype(str).apply(extract_gene)
    df["stouffer_z_score"] = signed_normal_equivalent_zscores(df)
    df = df.sort_values(["gene", "guide"], kind="mergesort").reset_index(drop=True)

    gene_df = (
        df.groupby("gene", sort=False)
        .apply(
            lambda sub: pd.Series(
                {
                    "combined_z_score": float(
                        np.sum(sub["stouffer_z_score"]) / np.sqrt(len(sub))
                    ),
                    "first_zscore": float(sub["stouffer_z_score"].iloc[0]),
                    "second_zscore": float(sub["stouffer_z_score"].iloc[1]),
                    "first_pvalue": float(sub["pvalue"].iloc[0]),
                    "second_pvalue": float(sub["pvalue"].iloc[1]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    gene_df["pvalue"] = 2 * norm.sf(np.abs(gene_df["combined_z_score"]))
    gene_df["bh_adj_pvalue"] = multipletests(gene_df["pvalue"], method="fdr_bh")[1]

    gene_df = gene_df.sort_values(
        ["combined_z_score", "pvalue"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return gene_df[
        [
            "gene",
            "combined_z_score",
            "first_zscore",
            "second_zscore",
            "first_pvalue",
            "second_pvalue",
            "pvalue",
            "bh_adj_pvalue",
        ]
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    guide_df = load_guide_table(input_path)
    gene_df = collapse_to_gene_level(guide_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(output_path, sep="\t", index=False, float_format="%.6e")

    print(f"Saved gene-level supplementary table to {output_path}")
    print(f"Rows: {len(gene_df)}")


if __name__ == "__main__":
    main()

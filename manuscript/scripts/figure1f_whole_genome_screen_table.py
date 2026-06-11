#!/usr/bin/env python
"""
Generate the Figure 1f whole-genome in vitro Tcf7 screen supplementary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from figure1f_from_old_copy import (
    REPO_ROOT,
    REQUIRED_GUIDES_PER_TARGET_GENE,
    categorize_gene,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "raw_data"
    / "CRISPR_sort_screens"
    / "2_26_2021_BRIE_run1"
    / "hypergeometric"
    / "All_Tcf7_hi-low_100_21-03-06-16-45.txt"
)

RENAME_COLUMNS = {
    "Gene Symbol": "Gene",
    "Average LFC": "source_average_lfc",
    "Average -log(p-values)": "average_neg_log10_pvalue",
    "Number of perturbations": "number_of_perturbations",
    "Individual LFCs": "inidivdual_lfcs",
    "Ascending ranks": "ascending_ranks",
    "Individual ascending -log(p-values)": "ascending_neg_log10_pvalues",
    "Descending ranks": "descending_ranks",
    "Individual descending -log(p-values)": "descending_neg_log10_pvalues",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "manuscript" / "supplementary_tables" / "1_3.tsv"),
    )
    return parser.parse_args()


def parse_float_list(value: object) -> list[float]:
    return [float(item) for item in str(value).split(";")]


def selected_tail(row: pd.Series) -> str:
    ascending_scores = parse_float_list(row["ascending_neg_log10_pvalues"])
    descending_scores = parse_float_list(row["descending_neg_log10_pvalues"])
    ascending_mean = sum(ascending_scores) / len(ascending_scores)
    descending_mean = sum(descending_scores) / len(descending_scores)
    if ascending_mean >= descending_mean:
        return "ascending"
    return "descending"


def load_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t").rename(columns=RENAME_COLUMNS)
    required = list(RENAME_COLUMNS.values())
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return table


def format_screen_table(path: Path) -> pd.DataFrame:
    table = load_table(path)
    table["category"] = table["Gene"].map(categorize_gene)
    table = table[
        (table["category"] != "target")
        | (table["number_of_perturbations"] == REQUIRED_GUIDES_PER_TARGET_GENE)
    ].copy()
    table["average_lfc_log2"] = table["source_average_lfc"]
    table["selected_tail"] = table.apply(selected_tail, axis=1)

    table = table.sort_values(
        "average_lfc_log2",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    return table[
        [
            "Gene",
            "average_lfc_log2",
            "inidivdual_lfcs",
            "average_neg_log10_pvalue",
            "selected_tail",
            "ascending_ranks",
            "ascending_neg_log10_pvalues",
            "descending_ranks",
            "descending_neg_log10_pvalues",
        ]
    ]


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    table = format_screen_table(Path(args.input))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False, float_format="%.6g")
    print(f"Saved Figure 1f supplementary table to {output}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()

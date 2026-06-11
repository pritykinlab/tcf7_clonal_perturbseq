#!/usr/bin/env python
"""
Generate the Figure 1e pilot in vitro Tcf7 screen supplementary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pilot_tcf7_validation_screens import (
    SCREENS,
    load_screen,
)


RENAME_COLUMNS = {
    "Gene Symbol": "gene_symbol",
    "Average LFC": "source_average_lfc",
    "Average -log(p-values)": "average_neg_log10_pvalue",
    "Individual LFCs": "inidivdual_lfcs",
    "Ascending ranks": "ascending_ranks",
    "Individual ascending -log(p-values)": "ascending_neg_log10_pvalues",
    "Descending ranks": "descending_ranks",
    "Individual descending -log(p-values)": "descending_neg_log10_pvalues",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-stem",
        choices=[config["stem"] for config in SCREENS],
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def screen_config(screen_stem: str) -> dict[str, object]:
    for config in SCREENS:
        if config["stem"] == screen_stem:
            return config
    raise ValueError(f"Unknown screen stem: {screen_stem}")


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


def format_screen_table(screen_stem: str) -> pd.DataFrame:
    config = screen_config(screen_stem)
    table = load_screen(Path(config["path"]))
    table = table.rename(columns=RENAME_COLUMNS).copy()
    table["average_lfc_log2"] = table["x"]
    table["selected_tail"] = table.apply(selected_tail, axis=1)
    table = table.sort_values(
        "average_lfc_log2",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    return table.rename(columns={"gene_symbol": "Gene"})[
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
    table = format_screen_table(args.screen_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False, float_format="%.6g")
    print(f"Saved Figure 1e {args.screen_stem} supplementary table to {output}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()

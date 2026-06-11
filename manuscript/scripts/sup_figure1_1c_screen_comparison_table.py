#!/usr/bin/env python
"""Generate the Figure S2a screen-comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from figure1f_whole_genome_screen_table import format_screen_table


REPO_ROOT = Path(__file__).resolve().parents[2]

SCREEN_SPECS = [
    {
        "comparison_label": "D7 Tcf7 high vs low, acute stimulation",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D7tcf7hi-low_100_21-03-01-18-00.txt",
    },
    {
        "comparison_label": "D7 Tcf7 high vs low, chronic stimulation",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D7conttcf7hi-low_100_21-03-01-18-04.txt",
    },
    {
        "comparison_label": "D5 Prdm1 high vs low, chronic stimulation",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D5contprdm1hi-low_100_21-03-01-18-17.txt",
    },
    {
        "comparison_label": "D7 Prdm1 high vs low",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D7prdm1hi-low_100_21-03-01-18-03.txt",
    },
    {
        "comparison_label": "D7 Eomes high vs low",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D7eomeshi-low_100_21-03-01-18-01.txt",
    },
    {
        "comparison_label": "D5 Tox high vs low, hypoxic stimulation",
        "path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "2_26_2021_BRIE_run1"
        / "hypergeometric"
        / "D5hyptoxhi-low_100_21-03-01-18-19.txt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "manuscript" / "supplementary_tables" / "1_4.tsv"),
    )
    return parser.parse_args()


def format_comparison_table() -> pd.DataFrame:
    frames = []
    for spec in SCREEN_SPECS:
        table = format_screen_table(spec["path"])
        table.insert(0, "comparison_label", spec["comparison_label"])
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    table = format_comparison_table()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False, float_format="%.6g")
    print(f"Saved Figure S2a screen-comparison table to {output}")
    print(f"Rows: {len(table)}")
    print(f"Comparisons: {table['comparison_label'].nunique()}")


if __name__ == "__main__":
    main()

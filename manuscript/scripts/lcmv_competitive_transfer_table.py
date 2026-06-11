#!/usr/bin/env python
"""Generate the Figure 1d LCMV competitive-transfer source-data table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


OUTPUT_COLUMNS = [
    "timepoint",
    "measurement",
    "metric",
    "group",
    "mouse",
    "input_control_pct",
    "input_comparator_pct",
    "input_comparator_to_control_ratio",
    "output_control_pct",
    "output_comparator_pct",
    "output_comparator_to_control_ratio",
    "output_to_input_ratio",
    "value",
]

NOT_APPLICABLE = "NA"

OUTPUT_COLUMN_RENAMES = {
    "control_input_pct": "input_control_pct",
    "experimental_input_pct": "input_comparator_pct",
    "input_ratio_experimental_over_control": "input_comparator_to_control_ratio",
    "control_output_pct": "output_control_pct",
    "experimental_output_pct": "output_comparator_pct",
    "output_ratio_experimental_over_control": "output_comparator_to_control_ratio",
    "output_over_input_ratio": "output_to_input_ratio",
}

TIMEPOINT_DISPLAY = {
    "D7": "day 7",
    "D30": "day 30",
}

GROUP_DISPLAY = {
    "Control": "Control",
    "Fresh": "Ctrl/Ctrl benchmark",
    "No Stim": "Lenti naive",
    "Activated": "Lenti activated",
}

MEASUREMENT_DISPLAY = {
    "relative_accumulation": "relative accumulation",
    "SLEC": "acute effector differentiation",
    "TCM": "central memory formation",
}

PANEL_ORDER = {
    ("day 7", "relative accumulation", "log2(output / input ratio)"): 0,
    ("day 7", "acute effector differentiation", "% SLEC among P14"): 1,
    ("day 30", "relative accumulation", "log2(output / input ratio)"): 2,
    ("day 30", "central memory formation", "% TCM among P14"): 3,
}

GROUP_ORDER = {
    "Control": 0,
    "Lenti naive": 1,
    "Lenti activated": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tidy-values", required=True, type=Path)
    parser.add_argument("--input-ratios", required=True, type=Path)
    parser.add_argument("--summary-workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(required_columns - set(df.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing columns in {path}: {missing_str}")
    return df


def format_float(value: float) -> str:
    return f"{value:.12g}"


def keep_figure1d_rows(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    relative_mask = (
        (table["assay"] == "relative_accumulation")
        & table["comparison"].isin(["No Stim", "Activated"])
    )
    phenotype_mask = (
        (table["assay"] == "phenotype")
        & table["condition"].isin(["Control", "No Stim", "Activated"])
        & table["comparison"].isin(["No Stim", "Activated"])
        & (
            ((table["timepoint"] == "D7") & (table["metric"] == "SLEC"))
            | ((table["timepoint"] == "D30") & (table["metric"] == "TCM"))
        )
    )
    table = table.loc[relative_mask | phenotype_mask].copy()

    table["group"] = table["condition"].map(GROUP_DISPLAY)
    table["measurement"] = table["assay"].map(MEASUREMENT_DISPLAY)
    phenotype_rows = table["assay"] == "phenotype"
    table.loc[phenotype_rows, "measurement"] = table.loc[phenotype_rows, "metric"].map(
        MEASUREMENT_DISPLAY
    )
    table.loc[phenotype_rows, "metric"] = (
        "% " + table.loc[phenotype_rows, "metric"] + " among P14"
    )
    table["timepoint"] = table["timepoint"].map(TIMEPOINT_DISPLAY)
    table["mouse"] = (
        table.groupby(["timepoint", "measurement", "metric", "group"]).cumcount() + 1
    )
    table["_panel_order"] = table.apply(
        lambda row: PANEL_ORDER[(row["timepoint"], row["measurement"], row["metric"])],
        axis=1,
    )
    table["_group_order"] = table["group"].map(GROUP_ORDER)
    table = table.sort_values(["_panel_order", "_group_order", "mouse"]).drop(
        columns=["_panel_order", "_group_order"]
    )
    return table


def read_relative_summary(summary_workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(summary_workbook, sheet_name="Sheet1", header=None)
    records: list[dict[str, str]] = []

    for timepoint in ("D7", "D30"):
        timepoint_rows = df.index[df[0].astype(str).str.strip() == timepoint].tolist()
        if not timepoint_rows:
            raise ValueError(f"Could not find {timepoint} section in {summary_workbook}")

        row_idx = timepoint_rows[0] + 1
        if str(df.iloc[row_idx, 0]).strip() == "Sample:":
            row_idx += 1

        mouse_counts: dict[str, int] = {}
        while row_idx < len(df) and pd.notna(df.iloc[row_idx, 0]):
            sample_label = str(df.iloc[row_idx, 0]).strip()
            if not sample_label:
                break
            label_parts = sample_label.split(maxsplit=1)
            if len(label_parts) != 2:
                raise ValueError(f"Unexpected sample label: {sample_label}")

            comparison = label_parts[1]
            mouse_counts[comparison] = mouse_counts.get(comparison, 0) + 1
            records.append(
                {
                    "timepoint": timepoint,
                    "comparison": comparison,
                    "mouse": str(mouse_counts[comparison]),
                    "control_output_pct": format_float(float(df.iloc[row_idx, 1])),
                    "experimental_output_pct": format_float(float(df.iloc[row_idx, 4])),
                }
            )
            row_idx += 1

    return pd.DataFrame.from_records(records)


def add_relative_accumulation_components(
    table: pd.DataFrame,
    input_ratios: pd.DataFrame,
    relative_summary: pd.DataFrame,
) -> pd.DataFrame:
    table = table.copy()
    key_columns = ["timepoint", "comparison", "mouse"]
    control_output = (
        relative_summary.set_index(key_columns)["control_output_pct"].to_dict()
    )
    experimental_output = (
        relative_summary.set_index(key_columns)["experimental_output_pct"].to_dict()
    )
    input_ratio_lookup = input_ratios.set_index("comparison").to_dict("index")

    p14_mask = (
        (table["assay"] == "percent_p14_cd8")
        & (table["metric"] == "% P14 among CD8")
    )
    for index, row in table.loc[p14_mask].iterrows():
        key = (row["timepoint"], row["comparison"], row["mouse"])
        if row["condition"] == "Control" and key in control_output:
            table.loc[index, "value"] = control_output[key]
        elif row["condition"] == row["comparison"] and key in experimental_output:
            table.loc[index, "value"] = experimental_output[key]

    mask = table["assay"] == "relative_accumulation"
    for index, row in table.loc[mask].iterrows():
        key = (row["timepoint"], row["comparison"], row["mouse"])
        if key not in control_output or key not in experimental_output:
            raise ValueError(f"Missing P14 output percentage for {key}")
        if row["comparison"] not in input_ratio_lookup:
            raise ValueError(f"Missing input ratio for {row['comparison']}")

        inputs = input_ratio_lookup[row["comparison"]]
        control_input_pct = float(inputs["control_input_pct"])
        experimental_input_pct = float(inputs["experimental_input_pct"])
        control_output_pct = float(control_output[key])
        experimental_output_pct = float(experimental_output[key])
        input_ratio = experimental_input_pct / control_input_pct
        output_ratio = experimental_output_pct / control_output_pct
        output_over_input_ratio = output_ratio / input_ratio
        calculated_value = math.log2(output_over_input_ratio)
        reported_value = float(row["value"])
        if not math.isclose(
            calculated_value,
            reported_value,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "Relative accumulation mismatch for "
                f"{key}: calculated {calculated_value}, reported {reported_value}"
            )

        table.loc[index, "control_input_pct"] = inputs["control_input_pct"]
        table.loc[index, "experimental_input_pct"] = inputs["experimental_input_pct"]
        table.loc[index, "input_ratio_experimental_over_control"] = format_float(
            input_ratio
        )
        table.loc[index, "control_output_pct"] = control_output[key]
        table.loc[index, "experimental_output_pct"] = experimental_output[key]
        table.loc[index, "output_ratio_experimental_over_control"] = format_float(
            output_ratio
        )
        table.loc[index, "output_over_input_ratio"] = format_float(
            output_over_input_ratio
        )

    return table


def build_measurement_rows(
    tidy_values: pd.DataFrame,
    input_ratios: pd.DataFrame,
    relative_summary: pd.DataFrame,
) -> pd.DataFrame:
    table = tidy_values.copy()
    table["record_type"] = table["assay"]
    table["control_input_pct"] = NOT_APPLICABLE
    table["experimental_input_pct"] = NOT_APPLICABLE
    table["input_ratio_experimental_over_control"] = NOT_APPLICABLE
    table["control_output_pct"] = NOT_APPLICABLE
    table["experimental_output_pct"] = NOT_APPLICABLE
    table["output_ratio_experimental_over_control"] = NOT_APPLICABLE
    table["output_over_input_ratio"] = NOT_APPLICABLE
    table = add_relative_accumulation_components(table, input_ratios, relative_summary)
    table = keep_figure1d_rows(table)
    table = table.rename(columns=OUTPUT_COLUMN_RENAMES)
    return table.loc[:, OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    tidy_values = read_tsv(
        args.tidy_values,
        {
            "timepoint",
            "assay",
            "metric",
            "condition",
            "comparison",
            "mouse",
            "value",
            "timepoint_label",
        },
    )
    input_ratios = read_tsv(
        args.input_ratios,
        {"comparison", "control_input_pct", "experimental_input_pct"},
    )
    relative_summary = read_relative_summary(args.summary_workbook)

    table = build_measurement_rows(
        tidy_values,
        input_ratios,
        relative_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()

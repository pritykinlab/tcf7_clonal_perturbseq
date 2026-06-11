#!/usr/bin/env python
"""Generate Figure 1d from the LCMV competitive-transfer Prism workbook.

The Prism workbook stores three pairwise competitive transfers:
1. Control vs Fresh (control-control benchmark)
2. Control vs No Stim (naive-transduced)
3. Control vs Activated (CD3/CD28 pre-activated)

For the one-way abundance tables, the control column is laid out as three
consecutive 5-mouse blocks in that same order. For the two-way phenotype
tables, the comparator rows each contain 5 mouse values while the control row
contains 15 values, again interpreted as three consecutive 5-mouse blocks.
This script makes that assumption explicit and exports the parsed data.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRISM = (
    REPO_ROOT
    / "Figure1_data_handoff"
    / "04_lcmv_competitive_transfer"
    / "prism_and_summary"
    / "20200311_Frequencies.pzfx"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"

COMPARISON_ORDER = ["Fresh", "No Stim", "Activated"]
COMPARISON_LABELS = {
    "Fresh": "Ctrl/Ctrl\nbenchmark",
    "No Stim": "Lenti\nnaive",
    "Activated": "Lenti\nactivated",
}
CONTROL_LABEL = "Control"

COLORS = {
    "Control": "#2F2F2F",
    "Fresh": "#A7B0B8",
    "No Stim": "#0E9384",
    "Activated": "#D26A3D",
}

TIMEPOINT_LABELS = {
    "D7": "day 7",
    "D30": "day 30",
}

FOCUSED_PANEL_SPECS = [
    ("D7", "Relative accumulation", "log2(output / input ratio)", ["No Stim", "Activated"]),
    ("D7", "Acute effector differentiation", "% SLEC among P14", ["Control", "No Stim", "Activated"]),
    ("D30", "Relative accumulation", "log2(output / input ratio)", ["No Stim", "Activated"]),
    ("D30", "Central memory formation", "% TCM among P14", ["Control", "No Stim", "Activated"]),
]

BENCHMARK_PANEL_SPECS = [
    ("D7", "Relative accumulation", "log2(output / input ratio)", ["Fresh", "No Stim", "Activated"]),
    ("D7", "Acute effector differentiation", "% SLEC among P14", ["Control", "Fresh", "No Stim", "Activated"]),
    ("D30", "Relative accumulation", "log2(output / input ratio)", ["Fresh", "No Stim", "Activated"]),
    ("D30", "Central memory formation", "% TCM among P14", ["Control", "Fresh", "No Stim", "Activated"]),
]

SELECTED_METRIC_MAP = {
    ("D7", "Relative accumulation"): ("relative_accumulation", None),
    ("D30", "Relative accumulation"): ("relative_accumulation", None),
    ("D7", "Acute effector differentiation"): ("phenotype", "SLEC"),
    ("D30", "Central memory formation"): ("phenotype", "TCM"),
}

PANEL_TITLE_LABELS = {
    "Relative accumulation": "relative accumulation",
    "Acute effector differentiation": "acute effector differentiation",
    "Central memory formation": "central memory formation",
}

DISPLAY_STAT_TESTS = [
    {
        "timepoint": "D7",
        "panel_title": "Relative accumulation",
        "test_kind": "welch_t",
        "assay": "relative_accumulation",
        "metric": None,
        "condition_a": "No Stim",
        "condition_b": "Activated",
        "comparison_scope": "between_groups",
    },
    {
        "timepoint": "D7",
        "panel_title": "Acute effector differentiation",
        "test_kind": "paired_t",
        "assay": "phenotype",
        "metric": "SLEC",
        "condition_a": "Control",
        "condition_b": "No Stim",
        "comparison_scope": "matched_within_mouse",
    },
    {
        "timepoint": "D7",
        "panel_title": "Acute effector differentiation",
        "test_kind": "paired_t",
        "assay": "phenotype",
        "metric": "SLEC",
        "condition_a": "Control",
        "condition_b": "Activated",
        "comparison_scope": "matched_within_mouse",
    },
    {
        "timepoint": "D30",
        "panel_title": "Relative accumulation",
        "test_kind": "welch_t",
        "assay": "relative_accumulation",
        "metric": None,
        "condition_a": "No Stim",
        "condition_b": "Activated",
        "comparison_scope": "between_groups",
    },
    {
        "timepoint": "D30",
        "panel_title": "Central memory formation",
        "test_kind": "paired_t",
        "assay": "phenotype",
        "metric": "TCM",
        "condition_a": "Control",
        "condition_b": "No Stim",
        "comparison_scope": "matched_within_mouse",
    },
    {
        "timepoint": "D30",
        "panel_title": "Central memory formation",
        "test_kind": "paired_t",
        "assay": "phenotype",
        "metric": "TCM",
        "condition_a": "Control",
        "condition_b": "Activated",
        "comparison_scope": "matched_within_mouse",
    },
]


@dataclass(frozen=True)
class InputRatio:
    comparison: str
    control_input_pct: float
    experimental_input_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1d from the LCMV Prism file.")
    parser.add_argument("--prism", type=Path, default=DEFAULT_PRISM, help="Path to the source Prism .pzfx file.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tidy extracted tables and summaries.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered figure outputs.",
    )
    parser.add_argument("--panel-stem", default="d", help="Output filename stem for the focused Figure 1d panel.")
    parser.add_argument(
        "--panel-version",
        choices=("focused", "benchmark"),
        default="focused",
        help="Panel specification to render as the primary panel-stem output.",
    )
    parser.add_argument("--write-benchmark", action="store_true", help="Also write the control-control benchmark version.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    parser.add_argument("--text-scale", type=float, default=1.0, help="Scale all plot text by this factor.")
    parser.add_argument(
        "--axis-text-scale",
        type=float,
        default=1.0,
        help="Additional scale factor for axis labels and tick labels only.",
    )
    parser.add_argument("--width-scale", type=float, default=1.0, help="Scale rendered figure width by this factor.")
    parser.add_argument("--height-scale", type=float, default=1.0, help="Scale rendered figure height by this factor.")
    parser.add_argument("--title-scale", type=float, default=1.0, help="Scale subplot title text by this factor.")
    parser.add_argument(
        "--layout",
        choices=("auto", "2x2", "vertical"),
        default="auto",
        help="Panel layout. Auto keeps the original 2x2 unless large text is combined with a narrow figure.",
    )
    return parser.parse_args()


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def get_table(root: ET.Element, title: str) -> ET.Element:
    for table in root.findall("Table"):
        if table.findtext("Title") == title:
            return table
    raise KeyError(f"Could not find Prism table titled {title!r}")


def ycolumn_title(ycolumn: ET.Element) -> str:
    title_node = ycolumn.find("Title")
    if title_node is None:
        return ""
    return "".join(title_node.itertext()).strip()


def extract_input_ratios(root: ET.Element) -> list[InputRatio]:
    table = get_table(root, "Input")
    control_values: list[float] = []
    experimental_values: dict[str, float] = {}

    for ycolumn in table.findall("YColumn"):
        title = ycolumn_title(ycolumn)
        values = [safe_float(d.text) for d in ycolumn.find("Subcolumn").findall("d")]
        numeric_values = [value for value in values if value is not None]
        if title == "Control":
            control_values = numeric_values
        else:
            if len(numeric_values) != 1:
                raise ValueError(f"Expected one input value for {title}, found {len(numeric_values)}")
            experimental_values[title] = numeric_values[0]

    if len(control_values) != len(COMPARISON_ORDER):
        raise ValueError(f"Expected {len(COMPARISON_ORDER)} control input values, found {len(control_values)}")

    ratios: list[InputRatio] = []
    for idx, comparison in enumerate(COMPARISON_ORDER):
        ratios.append(
            InputRatio(
                comparison=comparison,
                control_input_pct=control_values[idx],
                experimental_input_pct=experimental_values[comparison],
            )
        )
    return ratios


def extract_relative_accumulation(root: ET.Element) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for timepoint in ("D7", "D30"):
        table = get_table(root, f"{timepoint} Log2 from input")
        for ycolumn in table.findall("YColumn"):
            comparison = ycolumn_title(ycolumn)
            values = [safe_float(d.text) for d in ycolumn.find("Subcolumn").findall("d")]
            numeric_values = [value for value in values if value is not None]
            for mouse_idx, value in enumerate(numeric_values, start=1):
                records.append(
                    {
                        "timepoint": timepoint,
                        "assay": "relative_accumulation",
                        "metric": "log2(output / input ratio)",
                        "condition": comparison,
                        "comparison": comparison,
                        "mouse": mouse_idx,
                        "value": value,
                    }
                )
    return pd.DataFrame.from_records(records)


def extract_percent_p14(root: ET.Element) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for timepoint in ("D7", "D30"):
        table = get_table(root, f"{timepoint} Percent P14 CD8")
        for ycolumn in table.findall("YColumn"):
            condition = ycolumn_title(ycolumn)
            values = [safe_float(d.text) for d in ycolumn.find("Subcolumn").findall("d")]
            numeric_values = [value for value in values if value is not None]

            if condition == CONTROL_LABEL:
                expected = len(COMPARISON_ORDER) * 5
                if len(numeric_values) != expected:
                    raise ValueError(f"Expected {expected} control values in {timepoint} Percent P14 CD8")
                for block_idx, comparison in enumerate(COMPARISON_ORDER):
                    start = block_idx * 5
                    block = numeric_values[start : start + 5]
                    for mouse_idx, value in enumerate(block, start=1):
                        records.append(
                            {
                                "timepoint": timepoint,
                                "assay": "percent_p14_cd8",
                                "metric": "% P14 among CD8",
                                "condition": CONTROL_LABEL,
                                "comparison": comparison,
                                "mouse": mouse_idx,
                                "value": value,
                            }
                        )
            else:
                if len(numeric_values) != 5:
                    raise ValueError(f"Expected 5 values for {condition} in {timepoint} Percent P14 CD8")
                for mouse_idx, value in enumerate(numeric_values, start=1):
                    records.append(
                        {
                            "timepoint": timepoint,
                            "assay": "percent_p14_cd8",
                            "metric": "% P14 among CD8",
                            "condition": condition,
                            "comparison": condition,
                            "mouse": mouse_idx,
                            "value": value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def extract_two_way_phenotypes(root: ET.Element) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    row_order = [CONTROL_LABEL, "Fresh", "No Stim", "Activated"]

    for timepoint, table_title in (
        ("D7", "D7 TE/TEM vs TCM"),
        ("D7", "D7 SLEC vs MPEC"),
        ("D30", "D30 TE/TEM vs TCM"),
        ("D30", "D30 SLEC vs MPEC"),
    ):
        table = get_table(root, table_title)
        for ycolumn in table.findall("YColumn"):
            metric = ycolumn_title(ycolumn)
            subcolumns = ycolumn.findall("Subcolumn")
            if len(subcolumns) != 15:
                raise ValueError(f"Expected 15 subcolumns in {table_title} / {metric}")

            control_values = [safe_float(subcolumn.findall("d")[0].text) for subcolumn in subcolumns]
            if any(value is None for value in control_values):
                raise ValueError(f"Found missing control value in {table_title} / {metric}")

            for block_idx, comparison in enumerate(COMPARISON_ORDER):
                start = block_idx * 5
                control_block = control_values[start : start + 5]
                for mouse_idx, value in enumerate(control_block, start=1):
                    records.append(
                        {
                            "timepoint": timepoint,
                            "assay": "phenotype",
                            "metric": metric,
                            "condition": CONTROL_LABEL,
                            "comparison": comparison,
                            "mouse": mouse_idx,
                            "value": value,
                        }
                    )

                row_idx = row_order.index(comparison)
                experimental_values: list[float] = []
                for subcolumn in subcolumns[:5]:
                    d_nodes = subcolumn.findall("d")
                    value = safe_float(d_nodes[row_idx].text)
                    if value is None:
                        raise ValueError(f"Missing {comparison} value in {table_title} / {metric}")
                    experimental_values.append(value)

                for mouse_idx, value in enumerate(experimental_values, start=1):
                    records.append(
                        {
                            "timepoint": timepoint,
                            "assay": "phenotype",
                            "metric": metric,
                            "condition": comparison,
                            "comparison": comparison,
                            "mouse": mouse_idx,
                            "value": value,
                        }
                    )

    return pd.DataFrame.from_records(records)


def build_tidy_tables(root: ET.Element) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_ratios = pd.DataFrame([ratio.__dict__ for ratio in extract_input_ratios(root)])
    relative_accumulation = extract_relative_accumulation(root)
    percent_p14 = extract_percent_p14(root)
    phenotypes = extract_two_way_phenotypes(root)
    tidy = pd.concat([relative_accumulation, percent_p14, phenotypes], ignore_index=True)
    tidy["timepoint_label"] = tidy["timepoint"].map(TIMEPOINT_LABELS)
    return input_ratios, tidy, phenotypes


def build_control_split_summary(phenotypes: pd.DataFrame) -> pd.DataFrame:
    controls = phenotypes.loc[phenotypes["condition"] == CONTROL_LABEL].copy()
    comparators = phenotypes.loc[phenotypes["condition"] != CONTROL_LABEL].copy()

    merged = comparators.merge(
        controls,
        on=["timepoint", "metric", "comparison", "mouse"],
        suffixes=("_experimental", "_control"),
        validate="one_to_one",
    )
    merged["delta_vs_control"] = merged["value_experimental"] - merged["value_control"]
    merged = merged.rename(columns={"condition_experimental": "condition"})
    return merged[
        [
            "timepoint",
            "metric",
            "comparison",
            "condition",
            "mouse",
            "value_control",
            "value_experimental",
            "delta_vs_control",
        ]
    ]


def summarize_table(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    grouped = (
        df.groupby(["timepoint", "assay", "metric", "condition"], dropna=False)[value_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return grouped


def significance_stars(p_value: float) -> str:
    if p_value < 1.0e-4:
        return "****"
    if p_value < 1.0e-3:
        return "***"
    if p_value < 1.0e-2:
        return "**"
    if p_value < 5.0e-2:
        return "*"
    return "ns"


def compute_display_stats(tidy: pd.DataFrame, control_split: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for spec in DISPLAY_STAT_TESTS:
        if spec["test_kind"] == "welch_t":
            subset = tidy.loc[
                (tidy["timepoint"] == spec["timepoint"])
                & (tidy["assay"] == spec["assay"])
                & (tidy["condition"].isin([spec["condition_a"], spec["condition_b"]]))
            ].copy()
            group_a = subset.loc[subset["condition"] == spec["condition_a"], "value"].to_numpy(dtype=float)
            group_b = subset.loc[subset["condition"] == spec["condition_b"], "value"].to_numpy(dtype=float)
            result = stats.ttest_ind(group_a, group_b, equal_var=False)
            n_a = len(group_a)
            n_b = len(group_b)
        elif spec["test_kind"] == "paired_t":
            subset = control_split.loc[
                (control_split["timepoint"] == spec["timepoint"])
                & (control_split["metric"] == spec["metric"])
                & (control_split["comparison"] == spec["condition_b"])
            ].copy()
            group_a = subset["value_control"].to_numpy(dtype=float)
            group_b = subset["value_experimental"].to_numpy(dtype=float)
            result = stats.ttest_rel(group_a, group_b)
            n_a = len(group_a)
            n_b = len(group_b)
        else:
            raise ValueError(f"Unsupported test kind: {spec['test_kind']}")

        records.append(
            {
                "timepoint": spec["timepoint"],
                "panel_title": spec["panel_title"],
                "assay": spec["assay"],
                "metric": spec["metric"],
                "condition_a": spec["condition_a"],
                "condition_b": spec["condition_b"],
                "comparison_scope": spec["comparison_scope"],
                "test_kind": spec["test_kind"],
                "test_label": "Welch t-test" if spec["test_kind"] == "welch_t" else "Paired t-test",
                "n_a": n_a,
                "n_b": n_b,
                "statistic": float(result.statistic),
                "p_value_raw": float(result.pvalue),
            }
        )

    stats_df = pd.DataFrame.from_records(records)
    _, p_adj, _, _ = multipletests(stats_df["p_value_raw"], method="fdr_bh")
    stats_df["p_value_bh"] = p_adj
    stats_df["significance_raw"] = stats_df["p_value_raw"].map(significance_stars)
    stats_df["significance_bh"] = stats_df["p_value_bh"].map(significance_stars)
    return stats_df


def apply_plot_style(text_scale: float = 1.0, axis_text_scale: float = 1.0) -> None:
    text_scale = max(0.25, float(text_scale))
    axis_text_scale = max(0.25, float(axis_text_scale))
    reference_text_size = 9.5 * text_scale * axis_text_scale
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.default": "regular",
            "axes.linewidth": 0.8,
            "axes.labelsize": reference_text_size,
            "axes.titlesize": reference_text_size,
            "xtick.labelsize": reference_text_size,
            "ytick.labelsize": reference_text_size,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def display_label(condition: str) -> str:
    if condition == CONTROL_LABEL:
        return "Control"
    return COMPARISON_LABELS[condition]


def display_label_with_n(condition: str, n: int) -> str:
    return f"{display_label(condition)}\nn={n}"


def metric_series(
    tidy: pd.DataFrame,
    timepoint: str,
    panel_title: str,
    conditions: list[str],
) -> pd.DataFrame:
    assay, metric = SELECTED_METRIC_MAP[(timepoint, panel_title)]
    subset = tidy.loc[(tidy["timepoint"] == timepoint) & (tidy["assay"] == assay)].copy()
    if metric is not None:
        subset = subset.loc[subset["metric"] == metric].copy()
    subset = subset.loc[subset["condition"].isin(conditions)].copy()
    displayed_comparisons = [condition for condition in conditions if condition != CONTROL_LABEL]
    if displayed_comparisons and CONTROL_LABEL in conditions:
        control_rows = subset["condition"] == CONTROL_LABEL
        subset = subset.loc[
            (~control_rows) | subset["comparison"].isin(displayed_comparisons)
        ].copy()
    return subset


def panel_stats(stats_df: pd.DataFrame, timepoint: str, panel_title: str) -> pd.DataFrame:
    subset = stats_df.loc[
        (stats_df["timepoint"] == timepoint) & (stats_df["panel_title"] == panel_title)
    ].copy()
    if subset.empty:
        return subset
    subset["span_rank"] = subset.apply(
        lambda row: abs(
            ["Control", "Fresh", "No Stim", "Activated"].index(row["condition_a"])
            - ["Control", "Fresh", "No Stim", "Activated"].index(row["condition_b"])
        ),
        axis=1,
    )
    subset = subset.sort_values(["span_rank", "condition_b"]).reset_index(drop=True)
    return subset


def symmetric_limits(values: np.ndarray, pad_fraction: float = 0.12) -> tuple[float, float]:
    max_abs = max(1.0e-9, float(np.nanmax(np.abs(values))))
    padded = max_abs * (1.0 + pad_fraction)
    rounded = math.ceil(padded * 2) / 2
    return -rounded, rounded


def positive_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    upper = max(1.0e-9, float(np.nanmax(values)))
    lower = min(0.0, float(np.nanmin(values)))
    span = upper - lower
    if span <= 0:
        span = upper if upper > 0 else 1.0
    return lower - 0.03 * span, upper + pad_fraction * span


def condition_layout(conditions: list[str]) -> tuple[np.ndarray, tuple[float, float], float, float]:
    if len(conditions) == 2:
        xticks = np.array([0.0, 0.40], dtype=float)
        x_limits = (-0.16, 0.56)
        jitter_half_width = 0.07
        mean_half_width = 0.12
    else:
        xticks = np.arange(len(conditions), dtype=float)
        x_limits = (-0.45, len(conditions) - 0.55)
        jitter_half_width = 0.12
        mean_half_width = 0.20
    return xticks, x_limits, jitter_half_width, mean_half_width


def draw_significance_brackets(
    ax: plt.Axes,
    data: pd.DataFrame,
    conditions: list[str],
    stats_subset: pd.DataFrame,
    *,
    text_scale: float = 1.0,
) -> bool:
    if stats_subset.empty:
        return False

    show_ns = len(conditions) > 2
    if not show_ns:
        stats_subset = stats_subset.loc[stats_subset["significance_bh"] != "ns"].copy()
    if stats_subset.empty:
        return False

    stats_subset = stats_subset.sort_values(["span_rank", "condition_b"]).reset_index(drop=True)
    group_max = data.groupby("condition")["value"].max().to_dict()
    if not group_max:
        return False

    xticks, _, _, _ = condition_layout(conditions)
    x_lookup = {condition: xticks[idx] for idx, condition in enumerate(conditions)}
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    base_offset = span * 0.035
    step = span * (0.105 if len(stats_subset) > 1 else 0.055)
    bracket_height = span * 0.015
    ns_text_offset = 0.0
    star_text_offset = -span * 0.014
    baseline = max(group_max.values()) + base_offset
    label_top = baseline + (len(stats_subset) - 1) * step + bracket_height + max(ns_text_offset, star_text_offset)
    required_top = label_top + span * 0.14

    if required_top > ymax:
        ax.set_ylim(ymin, required_top)

    for idx, (_, row) in enumerate(stats_subset.iterrows()):
        if row["condition_a"] not in x_lookup or row["condition_b"] not in x_lookup:
            continue
        x1 = x_lookup[row["condition_a"]]
        x2 = x_lookup[row["condition_b"]]
        if x2 < x1:
            x1, x2 = x2, x1

        y = baseline + idx * step
        is_ns = row["significance_bh"] == "ns"
        color = "#6F767D" if is_ns else "#1F1F1F"
        text_offset = ns_text_offset if is_ns else star_text_offset
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            color=color,
            linewidth=0.9 if is_ns else 1.0,
            solid_capstyle="round",
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height + text_offset,
            row["significance_bh"],
            ha="center",
            va="bottom",
            fontsize=float(plt.rcParams["xtick.labelsize"]),
            fontweight="normal",
            color=color,
        )
    return True


def plot_group_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    conditions: list[str],
    ylabel: str,
    title: str,
    *,
    zero_line: bool = False,
    rng_seed: int = 0,
    text_scale: float = 1.0,
    title_scale: float = 1.0,
) -> None:
    rng = np.random.default_rng(rng_seed)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)

    xticks, x_limits, jitter_half_width, mean_half_width = condition_layout(conditions)
    all_values: list[float] = []

    for idx, condition in enumerate(conditions):
        group = data.loc[data["condition"] == condition, "value"].to_numpy(dtype=float)
        all_values.extend(group.tolist())
        jitter = rng.uniform(-jitter_half_width, jitter_half_width, size=len(group))
        ax.scatter(
            xticks[idx] + jitter,
            group,
            s=44,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
        mean_value = float(np.mean(group))
        ax.hlines(
            mean_value,
            xticks[idx] - mean_half_width,
            xticks[idx] + mean_half_width,
            color="#1F1F1F",
            linewidth=1.5,
            zorder=4,
        )

    if zero_line:
        ax.axhline(0, color="#6F767D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ymin, ymax = symmetric_limits(np.asarray(all_values, dtype=float))
    else:
        ymin, ymax = positive_limits(np.asarray(all_values, dtype=float))

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*x_limits)
    ax.set_xticks(xticks)
    group_sizes = {
        condition: int(data.loc[data["condition"] == condition, "value"].notna().sum())
        for condition in conditions
    }
    ax.set_xticklabels([display_label_with_n(condition, group_sizes[condition]) for condition in conditions])
    if text_scale > 1.25:
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(35)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")
    ax.set_ylabel(ylabel)
    title_text = ax.set_title(
        title,
        loc="left",
        pad=8,
        fontweight="normal",
        fontsize=float(plt.rcParams["axes.titlesize"]) * title_scale,
        linespacing=0.95,
    )
    title_text.set_multialignment("left")


def enforce_minimum_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=5))


def make_candidate_figure(
    tidy: pd.DataFrame,
    stats_df: pd.DataFrame,
    panel_specs: list[tuple[str, str, str, list[str]]],
    output_prefix: Path,
    *,
    dpi: int,
    text_scale: float = 1.0,
    axis_text_scale: float = 1.0,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
    title_scale: float = 1.0,
    layout: str = "auto",
) -> None:
    text_scale = max(0.25, float(text_scale))
    axis_text_scale = max(0.25, float(axis_text_scale))
    width_scale = max(0.25, float(width_scale))
    height_scale = max(0.25, float(height_scale))
    title_scale = max(0.25, float(title_scale))
    apply_plot_style(text_scale=text_scale, axis_text_scale=axis_text_scale)
    if layout == "auto":
        layout = "vertical" if text_scale > 1.25 and width_scale <= 0.85 else "2x2"

    if layout == "vertical":
        figure_height = (22.0 + max(0.0, text_scale - 2.0) * 2.0) * height_scale
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(10.4 * width_scale, figure_height),
            constrained_layout=False,
        )
        axes_flat = np.asarray(axes).flatten()
    else:
        figure_height = (9.8 + max(0.0, text_scale - 1.0) * 1.0) * height_scale
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.4 * width_scale, figure_height),
            constrained_layout=False,
            gridspec_kw={
                "width_ratios": [0.86, 1.14],
                "wspace": 0.22,
                "hspace": 0.44,
            },
        )
        if title_scale > 1.0:
            fig.subplots_adjust(left=0.14, right=0.985, bottom=0.13, top=0.86, wspace=0.32, hspace=1.02)
        else:
            fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.91, wspace=0.32, hspace=0.92)
        axes_flat = axes.flatten()

    for idx, (timepoint, panel_title, ylabel, conditions) in enumerate(panel_specs):
        panel_data = metric_series(tidy, timepoint, panel_title, conditions)
        title_timepoint = TIMEPOINT_LABELS[timepoint].capitalize()
        title_label = PANEL_TITLE_LABELS.get(panel_title, panel_title)
        title = f"{title_timepoint}: {title_label}"
        zero_line = panel_title == "Relative accumulation"
        panel_stat_rows = panel_stats(stats_df, timepoint, panel_title)
        plot_group_scatter(
            axes_flat[idx],
            panel_data,
            conditions,
            ylabel,
            title,
            zero_line=zero_line,
            rng_seed=20260403 + idx,
            text_scale=text_scale,
            title_scale=title_scale,
        )
        draw_significance_brackets(axes_flat[idx], panel_data, conditions, panel_stat_rows, text_scale=text_scale)
        enforce_minimum_y_ticks(axes_flat[idx])

    if layout == "vertical":
        fig.subplots_adjust(left=0.32, right=0.985, top=0.91, bottom=0.12, hspace=1.25)

    use_tight_bbox = abs(width_scale - 1.0) < 1e-9 and abs(text_scale - 1.0) < 1e-9
    for extension in ("png", "svg"):
        save_kwargs = {"dpi": dpi} if extension == "png" else {}
        if use_tight_bbox:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(output_prefix.with_suffix(f".{extension}"), **save_kwargs)
    strip_trailing_whitespace(output_prefix.with_suffix(".svg"))
    plt.close(fig)
    validate_file_sizes(output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    root = ET.fromstring(args.prism.read_text(encoding="utf-8"))
    input_ratios, tidy, phenotypes = build_tidy_tables(root)
    control_split = build_control_split_summary(phenotypes)
    summaries = summarize_table(tidy)
    display_stats = compute_display_stats(tidy, control_split)

    stem = args.panel_stem
    input_ratios.to_csv(args.analysis_dir / f"{stem}_lcmv_input_ratios.tsv", sep="\t", index=False)
    tidy.to_csv(args.analysis_dir / f"{stem}_lcmv_tidy_values.tsv", sep="\t", index=False)
    control_split.to_csv(args.analysis_dir / f"{stem}_lcmv_phenotype_control_split.tsv", sep="\t", index=False)
    summaries.to_csv(args.analysis_dir / f"{stem}_lcmv_metric_summary.tsv", sep="\t", index=False)
    display_stats.to_csv(args.analysis_dir / f"{stem}_lcmv_stats.tsv", sep="\t", index=False)

    primary_prefix = args.figure_dir / stem
    benchmark_prefix = args.figure_dir / f"{stem}_with_benchmark"
    primary_panel_specs = BENCHMARK_PANEL_SPECS if args.panel_version == "benchmark" else FOCUSED_PANEL_SPECS

    make_candidate_figure(
        tidy,
        display_stats,
        primary_panel_specs,
        primary_prefix,
        dpi=args.dpi,
        text_scale=args.text_scale,
        axis_text_scale=args.axis_text_scale,
        width_scale=args.width_scale,
        height_scale=args.height_scale,
        title_scale=args.title_scale,
        layout=args.layout,
    )
    if args.write_benchmark:
        make_candidate_figure(
            tidy,
            display_stats,
            BENCHMARK_PANEL_SPECS,
            benchmark_prefix,
            dpi=args.dpi,
            text_scale=args.text_scale,
            axis_text_scale=args.axis_text_scale,
            width_scale=args.width_scale,
            height_scale=args.height_scale,
            title_scale=args.title_scale,
            layout=args.layout,
        )

    print(f"Saved tidy values: {args.analysis_dir / f'{stem}_lcmv_tidy_values.tsv'}")
    print(f"Saved input ratios: {args.analysis_dir / f'{stem}_lcmv_input_ratios.tsv'}")
    print(f"Saved control-split phenotype table: {args.analysis_dir / f'{stem}_lcmv_phenotype_control_split.tsv'}")
    print(f"Saved summaries: {args.analysis_dir / f'{stem}_lcmv_metric_summary.tsv'}")
    print(f"Saved display stats: {args.analysis_dir / f'{stem}_lcmv_stats.tsv'}")
    print(f"Saved {args.panel_version} figure: {primary_prefix.with_suffix('.png')}")
    if args.write_benchmark:
        print(f"Saved benchmark figure: {benchmark_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()

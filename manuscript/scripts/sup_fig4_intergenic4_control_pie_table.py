#!/usr/bin/env python
"""
Generate Figure S7b: an INTERGENIC-4 versus other-control example
showing the cell-based Fisher comparison used in Figure 4c.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "manuscript" / "plots" / "Sup_Figure_7"
OUTPUT_PNG = OUTPUT_DIR / "b.png"
OUTPUT_SVG = OUTPUT_DIR / "b.svg"
OUTPUT_TABLE = OUTPUT_DIR / "b_contingency.tsv"
OUTPUT_STATS = OUTPUT_DIR / "b_stats.tsv"

sys.path.append(str(REPO_DIR / "scripts" / "utils"))
from figure_format import get_adata  # type: ignore  # pylint: disable=wrong-import-position


GUIDE_NAME = "INTERGENIC-4"
TARGET_CELL_TYPE = "Progenitor"
PIE_COLORS = ["#ff7f0e", "#c8c8c8"]
TABLE_ROWS = [GUIDE_NAME, "Other NT guides"]
TABLE_COLUMNS = [TARGET_CELL_TYPE, "Other cell states"]
PIE_TEXT_SIZE = 22
PIE_TITLE_SIZE = 26
TABLE_TITLE_SIZE = 28
TABLE_FONT_SIZE = 25
SUMMARY_FONT_SIZE = 24


def load_gene10_data():
    """Load the gene_10 dataset with the same preprocessing as Figure 4."""
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=False,
        get_raw_scaled=True,
    )

    if "guide" not in adata.obs.columns:
        raise KeyError("Expected column 'guide' not found in AnnData.obs")

    keep_mask = adata.obs["guide"] != "INTERGENIC-3"
    adata = adata[keep_mask].copy()

    for column in ["guide", "perturbation", "cell_type"]:
        adata.obs[column] = adata.obs[column].astype(str)

    return adata


def count_target_vs_other(adata, target_cell_type: str) -> tuple[int, int]:
    """Return counts of target cell type versus all other cell states."""
    mask = adata.obs["cell_type"] == target_cell_type
    target_count = int(mask.sum())
    other_count = int(adata.n_obs - target_count)
    return target_count, other_count


def build_example_table(adata, guide_name: str) -> pd.DataFrame:
    """Build the 2x2 contingency table for one control guide versus other controls."""
    control_mask = adata.obs["perturbation"] == "NT"
    controls = adata[control_mask].copy()

    guide_subset = controls[controls.obs["guide"] == guide_name].copy()
    other_controls = controls[controls.obs["guide"] != guide_name].copy()

    if guide_subset.n_obs == 0:
        raise ValueError(f"No cells found for guide '{guide_name}'")
    if other_controls.n_obs == 0:
        raise ValueError("No other control cells were found for comparison")

    guide_counts = count_target_vs_other(guide_subset, TARGET_CELL_TYPE)
    other_counts = count_target_vs_other(other_controls, TARGET_CELL_TYPE)

    return pd.DataFrame(
        [guide_counts, other_counts],
        index=TABLE_ROWS,
        columns=TABLE_COLUMNS,
    )


def compute_all_guide_statistics(
    adata,
    target_cell_type: str,
) -> pd.DataFrame:
    """Compute Figure-4c-style Fisher statistics for one cell type across all guides."""
    control_mask = adata.obs["perturbation"] == "NT"
    controls = adata[control_mask].copy()
    guide_names = sorted(adata.obs["guide"].unique())

    records = []
    for guide_name in guide_names:
        guide_subset = adata[adata.obs["guide"] == guide_name].copy()
        guide_is_control = bool((guide_subset.obs["perturbation"] == "NT").all())

        if guide_is_control:
            reference = controls[controls.obs["guide"] != guide_name].copy()
            reference_label = "Other NT guides"
        else:
            reference = controls.copy()
            reference_label = "All NT guides"

        table = pd.DataFrame(
            [
                count_target_vs_other(guide_subset, target_cell_type),
                count_target_vs_other(reference, target_cell_type),
            ],
            index=[guide_name, reference_label],
            columns=[target_cell_type, "Other cell states"],
        )

        raw_table = table.to_numpy()
        odds_ratio, p_value = fisher_exact(raw_table, alternative="two-sided")
        pseudocount_odds_ratio, _ = fisher_exact(raw_table + 1, alternative="two-sided")

        records.append(
            {
                "guide": guide_name,
                "guide_target": int(raw_table[0, 0]),
                "guide_other": int(raw_table[0, 1]),
                "other_target": int(raw_table[1, 0]),
                "other_other": int(raw_table[1, 1]),
                "reference_label": reference_label,
                "odds_ratio": odds_ratio,
                "log2_odds": np.log2(pseudocount_odds_ratio),
                "p_value": p_value,
            }
        )

    stats_df = pd.DataFrame.from_records(records)
    stats_df["p_adjusted_all_guides"] = multipletests(
        stats_df["p_value"].to_numpy(),
        method="fdr_bh",
    )[1]
    return stats_df


def autopct_formatter(values: Iterable[int]):
    """Format pie labels with percentages and counts."""
    values = list(values)
    total = sum(values)

    def _formatter(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        if count == 0:
            return ""
        return f"{pct:.1f}%\n(n={count})"

    return _formatter


def style_pie_axis(ax, values: list[int], title: str):
    """Draw one pie chart panel."""
    ax.pie(
        values,
        labels=TABLE_COLUMNS,
        colors=PIE_COLORS,
        startangle=90,
        counterclock=False,
        autopct=autopct_formatter(values),
        textprops={"fontsize": PIE_TEXT_SIZE},
    )
    ax.set_title(title, fontsize=PIE_TITLE_SIZE)


def add_contingency_table(ax, contingency_df: pd.DataFrame, stats_row: pd.Series):
    """Add the 2x2 table and the corresponding test summary."""
    ax.axis("off")
    ax.set_title("Counts used for the Fisher exact test", fontsize=TABLE_TITLE_SIZE, pad=12)

    table = ax.table(
        cellText=[[f"{value:,d}" for value in row] for row in contingency_df.to_numpy()],
        rowLabels=list(contingency_df.index),
        colLabels=list(contingency_df.columns),
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.09, 0.42, 0.82, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1.0, 1.55)

    header_fill = "#f2f2f2"
    row_label_fill = "#f7f7f7"
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#4d4d4d")
        cell.set_linewidth(1.0)

        if row_index == 0 and col_index >= 0:
            cell.set_facecolor(header_fill)
        elif col_index == -1 and row_index > 0:
            cell.set_facecolor(row_label_fill)
            cell.get_text().set_ha("left")
            cell.PAD = 0.08
        else:
            cell.set_facecolor("white")

    guide_target = int(contingency_df.iloc[0, 0])
    guide_other = int(contingency_df.iloc[0, 1])
    other_target = int(contingency_df.iloc[1, 0])
    other_other = int(contingency_df.iloc[1, 1])
    odds_ratio_formula = (
        "log2 fold change in proportion =\n"
        r"$\log_2\!\left(\frac{\mathrm{Proportion\ Progenitor\ in\ INTERGENIC\!-\!4}}"
        r"{\mathrm{Proportion\ Progenitor\ in\ NT\ guides}}\right)$"
        "\n"
        rf"$\approx \log_2\!\left(\frac{{({guide_target}+1)/({guide_other}+1)}}"
        rf"{{({other_target}+1)/({other_other}+1)}}\right)={stats_row['log2_odds']:.2f}$"
    )
    summary_text = (
        f"BH-adjusted P = {stats_row['p_adjusted_all_guides']:.2e}\n"
        f"{odds_ratio_formula}"
    )
    ax.text(
        0.5,
        0.12,
        summary_text,
        ha="center",
        va="center",
        fontsize=SUMMARY_FONT_SIZE,
    )


def validate_file_sizes(png_path: Path, svg_path: Path):
    """Validate figure file sizes against manuscript thresholds."""
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = load_gene10_data()
    contingency_df = build_example_table(adata, GUIDE_NAME)
    guide_stats = compute_all_guide_statistics(adata, TARGET_CELL_TYPE)
    stats_row = guide_stats.loc[guide_stats["guide"] == GUIDE_NAME].iloc[0]

    contingency_df.to_csv(OUTPUT_TABLE, sep="\t")
    stats_row.to_frame().T.to_csv(OUTPUT_STATS, sep="\t", index=False)

    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[3.1, 2.5], hspace=0.15)

    guide_values = contingency_df.loc[GUIDE_NAME].tolist()
    other_values = contingency_df.loc["Other NT guides"].tolist()

    style_pie_axis(
        fig.add_subplot(grid[0, 0]),
        guide_values,
        f"{GUIDE_NAME} (n={sum(guide_values):,d})",
    )
    style_pie_axis(
        fig.add_subplot(grid[0, 1]),
        other_values,
        f"Other NT guides (n={sum(other_values):,d})",
    )
    add_contingency_table(fig.add_subplot(grid[1, :]), contingency_df, stats_row)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(OUTPUT_PNG, OUTPUT_SVG)
    print(f"Saved contingency table to {OUTPUT_TABLE}")
    print(f"Saved statistics to {OUTPUT_STATS}")


if __name__ == "__main__":
    main()

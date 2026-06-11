#!/usr/bin/env python
"""
Generate Figure S6b.

Plot isotonic regression fits for NT clones split by Isotype and anti-PD-1,
using cell-type colors with condition-specific line styles and a combined
black progenitor line.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

sys.path.append('../scripts/utils')
from figure_format import cell_type_colors as base_cell_type_colors, cell_type_orders_gene10, get_adata

from isotonic_utils import prepare_clone_proportions_df

# Required manuscript font settings
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams["legend.fontsize"] = 16
plt.rcParams['axes.grid'] = False

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_6"
COMBINED_LABEL = "Progenitor + Progenitor Cycling"
COMBINED_CELL_TYPES = ("Progenitor", "Progenitor Cycling")
CELL_TYPE_COLORS = base_cell_type_colors.copy()
CELL_TYPE_COLORS[COMBINED_LABEL] = "black"
CELL_TYPE_ORDER = [COMBINED_LABEL] + list(cell_type_orders_gene10)
CONDITION_ORDER = ["Isotype", "PD1"]
CONDITION_LINESTYLES = {"Isotype": "-", "PD1": "--"}
CONDITION_DISPLAY = {"Isotype": "Isotype", "PD1": "anti-PD-1"}
SIGNIFICANT_TESTS_PATH = os.path.join(OUTPUT_DIR, "b_significant_tests.tsv")


def _add_combined_cell_type(clone_df):
    clone_df = clone_df.copy()
    clone_df[COMBINED_LABEL] = (
        clone_df[list(COMBINED_CELL_TYPES)]
        .fillna(0)
        .sum(axis=1)
        .clip(upper=1)
    )
    return clone_df


def _fit_best_isotonic(x, y):
    iso_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    iso_dec = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)

    mse_inc = np.mean((y - iso_inc.predict(x)) ** 2)
    mse_dec = np.mean((y - iso_dec.predict(x)) ** 2)
    return iso_inc if mse_inc <= mse_dec else iso_dec


def significance_label(p_adj):
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def plotted_cell_types(clone_df):
    nt_df = clone_df[clone_df["perturbation"] == "NT"].copy()
    present = []
    for cell_type in CELL_TYPE_ORDER:
        if cell_type not in nt_df.columns:
            continue
        y = nt_df[cell_type].to_numpy()
        if y.size == 0 or np.allclose(y, 0):
            continue
        present.append(cell_type)
    return present


def compute_condition_stats(clone_df):
    nt_df = clone_df[clone_df["perturbation"] == "NT"].copy()
    results = []

    for cell_type in plotted_cell_types(clone_df):
        pd1_data = nt_df.loc[nt_df["condition"] == "PD1", cell_type]
        isotype_data = nt_df.loc[nt_df["condition"] == "Isotype", cell_type]
        if pd1_data.empty or isotype_data.empty:
            continue

        stat, p_value = ranksums(pd1_data, isotype_data)
        results.append(
            {
                "cell_type": cell_type,
                "condition_1": CONDITION_DISPLAY["PD1"],
                "condition_2": CONDITION_DISPLAY["Isotype"],
                "n_condition_1": int(pd1_data.shape[0]),
                "n_condition_2": int(isotype_data.shape[0]),
                "test": "two-sided Wilcoxon rank-sum",
                "statistic": stat,
                "p_value": p_value,
            }
        )

    stats_df = pd.DataFrame(results)
    if stats_df.empty:
        return stats_df

    stats_df["p_adj"] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
    stats_df["label"] = stats_df["p_adj"].apply(significance_label)
    return stats_df


def plot_nt_condition_isotonic(clone_df):
    nt_df = clone_df[clone_df["perturbation"] == "NT"].copy()
    if nt_df.empty:
        raise ValueError("No NT clones available for Figure S6b.")

    nt_df["log2_size"] = np.log2(nt_df["num_cells"].clip(lower=1))
    nt_df = nt_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_size"])
    if nt_df.empty:
        raise ValueError("No valid NT clone sizes available for Figure S6b.")

    x_min = max(0.0, float(nt_df["log2_size"].min()))
    x_max = min(8.0, float(nt_df["log2_size"].max()))
    if x_max <= x_min:
        x_max = x_min + 1.0

    fig, ax = plt.subplots(figsize=(12 * 0.7, 6 * 0.8))

    for cell_type in plotted_cell_types(clone_df):
        if cell_type not in nt_df.columns:
            continue

        for condition in CONDITION_ORDER:
            subset = nt_df[nt_df["condition"] == condition].copy()
            if subset.empty:
                continue

            x = subset["log2_size"].to_numpy()
            y = subset[cell_type].to_numpy()
            if x.size == 0 or np.allclose(y, 0):
                continue

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            iso_model = _fit_best_isotonic(x, y)
            local_x_max = min(8.0, float(x.max()))
            if local_x_max <= float(x.min()):
                local_x_max = float(x.min()) + 1.0
            x_uniform = np.linspace(float(x.min()), local_x_max, num=200)
            y_uniform = iso_model.predict(x_uniform)

            legend_label = f"{cell_type} ({CONDITION_DISPLAY[condition]})"
            linewidth = 3 if cell_type == COMBINED_LABEL else 2

            ax.plot(
                x_uniform,
                y_uniform,
                color=CELL_TYPE_COLORS[cell_type],
                linestyle=CONDITION_LINESTYLES[condition],
                linewidth=linewidth,
                label=legend_label,
            )

    ax.set_title("isotonic regression fits in NT cells")
    ax.set_xlabel("log2(clone size)")
    ax.set_ylabel("proportion of cell type")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)

    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    clone_df = prepare_clone_proportions_df(adata, cell_type_orders_gene10)
    clone_df = _add_combined_cell_type(clone_df)
    stats_df = compute_condition_stats(clone_df)
    significant_stats_df = (
        stats_df.loc[stats_df["p_adj"] < 0.05]
        .sort_values(["p_adj", "p_value", "cell_type"])
        .reset_index(drop=True)
    )
    significant_stats_df.to_csv(SIGNIFICANT_TESTS_PATH, sep="\t", index=False)

    fig = plot_nt_condition_isotonic(clone_df)
    fig.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "b.png")
    svg_path = os.path.join(OUTPUT_DIR, "b.svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


if __name__ == "__main__":
    main()

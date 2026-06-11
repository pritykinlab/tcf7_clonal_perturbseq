#!/usr/bin/env python
"""
Generate Figure S6f.

Forest plot of odds ratios from Fisher exact tests comparing the Isotype:anti-PD-1
clone-count ratio for each perturbation against the corresponding NT ratio in
the 10-gene dataset.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio

sys.path.append("../scripts/utils")
from figure_format import get_adata

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = (7, 7)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "manuscript", "plots", "Sup_Figure_6")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "f.png")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "f.svg")

PERTURBATION_ORDER = [
    "Bach2",
    "Ifngr1",
    "Ifngr2",
    "Il18r1",
    "Jak2",
    "Jak3",
    "Prdm1",
    "Setdb1",
    "Tnfrsf1b",
    "Trim28",
]


def build_contingency_results(unique_clones_series):
    """Build Fisher exact-test summary statistics for each perturbation."""
    df = unique_clones_series.reset_index()
    df.columns = ["perturbation", "condition", "unique_clones"]

    lookup = {}
    for row in df.itertuples(index=False):
        lookup[(row.perturbation, row.condition)] = row.unique_clones

    nt_isotype = lookup.get(("NT", "Isotype"), 0)
    nt_pd1 = lookup.get(("NT", "PD1"), 0)
    results = []

    for perturbation in PERTURBATION_ORDER:
        if perturbation not in set(df["perturbation"]):
            continue

        perturbation_isotype = lookup.get((perturbation, "Isotype"), 0)
        perturbation_pd1 = lookup.get((perturbation, "PD1"), 0)

        table = [
            [nt_isotype, nt_pd1],
            [perturbation_isotype, perturbation_pd1],
        ]
        _, p_value = fisher_exact(table, alternative="two-sided")
        or_result = odds_ratio(table, kind="conditional")
        ci = or_result.confidence_interval(confidence_level=0.95)

        results.append(
            {
                "perturbation": perturbation,
                "table": table,
                "odds_ratio": or_result.statistic,
                "p_value": p_value,
                "lower_ci": ci.low,
                "upper_ci": ci.high,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df):
    """Render the forest plot."""
    if results_df.empty:
        raise ValueError("No perturbation results were generated for Figure S6f.")

    fig_width = max(16.0, 1.5 * len(results_df))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))

    x_positions = np.arange(len(results_df))
    lower_err = results_df["odds_ratio"] - results_df["lower_ci"]
    upper_err = results_df["upper_ci"] - results_df["odds_ratio"]

    ax.errorbar(
        x=x_positions,
        y=results_df["odds_ratio"],
        yerr=[lower_err, upper_err],
        fmt="o",
        color="black",
        ecolor="blue",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
    )
    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.8)

    xticklabels = [
        f"{row.perturbation}\n(p={row.p_value:.2f})"
        for row in results_df.itertuples(index=False)
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Conditional odds ratio\n(anti-PD-1 vs isotype, relative to NT)")

    y_bottom = max(0.0, min(results_df["lower_ci"].min(), results_df["odds_ratio"].min()) - 0.1)
    y_top = results_df["upper_ci"].max() + 0.25
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(-0.6, len(results_df) - 0.4)

    # Odds ratios > 1 indicate relatively more anti-PD-1 clones than expected from NT.
    ax.annotate(
        "More anti-PD-1",
        xy=(1, 1),
        xycoords=("axes fraction", "data"),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=21,
        clip_on=False,
    )
    ax.annotate(
        "More Isotype",
        xy=(1, 1),
        xycoords=("axes fraction", "data"),
        xytext=(-6, -8),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=21,
        clip_on=False,
    )

    fig.tight_layout()
    return fig


def validate_file_sizes(png_path, svg_path):
    """Report whether manuscript file-size targets are met."""
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    unique_clones = adata.obs.groupby(["perturbation", "condition"], observed=False)["clone"].nunique()
    results_df = build_contingency_results(unique_clones)
    fig = plot_results(results_df)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(OUTPUT_PNG, OUTPUT_SVG)
    print("Figure S6f completed successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Figure S10b: pooled clone-size histograms for Trim28 and Setdb1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import get_adata, perturbation_colors  # noqa: E402


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
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.edgecolor"] = "dimgray"


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_10"
OUTPUT_STEM = "b"
RESULTS_STEM = "b_clone_size_tests.tsv"
PERTURBATION_ORDER = ["Trim28", "Setdb1"]
LOG10_BIN_WIDTH = 0.5


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def resolve_mode(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def get_clone_df() -> pd.DataFrame:
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=False,
        remove_NG=True,
        remove_lymph_node=True,
    )

    obs = adata.obs.copy()
    for column in ["clone", "perturbation", "guide", "condition"]:
        obs[column] = obs[column].astype(str)

    grouped = obs.groupby("clone", observed=False)

    clone_df = pd.DataFrame(index=grouped.size().index)
    clone_df["num_cells"] = grouped.size()
    clone_df["perturbation"] = grouped["perturbation"].agg(resolve_mode)
    clone_df["guide"] = grouped["guide"].agg(resolve_mode)
    clone_df["condition"] = grouped["condition"].agg(resolve_mode)
    clone_df = clone_df.reset_index()
    clone_df["log10_clone_size"] = np.log10(clone_df["num_cells"])
    clone_df["log2_clone_size"] = np.log2(clone_df["num_cells"])
    return clone_df


def compute_stats(clone_df: pd.DataFrame) -> pd.DataFrame:
    nt_values = clone_df.loc[clone_df["perturbation"] == "NT", "log2_clone_size"].to_numpy()

    rows = []
    for perturbation in PERTURBATION_ORDER:
        perturbation_values = clone_df.loc[
            clone_df["perturbation"] == perturbation,
            "log2_clone_size",
        ].to_numpy()
        stat, p_value = ranksums(nt_values, perturbation_values, alternative="two-sided")
        rows.append(
            {
                "Condition": "All",
                "Perturbation": perturbation,
                "P-Value": p_value,
                "Corrected P-Value": np.nan,
                "TestStatistic": -stat,
                "NT_N": len(nt_values),
                "Perturbation_N": len(perturbation_values),
            }
        )

    results_df = pd.DataFrame(rows)
    results_df["Corrected P-Value"] = multipletests(
        results_df["P-Value"].to_numpy(),
        method="fdr_bh",
    )[1]
    return results_df


def plot_histograms(clone_df: pd.DataFrame, stats_df: pd.DataFrame) -> plt.Figure:
    plot_df = clone_df.loc[
        clone_df["perturbation"].isin(["NT", *PERTURBATION_ORDER])
    ].copy()

    max_log10 = plot_df["log10_clone_size"].max()
    upper_edge = LOG10_BIN_WIDTH * np.ceil(max_log10 / LOG10_BIN_WIDTH)
    bins = np.arange(0, upper_edge + LOG10_BIN_WIDTH, LOG10_BIN_WIDTH)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)

    for ax, perturbation in zip(axes, PERTURBATION_ORDER):
        nt_values = plot_df.loc[plot_df["perturbation"] == "NT", "log10_clone_size"]
        pert_values = plot_df.loc[
            plot_df["perturbation"] == perturbation,
            "log10_clone_size",
        ]
        stats_row = stats_df.loc[stats_df["Perturbation"] == perturbation].iloc[0]

        ax.hist(
            nt_values,
            bins=bins,
            color=perturbation_colors["NT"],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.7,
            density=True,
            label=f"NT (n={len(nt_values)})",
        )
        ax.hist(
            pert_values,
            bins=bins,
            color=perturbation_colors[perturbation],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.7,
            density=True,
            label=f"{perturbation} (n={len(pert_values)})",
        )

        p_value = float(stats_row["P-Value"])
        effect_size = float(stats_row["TestStatistic"])
        ax.set_title(f"{perturbation} vs NT\np={p_value:.2g}, z={effect_size:.2f}")
        ax.set_xlabel("log10(clone size)")
        ax.set_xlim(0, bins[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, fancybox=False)

    axes[0].set_ylabel("Density")
    fig.tight_layout()
    return fig


def save_results_table(results_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / RESULTS_STEM
    results_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved clone-size test table to {output_path}")


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    validate_file_sizes(png_path, svg_path)


def main() -> None:
    print("Loading trim28_setdb1 clone-level data...")
    clone_df = get_clone_df()
    print(
        clone_df.loc[clone_df["perturbation"].isin(["NT", *PERTURBATION_ORDER])]
        .groupby("perturbation", observed=False)["num_cells"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )

    print("Computing pooled clone-size tests...")
    stats_df = compute_stats(clone_df)
    print(stats_df.to_string(index=False))
    save_results_table(stats_df)

    print("Creating overlapping clone-size histograms...")
    fig = plot_histograms(clone_df, stats_df)
    save_figure(fig)
    plt.close(fig)
    print("Figure S10b completed successfully.")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()

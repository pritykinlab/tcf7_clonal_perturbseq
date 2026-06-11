#!/usr/bin/env python
"""
Generate Figure S6d clone-size perturbation heatmap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import get_adata  # noqa: E402


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
plt.rcParams["figure.figsize"] = (10, 3)


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_6"
OUTPUT_STEM = "d"
CONDITION_ORDER = ["All", "Isotype", "anti-PD-1"]
COLORBAR_LABEL = "Wilcoxon rank-sum\nz-statistic"
PADJ_MASK_THRESHOLD = 0.05


def truncate_colormap(
    cmap: LinearSegmentedColormap,
    minval: float = 0.3,
    maxval: float = 0.7,
    n: int = 100,
) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def resolve_mode(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def get_clone_df(adata) -> pd.DataFrame:
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
    return clone_df


def compare_perturbations(clone_df: pd.DataFrame) -> pd.DataFrame:
    clone_df = clone_df.copy()
    clone_df["log2_cells"] = np.log2(clone_df["num_cells"])

    clone_df_all = clone_df.copy()
    clone_df_all["condition"] = "All"
    extended_df = pd.concat([clone_df, clone_df_all], ignore_index=True)

    perturbations = sorted(p for p in clone_df["perturbation"].unique() if p != "NT")
    conditions = ["Isotype", "PD1", "All"]

    raw_results: list[dict[str, object]] = []

    for condition in conditions:
        nt_mask = (
            (extended_df["perturbation"] == "NT")
            & (extended_df["condition"] == condition)
        )
        nt_data = extended_df.loc[nt_mask, "log2_cells"]
        if nt_data.empty:
            continue

        for perturbation in perturbations:
            pert_mask = (
                (extended_df["perturbation"] == perturbation)
                & (extended_df["condition"] == condition)
            )
            pert_data = extended_df.loc[pert_mask, "log2_cells"]
            if pert_data.empty:
                continue

            stat, p_value = ranksums(nt_data, pert_data, alternative="two-sided")
            raw_results.append(
                {
                    "Condition": condition,
                    "Perturbation": perturbation,
                    "P-Value": p_value,
                    "TestStatistic": -stat,
                    "NT_N": len(nt_data),
                    "Perturbation_N": len(pert_data),
                }
            )

    results_df = pd.DataFrame(raw_results)
    results_df["Corrected P-Value"] = np.nan

    for condition, frame in results_df.groupby("Condition", sort=False):
        corrected = multipletests(frame["P-Value"].to_numpy(), method="fdr_bh")[1]
        results_df.loc[frame.index, "Corrected P-Value"] = corrected

    results_df["Condition"] = results_df["Condition"].replace({"PD1": "anti-PD-1"})
    return results_df


def plot_heatmap(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.copy()
    plot_df["Condition"] = pd.Categorical(
        plot_df["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    plot_df = plot_df.sort_values("Condition")

    pivot_table = plot_df.pivot_table(
        index="Condition",
        columns="Perturbation",
        values="TestStatistic",
        observed=False,
    )
    p_values_table = plot_df.pivot_table(
        index="Condition",
        columns="Perturbation",
        values="Corrected P-Value",
        observed=False,
    )

    ordered_columns = sorted(pivot_table.columns)
    pivot_table = pivot_table[ordered_columns]
    p_values_table = p_values_table[ordered_columns]
    pivot_table = pivot_table.reindex(CONDITION_ORDER)
    p_values_table = p_values_table.reindex(CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(10, 3))
    mask = p_values_table >= PADJ_MASK_THRESHOLD
    truncated_cmap = truncate_colormap(plt.cm.PuOr, minval=0.3, maxval=0.7)

    sns.heatmap(
        pivot_table,
        mask=mask,
        cmap=truncated_cmap,
        center=0,
        vmin=-2,
        vmax=2,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        cbar_kws={"label": COLORBAR_LABEL},
        linewidths=0.5,
        linecolor="lightgray",
        rasterized=True,
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(COLORBAR_LABEL, fontsize=20)

    ax.set_title("Perturbation effect on clone size", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    validate_file_sizes(png_path, svg_path)


def save_results_table(results_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{OUTPUT_STEM}_clone_size_tests.tsv"
    results_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved clone-size test table to {output_path}")


def main() -> None:
    print("Loading gene_10 AnnData for clone-size heatmap...")
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
    )

    print("Building clone-level table...")
    clone_df = get_clone_df(adata)
    print(
        clone_df.groupby(["condition", "perturbation"], observed=False)["clone"]
        .count()
        .rename("n_clones")
        .to_string()
    )

    print("Running clone-size perturbation tests...")
    results_df = compare_perturbations(clone_df)
    print(results_df.to_string(index=False))
    save_results_table(results_df)

    print("Creating heatmap...")
    fig = plot_heatmap(results_df)
    save_figure(fig)
    plt.close(fig)
    print("Figure S6d completed successfully.")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()

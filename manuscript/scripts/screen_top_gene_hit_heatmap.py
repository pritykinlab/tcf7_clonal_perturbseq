#!/usr/bin/env python
"""Generate the gene-level hit heatmap for Figure S2a."""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


SCREEN_FILES = [
    {
        "condition": "D7 Tcf7 low vs High Acute Stim",
        "label": "D7 Tcf7 acute",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7tcf7hi-low_100_21-03-01-18-00.txt",
    },
    {
        "condition": "D7 Tcf7 low vs High Chronic Stim",
        "label": "D7 Tcf7 chronic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7conttcf7hi-low_100_21-03-01-18-04.txt",
    },
    {
        "condition": "D5 Prdm1 low vs high Chronic Stim",
        "label": "D5 Prdm1 chronic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D5contprdm1hi-low_100_21-03-01-18-17.txt",
    },
    {
        "condition": "D7 Prdm1 low vs high",
        "label": "D7 Prdm1",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7prdm1hi-low_100_21-03-01-18-03.txt",
    },
    {
        "condition": "D7 Eomes low vs high",
        "label": "D7 Eomes",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7eomeshi-low_100_21-03-01-18-01.txt",
    },
    {
        "condition": "D5 Tox low vs high Hypoxic",
        "label": "D5 Tox hypoxic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D5hyptoxhi-low_100_21-03-01-18-19.txt",
    },
]

CONTROL_PREFIXES = ("NO_SITE", "INACTIVE", "POTENTIALLY_ACTIVE")
P_THRESHOLD = 0.05
NEG_LOG10_P_THRESHOLD = -math.log10(P_THRESHOLD)
TOP_N_PER_DIRECTION = 5
REQUIRED_GUIDES_PER_TARGET_GENE = 4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manuscript_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_target_gene(series: pd.Series) -> pd.Series:
    return (
        ~series.str.contains("INTERGENIC", na=False)
        & ~series.str.startswith(CONTROL_PREFIXES, na=False)
    )


def load_screen(path: Path, condition: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").rename(
        columns={
            "Gene Symbol": "gene",
            "Average LFC": "average_lfc",
            "Average -log(p-values)": "average_neg_log10_p",
            "Number of perturbations": "number_of_perturbations",
        }
    )
    required = [
        "gene",
        "average_lfc",
        "average_neg_log10_p",
        "number_of_perturbations",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df[required].copy()
    df["average_lfc"] = pd.to_numeric(df["average_lfc"], errors="coerce")
    df["average_neg_log10_p"] = pd.to_numeric(
        df["average_neg_log10_p"], errors="coerce"
    )
    df["number_of_perturbations"] = pd.to_numeric(
        df["number_of_perturbations"], errors="coerce"
    )
    df = df[np.isfinite(df["average_lfc"]) & np.isfinite(df["average_neg_log10_p"])]
    df = df[is_target_gene(df["gene"])]
    df = df[df["number_of_perturbations"] == REQUIRED_GUIDES_PER_TARGET_GENE]
    df["condition"] = condition
    df["label"] = label
    return df.drop_duplicates("gene", keep="first")


def significant_hits(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["average_neg_log10_p"] >= NEG_LOG10_P_THRESHOLD].copy()


def select_top_genes(frames: list[pd.DataFrame]) -> list[str]:
    selected: list[str] = []
    for df in frames:
        sig = significant_hits(df)
        positive = sig[sig["average_lfc"] > 0].nlargest(TOP_N_PER_DIRECTION, "average_lfc")
        negative = sig[sig["average_lfc"] < 0].nsmallest(TOP_N_PER_DIRECTION, "average_lfc")
        selected.extend(positive["gene"].tolist())
        selected.extend(negative["gene"].tolist())
    return list(dict.fromkeys(selected))


def prepare_heatmap() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        load_screen(
            repo_root() / spec["path"],
            condition=spec["condition"],
            label=spec["label"],
        )
        for spec in SCREEN_FILES
    ]
    selected_genes = select_top_genes(frames)
    labels = [spec["label"] for spec in SCREEN_FILES]

    heatmap = pd.DataFrame(index=selected_genes, columns=labels, dtype=float)
    neglogp = pd.DataFrame(index=selected_genes, columns=labels, dtype=float)

    for df in frames:
        label = df["label"].iloc[0]
        sig = significant_hits(df).set_index("gene")
        genes = [gene for gene in selected_genes if gene in sig.index]
        heatmap.loc[genes, label] = sig.loc[genes, "average_lfc"]
        neglogp.loc[genes, label] = sig.loc[genes, "average_neg_log10_p"]

    if heatmap.dropna(how="all").empty:
        raise ValueError("No significant target genes passed the selection criteria.")

    return heatmap, neglogp


def significance_stars(neg_log10_p: float) -> str:
    if neg_log10_p >= 3:
        return "***"
    if neg_log10_p >= 2:
        return "**"
    if neg_log10_p >= NEG_LOG10_P_THRESHOLD:
        return "*"
    return ""


def add_annotations(
    ax: plt.Axes,
    heatmap: pd.DataFrame,
    neglogp: pd.DataFrame,
    vmax: float,
) -> None:
    for y, gene in enumerate(heatmap.index):
        for x, label in enumerate(heatmap.columns):
            value = heatmap.loc[gene, label]
            neg_log10_p = neglogp.loc[gene, label]
            if pd.isna(value) or pd.isna(neg_log10_p):
                continue

            stars = significance_stars(float(neg_log10_p))
            text_color = "white" if abs(float(value)) > 0.58 * vmax else "black"
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{value:.1f}\n{stars}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.5,
                linespacing=0.82,
                zorder=4,
            )


def make_plot(heatmap: pd.DataFrame, neglogp: pd.DataFrame) -> sns.matrix.ClusterGrid:
    cluster_data = heatmap.fillna(0)
    vmax = float(np.nanmax(np.abs(heatmap.values)))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    grid = sns.clustermap(
        cluster_data,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        row_cluster=True,
        col_cluster=False,
        linewidths=0.5,
        linecolor="white",
        figsize=(4, 12),
        dendrogram_ratio=(0.12, 0.02),
        cbar_pos=None,
        xticklabels=True,
        yticklabels=True,
    )

    row_order = grid.dendrogram_row.reordered_ind
    ordered_heatmap = heatmap.iloc[row_order]
    ordered_neglogp = neglogp.iloc[row_order]

    grid.ax_heatmap.clear()
    sns.heatmap(
        ordered_heatmap,
        ax=grid.ax_heatmap,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        mask=ordered_heatmap.isna(),
        rasterized=True,
    )
    grid.ax_heatmap.set_facecolor("white")
    add_annotations(grid.ax_heatmap, ordered_heatmap, ordered_neglogp, vmax)

    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="x", rotation=90, labelsize=8, length=0)
    grid.ax_heatmap.tick_params(axis="y", rotation=0, labelsize=8, length=0)

    colorbar_axis = grid.ax_heatmap.inset_axes([1.18, 0.25, 0.025, 0.5])
    colorbar = grid.fig.colorbar(
        grid.ax_heatmap.collections[0],
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label("Average LFC", fontsize=12)
    colorbar.ax.tick_params(labelsize=9)

    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=manuscript_root() / "plots" / "Sup_Figure_2",
    )
    parser.add_argument("--panel-stem", default="a")
    return parser.parse_args()


def save_outputs(fig: plt.Figure, output_dir: Path, panel_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{panel_stem}.png"
    svg_path = output_dir / f"{panel_stem}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    args = parse_args()
    heatmap, neglogp = prepare_heatmap()
    print(f"Selected gene union: {heatmap.shape[0]} genes")
    grid = make_plot(heatmap, neglogp)
    save_outputs(grid.fig, args.output_dir, args.panel_stem)
    plt.close(grid.fig)


if __name__ == "__main__":
    main()

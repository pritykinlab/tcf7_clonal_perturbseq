#!/usr/bin/env python
"""
Generate Figure S4a.

This panel reproduces the legacy construct-count violin plot from the original
construct_filtering notebook while using the current manuscript workflow.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_4"
OLD_ADATA_PATH = REPO_ROOT / "data" / "gene10" / "preprocessing" / "05_filter_cells_adata.h5ad"
FINAL_ADATA_PATH = REPO_ROOT / "data" / "gene10" / "preprocessing" / "final.h5ad"
GUIDE_UMI_PATH = REPO_ROOT / "raw_data" / "new_data" / "exp49_CP8_old" / "pycashier" / "CBC_guide_umi.tsv"

OUTPUT_STEM = "a"
MIN_CLONE_SIZE = 100
TSV_CHUNKSIZE = 1_000_000

# This explicit guide order matches the legacy reference panel in image.png.
GUIDE_ORDER = [
    "Bach2-1",
    "Bach2-2",
    "Ifngr1-1",
    "Ifngr1-2",
    "Il18r1-1",
    "Il18r1-2",
    "Jak3-1",
    "NO_SITE_26",
    "NO_SITE_80",
    "NO_SITE_223",
    "NO_SITE_738",
    "NO_SITE_913",
    "ONE_INTERGENIC_SITE_1",
    "ONE_INTERGENIC_SITE_18",
    "ONE_INTERGENIC_SITE_21",
    "ONE_INTERGENIC_SITE_24",
    "ONE_INTERGENIC_SITE_5",
    "Prdm1-2",
    "Setdb1-2",
    "Tnfrsf1b-1",
    "Tnfrsf1b-2",
    "Trim28-1",
    "Trim28-2",
]


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 9
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


def load_clone_metadata() -> pd.DataFrame:
    old_adata = ad.read_h5ad(OLD_ADATA_PATH, backed="r")
    final_adata = ad.read_h5ad(FINAL_ADATA_PATH, backed="r")

    obs = pd.DataFrame(
        {
            "clone_old": old_adata.obs["clone"].astype(str),
            "num_constructs": pd.to_numeric(old_adata.obs["num_constructs"]),
            "mouse": final_adata.obs["mouse"].astype(str),
        },
        index=old_adata.obs_names,
    )

    old_adata.file.close()
    final_adata.file.close()

    obs["old_construct"] = obs["clone_old"].str.split("|").str[1]
    obs["old_guide"] = obs["old_construct"].str.split("__").str[0]
    obs["clone_label"] = obs["mouse"] + "|" + obs["old_construct"]
    return obs


def prepare_clone_obs(obs: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[float]]:
    clone_stats = (
        obs.groupby("clone_label", observed=False)
        .agg(
            clone_size=("clone_label", "size"),
            num_constructs=("num_constructs", "first"),
            mouse=("mouse", "first"),
            old_construct=("old_construct", "first"),
            old_guide=("old_guide", "first"),
        )
        .reset_index()
    )

    clone_stats = clone_stats.loc[
        (clone_stats["clone_size"] > MIN_CLONE_SIZE)
        & (clone_stats["num_constructs"] == 1)
        & (clone_stats["old_guide"].isin(GUIDE_ORDER))
    ].copy()

    filtered_obs = obs.loc[obs["clone_label"].isin(clone_stats["clone_label"])].copy()
    filtered_obs = filtered_obs.join(
        clone_stats.set_index("clone_label")[["clone_size"]],
        on="clone_label",
    )

    guide_rank = {guide: idx for idx, guide in enumerate(GUIDE_ORDER)}
    clone_stats["guide_rank"] = clone_stats["old_guide"].map(guide_rank)
    clone_stats["mouse_num"] = clone_stats["mouse"].str.split(":").str[1].astype(int)
    clone_stats["condition_rank"] = clone_stats["mouse"].str.startswith("aPD1").astype(int)
    clone_stats["mouse_display"] = clone_stats["mouse"].str.replace(
        "aPD1", "anti-PD-1", n=1, regex=False
    )
    clone_stats = clone_stats.sort_values(
        ["guide_rank", "condition_rank", "mouse_num", "clone_label"],
    )
    clone_stats["clone_label_display"] = (
        clone_stats["mouse_display"]
        + "|"
        + clone_stats["old_construct"]
        + " (n="
        + clone_stats["clone_size"].astype(str)
        + ")"
    )

    filtered_obs = filtered_obs.join(
        clone_stats.set_index("clone_label")[["clone_label_display", "mouse_display"]],
        on="clone_label",
    )

    clone_order = clone_stats["clone_label_display"].tolist()
    mouse_order_df = (
        clone_stats[["mouse", "mouse_display"]]
        .drop_duplicates()
        .assign(
            condition_rank=lambda df: df["mouse"].str.startswith("aPD1").astype(int),
            mouse_num=lambda df: df["mouse"].str.split(":").str[1].astype(int),
        )
        .sort_values(["condition_rank", "mouse_num", "mouse"])
    )
    mouse_order = mouse_order_df["mouse_display"].tolist()

    guide_counts = clone_stats["old_guide"].value_counts(sort=False)
    guide_boundaries: list[float] = []
    cumulative = 0
    for guide in clone_stats["old_guide"].drop_duplicates():
        cumulative += int(guide_counts.loc[guide])
        if cumulative < len(clone_stats):
            guide_boundaries.append(cumulative - 0.5)

    return filtered_obs, clone_order, mouse_order, guide_boundaries


def load_construct_counts(valid_cbcs: set[str], valid_constructs: set[str]) -> pd.DataFrame:
    grouped_chunks: list[pd.DataFrame] = []
    usecols = ["CBC", "clone_UMI", "gRNA"]

    for chunk in pd.read_csv(GUIDE_UMI_PATH, sep="\t", usecols=usecols, chunksize=TSV_CHUNKSIZE):
        chunk = chunk.loc[chunk["CBC"].isin(valid_cbcs)].copy()
        if chunk.empty:
            continue

        chunk["old_construct"] = chunk["gRNA"] + "__" + chunk["clone_UMI"]
        chunk = chunk.loc[chunk["old_construct"].isin(valid_constructs)]
        if chunk.empty:
            continue

        grouped = (
            chunk.groupby(["CBC", "old_construct"], observed=False)
            .size()
            .reset_index(name="construct_count")
        )
        grouped_chunks.append(grouped)

    if not grouped_chunks:
        raise RuntimeError("No construct counts matched the selected Figure S4 clones.")

    counts = pd.concat(grouped_chunks, ignore_index=True)
    counts = (
        counts.groupby(["CBC", "old_construct"], observed=False)["construct_count"]
        .sum()
        .reset_index()
    )
    return counts


def build_plot_df() -> tuple[pd.DataFrame, list[str], list[str], list[float]]:
    obs = load_clone_metadata()
    filtered_obs, clone_order, mouse_order, guide_boundaries = prepare_clone_obs(obs)

    counts = load_construct_counts(
        valid_cbcs=set(filtered_obs.index),
        valid_constructs=set(filtered_obs["old_construct"].unique()),
    )

    plot_df = filtered_obs.reset_index(names="CBC").merge(
        counts,
        on=["CBC", "old_construct"],
        how="left",
    )
    plot_df = plot_df.dropna(subset=["construct_count"]).copy()
    plot_df["log10_construct_count"] = np.log10(plot_df["construct_count"])
    return plot_df, clone_order, mouse_order, guide_boundaries


def make_palette(mouse_order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {mouse: cmap(idx) for idx, mouse in enumerate(mouse_order)}


def plot_violin_panel(
    plot_df: pd.DataFrame,
    clone_order: list[str],
    mouse_order: list[str],
    guide_boundaries: list[float],
) -> plt.Figure:
    palette = make_palette(mouse_order)

    fig, ax = plt.subplots(figsize=(20, 6.5))
    sns.violinplot(
        data=plot_df,
        x="clone_label_display",
        y="log10_construct_count",
        hue="mouse_display",
        order=clone_order,
        hue_order=mouse_order,
        palette=palette,
        dodge=False,
        cut=0,
        inner="box",
        linewidth=1,
        bw_method=0.3,
        density_norm="width",
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_rasterized(True)
    for boundary in guide_boundaries:
        ax.plot(
            [boundary, boundary],
            [-1.05, 1.0],
            transform=ax.get_xaxis_transform(),
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            zorder=0,
            clip_on=False,
        )
    ax.set_xlim(-0.5, len(clone_order) - 0.5)

    ax.set_xlabel("Clone", fontsize=12.5)
    ax.set_ylabel("log10(construct count)", fontsize=12.5)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(
        title="Condition:Mouse",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"Saved Figure S4a: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    print("Preparing legacy construct-count data for Figure S4...")
    plot_df, clone_order, mouse_order, guide_boundaries = build_plot_df()
    print(
        f"Plotting {len(clone_order)} large single-construct clones "
        f"across {plot_df['mouse'].nunique()} mice."
    )
    fig = plot_violin_panel(plot_df, clone_order, mouse_order, guide_boundaries)
    save_figure(fig)


if __name__ == "__main__":
    main()

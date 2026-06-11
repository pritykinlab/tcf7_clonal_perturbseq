#!/usr/bin/env python
"""
Generate Figure S6 treatment-condition histograms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ranksums


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import condition_colors_dict, get_adata  # noqa: E402


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
plt.rcParams["legend.fontsize"] = 17
plt.rcParams["legend.edgecolor"] = "dimgray"
plt.rcParams["figure.figsize"] = (7.2, 5.4)


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_6"
SIGNIFICANT_TESTS_PATH = OUTPUT_DIR / "b_significant_tests.tsv"
DISPLAY_LABELS = {"Isotype": "Isotype", "PD1": "anti-PD-1"}
DISPLAY_ORDER = ["Isotype", "anti-PD-1"]
DISPLAY_COLOR_LABELS = {"Isotype": "Isotype", "anti-PD-1": "aPD1"}
LOG10_BIN_WIDTH = 0.5
PROGENITOR_CELL_TYPES = {"Progenitor", "Progenitor Cycling"}
PROPORTION_BIN_WIDTH = 0.1


def _load_nt_obs(columns: list[str]) -> pd.DataFrame:
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    obs = adata.obs[columns].copy().astype(str)
    return obs.loc[obs["perturbation"] == "NT"].copy()


def prepare_clone_sizes() -> tuple[pd.DataFrame, pd.Series]:
    obs = _load_nt_obs(["condition", "perturbation", "clone"])
    clone_sizes = (
        obs.groupby(["condition", "perturbation", "clone"], observed=False)
        .size()
        .rename("clone_size")
        .reset_index()
    )
    clone_sizes["condition_label"] = clone_sizes["condition"].map(DISPLAY_LABELS)
    clone_sizes = clone_sizes.dropna(subset=["condition_label"]).copy()
    clone_sizes["log10_clone_size"] = np.log10(clone_sizes["clone_size"])
    clone_counts = clone_sizes["condition_label"].value_counts().reindex(DISPLAY_ORDER)
    return clone_sizes, clone_counts


def prepare_clone_progenitor_proportions() -> tuple[pd.DataFrame, pd.Series]:
    obs = _load_nt_obs(["condition", "perturbation", "clone", "cell_type"])
    cell_counts = obs["condition"].map(DISPLAY_LABELS).value_counts().reindex(DISPLAY_ORDER)

    clone_props = (
        obs.assign(
            progenitor_or_cycling=obs["cell_type"].isin(PROGENITOR_CELL_TYPES).astype(int),
        )
        .groupby(["condition", "perturbation", "clone"], observed=False)
        .agg(
            clone_size=("cell_type", "size"),
            progenitor_or_cycling_count=("progenitor_or_cycling", "sum"),
        )
        .reset_index()
    )
    clone_props["condition_label"] = clone_props["condition"].map(DISPLAY_LABELS)
    clone_props = clone_props.dropna(subset=["condition_label"]).copy()
    clone_props["proportion"] = (
        clone_props["progenitor_or_cycling_count"] / clone_props["clone_size"]
    )
    return clone_props, cell_counts


def prepare_clone_cell_type_proportions(cell_type: str) -> tuple[pd.DataFrame, pd.Series]:
    obs = _load_nt_obs(["condition", "perturbation", "clone", "cell_type"])
    cell_counts = obs["condition"].map(DISPLAY_LABELS).value_counts().reindex(DISPLAY_ORDER)

    clone_props = (
        obs.assign(
            cell_type_match=(obs["cell_type"] == cell_type).astype(int),
        )
        .groupby(["condition", "perturbation", "clone"], observed=False)
        .agg(
            clone_size=("cell_type", "size"),
            cell_type_count=("cell_type_match", "sum"),
        )
        .reset_index()
    )
    clone_props["condition_label"] = clone_props["condition"].map(DISPLAY_LABELS)
    clone_props = clone_props.dropna(subset=["condition_label"]).copy()
    clone_props["proportion"] = clone_props["cell_type_count"] / clone_props["clone_size"]
    return clone_props, cell_counts


def load_condition_padj(cell_type: str) -> float:
    if not SIGNIFICANT_TESTS_PATH.exists():
        raise FileNotFoundError(
            f"Expected significance table at {SIGNIFICANT_TESTS_PATH}. Run Figure S6b first."
        )

    stats_df = pd.read_csv(SIGNIFICANT_TESTS_PATH, sep="\t")
    match = stats_df.loc[stats_df["cell_type"] == cell_type, "p_adj"]
    if match.empty:
        raise ValueError(f"No adjusted p-value found for cell type '{cell_type}' in {SIGNIFICANT_TESTS_PATH}.")
    return float(match.iloc[0])


def plot_clone_size_histogram(clone_sizes: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()

    max_log10 = clone_sizes["log10_clone_size"].max()
    upper_edge = LOG10_BIN_WIDTH * np.ceil(max_log10 / LOG10_BIN_WIDTH)
    bins = np.arange(0, upper_edge + LOG10_BIN_WIDTH, LOG10_BIN_WIDTH)

    for label in DISPLAY_ORDER:
        values = clone_sizes.loc[
            clone_sizes["condition_label"] == label,
            "log10_clone_size",
        ]
        ax.hist(
            values,
            bins=bins,
            color=condition_colors_dict[DISPLAY_COLOR_LABELS[label]],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.6,
            density=True,
        )

    apd1_values = clone_sizes.loc[
        clone_sizes["condition_label"] == "anti-PD-1",
        "clone_size",
    ].to_numpy()
    isotype_values = clone_sizes.loc[
        clone_sizes["condition_label"] == "Isotype",
        "clone_size",
    ].to_numpy()
    _, p_value = ranksums(apd1_values, isotype_values)

    ax.set_title(f"Wilcoxon rank-sum test anti-PD-1 vs isotype in NT clones\n(p={p_value:.2g})")
    ax.set_xlabel("log10(clone size)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, bins[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_progenitor_proportion_histogram(clone_props: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()

    bins = np.arange(0, 1 + PROPORTION_BIN_WIDTH, PROPORTION_BIN_WIDTH)

    for label in DISPLAY_ORDER:
        values = clone_props.loc[
            clone_props["condition_label"] == label,
            "proportion",
        ]
        ax.hist(
            values,
            bins=bins,
            color=condition_colors_dict[DISPLAY_COLOR_LABELS[label]],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.8,
            density=True,
        )

    p_adj = load_condition_padj("Progenitor + Progenitor Cycling")
    ax.set_title(f"Wilcoxon rank-sum test anti-PD-1 vs isotype in NT clones\n(padj={p_adj:.2g})")
    ax.set_xlabel("proportion in Progenitor or Progenitor Cycling")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_cell_type_proportion_histogram(clone_props: pd.DataFrame, cell_type: str) -> plt.Figure:
    fig, ax = plt.subplots()

    bins = np.arange(0, 1 + PROPORTION_BIN_WIDTH, PROPORTION_BIN_WIDTH)

    for label in DISPLAY_ORDER:
        values = clone_props.loc[
            clone_props["condition_label"] == label,
            "proportion",
        ]
        ax.hist(
            values,
            bins=bins,
            color=condition_colors_dict[DISPLAY_COLOR_LABELS[label]],
            alpha=0.55,
            edgecolor="black",
            linewidth=0.8,
            density=True,
        )

    p_adj = load_condition_padj(cell_type)
    ax.set_title(f"Wilcoxon rank-sum test anti-PD-1 vs isotype in NT clones\n(padj={p_adj:.2g})")
    ax.set_xlabel(f"proportion in {cell_type}")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def create_legend(cell_counts: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    ax.axis("off")

    handles = [
        mpatches.Patch(
            color=condition_colors_dict[DISPLAY_COLOR_LABELS[label]],
            label=f"{label} (n={int(cell_counts.loc[label])})",
        )
        for label in DISPLAY_ORDER
    ]

    ax.legend(
        handles=handles,
        loc="center",
        title="Condition",
        frameon=True,
        fontsize=18,
        title_fontsize=20,
        edgecolor="black",
        fancybox=False,
        ncol=1,
        handletextpad=0.8,
        labelspacing=0.8,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["clone_size", "progenitor_proportion", "cell_type_proportion"],
        default="clone_size",
    )
    parser.add_argument("--stem", default="a")
    parser.add_argument("--cell-type")
    parser.add_argument("--no-legend", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.metric == "clone_size":
        print("Loading clone sizes by condition...")
        plot_df, cell_counts = prepare_clone_sizes()
        print(
            plot_df.groupby("condition_label")["clone_size"]
            .agg(["count", "min", "median", "max"])
            .to_string()
        )
        fig = plot_clone_size_histogram(plot_df)
    elif args.metric == "progenitor_proportion":
        print("Loading clone progenitor-or-cycling proportions by condition...")
        plot_df, cell_counts = prepare_clone_progenitor_proportions()
        print(
            plot_df.groupby("condition_label")["proportion"]
            .agg(["count", "min", "median", "max"])
            .to_string()
        )
        fig = plot_progenitor_proportion_histogram(plot_df)
    else:
        if not args.cell_type:
            raise ValueError("--cell-type is required when --metric cell_type_proportion is used.")
        print(f"Loading clone proportions for {args.cell_type} by condition...")
        plot_df, cell_counts = prepare_clone_cell_type_proportions(args.cell_type)
        print(
            plot_df.groupby("condition_label")["proportion"]
            .agg(["count", "min", "median", "max"])
            .to_string()
        )
        fig = plot_cell_type_proportion_histogram(plot_df, args.cell_type)

    print(cell_counts.to_string())

    print(f"Plotting Figure S6{args.stem}...")
    save_figure(fig, args.stem)
    plt.close(fig)

    if not args.no_legend:
        print(f"Creating standalone legend for Figure S6{args.stem}...")
        legend_fig = create_legend(cell_counts)
        save_figure(legend_fig, f"{args.stem}_legend")
        plt.close(legend_fig)

    print(f"Figure S6{args.stem} completed successfully.")


if __name__ == "__main__":
    main()

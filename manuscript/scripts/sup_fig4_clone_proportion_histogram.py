#!/usr/bin/env python
"""
Generate Figure S7 clone-level histogram examples.

This script plots side-by-side histograms of per-clone cell-type proportions for
NT clones versus the specific guide selected from a requested perturbation in the
gene_10 dataset. When multiple guides for a perturbation are significant for the
requested cell state, the guide with the largest absolute Wilcoxon rank-sum
z-statistic is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))
sys.path.append(str(REPO_ROOT / "manuscript" / "scripts"))

from figure_format import get_adata, perturbation_colors, set_rc_params  # type: ignore
from proportion_heatmap_utils import SIG_THRESHOLD  # type: ignore

OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_7"
NT_COLOR = "#808080"
BINS = np.linspace(0, 1, 21)


def load_gene10() -> pd.DataFrame:
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=False,
        get_raw_scaled=True,
    )
    obs = adata.obs.copy()
    obs = obs.loc[obs["guide"] != "INTERGENIC-3"].copy()
    obs = obs.loc[obs["clone"].notna()].copy()
    obs["clone"] = obs["clone"].astype(str)
    obs["guide"] = obs["guide"].astype(str)
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    return obs


def compute_clone_proportions(obs: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    clone_props = (
        obs.assign(cell_type_match=(obs["cell_type"] == cell_type).astype(int))
        .groupby(["guide", "perturbation", "clone"], observed=False)
        .agg(
            clone_size=("cell_type", "size"),
            cell_type_count=("cell_type_match", "sum"),
        )
        .reset_index()
    )
    clone_props["proportion"] = clone_props["cell_type_count"] / clone_props["clone_size"]
    return clone_props


def compute_adjusted_pvalues(clone_props: pd.DataFrame) -> pd.DataFrame:
    nt_guides = set(
        clone_props.loc[clone_props["perturbation"] == "NT", "guide"].drop_duplicates().tolist()
    )
    guides = sorted(clone_props["guide"].unique())
    rows = []
    for guide in guides:
        guide_rows = clone_props.loc[clone_props["guide"] == guide]
        perturbation = str(guide_rows["perturbation"].iloc[0])
        guide_props = guide_rows["proportion"].to_numpy()

        if perturbation == "NT":
            reference_props = clone_props.loc[
                (clone_props["perturbation"] == "NT") & (clone_props["guide"] != guide),
                "proportion",
            ].to_numpy()
        else:
            reference_props = clone_props.loc[
                clone_props["perturbation"] == "NT", "proportion"
            ].to_numpy()

        statistic, pvalue = ranksums(reference_props, guide_props, alternative="two-sided")
        rows.append(
            {
                "guide": guide,
                "perturbation": perturbation,
                "p_value": pvalue,
                "z_statistic": -statistic,
                "n_nt": len(reference_props),
                "n_guide": len(guide_props),
                "mean_nt": float(reference_props.mean()),
                "mean_guide": float(guide_props.mean()),
                "is_control_guide": guide in nt_guides,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["p_adj"] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
    return stats_df


def select_target_guide(
    stats_df: pd.DataFrame,
    perturbation: str | None,
    guide: str | None,
) -> pd.Series:
    if guide is not None:
        selected = stats_df.loc[stats_df["guide"] == guide]
        if selected.empty:
            raise ValueError(f"Guide '{guide}' was not found.")
        return selected.iloc[0]

    if perturbation is None:
        raise ValueError("Either --perturbation or --guide must be provided.")

    selected = stats_df.loc[stats_df["perturbation"] == perturbation].copy()
    if selected.empty:
        raise ValueError(f"Perturbation '{perturbation}' was not found.")

    selected = selected.loc[selected["p_adj"] < SIG_THRESHOLD].copy()
    if selected.empty:
        raise ValueError(
            f"No guides for perturbation '{perturbation}' were significant at "
            f"BH-adjusted p < {SIG_THRESHOLD:.2f}."
        )

    selected["abs_effect_size"] = selected["z_statistic"].abs()
    selected = selected.sort_values(
        ["abs_effect_size", "z_statistic", "guide"], ascending=[False, False, True]
    )
    return selected.iloc[0]


def plot_histogram(
    nt_props: np.ndarray,
    guide_props: np.ndarray,
    guide: str,
    perturbation: str,
    cell_type: str,
    p_adj: float,
) -> plt.Figure:
    color = perturbation_colors.get(perturbation, "#d62728")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    ax.hist(
        nt_props,
        bins=BINS,
        color=NT_COLOR,
        alpha=0.5,
        density=True,
        edgecolor=NT_COLOR,
        linewidth=1.0,
        label=f"NT (n={len(nt_props)})",
    )
    ax.hist(
        guide_props,
        bins=BINS,
        color=color,
        alpha=0.5,
        density=True,
        edgecolor=color,
        linewidth=1.0,
        label=f"{guide} (n={len(guide_props)})",
    )

    ax.set_xlabel(f"fraction of clone in {cell_type}")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Wilcoxon rank-sum, BH-adjusted p={p_adj:.2g}")

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--perturbation")
    group.add_argument("--guide")
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--stem", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_rc_params()
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["axes.grid"] = False

    obs = load_gene10()
    clone_props = compute_clone_proportions(obs, args.cell_type)
    stats_df = compute_adjusted_pvalues(clone_props)
    stat_row = select_target_guide(
        stats_df=stats_df,
        perturbation=args.perturbation,
        guide=args.guide,
    )

    selected_guide = str(stat_row["guide"])
    selected_perturbation = str(stat_row["perturbation"])
    if selected_perturbation == "NT":
        nt_props = clone_props.loc[
            (clone_props["perturbation"] == "NT") & (clone_props["guide"] != selected_guide),
            "proportion",
        ].to_numpy()
    else:
        nt_props = clone_props.loc[clone_props["perturbation"] == "NT", "proportion"].to_numpy()
    guide_props = clone_props.loc[clone_props["guide"] == selected_guide, "proportion"].to_numpy()

    fig = plot_histogram(
        nt_props=nt_props,
        guide_props=guide_props,
        guide=selected_guide,
        perturbation=selected_perturbation,
        cell_type=args.cell_type,
        p_adj=float(stat_row["p_adj"]),
    )
    save_figure(fig, args.stem)
    plt.close(fig)

    print(
        f"{args.stem}: selected {selected_guide} ({selected_perturbation}) for {args.cell_type} "
        f"(n_nt={len(nt_props)}, n_guide={len(guide_props)}, "
        f"z={stat_row['z_statistic']:.3g}, p={stat_row['p_value']:.3g}, "
        f"p_adj={stat_row['p_adj']:.3g})"
    )


if __name__ == "__main__":
    main()

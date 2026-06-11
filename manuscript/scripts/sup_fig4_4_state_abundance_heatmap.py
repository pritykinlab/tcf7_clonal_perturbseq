#!/usr/bin/env python
"""
Generate deprecated internal Sup_Figure_4_4 per-state abundance heatmaps.

Panels reuse the Figure 4a/4b negative-binomial guide-abundance framework,
but repeat the test separately for each cell state.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import nbinom
from statsmodels.stats.multitest import multipletests

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

sys.path.append("../manuscript/scripts")
from guide_abundance_correlation_analysis import (  # noqa: E402
    convert_long_to_short,
)
from proportion_heatmap_utils import truncate_colormap  # noqa: E402

sys.path.append("../scripts/utils")
from figure_format import cell_type_orders_gene10, get_adata  # noqa: E402


GUIDES_TO_REMOVE = {"INTERGENIC-3"}
SIG_THRESHOLD = 0.05
ZSCORE_LIMIT = 5
PROGENITOR_SIDE = {"Progenitor", "Progenitor Cycling"}

PANEL_TITLES = {
    ("individual", "cells"): "Cell abundance by state",
    ("individual", "clones"): "clone abundance by state",
    ("grouped", "cells"): "Cell abundance by grouped state",
    ("grouped", "clones"): "clone abundance by grouped state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../manuscript/plots/Sup_Figure_4_4",
    )
    parser.add_argument("--panel-letter", required=True)
    parser.add_argument(
        "--analysis-type",
        choices=["cells", "clones"],
        required=True,
    )
    parser.add_argument(
        "--state-mode",
        choices=["individual", "grouped"],
        default="individual",
        help="Use individual cell states or grouped Progenitor-side versus differentiated states.",
    )
    return parser.parse_args()


def fit_negative_binomial(data_df: pd.DataFrame) -> tuple[float, float]:
    control_df = data_df[data_df["is_control_guide"]].copy()
    obs_control = control_df["count"].astype(float)
    control_conc = control_df["Concentration"].astype(float)

    def neg_ll_nb(params, obs, x_conc):
        r, slope = params
        mu = x_conc * slope
        p = r / (r + mu + 1e-10)
        return -np.sum(nbinom.logpmf(obs, r, p))

    r_init = 1.0
    if control_conc.mean() > 0:
        slope_init = max(obs_control.mean() / control_conc.mean(), 0.001)
    else:
        slope_init = 0.001

    res = minimize(
        neg_ll_nb,
        [r_init, slope_init],
        args=(obs_control, control_conc),
        bounds=[(0.001, None), (0.001, None)],
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])


def nb_two_sided_pvalue(counts: pd.Series, mus: pd.Series, r_star: float) -> np.ndarray:
    p_param = r_star / (r_star + mus + 1e-10)
    lower_tail = nbinom.cdf(counts, r_star, p_param)
    upper_tail = nbinom.sf(counts - 1, r_star, p_param)
    return np.clip(2 * np.minimum(lower_tail, upper_tail), 0, 1)


def build_count_df(
    obs: pd.DataFrame,
    state_label: str,
    state_cell_types: set[str],
    analysis_type: str,
    guide_conc_dict: dict[str, float],
    guide_names: list[str],
) -> pd.DataFrame:
    records = [
        {"Guide": guide, "Concentration": guide_conc_dict.get(guide, 0)}
        for guide in guide_names
    ]
    data_df = pd.DataFrame(records)

    state_mask = obs["cell_type"].isin(state_cell_types)
    state_obs = obs.loc[state_mask, ["guide", "all_constructs_str"]]
    if analysis_type == "cells":
        counts = state_obs["guide"].value_counts().to_dict()
    else:
        counts = {}
        clones = state_obs["all_constructs_str"].dropna().astype(str).unique()
        for clone in clones:
            guide = clone.split("__")[0]
            counts[guide] = counts.get(guide, 0) + 1

    data_df["count"] = data_df["Guide"].map(lambda guide: counts.get(guide, 0))
    data_df["is_control_guide"] = data_df["Guide"].map(
        lambda guide: ("NO_SITE" in guide) or ("INTERGENIC" in guide)
    )
    return data_df


def run_state_test(
    obs: pd.DataFrame,
    state_label: str,
    state_cell_types: set[str],
    analysis_type: str,
    guide_conc_dict: dict[str, float],
    guide_names: list[str],
) -> pd.DataFrame:
    data_df = build_count_df(
        obs=obs,
        state_label=state_label,
        state_cell_types=state_cell_types,
        analysis_type=analysis_type,
        guide_conc_dict=guide_conc_dict,
        guide_names=guide_names,
    )

    r_star, slope = fit_negative_binomial(data_df)
    alpha_nb = 1 / r_star

    counts = data_df["count"].astype(float)
    expected_mean = data_df["Concentration"].astype(float) * slope
    stdev = np.sqrt(expected_mean + alpha_nb * (expected_mean**2))
    z_score = (counts - expected_mean) / (stdev + 1e-10)
    pvalues = nb_two_sided_pvalue(counts, expected_mean, r_star)
    _, adjusted, _, _ = multipletests(pvalues, method="fdr_bh")

    results = pd.DataFrame(
        {
            "cell_type": state_label,
            "guide": data_df["Guide"].values,
            "analysis_type": analysis_type,
            "count": counts.values,
            "expected_mean": expected_mean.values,
            "z_score": z_score.values,
            "pvalue": pvalues,
            "bh_adj_pvalue": adjusted,
            "is_control_guide": data_df["is_control_guide"].values,
            "nb_r": r_star,
            "nb_alpha": alpha_nb,
            "nb_slope": slope,
        }
    )
    return results


def ordered_guides(guides: pd.Index) -> list[str]:
    no_site = sorted([g for g in guides if g.startswith("NO_SITE")])
    intergenic = sorted([g for g in guides if "INTERGENIC" in g])
    others = sorted(set(guides) - set(no_site) - set(intergenic))
    return no_site + intergenic + others


def state_definitions(state_mode: str) -> list[tuple[str, set[str]]]:
    if state_mode == "grouped":
        differentiated = set(cell_type_orders_gene10) - PROGENITOR_SIDE
        return [
            ("Progenitor + Progenitor Cycling", PROGENITOR_SIDE),
            ("Differentiated", differentiated),
        ]
    return [(cell_type, {cell_type}) for cell_type in cell_type_orders_gene10]


def create_heatmap(
    results: pd.DataFrame,
    output_dir: str,
    panel_letter: str,
    analysis_type: str,
    state_order: list[str],
    state_mode: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    z_scores = results.pivot_table(
        index="cell_type",
        columns="guide",
        values="z_score",
        aggfunc="mean",
    )
    adjusted_pvalues = results.pivot_table(
        index="cell_type",
        columns="guide",
        values="bh_adj_pvalue",
        aggfunc="min",
    )

    guide_order = ordered_guides(z_scores.columns)
    z_scores = z_scores.reindex(index=state_order, columns=guide_order)
    adjusted_pvalues = adjusted_pvalues.reindex(index=state_order, columns=guide_order)
    mask = adjusted_pvalues >= SIG_THRESHOLD

    fig_height = 5.8 if state_mode == "individual" else 3.8
    fig, ax = plt.subplots(figsize=(19, fig_height))
    cmap = truncate_colormap(plt.cm.PuOr, 0.2, 0.8)
    sns.heatmap(
        z_scores,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-ZSCORE_LIMIT,
        vmax=ZSCORE_LIMIT,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 10.8},
        cbar_kws={"label": "NB z-score", "shrink": 0.8, "aspect": 10},
        linewidths=0.5,
        linecolor="lightgray",
        rasterized=True,
        xticklabels=list(z_scores.columns),
        yticklabels=list(z_scores.index),
        ax=ax,
    )

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=20)
    colorbar.set_label("NB z-score", fontsize=20)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(PANEL_TITLES[(state_mode, analysis_type)], fontsize=28, pad=50)
    ax.tick_params(axis="x", labelrotation=90, labelsize=16)
    ax.tick_params(axis="y", labelrotation=0, labelsize=18 if state_mode == "grouped" else 20)

    control_guides = [g for g in z_scores.columns if "NO_SITE" in g or "INTERGENIC" in g]
    ko_guides = [g for g in z_scores.columns if g not in control_guides]
    control_sig_count = int((adjusted_pvalues[control_guides] < SIG_THRESHOLD).sum().sum())
    ko_sig_count = int((adjusted_pvalues[ko_guides] < SIG_THRESHOLD).sum().sum())
    total_control_tests = len(z_scores.index) * len(control_guides)
    total_ko_tests = len(z_scores.index) * len(ko_guides)

    ax.text(
        0.00,
        1.04,
        f"{control_sig_count}/{total_control_tests} control tests\nare significant",
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="left",
        multialignment="center",
        clip_on=False,
    )
    ax.text(
        1.00,
        1.04,
        f"{ko_sig_count}/{total_ko_tests} guide tests\nare significant",
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="right",
        multialignment="center",
        clip_on=False,
    )

    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{panel_letter}.png")
    svg_path = os.path.join(output_dir, f"{panel_letter}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    args = parse_args()

    print("Loading gene10 data...")
    adata = get_adata(
        dataset="gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=False,
    )
    adata.obs["guide"] = adata.obs["guide"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    obs = adata.obs.loc[~adata.obs["guide"].isin(GUIDES_TO_REMOVE)].copy()
    del adata
    guide_names = sorted(obs["guide"].unique())

    print("Loading guide concentration data...")
    guide_conc_df = pd.read_csv(
        "../data/guide_concentration/gene10.csv"
    )
    guide_conc_df["short_guide"] = guide_conc_df["Row Barcode IDs"].apply(convert_long_to_short)
    guide_conc_dict = guide_conc_df.set_index("short_guide")["cp8 raw counts"].to_dict()

    all_results = []
    states = state_definitions(args.state_mode)
    for state_label, state_cell_types in states:
        print(f"Testing {state_label} {args.analysis_type}...")
        all_results.append(
            run_state_test(
                obs=obs,
                state_label=state_label,
                state_cell_types=state_cell_types,
                analysis_type=args.analysis_type,
                guide_conc_dict=guide_conc_dict,
                guide_names=guide_names,
            )
        )

    results = pd.concat(all_results, ignore_index=True)
    stats_path = os.path.join(args.output_dir, f"{args.panel_letter}_stats.tsv")
    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(stats_path, sep="\t", index=False, float_format="%.6e")
    print(f"Saved statistics to {stats_path}")

    create_heatmap(
        results,
        args.output_dir,
        args.panel_letter,
        args.analysis_type,
        [state_label for state_label, _ in states],
        args.state_mode,
    )
    print(f"deprecated internal Sup_Figure_4_4{args.panel_letter} completed successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Figure S4b.

This panel shows an UpSet-style summary of construct-sharing patterns among
putative clones linked to the largest retained clones in the gene10 dataset.
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "supplementary_analysis" / "construct_threshold"))

from sweep_utils import compute_triple_counts, constructs_str_by_cbc, load_singlet_adata  # noqa: E402


FINAL_H5AD_PATH = REPO_ROOT / "data" / "gene10" / "preprocessing" / "final.h5ad"
COMBINED_H5AD_PATH = REPO_ROOT / "data" / "gene10" / "preprocessing" / "01_combined.h5ad"
HASHTAG_METADATA_DIR = REPO_ROOT / "data" / "gene10" / "preprocessing" / "hashtag_metadata"
RAW_GUIDE_UMI_TSV = REPO_ROOT / "raw_data" / "new_data" / "exp49_CP8" / "pycashier" / "CBC_guide_umi.tsv"
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_4"

MIN_COUNT = 10
MIN_PUTATIVE_CLONE_CELLS = 3
N_TOP_CLONES = 20

QUALITATIVE_CONSTRUCT_COLORS = [
    "#1F4E79",
    "#C06C2B",
    "#3A7D44",
    "#8E4B9B",
    "#B33C2E",
    "#2E7C9A",
    "#8C6D1F",
    "#4B5D16",
    "#A04A73",
    "#5B4B8A",
]


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["axes.grid"] = False
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["legend.title_fontsize"] = 11
plt.rcParams["legend.edgecolor"] = "dimgray"


def load_final_obs(final_h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(final_h5ad_path, backed="r")
    obs = adata.obs.copy()
    obs.index = adata.obs_names.copy()
    return obs


def get_top_retained_clones(obs: pd.DataFrame) -> pd.DataFrame:
    clone_obs = obs.loc[:, ["clone", "all_constructs_str"]].copy()
    clone_obs["clone"] = clone_obs["clone"].astype(str)
    clone_obs["all_constructs_str"] = clone_obs["all_constructs_str"].astype(str)
    clone_obs = clone_obs.loc[clone_obs["clone"] != ""].copy()

    clone_sizes = clone_obs.groupby("clone", observed=False).size().rename("clone_size")
    top_clones = (
        clone_obs.loc[:, ["clone", "all_constructs_str"]]
        .drop_duplicates()
        .join(clone_sizes, on="clone")
        .sort_values(["clone_size", "clone"], ascending=[False, True])
        .head(N_TOP_CLONES)
        .reset_index(drop=True)
    )

    print("Top retained clones used to define focal constructs:")
    print(top_clones.to_string(index=False))
    return top_clones


def get_raw_construct_calls() -> pd.Series:
    adata_base = load_singlet_adata(
        adata_path=str(COMBINED_H5AD_PATH),
        metadata_dir=str(HASHTAG_METADATA_DIR),
        num_lanes=8,
    )
    valid_cbcs = set(adata_base.obs_names)
    triple_counts = compute_triple_counts(str(RAW_GUIDE_UMI_TSV), valid_cbcs=valid_cbcs)
    raw_constructs = constructs_str_by_cbc(triple_counts, min_count=MIN_COUNT)
    return raw_constructs


def get_focus_constructs(top_clones: pd.DataFrame, obs: pd.DataFrame, raw_constructs: pd.Series) -> pd.DataFrame:
    focus_rows: list[dict[str, object]] = []
    seen_raw_constructs: set[str] = set()

    for row in top_clones.itertuples(index=False):
        clone_cells = obs.index[obs["clone"].astype(str) == row.clone]
        raw_mode = raw_constructs.reindex(clone_cells).dropna().value_counts()
        if raw_mode.empty:
            raise ValueError(f"Could not recover raw construct call for retained clone: {row.clone}")
        raw_construct = str(raw_mode.index[0])
        if raw_construct in seen_raw_constructs:
            continue
        seen_raw_constructs.add(raw_construct)
        focus_rows.append(
            {
                "raw_construct": raw_construct,
                "display_label": str(row.all_constructs_str),
                "retained_clone_size": int(row.clone_size),
            }
        )

    focus_df = pd.DataFrame(focus_rows)

    print("\nFocal constructs carried by the top retained clones:")
    print(focus_df.to_string(index=False))
    return focus_df


def build_selected_putative_clones(
    raw_constructs: pd.Series,
    focus_raw_constructs: list[str],
    summary_prefix: str = "Selected",
) -> pd.DataFrame:
    focus_set = set(focus_raw_constructs)
    clone_table = raw_constructs.value_counts().rename_axis("putative_clone").reset_index(name="clone_size")
    clone_table["n_constructs"] = clone_table["putative_clone"].str.count("___") + 1
    clone_table["uses_only_focus_constructs"] = clone_table["putative_clone"].map(
        lambda clone: set(clone.split("___")).issubset(focus_set)
    )
    clone_table = clone_table.loc[clone_table["uses_only_focus_constructs"]].copy()
    clone_table = clone_table.loc[clone_table["clone_size"] >= MIN_PUTATIVE_CLONE_CELLS].copy()
    clone_table = clone_table.sort_values(
        ["clone_size", "n_constructs", "putative_clone"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    print(
        f"\n{summary_prefix} "
        f"{len(clone_table)} putative clones with >= {MIN_PUTATIVE_CLONE_CELLS} cells "
        "composed only of the focal constructs."
    )
    return clone_table


def filter_focus_constructs_by_observed_sharing(
    focus_df: pd.DataFrame,
    candidate_putative_clones: pd.DataFrame,
) -> pd.DataFrame:
    shared_constructs = sorted(
        {
            construct
            for putative_clone in candidate_putative_clones.loc[
                candidate_putative_clones["n_constructs"] >= 2, "putative_clone"
            ]
            for construct in putative_clone.split("___")
        }
    )
    if not shared_constructs:
        raise ValueError(
            "No focal constructs were observed in a multi-construct putative clone with another focal construct."
        )

    filtered_focus_df = (
        focus_df.loc[focus_df["raw_construct"].isin(shared_constructs)]
        .copy()
        .reset_index(drop=True)
    )

    print("\nFocal constructs retained after requiring observed sharing with another focal construct:")
    print(filtered_focus_df.to_string(index=False))
    return filtered_focus_df


def build_construct_color_map(all_constructs: list[str]) -> dict[str, tuple[float, float, float, float] | str]:
    colors: list[str] = []
    for i, construct in enumerate(all_constructs):
        if i < len(QUALITATIVE_CONSTRUCT_COLORS):
            colors.append(QUALITATIVE_CONSTRUCT_COLORS[i])
            continue

        # Keep overflow colors in a dark, saturated range so the small text stays readable.
        hue = (0.11 + (i - len(QUALITATIVE_CONSTRUCT_COLORS)) * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.38, 0.62)
        colors.append((red, green, blue))

    return {construct: colors[i] for i, construct in enumerate(all_constructs)}


def build_upset_table(putative_clones: pd.DataFrame, focus_df: pd.DataFrame) -> pd.DataFrame:
    construct_order = focus_df["raw_construct"].tolist()
    upset_table = putative_clones.copy()
    upset_table["membership_signature"] = upset_table["putative_clone"].map(
        lambda clone: tuple(idx for idx, construct in enumerate(construct_order) if construct in clone.split("___"))
    )
    for construct in construct_order:
        upset_table[construct] = upset_table["putative_clone"].map(
            lambda clone, current_construct=construct: current_construct in clone.split("___")
        )

    upset_table = upset_table.sort_values(
        ["n_constructs", "membership_signature", "clone_size", "putative_clone"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return upset_table


def plot_panel(
    putative_clones: pd.DataFrame,
    focus_df: pd.DataFrame,
) -> plt.Figure:
    construct_order = focus_df["raw_construct"].tolist()
    display_label_map = dict(zip(focus_df["raw_construct"], focus_df["display_label"]))
    construct_color_map = build_construct_color_map(construct_order)
    upset_table = build_upset_table(putative_clones, focus_df)

    n_columns = len(upset_table)
    n_rows = len(construct_order)
    x_positions = np.arange(n_columns)
    row_positions = np.arange(n_rows)[::-1]
    row_position_map = {construct: row_positions[idx] for idx, construct in enumerate(construct_order)}

    fig_width = max(9.5, 0.34 * n_columns)
    fig_height = max(5.4, 2.6 + 0.34 * n_rows)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gridspec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.5, max(1.8, 0.28 * n_rows)], hspace=0.04)
    bar_ax = fig.add_subplot(gridspec[0])
    matrix_ax = fig.add_subplot(gridspec[1], sharex=bar_ax)

    bar_ax.bar(
        x_positions,
        upset_table["clone_size"].to_numpy(),
        width=0.72,
        color="0.65",
        edgecolor="0.4",
        linewidth=0.8,
    )

    bar_ax.set_ylabel("# cells")
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    bar_ax.set_xlim(-0.6, n_columns - 0.4)

    matrix_ax.scatter(
        np.repeat(x_positions, n_rows),
        np.tile(row_positions, n_columns),
        s=16,
        color="0.88",
        zorder=1,
    )
    for column_idx, row in upset_table.iterrows():
        present_constructs = [construct for construct in construct_order if row[construct]]
        present_y = [row_position_map[construct] for construct in present_constructs]
        if len(present_y) >= 2:
            matrix_ax.plot(
                [column_idx, column_idx],
                [min(present_y), max(present_y)],
                color="0.35",
                linewidth=1.0,
                zorder=2,
            )
        for construct in present_constructs:
            matrix_ax.scatter(
                column_idx,
                row_position_map[construct],
                s=34,
                color=construct_color_map[construct],
                edgecolor="none",
                zorder=3,
            )

    matrix_ax.set_yticks(row_positions)
    matrix_ax.set_yticklabels([display_label_map[construct] for construct in construct_order])
    for tick_label, construct in zip(matrix_ax.get_yticklabels(), construct_order):
        tick_label.set_color(construct_color_map[construct])
    matrix_ax.tick_params(axis="y", length=0, pad=6)
    matrix_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    matrix_ax.spines["top"].set_visible(False)
    matrix_ax.spines["right"].set_visible(False)
    matrix_ax.spines["bottom"].set_visible(False)
    matrix_ax.set_ylim(-0.7, n_rows - 0.3)
    matrix_ax.set_xlim(-0.6, n_columns - 0.4)

    fig.subplots_adjust(top=0.98, bottom=0.08, left=0.23, right=0.98)
    return fig


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "b.png"
    svg_path = OUTPUT_DIR / "b.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    if not FINAL_H5AD_PATH.exists():
        raise FileNotFoundError(f"Missing final AnnData: {FINAL_H5AD_PATH}")
    if not COMBINED_H5AD_PATH.exists():
        raise FileNotFoundError(f"Missing combined AnnData: {COMBINED_H5AD_PATH}")
    if not HASHTAG_METADATA_DIR.exists():
        raise FileNotFoundError(f"Missing hashtag metadata directory: {HASHTAG_METADATA_DIR}")
    if not RAW_GUIDE_UMI_TSV.exists():
        raise FileNotFoundError(f"Missing raw guide/UMI TSV: {RAW_GUIDE_UMI_TSV}")

    obs = load_final_obs(FINAL_H5AD_PATH)
    top_clones = get_top_retained_clones(obs)
    raw_constructs = get_raw_construct_calls()
    focus_df = get_focus_constructs(top_clones, obs, raw_constructs)
    candidate_putative_clones = build_selected_putative_clones(
        raw_constructs,
        focus_df["raw_construct"].tolist(),
        summary_prefix="Candidate",
    )
    focus_df = filter_focus_constructs_by_observed_sharing(focus_df, candidate_putative_clones)
    putative_clones = build_selected_putative_clones(raw_constructs, focus_df["raw_construct"].tolist())
    if putative_clones.empty:
        raise ValueError("No putative clones matched the selected focal constructs.")

    fig = plot_panel(putative_clones, focus_df)
    save_figure(fig)


if __name__ == "__main__":
    main()

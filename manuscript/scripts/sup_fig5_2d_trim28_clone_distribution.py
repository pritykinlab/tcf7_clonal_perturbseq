#!/usr/bin/env python
"""
Generate the Figure S11d source panel: clone-level cell-state compositions
for expanded Trim28 clones in the Trim28-Setdb1 dataset.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import cell_type_colors, cell_type_orders_trim28_setdb1, get_adata  # noqa: E402

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_11_shared_source"
N_CLONES_TO_SHOW = 9


def validate_file_sizes(png_path: str, svg_path: str) -> None:
    """Report whether PNG/SVG outputs meet manuscript size requirements."""
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Consider rasterization.")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trim28_setdb1 data...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=False,
        remove_NG=True,
        remove_lymph_node=False,
    )

    obs = adata.obs.copy()
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["clone"] = obs["clone"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)

    trim28_obs = obs[obs["perturbation"] == "Trim28"].copy()
    clone_sizes = trim28_obs.groupby("clone").size().rename("clone_size")
    trim28_obs = trim28_obs[trim28_obs["clone"].isin(clone_sizes.index)].copy()

    print(f"Retained {len(clone_sizes)} Trim28 clones")

    unique_clones_sorted = (
        clone_sizes
        .reset_index()
        .sort_values("clone_size", ascending=False)
    )

    clone_sizes_array = unique_clones_sorted["clone_size"].to_numpy()
    log2_clone_sizes = np.log2(clone_sizes_array + 0.5)
    weights = log2_clone_sizes / log2_clone_sizes.sum()
    cumulative_weights = np.cumsum(weights)

    if len(unique_clones_sorted) >= N_CLONES_TO_SHOW:
        bin_edges = np.linspace(0, 1, N_CLONES_TO_SHOW + 1)
        target_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

        selected_indices = []
        for target in target_positions:
            idx = np.searchsorted(cumulative_weights, target)
            idx = min(idx, len(cumulative_weights) - 1)
            selected_indices.append(idx)

        selected_indices = list(dict.fromkeys(selected_indices))
        selected_clones = unique_clones_sorted.iloc[selected_indices]
    else:
        selected_clones = unique_clones_sorted

    selected_clone_ids = selected_clones["clone"].tolist()

    count_data = (
        trim28_obs
        .groupby(["clone", "cell_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cell_type_orders_trim28_setdb1, fill_value=0)
    )
    proportion_data = count_data.div(count_data.sum(axis=1), axis=0)
    proportion_data["clone_size"] = clone_sizes.loc[proportion_data.index]
    proportion_data["log2_clone_size"] = np.log2(proportion_data["clone_size"])

    sort_cols = ["log2_clone_size"] + cell_type_orders_trim28_setdb1[::-1]
    proportion_data = proportion_data.sort_values(by=sort_cols, ascending=False)

    clone_to_x_position = {
        clone_id: idx
        for idx, clone_id in enumerate(proportion_data.index.tolist())
    }
    selected_x_positions = [clone_to_x_position[clone_id] for clone_id in selected_clone_ids if clone_id in clone_to_x_position]
    selected_clone_sizes = [int(clone_sizes.loc[clone_id]) for clone_id in selected_clone_ids if clone_id in clone_to_x_position]

    sorted_clone_sizes = proportion_data["clone_size"].to_numpy()
    log2_sorted_clone_sizes = np.log2(sorted_clone_sizes)
    stacked_props = proportion_data[cell_type_orders_trim28_setdb1]

    print("Creating Figure S11d source panel...")
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(20 * 1.1, 4 * 1.1),
        gridspec_kw={"height_ratios": [0.5, 1], "hspace": 0.01},
    )

    ax_top.plot(log2_sorted_clone_sizes, linestyle="-", color="gray", rasterized=True)
    ax_top.set_ylabel("log2(clone size)", fontsize=20, rotation=45, labelpad=50)
    ax_top.set_xlim(0, len(sorted_clone_sizes))
    ax_top.set_yticks([])
    ax_top.set_xticks([])

    for x_pos, clone_size in zip(selected_x_positions, selected_clone_sizes):
        y_value = log2_sorted_clone_sizes[x_pos]
        ax_top.plot(x_pos, y_value, 'ro', markersize=8, rasterized=True)
        ax_top.text(
            x_pos,
            y_value,
            f"{clone_size}",
            fontsize=24,
            verticalalignment="bottom",
            horizontalalignment="center",
            color="red",
        )

    stacked_props.plot(
        kind="bar",
        stacked=True,
        ax=ax_bottom,
        width=1,
        color=[cell_type_colors[cell_type] for cell_type in cell_type_orders_trim28_setdb1],
        legend=False,
    )

    for patch in ax_bottom.patches:
        patch.set_rasterized(True)

    ax_bottom.set_xlabel("")
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([0, 0.5, 1.0])
    ax_bottom.set_ylabel("Cell type\nproportion")
    ax_bottom.set_ylim(0, 1)
    ax_bottom.text(
        0.5,
        -0.25,
        f"Trim28 clone (n={len(stacked_props)})",
        transform=ax_bottom.transAxes,
        ha="center",
        fontsize=30,
    )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.98, bottom=0.4, hspace=0.01)

    png_path = os.path.join(OUTPUT_DIR, "c.png")
    svg_path = os.path.join(OUTPUT_DIR, "c.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(png_path, svg_path)
    print("Figure S11d source panel completed successfully!")


if __name__ == "__main__":
    main()

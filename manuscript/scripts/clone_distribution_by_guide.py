#!/usr/bin/env python
"""
Generate Figure S5 clone distributions for guide sets.

Panel a_1: gene-targeting guides.
Panel a_2: NT guides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import cell_type_colors, cell_type_orders_gene10, get_adata  # noqa: E402


plt.rcParams["axes.titlesize"] = 22
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
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_5"
TARGETING_TITLE = "Gene-targeting guides"
CONTROL_TITLE = "NT Guides"
ANNOTATED_CLONE_SIZES = [1000, 50, 5, 1]
GUIDE_LABEL_FONTSIZE = 14


def create_guide_proportion_data(adata, guide_name: str):
    """Create proportion and clone size data for a specific guide."""
    guide_mask = adata.obs["guide"].astype(str) == guide_name
    guide_data = adata[guide_mask].copy()
    if len(guide_data) == 0:
        return None

    guide_data.obs["clone"] = guide_data.obs["clone"].astype(str)
    guide_data.obs["cell_type"] = guide_data.obs["cell_type"].astype(str)

    count_data = (
        guide_data.obs.groupby(["clone", "cell_type"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    proportion_data = count_data.div(count_data.sum(axis=1), axis=0)
    proportion_data = proportion_data.reindex(
        columns=cell_type_orders_gene10,
        fill_value=0,
    )

    clone_sizes = (
        guide_data.obs.groupby("clone", observed=False)
        .size()
        .sort_values(ascending=False)
    )
    proportion_data["clone_size"] = clone_sizes.reindex(proportion_data.index)
    proportion_data["log2_clone_size"] = np.log2(proportion_data["clone_size"] + 1)

    sort_cols = ["log2_clone_size"] + list(cell_type_orders_gene10[::-1])
    proportion_data = (
        proportion_data.reset_index()
        .sort_values(by=sort_cols, ascending=False)
        .set_index("clone")
    )
    proportion_data = proportion_data[cell_type_orders_gene10]

    sorted_clone_sizes = clone_sizes.reset_index(drop=True)
    log2_sorted_clone_sizes = np.log2(sorted_clone_sizes)

    return proportion_data, sorted_clone_sizes, log2_sorted_clone_sizes


def build_guide_data(adata):
    guide_data = {}
    for guide in sorted(adata.obs["guide"].astype(str).unique()):
        result = create_guide_proportion_data(adata, guide)
        if result is not None:
            guide_data[guide] = result
    return guide_data


def add_clone_size_annotations(ax, sorted_clone_sizes, log2_sorted_clone_sizes):
    for size in ANNOTATED_CLONE_SIZES:
        for idx, clone_size in enumerate(sorted_clone_sizes):
            if clone_size <= size:
                y_value = log2_sorted_clone_sizes[idx]
                ax.plot(idx, y_value, "ro")
                ax.text(
                    idx,
                    y_value,
                    f"{clone_size}",
                    fontsize=11,
                    verticalalignment="bottom",
                    color="red",
                )
                break


def set_bottom_clone_index_ticks(ax, max_x: int, xlabel: str):
    if max_x <= 0:
        ax.set_xticks([])
        return
    tick_values = np.arange(25, max_x, 25, dtype=int)
    tick_positions = tick_values - 1
    ax.set_xlim(-0.5, max_x - 0.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(value) for value in tick_values], fontsize=GUIDE_LABEL_FONTSIZE)
    ax.tick_params(axis="x", length=3, pad=1)
    ax.set_xlabel(xlabel, fontsize=GUIDE_LABEL_FONTSIZE, labelpad=2)
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)


def rasterize_bar_patches(ax):
    for container in ax.containers:
        for patch in container:
            patch.set_rasterized(True)


def plot_guide_group(fig, outer_spec, guides, guide_data, title: str, total_slots: int, xlabel: str):
    guides = [guide for guide in guides if guide in guide_data]
    if not guides:
        ax = fig.add_subplot(outer_spec)
        ax.text(0.5, 0.5, f"No guides found for\n{title}", ha="center", va="center")
        ax.axis("off")
        return

    height_ratios = []
    for idx in range(total_slots):
        height_ratios.extend([0.3, 1.0])
        if idx < total_slots - 1:
            height_ratios.append(0.1)
    height_ratios.append(0.12)

    inner_spec = outer_spec.subgridspec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=0.01,
    )
    plot_colors = [cell_type_colors.get(cell_type, "gray") for cell_type in cell_type_orders_gene10]
    max_x = max(len(guide_data[guide][0]) for guide in guides)

    row_idx = 0
    first_line_ax = None
    for slot_idx in range(total_slots):
        ax_line = fig.add_subplot(inner_spec[row_idx, 0])
        ax_bar = fig.add_subplot(inner_spec[row_idx + 1, 0], sharex=ax_line)
        if first_line_ax is None:
            first_line_ax = ax_line
        row_idx += 2

        if slot_idx < total_slots - 1:
            ax_blank = fig.add_subplot(inner_spec[row_idx, 0])
            ax_blank.axis("off")
            row_idx += 1

        if slot_idx >= len(guides):
            ax_line.set_visible(False)
            ax_bar.set_visible(False)
            continue

        guide = guides[slot_idx]
        proportion_data, sorted_clone_sizes, log2_sorted_clone_sizes = guide_data[guide]

        ax_line.plot(log2_sorted_clone_sizes, linestyle="-", color="black", linewidth=1.0)
        ax_line.set_xlim(-0.5, max_x - 0.5)
        ax_line.set_xticks([])
        ax_line.set_yticks([])
        add_clone_size_annotations(ax_line, sorted_clone_sizes, log2_sorted_clone_sizes)

        proportion_data.plot(
            kind="bar",
            stacked=True,
            ax=ax_bar,
            width=1,
            color=plot_colors,
            legend=False,
        )
        rasterize_bar_patches(ax_bar)
        ax_bar.set_xlim(-0.5, max_x - 0.5)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("")
        ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)

        box_line = ax_line.get_position()
        box_bar = ax_bar.get_position()
        # Center the rotated guide label across the full line+bar block.
        y_mid = 0.5 * (box_line.y1 + box_bar.y0)
        x_label = box_line.x0 - 0.03
        fig.text(
            x_label,
            y_mid,
            f"{guide}\n(n={len(proportion_data)})",
            rotation=90,
            va="center",
            ha="center",
            fontsize=GUIDE_LABEL_FONTSIZE,
        )

    if first_line_ax is not None:
        first_line_ax.set_title(title, pad=12)
    xaxis_ax = fig.add_subplot(inner_spec[-1, 0])
    set_bottom_clone_index_ticks(xaxis_ax, max_x, xlabel)


def save_figure(fig, stem: str):
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def create_group_figure(guides, guide_data, title: str, xlabel: str, stem: str):
    total_slots = max(len(guides), 1)
    fig = plt.figure(figsize=(9, 2.25 * total_slots))
    outer_spec = fig.add_gridspec(1, 1)

    plot_guide_group(
        fig,
        outer_spec[0, 0],
        guides,
        guide_data,
        title,
        total_slots,
        xlabel,
    )

    save_figure(fig, stem)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    print("Building guide-level clone distributions...")
    guide_data = build_guide_data(adata)
    targeting_guides = [
        guide
        for guide in sorted(adata.obs["guide"].astype(str).value_counts().index)
        if guide in guide_data
        and guide not in {"", "nan", "None"}
        and not guide.startswith("NO_SITE")
        and not guide.startswith("INTERGENIC")
    ]
    control_guides = [
        guide
        for guide in sorted(adata.obs["guide"].astype(str).value_counts().index)
        if guide in guide_data
        and (guide.startswith("NO_SITE") or guide.startswith("INTERGENIC"))
    ]

    print(f"Targeting guides: {len(targeting_guides)}")
    print(f"Control guides: {len(control_guides)}")

    print("Creating visualization for panel a_1...")
    create_group_figure(
        targeting_guides,
        guide_data,
        TARGETING_TITLE,
        "clone index",
        "a_1",
    )

    print("Creating visualization for panel a_2...")
    create_group_figure(
        control_guides,
        guide_data,
        CONTROL_TITLE,
        "clone index",
        "a_2",
    )

    print("Figure S5 clone distribution panels completed successfully!")


if __name__ == "__main__":
    main()

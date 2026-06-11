#!/usr/bin/env python
"""
Generate source ordered guide-proportion panel: guide-level cell-type proportions in the
Trim28/Setdb1 dataset ordered by guide class.
"""

import os
import sys

import matplotlib.pyplot as plt
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
plt.rcParams['figure.figsize'] = (20, 8)

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_11_shared_source"


def validate_file_sizes(png_path: str, svg_path: str) -> None:
    """Report whether PNG/SVG outputs meet manuscript size requirements."""
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Consider simplifying or rasterizing.")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def order_by_numeric_suffix(guides: list[str], prefix: str) -> list[str]:
    """Order guides by numeric suffix to avoid lexicographic sorting."""
    ordered = []
    others = []
    prefix_lower = prefix.lower()
    for guide in guides:
        guide_lower = guide.lower()
        if guide_lower.startswith(prefix_lower):
            suffix = guide_lower.replace(prefix_lower, "", 1).lstrip("-_")
            try:
                ordered.append((int(suffix), guide))
            except ValueError:
                others.append(guide)
        else:
            others.append(guide)
    return [guide for _, guide in sorted(ordered)] + sorted(others)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trim28_setdb1 data...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=True,
        remove_NG=True,
        remove_lymph_node=False,
    )

    print("Computing guide-level cell-type proportions...")
    df = pd.DataFrame(
        {
            "cell_type": adata.obs["cell_type"].astype(str),
            "guide": adata.obs["guide"].astype(str),
        }
    )

    counts = df.groupby(["guide", "cell_type"]).size().unstack(fill_value=0)
    proportions = counts.divide(counts.sum(axis=1), axis=0)
    guide_cell_counts = df.groupby("guide").size()

    no_site_guides = order_by_numeric_suffix(
        [guide for guide in proportions.index if guide.upper().startswith("NO_SITE")],
        "no_site",
    )
    setdb1_guides = order_by_numeric_suffix(
        [guide for guide in proportions.index if guide.lower().startswith("setdb1")],
        "setdb1",
    )
    trim28_guides = order_by_numeric_suffix(
        [guide for guide in proportions.index if guide.lower().startswith("trim28")],
        "trim28",
    )
    other_guides = sorted(
        set(proportions.index) - set(no_site_guides) - set(setdb1_guides) - set(trim28_guides)
    )
    ordered_guides = no_site_guides + setdb1_guides + trim28_guides + other_guides
    ordered_proportions = proportions.loc[ordered_guides]

    cell_types = [
        cell_type
        for cell_type in cell_type_orders_trim28_setdb1
        if cell_type in ordered_proportions.columns
    ]
    ordered_proportions = ordered_proportions[cell_types[::-1]]

    print("Creating stacked bar chart with dotted class separators...")
    fig, ax_main = plt.subplots(figsize=(20, 8))
    ordered_proportions.plot(
        kind="bar",
        stacked=True,
        ax=ax_main,
        legend=False,
        color=[cell_type_colors[cell_type] for cell_type in ordered_proportions.columns],
        width=0.8,
    )

    ax_main.set_ylabel("Cell type proportion")
    ax_main.set_xlabel("")
    ax_main.set_ylim(0, 1)

    tick_labels = [
        f"{guide} (n={guide_cell_counts[guide]})"
        for guide in ordered_proportions.index
    ]
    ax_main.set_xticks(range(len(tick_labels)))
    plt.draw()
    labels = [label for label in ax_main.get_xticklabels()]
    for idx, label in enumerate(labels):
        label.set_text(tick_labels[idx])
        label.set_rotation(90)
        label.set_ha("center")
    ax_main.set_xticklabels(labels, fontsize=19.5)
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "b.png")
    svg_path = os.path.join(OUTPUT_DIR, "b.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(png_path, svg_path)
    print("source ordered guide-proportion panel completed successfully!")


if __name__ == "__main__":
    main()

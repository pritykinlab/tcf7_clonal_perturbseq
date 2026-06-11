#!/usr/bin/env python
"""
Temporary internal Sup_Figure_5_tmp panel.
UMAPs highlighting the three clones with the largest number of TRM-like cells.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "scripts" / "utils"))

from figure_format import get_adata, set_rc_params  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_5_tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CELL_TYPE = "TRM-like"
N_CLONES = 3
BACKGROUND_COLOR = (0.8, 0.8, 0.8)
BACKGROUND_SIZE = 3.0
HIGHLIGHT_ALPHA = 0.95
CLONE_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]

set_rc_params()
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


def load_data():
    print("Loading AnnData for temporary internal Sup_Figure_5_tmp panel...")
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=False,
    )
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates not found in adata.obsm['X_umap'].")
    return adata


def top_trm_clones(adata):
    print(f"Selecting top {N_CLONES} clones with {TARGET_CELL_TYPE} cells...")
    trm_mask = adata.obs["cell_type"] == TARGET_CELL_TYPE
    clone_series = adata.obs["clone"].astype(str)
    perturbation_series = adata.obs["perturbation"].astype(str)
    counts = (
        clone_series[trm_mask]
        .value_counts()
        .head(N_CLONES)
    )
    if counts.empty:
        raise ValueError(f"No clones contain cells labeled {TARGET_CELL_TYPE}.")
    info = []
    for color, (clone, count) in zip(CLONE_COLORS, counts.items()):
        clone_mask = clone_series == clone
        perturbation_counts = perturbation_series[clone_mask].value_counts()
        perturbation = (
            perturbation_counts.index[0]
            if not perturbation_counts.empty
            else "Unknown"
        )
        info.append(
            {
                "clone": clone,
                "count": int(count),
                "color": color,
                "perturbation": perturbation,
            }
        )
        print(f"  {clone}: {count} cells ({perturbation})")
    return info


def short_clone_label(clone: str) -> str:
    if "__" in clone:
        return clone.split("__")[-1]
    return clone


def plot_umaps(adata, clones_info):
    n = len(clones_info)
    fig, axes = plt.subplots(
        1, n, figsize=(4 * n, 4), squeeze=False
    )
    axes = axes[0]
    umap = adata.obsm["X_umap"]

    for ax, info in zip(axes, clones_info):
        clone = info["clone"]
        color = info["color"]

        clone_mask = adata.obs["clone"].astype(str) == clone
        trm_mask = (adata.obs["cell_type"] == TARGET_CELL_TYPE) & clone_mask

        ax.scatter(
            umap[:, 0],
            umap[:, 1],
            color=BACKGROUND_COLOR,
            s=BACKGROUND_SIZE,
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
        )

        if trm_mask.sum() > 0:
            ax.scatter(
                umap[trm_mask, 0],
                umap[trm_mask, 1],
                color=color,
                s=18,
                alpha=HIGHLIGHT_ALPHA,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_title(
            f"{info['perturbation']}: {short_clone_label(clone)}\nTRM-like n={info['count']}",
            fontsize=17,
            pad=4,
        )
        ax.axis("off")

    fig.tight_layout()
    return fig


def save(fig, stem: str):
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", dpi=85, bbox_inches="tight")


def main():
    adata = load_data()
    clones = top_trm_clones(adata)
    fig = plot_umaps(adata, clones)
    save(fig, "a")
    plt.close(fig)
    print("Temporary internal Sup_Figure_5_tmp panel generated.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate Figure S13c: multiome PCA-space distances from TRM-like cells.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.append('../scripts/utils')

from figure_format import get_adata  # noqa: E402


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 27
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (12.8, 5.4)


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_13"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "c.png")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "c.svg")
REFERENCE_CELL_TYPE = "TRM-like"
BAR_COLOR = "#8c8c8c"

REPRESENTATIONS = (
    ("X_pca_rna", 30, "RNA PCA"),
    ("X_pca_atac", 40, "ATAC PCA"),
)


def compute_average_distance(adata, obsm_key, n_pcs, reference_cell_type=REFERENCE_CELL_TYPE):
    """Compute mean Euclidean distance from TRM-like cells to each other cell state."""
    if obsm_key not in adata.obsm:
        raise ValueError(f"Expected embedding {obsm_key!r} in adata.obsm.")

    obs = adata.obs.copy()
    obs["cell_type"] = obs["cell_type"].astype(str)
    coords = np.asarray(adata.obsm[obsm_key][:, :n_pcs])

    ref_mask = obs["cell_type"] == reference_cell_type
    if ref_mask.sum() == 0:
        raise ValueError(f"No cells found for reference cell type {reference_cell_type!r}.")

    ref_coords = coords[ref_mask.to_numpy()]
    rows = []
    for cell_type in obs["cell_type"].unique():
        if cell_type == reference_cell_type:
            continue

        cell_type_mask = (obs["cell_type"] == cell_type).to_numpy()
        pairwise_distances = cdist(ref_coords, coords[cell_type_mask], metric="euclidean")
        rows.append(
            {
                "cell_type": cell_type,
                "distance": float(pairwise_distances.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def plot_distances(ax, distance_df, title):
    """Render a single bar plot for one PCA representation."""
    ax.bar(distance_df["cell_type"], distance_df["distance"], color=BAR_COLOR, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Average distance\n(PCA space)")
    ax.set_ylim(0, distance_df["distance"].max() * 1.12)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def validate_file_sizes(png_path, svg_path):
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading multiome RNA data...")
    adata = get_adata("multiome_rna")

    print("Restricting to Trim28 cells...")
    trim28_adata = adata[adata.obs["condition"] == "Trim28"].copy()
    print(f"Using {trim28_adata.n_obs} Trim28 cells")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))

    for ax, (obsm_key, n_pcs, title) in zip(axes, REPRESENTATIONS):
        print(f"Computing distances for {obsm_key} using first {n_pcs} PCs...")
        distance_df = compute_average_distance(trim28_adata, obsm_key=obsm_key, n_pcs=n_pcs)
        print(distance_df.to_string(index=False))
        plot_distances(ax, distance_df, title)

    fig.suptitle(
        "Average distance (in PCA space) from TRM-like Trim28 cells\n"
        "to Trim28 cells of each state",
        fontsize=24,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.78, wspace=0.62)

    print("Saving figures...")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(OUTPUT_PNG, OUTPUT_SVG)
    print("Figure S13c completed successfully!")


if __name__ == "__main__":
    main()

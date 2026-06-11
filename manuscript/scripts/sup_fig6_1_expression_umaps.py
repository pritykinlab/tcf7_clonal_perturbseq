#!/usr/bin/env python
"""
Generate a marker-gene UMAP grid for Figure S13a.
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import issparse

sys.path.append('../scripts/utils')

from figure_format import get_adata, dotplot_genes


plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["lines.markersize"] = 5
plt.rcParams["axes.grid"] = False
plt.rcParams["xtick.bottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["ytick.major.size"] = 0
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "#c7c7c7"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.title_fontsize"] = 10
plt.rcParams["legend.edgecolor"] = "dimgray"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_13"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "a.png")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "a.svg")
UMAP_KEY = "X_umap"
LAYER = "norm_log"
N_COLUMNS = 9
POINT_SIZE = 3
BACKGROUND_COLOR = "#ececec"
EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "multiome_expression_umap",
    [BACKGROUND_COLOR, "#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"],
)


def extract_expression(layer, gene_index):
    values = layer[:, gene_index]
    if issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#c7c7c7")


def format_tick_label(value):
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def add_expression_legend(ax, vmin, vmax):
    legend_ax = ax.inset_axes([1.02, 0.08, 0.07, 0.82])
    gradient = np.linspace(vmin, vmax, 128).reshape(-1, 1)
    legend_ax.imshow(
        gradient,
        cmap=EXPRESSION_CMAP,
        aspect="auto",
        origin="lower",
        extent=(0, 1, vmin, vmax),
    )
    mid = (vmin + vmax) / 2
    legend_ax.set_xticks([])
    legend_ax.yaxis.tick_right()
    legend_ax.set_yticks([vmin, mid, vmax])
    legend_ax.set_yticklabels(
        [format_tick_label(vmin), format_tick_label(mid), format_tick_label(vmax)]
    )
    legend_ax.tick_params(axis="y", labelsize=4.5, length=1.5, pad=1)
    for spine in legend_ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#c7c7c7")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading multiome RNA data...")
    adata = get_adata("multiome_rna")

    genes = [gene for gene in dotplot_genes if gene in adata.var_names]
    if len(genes) != len(dotplot_genes):
        missing = sorted(set(dotplot_genes) - set(genes))
        print(f"Skipping missing genes: {missing}")

    if UMAP_KEY not in adata.obsm:
        raise ValueError(f"UMAP coordinates not found in adata.obsm['{UMAP_KEY}']")
    if LAYER not in adata.layers:
        raise ValueError(f"Expected expression layer '{LAYER}' not found in adata.layers")

    coords = adata.obsm[UMAP_KEY]
    layer = adata.layers[LAYER]

    n_rows = math.ceil(len(genes) / N_COLUMNS)
    fig_width = 18.5
    fig_height = 1.55 * n_rows
    fig, axes = plt.subplots(n_rows, N_COLUMNS, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).ravel()

    print(f"Rendering {len(genes)} marker UMAPs...")
    for ax, gene in zip(axes, genes):
        gene_index = adata.var_names.get_loc(gene)
        expression = extract_expression(layer, gene_index)

        finite_values = expression[np.isfinite(expression)]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(finite_values, 2))
            vmax = float(np.percentile(finite_values, 99))
            if vmax <= vmin:
                vmax = vmin + 1e-6

        order = np.argsort(expression)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=expression[order],
            s=POINT_SIZE,
            cmap=EXPRESSION_CMAP,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(gene, pad=3)
        style_axis(ax)
        add_expression_legend(ax, vmin, vmax)

    for ax in axes[len(genes):]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.035,
        right=0.992,
        top=0.992,
        bottom=0.03,
        wspace=0.22,
        hspace=0.24,
    )

    print("Saving figures...")
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, dpi=35, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = os.path.getsize(OUTPUT_PNG) / 1024 / 1024
    svg_size = os.path.getsize(OUTPUT_SVG) / 1024
    print(f"Saved {OUTPUT_PNG} ({png_size:.2f} MB)")
    print(f"Saved {OUTPUT_SVG} ({svg_size:.1f} KB)")


if __name__ == "__main__":
    main()
